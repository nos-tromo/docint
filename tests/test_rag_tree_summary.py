"""Tests for the tree-summary orchestration layer on ``RAG``.

Exercises the seam between the pure map-reduce pipeline
(:mod:`docint.core.summary.tree`) and the real Qdrant client, chat model, and
persistent per-collection map cache: ``RAG.build_tree_summary`` and
``RAG.cached_collection_summary``. Follows the fixture approach already
proven in ``tests/test_summary_image_evidence.py`` — a fake Qdrant client and
a stubbed chat model, no network, no real inference.
"""

from __future__ import annotations

import types
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from typing_extensions import override

from docint.core.rag import RAG

DOC_ONE_TEXT = "Alpha bravo charlie delta echo foxtrot golf hotel."
DOC_TWO_TEXT = "Indigo juliet kilo lima mike november oscar papa."

# The fixture overrides rag.summary_map_prompt / summary_fold_prompt with
# these fixed, marker-bearing templates rather than relying on whichever
# locale's real prompt file __post_init__ happened to load (RESPONSE_LANGUAGE
# is not cleared by the hermetic-env fixture, unlike ENABLE_HYBRID) — the
# stub model needs a stable, language-independent way to tell map calls,
# fold calls, and the final synthesis call apart.
_MAP_PROMPT_MARKER = "TEST_MAP_PROMPT_MARKER"
_FOLD_PROMPT_MARKER = "TEST_FOLD_PROMPT_MARKER"
_MAP_PROMPT_TEMPLATE = f"{_MAP_PROMPT_MARKER}\nUnit: {{label}}\n\n{{chunk_block}}\n\nEVIDENCE_INDICES: 1,2"
_FOLD_PROMPT_TEMPLATE = f"{_FOLD_PROMPT_MARKER}\n\n{{summaries_block}}"


class _StubCompletion:
    """Minimal stand-in for a llama-index ``CompletionResponse``."""

    def __init__(self, text: str) -> None:
        """Store the canned response text.

        Args:
            text: The text ``.complete()`` should appear to have returned.
        """
        self.text = text


class _StubTextModel:
    """Fake chat model that classifies calls by prompt content and counts them."""

    def __init__(self) -> None:
        """Initialize an empty call log."""
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> _StubCompletion:
        """Return a canned response appropriate to the prompt's stage.

        Args:
            prompt: The fully rendered prompt text.

        Returns:
            _StubCompletion: A map-stage summary (with an ``EVIDENCE_INDICES``
            line), a folded summary, or a final synthesized response.
        """
        self.prompts.append(prompt)
        if _MAP_PROMPT_MARKER in prompt:
            return _StubCompletion("Synthetic unit summary.\nEVIDENCE_INDICES: 1")
        if _FOLD_PROMPT_MARKER in prompt:
            return _StubCompletion("Synthetic folded summary.")
        return _StubCompletion("Synthetic final response covering both documents.")

    @property
    def map_call_count(self) -> int:
        """Number of calls whose prompt was a map-stage call.

        Returns:
            int: Count of logged prompts containing the map-prompt marker.
        """
        return sum(1 for p in self.prompts if _MAP_PROMPT_MARKER in p)


class _FakeQdrant:
    """Qdrant stub serving a fixed set of points for scroll and retrieve.

    ``collection_exists`` always reports ``False`` so the images-companion
    lane in ``_summary_image_nodes_for_document`` declines immediately —
    these tests are about the tree-summary orchestration, not multimodal
    evidence (covered separately by ``test_summary_image_evidence.py``).
    """

    def __init__(self, points: list[Any], *, reverse_retrieve: bool = False) -> None:
        """Store the fixed point set.

        Args:
            points: Points the main-collection scroll returns.
            reverse_retrieve: When ``True``, ``retrieve()`` returns matching
                points in the reverse of the requested id order — Qdrant's
                real ``retrieve()`` makes no ordering promise at all, and
                this flag exists to prove callers do not rely on one.
        """
        self._points = points
        self._by_id = {str(p.id): p for p in points}
        self.retrieve_calls: list[list[str]] = []
        self._reverse_retrieve = reverse_retrieve

    def collection_exists(self, collection_name: str) -> bool:
        """Report no companion collections as present.

        Args:
            collection_name: The collection being probed.

        Returns:
            bool: Always ``False``.
        """
        return False

    def scroll(
        self,
        *,
        collection_name: str,
        limit: int = 256,
        offset: Any = None,
        scroll_filter: Any = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[Any], Any]:
        """Return all points on the first page, then signal exhaustion.

        Args:
            collection_name: Collection being scrolled (ignored).
            limit: Page size (ignored — the fixture is small enough for one page).
            offset: Continuation offset from a previous call.
            scroll_filter: Optional Qdrant filter (ignored).
            with_payload: Whether payloads were requested (ignored).
            with_vectors: Whether vectors were requested (ignored).

        Returns:
            tuple[list[Any], Any]: The points and a null continuation offset
            on the first call; ``([], None)`` on any subsequent call.
        """
        if offset is not None:
            return ([], None)
        return (list(self._points), None)

    def retrieve(
        self,
        *,
        collection_name: str,
        ids: list[Any],
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[Any]:
        """Return the stored points matching ``ids``.

        Real Qdrant makes no promise that the response is ordered like the
        requested ``ids``; when ``reverse_retrieve`` is set this stub
        deliberately returns matches in the opposite order to catch callers
        that assume otherwise.

        Args:
            collection_name: Collection being retrieved from (ignored).
            ids: Point ids to fetch.
            with_payload: Whether payloads were requested (ignored).
            with_vectors: Whether vectors were requested (ignored).

        Returns:
            list[Any]: The matching points.
        """
        str_ids = [str(i) for i in ids]
        self.retrieve_calls.append(str_ids)
        matched = [self._by_id[i] for i in str_ids if i in self._by_id]
        return list(reversed(matched)) if self._reverse_retrieve else matched


def _two_document_points() -> list[Any]:
    """Build two synthetic single-chunk documents as fake Qdrant points.

    Returns:
        list[Any]: Two points, each carrying a distinct ``file_hash`` /
        ``filename`` / ``text`` — exactly what ``partition_units`` needs to
        form two ``document``-kind units.
    """
    return [
        types.SimpleNamespace(
            id="pt-doc-one",
            payload={
                "file_hash": "hash-aaa",
                "filename": "doc-one.txt",
                "text": DOC_ONE_TEXT,
            },
        ),
        types.SimpleNamespace(
            id="pt-doc-two",
            payload={
                "file_hash": "hash-bbb",
                "filename": "doc-two.txt",
                "text": DOC_TWO_TEXT,
            },
        ),
    ]


def _build_rag(
    tmp_path: Path,
    *,
    points: list[Any],
    collection: str = "tree-summary-fixture",
    reverse_retrieve: bool = False,
) -> RAG:
    """Build a ``RAG`` wired to a fake Qdrant collection over ``points``.

    Mirrors the construction style of ``test_summary_image_evidence.py``: a
    bare ``RAG(qdrant_collection=...)`` with its Qdrant client and image
    service swapped for stubs. The map cache and final summary cache are
    backed by a real ``SQLiteKVStore`` rooted at ``tmp_path`` (via
    ``rag._qdrant_src_dir``), so cache round-trips in these tests are genuine
    rather than mocked.

    Args:
        tmp_path: Pytest's per-test temporary directory.
        points: Points the fake Qdrant client's scroll/retrieve serve.
        collection: The (logical == physical here) collection name.
        reverse_retrieve: Forwarded to ``_FakeQdrant`` — see its docstring.

    Returns:
        RAG: A ready-to-use instance.
    """
    rag = RAG(qdrant_collection=collection)
    rag._qdrant_client = cast(Any, _FakeQdrant(points, reverse_retrieve=reverse_retrieve))
    rag._qdrant_src_dir = tmp_path
    rag._post_retrieval_text_model = cast(Any, _StubTextModel())
    # The stub sits in the reasoning slot, which ``post_retrieval_text_model``
    # only selects when thinking is on (env default or request override) —
    # so opt the env default in, or the property falls through to a real
    # ``text_model`` and the build dials out.
    rag.openai_config = replace(rag.openai_config, thinking_enabled=True)
    # Deterministic, locale-independent map/fold prompts (see the module
    # docstring comment above the marker constants).
    rag.summary_map_prompt = _MAP_PROMPT_TEMPLATE
    rag.summary_fold_prompt = _FOLD_PROMPT_TEMPLATE
    rag._image_ingestion_service = cast(
        Any,
        types.SimpleNamespace(
            _resolve_collection_name=lambda source_collection=None: f"{source_collection}_images",
            img_ingestion_config=types.SimpleNamespace(rerank_min_score=0.05, retrieve_top_k=5),
        ),
    )
    return rag


@pytest.fixture
def rag_with_fake_collection(tmp_path: Path) -> RAG:
    """Build a ``RAG`` wired to a fake two-document Qdrant collection.

    Args:
        tmp_path: Pytest's per-test temporary directory.

    Returns:
        RAG: A ready-to-use instance with two synthetic documents.
    """
    return _build_rag(tmp_path, points=_two_document_points())


def test_build_tree_summary_end_to_end(rag_with_fake_collection: RAG) -> None:
    """Two synthetic documents -> two map calls -> synthesis; payload shape intact."""
    rag = rag_with_fake_collection

    payload = rag.build_tree_summary()

    assert payload["response"]
    diag = payload["summary_diagnostics"]
    assert diag["total_documents"] == 2
    assert diag["covered_documents"] == 2
    assert diag["coverage_ratio"] == 1.0
    assert diag["coverage_unit"] == "documents"
    assert diag["partial"] is False
    assert payload["sources"], "evidence sources must be populated"
    assert payload["sources"][0]["citation_index"] == 1

    stub = cast(_StubTextModel, rag._post_retrieval_text_model)
    assert stub.map_call_count == 2


def test_build_tree_summary_incremental_uses_map_cache(rag_with_fake_collection: RAG) -> None:
    """Second build with unchanged content performs zero map LLM calls."""
    rag = rag_with_fake_collection

    first_payload = rag.build_tree_summary()
    assert first_payload["summary_diagnostics"]["partial"] is False

    # Fresh stub so the second build's call log starts empty; the map cache
    # lives in the SQLite KV store on disk, not on the stub, so this proves
    # the cache — not stub state — is what's carrying the map results.
    second_stub = _StubTextModel()
    rag._post_retrieval_text_model = cast(Any, second_stub)

    second_payload = rag.build_tree_summary()

    assert second_stub.map_call_count == 0
    # The synthesis call still happens every build (it is not cached itself).
    assert len(second_stub.prompts) == 1
    assert second_payload["summary_diagnostics"]["covered_documents"] == 2
    assert second_payload["response"]


def test_partial_build_is_cached_and_keeps_its_flag(rag_with_fake_collection: RAG) -> None:
    """A capped build is cached, flag intact, so /summarize can serve it.

    Withholding a completed-but-partial build from the cache made it
    unreachable: ``POST /summarize`` answers 200 only from this cache, so the
    client's post-completion refetch missed and silently queued another full
    build. Honesty is carried by ``partial`` surviving the round-trip, not by
    refusing to cache.
    """
    import dataclasses

    rag = rag_with_fake_collection
    rag.summary_config = dataclasses.replace(rag.summary_config, max_llm_calls=1)

    payload = rag.build_tree_summary()

    diag = payload["summary_diagnostics"]
    assert diag["partial"] is True
    assert diag["covered_documents"] == 1
    assert diag["total_documents"] == 2

    cached = rag.cached_collection_summary()
    assert cached is not None
    assert cached["response"] == payload["response"]
    assert cached["summary_diagnostics"]["partial"] is True
    assert cached["summary_diagnostics"]["covered_documents"] == 1


def test_build_tree_summary_zero_covered_is_flagged_partial(rag_with_fake_collection: RAG) -> None:
    """A non-empty collection where every unit fails to map is flagged partial.

    ``covered_units == 0`` produces the bare "unable to extract grounded
    evidence" ``response_text`` with no per-unit diagnostics to explain it.
    Without ``partial`` set, ``CoverageBanner`` has nothing to flag and the
    non-answer looks like a normal, complete summary until Refresh or the
    next revision bump. This scenario reaches zero coverage via per-unit map
    failures (not the LLM-call cap), so ``TreeSummaryResult.partial`` on its
    own is ``False`` — the orchestration layer must force the flag itself.
    """
    rag = rag_with_fake_collection

    class _AllMapsFailModel:
        """Raises on every map-stage call; answers the final synthesis call."""

        def complete(self, prompt: str) -> _StubCompletion:
            """Fail every map call, succeed on the exempt synthesis call.

            Args:
                prompt: The fully rendered prompt text.

            Returns:
                _StubCompletion: A canned final-response text for any
                non-map prompt.

            Raises:
                RuntimeError: For any map-stage prompt.
            """
            if _MAP_PROMPT_MARKER in prompt:
                raise RuntimeError("map model unreachable")
            return _StubCompletion("Synthetic final response.")

    rag._post_retrieval_text_model = cast(Any, _AllMapsFailModel())

    payload = rag.build_tree_summary()

    diag = payload["summary_diagnostics"]
    assert diag["covered_documents"] == 0
    assert diag["total_documents"] == 2
    assert diag["partial"] is True
    assert payload["response"] == "Unable to extract grounded evidence from the selected collection."


def test_build_tree_summary_truncated_unit_not_double_counted(tmp_path: Path) -> None:
    """A cap-truncated unit is covered exactly once, not also listed as uncovered.

    ``_KVMapCache`` deliberately never writes a truncated unit's result to
    the map cache (it would be stored under the unit's *full* content
    fingerprint and served as complete on every later build), so
    ``covered_keys`` — which only tracks cache get/put activity — never
    learns about it. Deriving ``uncovered_documents`` from ``covered_keys``
    therefore both counted the truncated unit in ``covered_documents`` (it
    has a ``UnitMapResult``) AND listed its label in ``uncovered_documents``
    (the cache never saw it) — a self-contradictory "N/N covered · show M
    uncovered" banner.
    """
    import dataclasses

    points = [
        types.SimpleNamespace(
            id="pt-big-1",
            payload={"file_hash": "hash-aaa-big", "filename": "big-doc.txt", "text": "A" * 30},
        ),
        types.SimpleNamespace(
            id="pt-big-2",
            payload={"file_hash": "hash-aaa-big", "filename": "big-doc.txt", "text": "B" * 30},
        ),
        types.SimpleNamespace(
            id="pt-small",
            payload={"file_hash": "hash-zzz-small", "filename": "small-doc.txt", "text": "small doc text"},
        ),
    ]
    rag = _build_rag(tmp_path, points=points)
    # window_chars = map_window_tokens * 4 = 40: the big doc's two 30-char
    # chunks cannot share one window (30 + 30 > 40), so mapping it costs two
    # windows/calls; capping the budget at 1 call truncates it mid-unit
    # before the small doc (sorted after it by unit_key) is ever attempted.
    rag.summary_config = dataclasses.replace(rag.summary_config, map_window_tokens=10, max_llm_calls=1)

    payload = rag.build_tree_summary()

    diag = payload["summary_diagnostics"]
    assert diag["partial"] is True
    assert diag["covered_documents"] == 1
    assert diag["total_documents"] == 2
    assert "big-doc.txt" not in diag["uncovered_documents"]
    assert "small-doc.txt" in diag["uncovered_documents"]


def test_empty_build_is_cached(tmp_path: Path) -> None:
    """An empty collection's build is cached, so it stops re-queueing forever."""
    rag = _build_rag(tmp_path, points=[])

    payload = rag.build_tree_summary()

    cached = rag.cached_collection_summary()
    assert cached is not None
    assert cached["response"] == payload["response"]
    assert cached["summary_diagnostics"]["total_documents"] == 0
    assert cached["summary_diagnostics"]["partial"] is False


def test_cached_collection_summary_roundtrip(rag_with_fake_collection: RAG) -> None:
    """After a full build, cached_collection_summary() returns the stored payload."""
    rag = rag_with_fake_collection

    built = rag.build_tree_summary()
    cached = rag.cached_collection_summary()

    assert cached is not None
    assert cached["response"] == built["response"]
    assert cached["summary_diagnostics"]["covered_documents"] == built["summary_diagnostics"]["covered_documents"]
    assert [s.get("chunk_id") for s in cached["sources"]] == [s.get("chunk_id") for s in built["sources"]]


def test_build_tree_summary_revision_bump_invalidates_cache(rag_with_fake_collection: RAG) -> None:
    """Bumping the summary revision invalidates a previously cached tree summary.

    ``_bump_summary_revision`` / ``_load_cached_collection_summary`` are
    shared cache/revision machinery retained from the now-removed sampling
    summarizer; this exercises them against ``build_tree_summary``, its
    map-reduce successor.
    """
    rag = rag_with_fake_collection

    rag.build_tree_summary()
    assert rag.cached_collection_summary() is not None

    rag._bump_summary_revision()

    assert rag.cached_collection_summary() is None


def test_fingerprint_includes_map_prompt(rag_with_fake_collection: RAG) -> None:
    """Mutating rag.summary_map_prompt changes _summary_prompt_fingerprint()."""
    rag = rag_with_fake_collection

    before = rag._summary_prompt_fingerprint()
    rag.summary_map_prompt = rag.summary_map_prompt + "\nEXTRA INSTRUCTION"
    after = rag._summary_prompt_fingerprint()

    assert before != after


def test_cached_collection_summary_requires_selected_collection() -> None:
    """cached_collection_summary() raises the same guard build_tree_summary() uses."""
    rag = RAG(qdrant_collection="")

    with pytest.raises(ValueError, match="No collection selected"):
        rag.cached_collection_summary()


def test_build_tree_summary_requires_collection() -> None:
    """build_tree_summary() raises ValueError if no collection is selected."""
    rag = RAG(qdrant_collection="")

    with pytest.raises(ValueError, match="No collection selected"):
        rag.build_tree_summary()


def test_build_tree_summary_sources_follow_evidence_order_despite_retrieve_reordering(tmp_path: Path) -> None:
    """Sources are ordered by covered-unit evidence order, not retrieve()'s response order.

    Qdrant's ``retrieve()`` does not promise its response is ordered like the
    requested ids — the same invariant ``_fetch_unit_chunks`` already guards
    against by re-indexing before use. The fake client here is configured to
    return points in the *reverse* of the requested order, which would have
    swapped ``doc-one``/``doc-two`` and misnumbered their citations under the
    naive "append points as retrieve() returns them" implementation.
    """
    rag = _build_rag(tmp_path, points=_two_document_points(), reverse_retrieve=True)

    payload = rag.build_tree_summary()

    sources = payload["sources"]
    assert [source.get("filename") for source in sources] == ["doc-one.txt", "doc-two.txt"]
    assert [source.get("citation_index") for source in sources] == [1, 2]


class _FlakyScrollQdrant(_FakeQdrant):
    """Qdrant stub whose scroll serves one page and then fails.

    Models the real failure C1 describes: a healthy first page, then a
    transport blip partway through a long scroll.
    """

    def __init__(self, points: list[Any]) -> None:
        """Store the points and split them across two scroll pages.

        Args:
            points: Points the scroll would serve if it stayed healthy.
        """
        super().__init__(points)
        self.scroll_calls = 0

    @override
    def scroll(
        self,
        *,
        collection_name: str,
        limit: int = 256,
        offset: Any = None,
        scroll_filter: Any = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[Any], Any]:
        """Return the first point, then raise on the next page.

        Args:
            collection_name: Collection being scrolled (ignored).
            limit: Page size (ignored).
            offset: Continuation offset from a previous call.
            scroll_filter: Optional Qdrant filter (ignored).
            with_payload: Whether payloads were requested (ignored).
            with_vectors: Whether vectors were requested (ignored).

        Returns:
            tuple[list[Any], Any]: The first page and a non-null continuation
            offset.

        Raises:
            RuntimeError: On every call after the first.
        """
        self.scroll_calls += 1
        if self.scroll_calls == 1:
            return ([self._points[0]], "page-2")
        raise RuntimeError("qdrant scroll blip")


def test_build_tree_summary_raises_when_scroll_fails_midway(tmp_path: Path) -> None:
    """A mid-scroll Qdrant failure fails the build instead of shrinking the universe.

    ``iter_scroll``'s default ``on_error="warn"`` logs and stops *cleanly*, so
    the truncated point set would look like the whole collection:
    ``partition_units`` sees only the pages that arrived, every unit maps, and
    the build reports ``coverage_ratio: 1.0``/``partial: False`` before caching
    a fraction of the collection as complete. The summary scroll therefore
    raises.
    """
    rag = _build_rag(tmp_path, points=_two_document_points())
    flaky = _FlakyScrollQdrant(_two_document_points())
    rag._qdrant_client = cast(Any, flaky)

    with pytest.raises(RuntimeError, match="qdrant scroll blip"):
        rag.build_tree_summary()

    assert flaky.scroll_calls >= 2
    # Nothing partial was published as the collection's summary.
    assert rag.cached_collection_summary() is None


def test_build_tree_summary_skips_cache_when_revision_moves_mid_build(rag_with_fake_collection: RAG) -> None:
    """An ingest landing mid-build must not have the stale summary stamped current.

    Summary and ingest jobs for one collection run concurrently by design. If
    the cache write read the revision at write time, a build that started at
    revision R would be stamped with the R+1 an ingest bumped it to mid-build,
    overwriting the newer summary and validating as current forever.
    """
    rag = rag_with_fake_collection
    stub = cast(_StubTextModel, rag._post_retrieval_text_model)
    original_complete = stub.complete
    bumped: list[int] = []

    def _complete_and_bump(prompt: str) -> _StubCompletion:
        """Simulate a concurrent ingest completing during the final synthesis call."""
        if _MAP_PROMPT_MARKER not in prompt and _FOLD_PROMPT_MARKER not in prompt and not bumped:
            bumped.append(rag._bump_summary_revision())
        return original_complete(prompt)

    rag._post_retrieval_text_model = cast(Any, types.SimpleNamespace(complete=_complete_and_bump))

    payload = rag.build_tree_summary()

    assert bumped, "the test must actually have bumped the revision mid-build"
    assert payload["response"], "the caller still receives its own build's payload"
    # The stale build did not become the served summary.
    assert rag.cached_collection_summary() is None


def test_build_tree_summary_empty_collection(tmp_path: Path) -> None:
    """An empty collection yields the sampling path's "no documents" message and coverage_unit."""
    rag = _build_rag(tmp_path, points=[])

    payload = rag.build_tree_summary()

    assert payload["response"] == "No documents available in the selected collection."
    assert payload["sources"] == []
    diag = payload["summary_diagnostics"]
    assert diag["total_documents"] == 0
    assert diag["covered_documents"] == 0
    # No units exist to derive a kind from; "documents" is the sensible
    # default for an empty document collection, rather than the meaningless
    # "units" a bare `else` branch would fall through to.
    assert diag["coverage_unit"] == "documents"
    assert diag["partial"] is False
