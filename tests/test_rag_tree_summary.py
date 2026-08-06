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
from pathlib import Path
from typing import Any, cast

import pytest

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


def test_partial_build_not_cached(rag_with_fake_collection: RAG) -> None:
    """With SUMMARY_MAX_LLM_CALLS=1 the result is partial and cached_collection_summary() stays None."""
    import dataclasses

    rag = rag_with_fake_collection
    rag.summary_config = dataclasses.replace(rag.summary_config, max_llm_calls=1)

    payload = rag.build_tree_summary()

    diag = payload["summary_diagnostics"]
    assert diag["partial"] is True
    assert diag["covered_documents"] == 1
    assert diag["total_documents"] == 2

    assert rag.cached_collection_summary() is None


def test_cached_collection_summary_roundtrip(rag_with_fake_collection: RAG) -> None:
    """After a full build, cached_collection_summary() returns the stored payload."""
    rag = rag_with_fake_collection

    built = rag.build_tree_summary()
    cached = rag.cached_collection_summary()

    assert cached is not None
    assert cached["response"] == built["response"]
    assert cached["summary_diagnostics"]["covered_documents"] == built["summary_diagnostics"]["covered_documents"]
    assert [s.get("chunk_id") for s in cached["sources"]] == [s.get("chunk_id") for s in built["sources"]]


def test_fingerprint_includes_map_prompt(rag_with_fake_collection: RAG) -> None:
    """Mutating rag.summary_map_prompt changes _summary_prompt_fingerprint()."""
    rag = rag_with_fake_collection

    before = rag._summary_prompt_fingerprint()
    rag.summary_map_prompt = rag.summary_map_prompt + "\nEXTRA INSTRUCTION"
    after = rag._summary_prompt_fingerprint()

    assert before != after


def test_cached_collection_summary_requires_selected_collection() -> None:
    """cached_collection_summary() raises the same guard summarize_collection() uses."""
    rag = RAG(qdrant_collection="")

    with pytest.raises(ValueError, match="No collection selected"):
        rag.cached_collection_summary()


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
    # default (matching what summarize_collection reports for an empty
    # document collection) rather than the meaningless "units" a bare
    # `else` branch would fall through to.
    assert diag["coverage_unit"] == "documents"
    assert diag["partial"] is False
