"""Relevance gating for CLIP-retrieved image sources.

CLIP text->image cosine similarity is not calibrated across queries: on the live
stack an unrelated query and a genuinely matching one both land in a ~0.20-0.30
band, so the top-k nearest images were surfaced for *every* question. These tests
pin the two-stage fix:

* the CLIP tower is English-only, so the query is translated before embedding;
* the multilingual reranker scores the image captions and a floor drops the
  ones that are merely nearest rather than relevant.

The stages now live in different places. The image lane
(``RAG._retrieve_image_nodes``) produces caption nodes; the query engine's own
reranker scores them against the text chunks; and
``ImageRelevanceFloorPostprocessor`` applies the floor. ``_lane_through_rerank``
below replays exactly that chain.

One case from the pre-fusion design is deliberately gone: the gate used to call
the reranker a second time, outside the query engine, and caught exceptions from
that extra call. There is now a single rerank -- the one text sources already
went through -- so an image-specific raise is no longer a distinct failure mode.
The silent-degradation path (``VLLMRerankPostprocessor`` swallowing its own
transport error and returning the nodes with their retrieval scores, stamped as
not reranked) is still covered below, because that is the one the live stack
actually takes.
"""

from __future__ import annotations

import json
import types
import urllib.error
import urllib.request
from typing import Any, cast

import pytest
from _pytest.logging import LogCaptureFixture
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from typing_extensions import override

import docint.utils.image_description as image_description
from docint.core.rag import (
    IMAGE_LANE_METADATA_KEY,
    RAG,
    RERANK_APPLIED_KEY,
    RERANK_ERROR_KEY,
    ImageRelevanceFloorPostprocessor,
    VLLMRerankPostprocessor,
)
from docint.utils.translate_client import TranslateResult


class StubImageService:
    """Image service stub that records the query text CLIP was asked to embed."""

    def __init__(self, matches: list[dict[str, Any]] | None = None) -> None:
        """Store the canned matches and prepare the call recorder.

        Args:
            matches: Payload dicts returned from a text->image query.
        """
        self.matches = matches if matches is not None else []
        self.seen_query_text: str | None = None
        self.img_ingestion_config = types.SimpleNamespace(rerank_min_score=0.05, retrieve_top_k=5)

    def _resolve_collection_name(self, source_collection: str | None = None) -> str:
        """Return the images companion name for a source collection.

        Args:
            source_collection: The base collection name.
        """
        return f"{source_collection}_images"

    def query_similar_images_by_text(
        self,
        query_text: str,
        top_k: int = 3,
        *,
        source_collection: str | None = None,
    ) -> list[dict[str, Any]]:
        """Record the embedded query text and return the canned matches.

        Args:
            query_text: Text handed to the CLIP text tower.
            top_k: Requested match count.
            source_collection: The base collection name.
        """
        self.seen_query_text = query_text
        return list(self.matches)


class StubReranker:
    """Reranker stub returning caller-supplied scores keyed by caption text."""

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        """Store the caption->score map and prepare the call recorder.

        Args:
            scores_by_text: Reranker relevance score for each caption.
        """
        self.scores_by_text = scores_by_text
        self.seen_query: str | None = None

    def postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Any = None,
    ) -> list[NodeWithScore]:
        """Score each node from the caption map, recording the rerank query.

        Args:
            nodes: Caption nodes built from the CLIP matches.
            query_bundle: Bundle carrying the rerank query string.
        """
        self.seen_query = getattr(query_bundle, "query_str", None)
        return [NodeWithScore(node=n.node, score=self.scores_by_text.get(n.node.get_content(), 0.0)) for n in nodes]


def _build_rag(
    *,
    image_service: StubImageService,
    reranker: Any = None,
    locale: str = "en",
) -> RAG:
    """Assemble a RAG wired to stubs, with the images collection present.

    Args:
        image_service: The stub CLIP-backed image service.
        reranker: Optional stub reranker.
        locale: Active ``RESPONSE_LANGUAGE`` for the instance. Defaults to
            ``en``, which short-circuits translation so tests that are not
            about translation never reach for the chat model.
    """
    rag = RAG(qdrant_collection="testbatch")
    rag.language_code = locale
    rag._image_ingestion_service = cast(Any, image_service)
    rag._qdrant_client = cast(Any, types.SimpleNamespace(collection_exists=lambda collection_name: True))
    if reranker is not None:
        rag.rerank_model_id = "stub-reranker"
        rag._reranker = cast(Any, reranker)
    return rag


def _lane_through_rerank(rag: RAG, query: str, *, top_k: int = 2) -> list[dict[str, Any]]:
    """Run the image lane the way the query engine does, and normalize.

    Replays retrieve -> rerank -> relevance floor, then normalizes what
    survives, so a test can assert on the sources a reader would actually see.

    Args:
        rag: The RAG under test.
        query: The user's original query.
        top_k: CLIP candidate depth.

    Returns:
        list[dict[str, Any]]: Normalized image sources that survived the floor.
    """
    nodes = rag._retrieve_image_nodes(query, top_k=top_k)
    if rag.rerank_model_id:
        nodes = rag.reranker.postprocess_nodes(nodes, QueryBundle(query_str=query))
    kept = ImageRelevanceFloorPostprocessor(min_score=rag._image_relevance_floor())._postprocess_nodes(nodes, None)
    return [source for source in (rag._source_from_node_with_score(node) for node in kept) if source is not None]


def _match(file_name: str, description: str, score: float) -> dict[str, Any]:
    """Build a CLIP match payload.

    Args:
        file_name: The image file name.
        description: The stored caption.
        score: The raw CLIP cosine similarity.
    """
    return {
        "image_id": file_name,
        "file_name": file_name,
        "llm_description": description,
        "score": score,
    }


PARTY = _match("invitation.jpg", "A German party invitation with a jungle background.", 0.2853)
TRANSFORMER = _match("transformer.png", "A diagram of the Transformer model architecture.", 0.2480)


def test_images_below_the_rerank_floor_are_dropped() -> None:
    """An image that is merely CLIP-nearest, not relevant, must not surface."""
    image_service = StubImageService([PARTY])
    # 0.0037 is what the live reranker scored this caption for an AI question.
    reranker = StubReranker({PARTY["llm_description"]: 0.0037})
    rag = _build_rag(image_service=image_service, reranker=reranker)

    sources = _lane_through_rerank(rag, "Gibt es Dokumente über KI?")

    assert sources == []


def test_images_above_the_rerank_floor_survive_carrying_the_rerank_score() -> None:
    """A relevant image survives and reports the reranker's score, not CLIP cosine."""
    image_service = StubImageService([TRANSFORMER])
    reranker = StubReranker({TRANSFORMER["llm_description"]: 0.8978})
    rag = _build_rag(image_service=image_service, reranker=reranker)

    sources = _lane_through_rerank(rag, "Zeig mir die Transformer-Architektur")

    assert len(sources) == 1
    assert sources[0]["filename"] == "transformer.png"
    # The raw CLIP cosine (0.2480) is not comparable with text source scores.
    assert sources[0]["score"] == pytest.approx(0.8978)


def test_clip_receives_an_english_translation_of_the_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """The English-only CLIP text tower must be embedded with English, not German.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    monkeypatch.setattr(
        "docint.core.rag.translate_text",
        lambda text, target_lang=None: TranslateResult(
            ok=True,
            translation="Show me the Transformer architecture",
            model="stub",
            target_lang="en",
        ),
    )
    image_service = StubImageService([TRANSFORMER])
    rag = _build_rag(image_service=image_service, locale="de")

    rag._retrieve_image_nodes("Zeig mir die Transformer-Architektur", top_k=2)

    assert image_service.seen_query_text == "Show me the Transformer architecture"


def test_reranking_uses_the_original_query_not_the_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reranker is multilingual, so it should judge against the user's own words.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    monkeypatch.setattr(
        "docint.core.rag.translate_text",
        lambda text, target_lang=None: TranslateResult(
            ok=True,
            translation="Show me the Transformer architecture",
            model="stub",
            target_lang="en",
        ),
    )
    image_service = StubImageService([TRANSFORMER])
    reranker = StubReranker({TRANSFORMER["llm_description"]: 0.9})
    rag = _build_rag(image_service=image_service, reranker=reranker, locale="de")

    _lane_through_rerank(rag, "Zeig mir die Transformer-Architektur")

    assert reranker.seen_query == "Zeig mir die Transformer-Architektur"


def test_translation_failure_falls_back_to_the_original_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """A translation outage degrades to the untranslated query rather than dropping images.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    monkeypatch.setattr(
        "docint.core.rag.translate_text",
        lambda text, target_lang=None: TranslateResult(
            ok=False, translation=None, model="stub", target_lang="en", error="upstream_error"
        ),
    )
    image_service = StubImageService([PARTY])
    rag = _build_rag(image_service=image_service, locale="de")

    rag._retrieve_image_nodes("Gibt es Dokumente über KI?", top_k=2)

    assert image_service.seen_query_text == "Gibt es Dokumente über KI?"


def test_query_already_in_english_is_embedded_as_is_without_a_warning(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """An English query in a German deployment is embedded unchanged, and is no outage.

    The translator answers such a query with ``same_language`` and no
    translation; logging that as "translation unavailable" would report a false
    outage on every English query.

    Args:
        monkeypatch: The monkeypatch fixture.
        loguru_caplog: Captures loguru WARNING records.
    """
    monkeypatch.setattr(
        "docint.core.rag.translate_text",
        lambda text, target_lang=None: TranslateResult(
            ok=True, translation=None, model="stub", target_lang="en", same_language=True
        ),
    )
    image_service = StubImageService([TRANSFORMER])
    rag = _build_rag(image_service=image_service, locale="de")

    rag._retrieve_image_nodes("Show me the Transformer architecture", top_k=2)

    assert image_service.seen_query_text == "Show me the Transformer architecture"
    assert not [r.getMessage() for r in loguru_caplog.records if r.funcName == "_image_query_for_clip"]


def test_english_locale_skips_the_translation_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """An English deployment must not pay an LLM call per query to translate English.

    Args:
        monkeypatch: The monkeypatch fixture.
    """

    def _fail(text: str, target_lang: str | None = None) -> TranslateResult:
        """Fail if the translator is reached at all.

        Args:
            text: The text that would be translated.
            target_lang: The requested destination language.

        Raises:
            AssertionError: Always -- reaching this is the failure.
        """
        raise AssertionError("translate_text should not be called for an English locale")

    monkeypatch.setattr("docint.core.rag.translate_text", _fail)
    image_service = StubImageService([TRANSFORMER])
    rag = _build_rag(image_service=image_service, locale="en")

    rag._retrieve_image_nodes("Are there documents about AI?", top_k=2)

    assert image_service.seen_query_text == "Are there documents about AI?"


def test_captionless_images_are_dropped_rather_than_reranked_on_empty_text() -> None:
    """An image with no caption cannot be judged for relevance, so it must not surface."""
    uncaptioned = {"image_id": "blank.png", "file_name": "blank.png", "llm_description": "", "score": 0.29}
    rag = _build_rag(image_service=StubImageService([uncaptioned]))

    nodes = rag._retrieve_image_nodes("Gibt es Dokumente über KI?", top_k=2)

    assert nodes == []


def test_silent_rerank_fallback_does_not_blank_the_image_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """``VLLMRerankPostprocessor`` swallows transport errors and keeps the retrieval scores.

    That internal degradation must not be mistaken for "every image scored below
    the floor" — otherwise a rerank outage silently deletes all image sources.
    The nodes come back scored (a CLIP cosine, on another scale than the floor)
    and stamped as not reranked, so the floor is set above the CLIP band here:
    that is the configuration in which comparing the two empties the lane.
    """

    def _refuse(request: object, timeout: float) -> object:
        raise urllib.error.URLError("rerank endpoint down")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)
    image_service = StubImageService([TRANSFORMER])
    image_service.img_ingestion_config.rerank_min_score = 0.5
    reranker = VLLMRerankPostprocessor(api_base="http://router:8000/v1", model="stub-reranker", top_n=5)
    rag = _build_rag(image_service=image_service, reranker=reranker)

    sources = _lane_through_rerank(rag, "Zeig mir die Transformer-Architektur")

    assert len(sources) == 1
    assert sources[0]["filename"] == "transformer.png"


def _identified_match(**extra: Any) -> dict[str, Any]:
    """Build a CLIP match payload carrying identity fields.

    Args:
        **extra: Payload overrides merged over the base match.
    """
    return {
        "image_id": "105cc611aabb",
        "file_name": "transformer.png",
        "llm_description": "A diagram of the Transformer model architecture.",
        "score": 0.2480,
        **extra,
    }


def test_image_sources_carry_the_retrieved_node_id_as_citation_id() -> None:
    """An image source must be traceable to the exact retrieved point, like a text source."""
    match = _identified_match(node_id="0298c8c6-aaab-559b-bd58-2bb428b853b2")
    reranker = StubReranker({match["llm_description"]: 0.9})
    rag = _build_rag(image_service=StubImageService([match]), reranker=reranker)

    sources = _lane_through_rerank(rag, "Zeig mir die Transformer-Architektur")

    assert sources[0]["id"] == "0298c8c6-aaab-559b-bd58-2bb428b853b2"


def test_image_sources_carry_the_image_hash_as_chunk_id() -> None:
    """The durable per-image identity is its content hash, mirroring a text chunk id."""
    match = _identified_match(node_id="0298c8c6-aaab-559b-bd58-2bb428b853b2")
    reranker = StubReranker({match["llm_description"]: 0.9})
    rag = _build_rag(image_service=StubImageService([match]), reranker=reranker)

    sources = _lane_through_rerank(rag, "Zeig mir die Transformer-Architektur")

    assert sources[0]["chunk_id"] == "105cc611aabb"


def test_image_source_id_falls_back_to_the_image_hash() -> None:
    """A match with no node id still cites something stable rather than nothing."""
    match = _identified_match()
    reranker = StubReranker({match["llm_description"]: 0.9})
    rag = _build_rag(image_service=StubImageService([match]), reranker=reranker)

    sources = _lane_through_rerank(rag, "Zeig mir die Transformer-Architektur")

    assert sources[0]["id"] == "105cc611aabb"


QUESTION = "Wie ist das Wetter am Hafen?"
KEYFRAME_CAPTION = "Zwei Personen in Regenjacken am Hafen, bedeckter Himmel."
PIER_CAPTION = "Ein leerer Steg bei Sonnenschein, ein Rettungsring an einem Pfosten."
DESCRIPTION = "Himmel über dem Hafen mit Wolken, Sonne oder Regen, Personen in Wetterkleidung"


def _reranked_keyframe(node_id: str, caption: str, score: float) -> NodeWithScore:
    """Build a keyframe caption node as the turn's healthy rerank left it.

    Args:
        node_id: The companion point id.
        caption: The stored caption.
        score: The rerank score the turn gave it.
    """
    metadata = {"file_name": "harbour_walk.webm", IMAGE_LANE_METADATA_KEY: True, RERANK_APPLIED_KEY: True}
    return NodeWithScore(node=TextNode(id_=node_id, text=caption, metadata=metadata), score=score)


class _Body:
    """Response body a faked ``urlopen`` hands the reranker."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Store the JSON payload.

        Args:
            payload: What the endpoint answers.
        """
        self.payload = payload

    def __enter__(self) -> _Body:
        """Enter the response context."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Leave the response context.

        Args:
            *exc: Exception details, unused.
        """

    def read(self) -> bytes:
        """Return the payload as the endpoint would send it."""
        return json.dumps(self.payload).encode("utf-8")


class _RerankEndpoint:
    """Rerank endpoint stand-in for ``urlopen``: scores by document text, every request recorded."""

    def __init__(self, scores_by_document: dict[str, float]) -> None:
        """Store the scores and prepare the request log.

        Args:
            scores_by_document: Relevance score per document text.
        """
        self.scores_by_document = scores_by_document
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: urllib.request.Request, timeout: float) -> _Body:
        """Record the request and answer it the way the rerank endpoint does.

        Args:
            request: The rerank request.
            timeout: The request timeout, unused.
        """
        body = json.loads(cast(bytes, request.data).decode("utf-8"))
        self.requests.append(body)
        results: list[dict[str, Any]] = [
            {"index": index, "relevance_score": self.scores_by_document.get(document, 0.0)}
            for index, document in enumerate(body["documents"])
        ]
        results.sort(key=lambda result: result["relevance_score"], reverse=True)
        return _Body({"results": results[: body["top_n"]]})


def _endpoint_rag(monkeypatch: pytest.MonkeyPatch, endpoint: _RerankEndpoint, *, top_n: int = 5) -> RAG:
    """Build a RAG whose real reranker talks to ``endpoint``.

    Args:
        monkeypatch: The monkeypatch fixture.
        endpoint: The faked rerank endpoint.
        top_n: The reranker's own cut.
    """
    monkeypatch.setattr(urllib.request, "urlopen", endpoint)
    reranker = VLLMRerankPostprocessor(api_base="http://router:8000/v1", model="stub-reranker", top_n=top_n)
    return _build_rag(image_service=StubImageService(), reranker=reranker)


def test_rescore_reranks_against_the_description_not_the_question(monkeypatch: pytest.MonkeyPatch) -> None:
    """The second look scores a caption against what a matching picture would show.

    Against the question itself the right keyframe scores in the band of an
    unrelated caption; against a caption-style description of the question the
    two separate. So the description, never the question, is the rerank query.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    asked: list[str] = []

    def _describe(query: str) -> str:
        """Record the query and describe it.

        Args:
            query: The query the images are scored for.
        """
        asked.append(query)
        return DESCRIPTION

    monkeypatch.setattr(image_description, "describe_for_image_match", _describe)
    endpoint = _RerankEndpoint({KEYFRAME_CAPTION: 0.86})
    rag = _endpoint_rag(monkeypatch, endpoint)

    scores = rag._rescore_images_by_description(QUESTION, [_reranked_keyframe("kf-0", KEYFRAME_CAPTION, 0.0044)])

    assert asked == [QUESTION]
    assert [(request["query"], request["documents"]) for request in endpoint.requests] == [
        (DESCRIPTION, [KEYFRAME_CAPTION])
    ]
    assert scores == {"kf-0": pytest.approx(0.86)}


def test_the_rescore_scores_every_candidate_whatever_the_rerank_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reranker's own top-n must not cut the candidates: an unscored one could never be rescued.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    monkeypatch.setattr(image_description, "describe_for_image_match", lambda query: DESCRIPTION)
    endpoint = _RerankEndpoint({KEYFRAME_CAPTION: 0.86, PIER_CAPTION: 0.39})
    rag = _endpoint_rag(monkeypatch, endpoint, top_n=1)
    candidates = [
        _reranked_keyframe("kf-0", KEYFRAME_CAPTION, 0.0044),
        _reranked_keyframe("kf-1", PIER_CAPTION, 0.0039),
    ]

    scores = rag._rescore_images_by_description(QUESTION, candidates)

    assert endpoint.requests[0]["top_n"] == 2
    assert scores == {"kf-0": pytest.approx(0.86), "kf-1": pytest.approx(0.39)}


def test_no_description_rescues_nothing_and_skips_the_rerank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a description there is nothing to rerank against, so no rerank call is made.

    The suite-wide stub in ``conftest.py`` answers no description.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    endpoint = _RerankEndpoint({KEYFRAME_CAPTION: 0.86})
    rag = _endpoint_rag(monkeypatch, endpoint)

    scores = rag._rescore_images_by_description(QUESTION, [_reranked_keyframe("kf-0", KEYFRAME_CAPTION, 0.0044)])

    assert scores == {}
    assert endpoint.requests == []


class _UnappliedReranker(BaseNodePostprocessor):
    """Reranker stand-in that scores every node yet stamps it as not reranked."""

    top_n: int = 5

    @override
    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Score each node 0.9 and stamp it as a failed rerank.

        Args:
            nodes: The nodes to rerank.
            query_bundle: The rerank query, unused.
        """
        scored = [NodeWithScore(node=node.node, score=0.9) for node in nodes]
        for node in scored:
            node.node.metadata[RERANK_APPLIED_KEY] = False
            node.node.metadata[RERANK_ERROR_KEY] = "rerank endpoint down"
        return scored


def test_the_rescore_never_marks_the_turn_unreranked(monkeypatch: pytest.MonkeyPatch) -> None:
    """The second look reranks clones, so its outcome never reaches the turn's sources.

    The turn reports its rerank from the stamps on its sources: a stamp from
    the re-score landing there would report a healthy turn as unreranked. And a
    clone stamped as not reranked rescues nothing, whatever score it carries.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    monkeypatch.setattr(image_description, "describe_for_image_match", lambda query: DESCRIPTION)
    rag = _build_rag(image_service=StubImageService(), reranker=_UnappliedReranker())
    candidate = _reranked_keyframe("kf-0", KEYFRAME_CAPTION, 0.0044)
    before = dict(candidate.node.metadata)

    scores = rag._rescore_images_by_description(QUESTION, [candidate])

    assert scores == {}
    assert candidate.node.metadata == before
    assert candidate.score == 0.0044


def test_the_floor_rescues_through_the_bound_rag_rescore(monkeypatch: pytest.MonkeyPatch) -> None:
    """The floor and ``RAG._rescore_images_by_description`` agree on the call, end to end.

    The floor's ``rescore`` is typed ``Any``, and its fail-soft ``except`` turns
    a signature that drifted into a quiet drop: the images would just stop being
    rescued while every test on either side still passed. So this runs the floor
    with the real bound method and the real reranker, faking only the chat
    description and the rerank endpoint.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    monkeypatch.setattr(image_description, "describe_for_image_match", lambda query: DESCRIPTION)
    endpoint = _RerankEndpoint({KEYFRAME_CAPTION: 0.86, PIER_CAPTION: 0.0072})
    rag = _endpoint_rag(monkeypatch, endpoint)
    floor = ImageRelevanceFloorPostprocessor(
        min_score=0.05, top_n=5, rescore=rag._rescore_images_by_description, rescore_min_score=0.02
    )
    text = NodeWithScore(
        node=TextNode(id_="txt-0", text="Der Hafen liegt im Norden.", metadata={RERANK_APPLIED_KEY: True}),
        score=0.31,
    )
    nodes = [
        text,
        _reranked_keyframe("kf-0", KEYFRAME_CAPTION, 0.0044),
        _reranked_keyframe("kf-1", PIER_CAPTION, 0.0039),
    ]

    kept = floor._postprocess_nodes(nodes, QueryBundle(query_str=QUESTION))

    assert [node.node.node_id for node in kept] == ["txt-0", "kf-0"]
    assert kept[1].score == 0.0044
    assert [(request["query"], request["documents"]) for request in endpoint.requests] == [
        (DESCRIPTION, [KEYFRAME_CAPTION, PIER_CAPTION])
    ]
