"""Relevance gating for CLIP-retrieved image sources.

CLIP text->image cosine similarity is not calibrated across queries: on the live
stack an unrelated query and a genuinely matching one both land in a ~0.20-0.30
band, so the top-k nearest images were surfaced for *every* question. These tests
pin the two-stage fix:

* the CLIP tower is English-only, so the query is translated before embedding;
* the multilingual reranker scores the image captions and a floor drops the
  ones that are merely nearest rather than relevant.
"""

from __future__ import annotations

import types
from typing import Any, cast

import pytest
from llama_index.core.schema import NodeWithScore

from docint.core.rag import RAG
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
        self.img_ingestion_config = types.SimpleNamespace(rerank_min_score=0.05)

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
    reranker: StubReranker | None = None,
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

    sources = rag._retrieve_image_sources("Gibt es Dokumente über KI?", top_k=2)

    assert sources == []


def test_images_above_the_rerank_floor_survive_carrying_the_rerank_score() -> None:
    """A relevant image survives and reports the reranker's score, not CLIP cosine."""
    image_service = StubImageService([TRANSFORMER])
    reranker = StubReranker({TRANSFORMER["llm_description"]: 0.8978})
    rag = _build_rag(image_service=image_service, reranker=reranker)

    sources = rag._retrieve_image_sources("Zeig mir die Transformer-Architektur", top_k=2)

    assert len(sources) == 1
    assert sources[0]["filename"] == "transformer.png"
    # The raw CLIP cosine (0.2480) is not comparable with text source scores.
    assert sources[0]["score"] == pytest.approx(0.8978)


def test_clip_receives_an_english_translation_of_the_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """The English-only CLIP text tower must be embedded with English, not German."""
    monkeypatch.setattr(
        "docint.core.rag.translate_text",
        lambda text, target_lang=None: TranslateResult(
            ok=True,
            translation="Are there documents about artificial intelligence?",
            model="stub",
            target_lang="en",
        ),
    )
    image_service = StubImageService([])
    rag = _build_rag(image_service=image_service, locale="de")

    rag._retrieve_image_sources("Gibt es Dokumente über KI?", top_k=2)

    assert image_service.seen_query_text == "Are there documents about artificial intelligence?"


def test_reranking_uses_the_original_query_not_the_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reranker is multilingual, so it should judge against the user's own words."""
    monkeypatch.setattr(
        "docint.core.rag.translate_text",
        lambda text, target_lang=None: TranslateResult(
            ok=True, translation="Show me the transformer architecture", model="stub", target_lang="en"
        ),
    )
    image_service = StubImageService([TRANSFORMER])
    reranker = StubReranker({TRANSFORMER["llm_description"]: 0.9})
    rag = _build_rag(image_service=image_service, reranker=reranker, locale="de")

    rag._retrieve_image_sources("Zeig mir die Transformer-Architektur", top_k=2)

    assert reranker.seen_query == "Zeig mir die Transformer-Architektur"


def test_translation_failure_falls_back_to_the_original_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """A translation outage degrades to the untranslated query rather than dropping images."""
    monkeypatch.setattr(
        "docint.core.rag.translate_text",
        lambda text, target_lang=None: TranslateResult(
            ok=False, translation=None, model="stub", target_lang="en", error="unavailable"
        ),
    )
    image_service = StubImageService([])
    rag = _build_rag(image_service=image_service, locale="de")

    rag._retrieve_image_sources("Gibt es Dokumente über KI?", top_k=2)

    assert image_service.seen_query_text == "Gibt es Dokumente über KI?"


def test_english_locale_skips_the_translation_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """An English deployment must not pay an LLM call per query to translate English."""

    def _fail(text: str, target_lang: str | None = None) -> TranslateResult:
        raise AssertionError("translate_text should not be called for an English locale")

    monkeypatch.setattr("docint.core.rag.translate_text", _fail)
    image_service = StubImageService([])
    rag = _build_rag(image_service=image_service, locale="en")

    rag._retrieve_image_sources("Are there documents about AI?", top_k=2)

    assert image_service.seen_query_text == "Are there documents about AI?"


def test_captionless_images_are_dropped_rather_than_reranked_on_empty_text() -> None:
    """An image with no caption cannot be judged for relevance, so it must not surface."""
    captionless = _match("scan.jpg", "", 0.29)
    image_service = StubImageService([captionless])
    reranker = StubReranker({})
    rag = _build_rag(image_service=image_service, reranker=reranker)

    sources = rag._retrieve_image_sources("Gibt es Dokumente über KI?", top_k=2)

    assert sources == []


def test_reranker_outage_degrades_to_returning_the_clip_matches() -> None:
    """A rerank transport failure must not silently blank the image lane."""

    class ExplodingReranker:
        """Reranker stub standing in for an unreachable rerank endpoint."""

        def postprocess_nodes(self, nodes: list[NodeWithScore], query_bundle: Any = None) -> list[NodeWithScore]:
            """Raise as an unreachable endpoint would.

            Args:
                nodes: Caption nodes built from the CLIP matches.
                query_bundle: Bundle carrying the rerank query string.
            """
            raise RuntimeError("rerank endpoint unreachable")

    image_service = StubImageService([TRANSFORMER])
    rag = _build_rag(image_service=image_service)
    rag.rerank_model_id = "stub-reranker"
    rag._reranker = cast(Any, ExplodingReranker())

    sources = rag._retrieve_image_sources("Zeig mir die Transformer-Architektur", top_k=2)

    assert len(sources) == 1
    assert sources[0]["filename"] == "transformer.png"


def test_silent_rerank_fallback_does_not_blank_the_image_lane() -> None:
    """``VLLMRerankPostprocessor`` swallows transport errors and returns nodes unscored.

    That internal degradation must not be mistaken for "every image scored below
    the floor" — otherwise a rerank outage silently deletes all image sources.
    """

    class FallbackReranker:
        """Reranker stub reproducing ``_fallback_nodes``: original nodes, unscored."""

        def postprocess_nodes(self, nodes: list[NodeWithScore], query_bundle: Any = None) -> list[NodeWithScore]:
            """Return the nodes exactly as handed in.

            Args:
                nodes: Caption nodes built from the CLIP matches.
                query_bundle: Bundle carrying the rerank query string.
            """
            return nodes

    image_service = StubImageService([TRANSFORMER])
    rag = _build_rag(image_service=image_service)
    rag.rerank_model_id = "stub-reranker"
    rag._reranker = cast(Any, FallbackReranker())

    sources = rag._retrieve_image_sources("Zeig mir die Transformer-Architektur", top_k=2)

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
    image_service = StubImageService([match])
    reranker = StubReranker({match["llm_description"]: 0.9})
    rag = _build_rag(image_service=image_service, reranker=reranker)

    sources = rag._retrieve_image_sources("Zeig mir die Transformer-Architektur", top_k=2)

    assert sources[0]["id"] == "0298c8c6-aaab-559b-bd58-2bb428b853b2"


def test_image_sources_carry_the_image_hash_as_chunk_id() -> None:
    """The durable per-image identity is its content hash, mirroring a text chunk id."""
    match = _identified_match(node_id="0298c8c6-aaab-559b-bd58-2bb428b853b2")
    image_service = StubImageService([match])
    reranker = StubReranker({match["llm_description"]: 0.9})
    rag = _build_rag(image_service=image_service, reranker=reranker)

    sources = rag._retrieve_image_sources("Zeig mir die Transformer-Architektur", top_k=2)

    assert sources[0]["chunk_id"] == "105cc611aabb"


def test_image_source_id_falls_back_to_the_image_hash() -> None:
    """A match with no node id still cites something stable rather than nothing."""
    match = _identified_match()
    image_service = StubImageService([match])
    reranker = StubReranker({match["llm_description"]: 0.9})
    rag = _build_rag(image_service=image_service, reranker=reranker)

    sources = rag._retrieve_image_sources("Zeig mir die Transformer-Architektur", top_k=2)

    assert sources[0]["id"] == "105cc611aabb"
