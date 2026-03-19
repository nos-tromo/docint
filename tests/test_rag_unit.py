"""Unit tests for the RAG engine.

Covers response normalisation, source extraction, ingestion hashing,
session management, collection selection, NER/hate-speech helpers,
summarisation with caching and diagnostics, sparse model resolution,
reranker configuration, and summary revision invalidation.
"""

from __future__ import annotations

import asyncio
import json
import types
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from llama_index.core import Document
from llama_index.core.schema import NodeWithScore, TextNode

from docint.core import rag as rag_module
from docint.core.rag import RAG
from docint.core.retrieval_filters import (
    build_metadata_filters,
    build_qdrant_filter,
    matches_metadata_filters,
)
from docint.utils.env_cfg import OpenAIConfig
from docint.utils.hashing import compute_file_hash


class DummyNode:
    """A dummy node class to simulate LlamaIndex nodes."""

    def __init__(self, text: str, metadata: dict[str, object]) -> None:
        """Initializes a DummyNode with text and metadata.

        Args:
            text: The text content of the node.
            metadata: Metadata associated with the node.
        """
        self.text = text
        self.metadata = metadata


class DummyNodeWithScore:
    """A dummy node with score class to simulate LlamaIndex nodes with scores."""

    def __init__(self, node: DummyNode, score: float = 0.0) -> None:
        """Initializes a DummyNodeWithScore with a DummyNode.

        Args:
            node: The dummy node associated with this score.
            score: The relevance score for this node. Defaults to 0.0.
        """
        self.node = node
        self.score = score


class DummyResponse:
    """Minimal stand-in for a LlamaIndex query response."""

    def __init__(self, text: str, nodes: list[DummyNodeWithScore]):
        """Initializes a DummyResponse with text and source nodes.

        Args:
            text: The response text.
            nodes: The source nodes associated with the response.
        """
        self.response = text
        self.source_nodes = nodes


def test_normalize_response_data_extracts_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that _normalize_response_data correctly extracts source information.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    monkeypatch.setattr(
        RAG,
        "_retrieve_image_sources",
        lambda self, query, top_k=3, metadata_filter_rules=None: [],
    )
    node = DummyNode(
        "Example text",
        {
            "origin": {
                "filename": "doc.pdf",
                "mimetype": "application/pdf",
                "file_hash": "abc",
            },
            "page_number": 3,
            "source": "document",
            "reference_metadata": {
                "network": "Telegram",
                "type": "comment",
                "timestamp": "2026-01-02T10:00:00Z",
                "author": "Alice",
                "author_id": "a1",
                "vanity": "alice-v",
                "text": "Example text",
                "text_id": "c1",
            },
        },
    )
    result = DummyResponse("<think>reason</think>Answer", [DummyNodeWithScore(node)])
    normalized = rag._normalize_response_data("query", result)
    assert normalized["response"] == "Answer"
    assert normalized["reasoning"] == "reason"

    sources = normalized["sources"]
    assert len(sources) == 1
    first_source = sources[0]

    expected = {
        "text": "Example text",
        "filename": "doc.pdf",
        "filetype": "application/pdf",
        "source": "document",
        "file_hash": "abc",
        "page": 3,
    }
    for key, value in expected.items():
        assert first_source.get(key) == value
    assert first_source["reference_metadata"]["type"] == "comment"
    assert first_source["reference_metadata"]["author"] == "Alice"

    # ensure preview helpers are attached when file hashes are present
    assert first_source.get("preview_text") == "Example text"
    assert first_source.get("preview_url")
    assert first_source.get("document_url") == first_source.get("preview_url")


def test_create_text_model_passes_reasoning_effort_for_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RAG text model creation should enable reasoning only for OpenAI provider.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    captured: dict[str, Any] = {}

    class FakeLocalOpenAI:
        """Capture constructor kwargs passed by ``RAG._create_text_model``."""

        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(rag_module, "LocalOpenAI", FakeLocalOpenAI)

    rag = RAG(qdrant_collection="test")
    rag.text_model_id = "gpt-5-mini"
    rag.openai_config = OpenAIConfig(
        api_base="https://api.openai.com/v1",
        api_key="sk-test",
        ctx_window=200000,
        dimensions=1024,
        max_retries=2,
        model_provider="openai",
        reuse_client=False,
        seed=42,
        temperature=0.0,
        thinking_effort="high",
        thinking_enabled=True,
        timeout=300.0,
        top_p=0.0,
    )
    rag.openai_api_base = rag.openai_config.api_base
    rag.openai_api_key = rag.openai_config.api_key
    rag.openai_ctx_window = rag.openai_config.ctx_window
    rag.openai_max_retries = rag.openai_config.max_retries
    rag.openai_reuse_client = rag.openai_config.reuse_client
    rag.openai_seed = rag.openai_config.seed
    rag.openai_temperature = rag.openai_config.temperature
    rag.openai_timeout = rag.openai_config.timeout
    rag.openai_top_p = rag.openai_config.top_p

    rag._create_text_model()

    assert captured["reasoning_effort"] == "high"


def test_normalize_response_data_appends_image_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image retrieval results should be appended to normalized sources.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    monkeypatch.setattr(
        RAG,
        "_retrieve_image_sources",
        lambda self, query, top_k=3, metadata_filter_rules=None: [
            {
                "text": "Image shows transformer blocks.",
                "filename": "attention_is_all_you_need.pdf",
                "source": "image",
                "image_id": "img-1",
                "score": 0.91,
            }
        ],
    )
    result = DummyResponse("Answer", [])

    normalized = rag._normalize_response_data("transformer diagram", result)
    sources = normalized["sources"]

    assert len(sources) == 1
    assert sources[0]["source"] == "image"
    assert sources[0]["image_id"] == "img-1"


def test_normalize_response_data_skips_aux_image_sources_when_filters_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request-scoped metadata filters should be forwarded to image retrieval.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    captured_rules: list[Any] = []

    def _fake_retrieve_image_sources(
        self: RAG,
        query: str,
        top_k: int = 3,
        metadata_filter_rules: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        _ = self
        _ = query
        _ = top_k
        captured_rules.extend(metadata_filter_rules or [])
        return [
            {
                "text": "Image-only source.",
                "filename": "img.png",
                "source": "image",
            }
        ]

    monkeypatch.setattr(
        RAG,
        "_retrieve_image_sources",
        _fake_retrieve_image_sources,
    )
    result = DummyResponse("Answer", [])

    normalized = rag._normalize_response_data(
        "transformer diagram",
        result,
        metadata_filters_active=True,
        metadata_filter_rules=[
            {
                "field": "mimetype",
                "operator": "mime_match",
                "value": "image/*",
            }
        ],
    )

    assert normalized["sources"] == [
        {
            "text": "Image-only source.",
            "filename": "img.png",
            "source": "image",
        }
    ]
    assert captured_rules == [
        {
            "field": "mimetype",
            "operator": "mime_match",
            "value": "image/*",
        }
    ]


def test_normalize_response_data_falls_back_for_empty_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalization should return a non-empty fallback when model output is empty.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    monkeypatch.setattr(
        RAG,
        "_retrieve_image_sources",
        lambda self, query, top_k=3, metadata_filter_rules=None: [],
    )

    # Simulates models that emit only hidden reasoning.
    result = DummyResponse("<think>internal reasoning</think>", [])
    normalized = rag._normalize_response_data("frage", result)

    assert normalized["response"] == rag_module.EMPTY_RESPONSE_FALLBACK
    assert normalized["reasoning"] == "internal reasoning"


def test_normalize_response_data_falls_back_for_empty_response_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalization should hide upstream empty-response sentinels."""
    rag = RAG(qdrant_collection="test")
    monkeypatch.setattr(
        RAG,
        "_retrieve_image_sources",
        lambda self, query, top_k=3, metadata_filter_rules=None: [],
    )

    normalized = rag._normalize_response_data(
        "frage", DummyResponse("Empty Response", [])
    )

    assert normalized["response"] == rag_module.EMPTY_RESPONSE_FALLBACK


def test_normalize_response_data_builds_source_backed_answer_when_sources_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalization should summarize matching sources instead of claiming no context."""
    rag = RAG(qdrant_collection="test")
    monkeypatch.setattr(
        RAG,
        "_retrieve_image_sources",
        lambda self, query, top_k=3, metadata_filter_rules=None: [
            {
                "text": "Transformer attention diagram.",
                "filename": "image-3-b65a08ee-0.png",
                "page": 4,
                "source": "image",
            },
            {
                "text": "Transformer architecture diagram.",
                "filename": "image-2-4fd97d25-0.png",
                "page": 3,
                "source": "image",
            },
        ],
    )

    normalized = rag._normalize_response_data("frage", DummyResponse("", []))

    assert normalized["response"] == (
        "I found 2 matching sources: image-3-b65a08ee-0.png (page 4), "
        "image-2-4fd97d25-0.png (page 3)."
    )


def test_build_metadata_filters_supports_mime_and_date_rules() -> None:
    """Metadata filter builder should compile MIME and date request rules."""
    compiled = build_metadata_filters(
        [
            {
                "field": "mimetype",
                "operator": "mime_match",
                "value": "image/*",
            },
            {
                "field": "reference_metadata.timestamp",
                "operator": "date_on_or_after",
                "value": "2026-01-01",
            },
        ]
    )

    assert compiled is not None
    assert len(compiled.filters) == 2
    first_filter = cast(Any, compiled.filters[0])
    second_filter = cast(Any, compiled.filters[1])
    assert first_filter.operator.value == "text_match_insensitive"
    assert second_filter.operator.value == ">="
    assert str(second_filter.value).startswith("2026-01-01T00:00:00")


def test_build_qdrant_filter_supports_boolean_rules() -> None:
    """Native Qdrant filters should support boolean metadata matches."""
    compiled = build_qdrant_filter(
        [
            {
                "field": "hate_speech.hate_speech",
                "operator": "eq",
                "value": True,
            }
        ]
    )

    assert compiled is not None
    assert compiled.must is not None
    must = compiled.must
    if isinstance(must, list):
        assert must
        condition = cast(Any, must[0])
    else:
        condition = cast(Any, must)
    assert condition.key == "hate_speech.hate_speech"
    assert condition.match.value is True


def test_build_metadata_filters_skips_invalid_date_values() -> None:
    """Invalid date filter inputs should be ignored rather than raising."""
    compiled = build_metadata_filters(
        [
            {
                "field": "timestamp",
                "operator": "date_on_or_after",
                "value": "not-a-date",
            }
        ]
    )

    assert compiled is None


def test_matches_metadata_filters_supports_mime_and_dates() -> None:
    """In-memory metadata matching should support MIME wildcards and dates."""
    payload = {
        "mimetype": "image/png",
        "created_at": "2026-03-17T12:00:00Z",
        "reference_metadata": {"timestamp": "2026-03-10T09:00:00Z"},
    }

    assert matches_metadata_filters(
        payload,
        [
            {"field": "mimetype", "operator": "mime_match", "value": "image/*"},
            {
                "field": "created_at",
                "operator": "date_on_or_after",
                "value": "2026-03-01",
            },
            {
                "field": "reference_metadata.timestamp",
                "operator": "date_on_or_before",
                "value": "2026-03-10",
            },
        ],
    )
    assert not matches_metadata_filters(
        payload,
        [
            {
                "field": "reference_metadata.timestamp",
                "operator": "date_on_or_after",
                "value": "2026-03-11",
            }
        ],
    )


def test_retrieve_image_sources_skips_when_image_collection_missing() -> None:
    """Image retrieval should short-circuit when the image collection is absent."""

    class DummyImageService:
        """Image service stub that raises if unexpectedly queried."""

        def __init__(self) -> None:
            self.called = False

        def _resolve_collection_name(self, source_collection: str | None = None) -> str:
            return f"{source_collection}_images"

        def query_similar_images_by_text(
            self,
            query_text: str,
            top_k: int = 3,
            *,
            source_collection: str | None = None,
        ) -> list[dict[str, Any]]:
            self.called = True
            raise AssertionError("query_similar_images_by_text should not be called")

    rag = RAG(qdrant_collection="spiegel-data")
    image_service = DummyImageService()
    rag._image_ingestion_service = cast(Any, image_service)

    rag._qdrant_client = cast(
        Any,
        types.SimpleNamespace(
            collection_exists=lambda collection_name: False,
        ),
    )

    sources = rag._retrieve_image_sources("any query", top_k=3)

    assert sources == []
    assert image_service.called is False


def test_directory_ingestion_attaches_file_hash(tmp_path: Path) -> None:
    """Test that directory ingestion attaches file hashes to documents.

    Args:
        tmp_path: The temporary path fixture.
    """
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello world")

    rag = RAG(qdrant_collection="test")
    rag.data_dir = tmp_path
    pipeline = rag._build_ingestion_pipeline()

    docs = []
    for batch_docs, _ in pipeline.build(existing_hashes=None):
        docs.extend(batch_docs)

    digest = compute_file_hash(file_path)

    assert docs
    assert all(getattr(doc, "metadata", {}).get("file_hash") == digest for doc in docs)


def test_start_session_requires_query_engine(tmp_path: Path) -> None:
    """Test that start_session raises RuntimeError if query_engine is not initialized.

    Args:
        tmp_path: The temporary path fixture.
    """
    rag = RAG(qdrant_collection="test")
    rag.init_session_store(f"sqlite:///{tmp_path / 'sessions.db'}")
    rag.query_engine = None
    with pytest.raises(RuntimeError):
        rag.start_session()


def test_select_collection_resets_image_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting another collection should reset image ingestion service state.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="alpha")
    rag._image_ingestion_service = object()  # type: ignore[assignment]
    monkeypatch.setattr(
        RAG,
        "list_collections",
        lambda self, prefer_api=True: ["alpha", "beta"],
    )

    rag.select_collection("beta")

    assert rag.qdrant_collection == "beta"
    assert rag._image_ingestion_service is None


def test_select_collection_invalidates_ner_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting a collection should clear stale NER caches.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="alpha")
    rag.ner_sources = [{"filename": "a.pdf", "entities": [{"text": "Acme"}]}]
    rag.ner_aggregate_cache["alpha"] = {"entities": []}
    rag.ner_graph_cache[("alpha", 100, 1)] = {"nodes": [], "edges": [], "meta": {}}
    monkeypatch.setattr(
        RAG,
        "list_collections",
        lambda self, prefer_api=True: ["alpha", "beta"],
    )

    rag.select_collection("beta")

    assert rag.ner_sources == []
    assert rag.ner_aggregate_cache.get("alpha") is None
    assert ("alpha", 100, 1) not in rag.ner_graph_cache


def test_get_collection_ner_refresh_bypasses_cache() -> None:
    """Refreshing collection NER should re-fetch data instead of returning stale cache."""
    rag = RAG(qdrant_collection="test")
    rag._qdrant_client = MagicMock()

    point1 = MagicMock()
    point1.payload = {"filename": "doc1.pdf", "entities": [{"text": "Acme"}]}
    point2 = MagicMock()
    point2.payload = {"filename": "doc2.pdf", "entities": [{"text": "Widget"}]}

    rag._qdrant_client.scroll = MagicMock(
        side_effect=[([point1], None), ([point2], None)]
    )

    first = rag.get_collection_ner()
    second = rag.get_collection_ner()
    refreshed = rag.get_collection_ner(refresh=True)

    assert first == second
    assert rag._qdrant_client.scroll.call_count == 2
    assert refreshed != first
    assert refreshed[0]["filename"] == "doc2.pdf"
    assert "chunk_text" in refreshed[0]


def test_get_collection_ner_extracts_chunk_text_from_node_content() -> None:
    """NER collection helper should recover text from serialized node payloads."""
    rag = RAG(qdrant_collection="test")
    rag._qdrant_client = MagicMock()

    point = MagicMock()
    point.id = "pt-1"
    point.payload = {
        "filename": "table.csv",
        "table": {"row_index": 25},
        "entities": [{"text": "Deutschland"}],
        "reference_metadata": {
            "network": "Telegram",
            "type": "comment",
            "timestamp": "2026-01-02T10:00:00Z",
            "author": "Alice",
            "author_id": "a1",
            "vanity": "alice-v",
            "text": "Deutschland",
            "text_id": "c1",
        },
        "_node_content": json.dumps({"text": "Deutschland"}),
    }
    rag._qdrant_client.scroll = MagicMock(side_effect=[([point], None)])

    rows = rag.get_collection_ner()

    assert len(rows) == 1
    assert rows[0]["chunk_id"] == "pt-1"
    assert rows[0]["chunk_text"] == "Deutschland"
    assert rows[0]["reference_metadata"]["text_id"] == "c1"


def test_get_collection_hate_speech_filters_flagged_rows() -> None:
    """Hate-speech collection helper should return only flagged chunks."""
    rag = RAG(qdrant_collection="test")
    rag._qdrant_client = MagicMock()

    flagged = MagicMock()
    flagged.id = "pt-1"
    flagged.payload = {
        "text": "flagged text",
        "page": 1,
        "filename": "doc1.pdf",
        "reference_metadata": {
            "network": "Facebook",
            "type": "posting",
            "timestamp": "2026-01-02T10:00:00Z",
            "author": "Alice",
            "author_id": "a1",
            "vanity": "alice-v",
            "text": "flagged text",
            "text_id": "p1",
        },
        "hate_speech": {
            "hate_speech": True,
            "category": "ethnicity",
            "confidence": "high",
            "reason": "Contains a slur",
        },
    }
    clean = MagicMock()
    clean.payload = {
        "text": "clean text",
        "hate_speech": {"hate_speech": False},
    }
    rag._qdrant_client.scroll = MagicMock(side_effect=[([flagged, clean], None)])

    rows = rag.get_collection_hate_speech()

    assert len(rows) == 1
    assert rows[0]["chunk_text"] == "flagged text"
    assert rows[0]["category"] == "ethnicity"
    assert rows[0]["reference_metadata"]["network"] == "Facebook"


def test_get_collection_hate_speech_extracts_chunk_text_from_node_content() -> None:
    """Hate-speech helper should recover text when payload text field is empty."""
    rag = RAG(qdrant_collection="test")
    rag._qdrant_client = MagicMock()

    flagged = MagicMock()
    flagged.id = "pt-1"
    flagged.payload = {
        "hate_speech": {
            "hate_speech": True,
            "category": "ethnicity",
            "confidence": "high",
            "reason": "Contains a slur",
        },
        "_node_content": json.dumps({"text": "table cell value"}),
    }
    rag._qdrant_client.scroll = MagicMock(side_effect=[([flagged], None)])

    rows = rag.get_collection_hate_speech()

    assert len(rows) == 1
    assert rows[0]["chunk_text"] == "table cell value"


def test_collection_ner_stats_and_search() -> None:
    """Stats/search should canonicalize entities and support filtering."""
    rag = RAG(qdrant_collection="test")
    sources = [
        {
            "filename": "a.pdf",
            "entities": [{"text": "Acme", "type": "ORG", "score": 0.8}],
        },
        {
            "filename": "a.pdf",
            "entities": [{"text": "acme", "type": "ORG", "score": 0.9}],
        },
        {"filename": "b.pdf", "entities": [{"text": "Rivertown", "type": "LOC"}]},
        {
            "filename": "b.pdf",
            "entities": [{"text": "Acme", "type": "ORG"}],
            "relations": [{"head": "Acme", "label": "located_in", "tail": "Rivertown"}],
        },
    ]
    rag.ner_sources = sources

    stats = rag.get_collection_ner_stats(top_k=10, min_mentions=2)
    assert stats["totals"]["unique_entities"] == 2
    assert stats["totals"]["entity_mentions"] == 4
    assert stats["top_entities"][0]["text"] == "Acme"
    assert stats["top_entities"][0]["mentions"] == 3

    loc_stats = rag.get_collection_ner_stats(
        top_k=10, min_mentions=1, entity_type="loc", include_relations=False
    )
    assert loc_stats["totals"]["unique_entities"] == 1
    assert loc_stats["totals"]["unique_relations"] == 0

    results = rag.search_collection_ner_entities(q="ac", limit=10)
    assert results[0]["text"] == "Acme"
    assert results[0]["mentions"] == 3


def test_collection_ner_search_matches_entity_acronyms() -> None:
    """Entity search should match acronym queries against multi-word entities."""
    rag = RAG(qdrant_collection="test")
    rag.ner_sources = [
        {
            "filename": "a.pdf",
            "entities": [{"text": "European Union", "type": "ORG"}],
        },
        {
            "filename": "b.pdf",
            "entities": [{"text": "European Union", "type": "ORG"}],
        },
        {"filename": "c.pdf", "entities": [{"text": "Europe", "type": "LOC"}]},
    ]

    results = rag.search_collection_ner_entities(q="eu", limit=10)

    assert results
    assert results[0]["text"] == "European Union"
    assert results[0]["mentions"] == 2


def test_run_entity_occurrence_query_returns_matching_sources() -> None:
    """Entity occurrence mode should return mention-level source rows."""
    rag = RAG(qdrant_collection="test")
    rag.ner_sources = [
        {
            "filename": "a.pdf",
            "file_hash": "hash-a",
            "chunk_id": "chunk-1",
            "chunk_text": "Remigration appears here.",
            "text": "Remigration appears here.",
            "page": 1,
            "entities": [{"text": "Remigration", "type": "IDEOLOGY"}],
        },
        {
            "filename": "b.pdf",
            "file_hash": "hash-b",
            "chunk_id": "chunk-2",
            "chunk_text": "Another Remigration mention.",
            "text": "Another Remigration mention.",
            "page": 2,
            "entities": [{"text": "Remigration", "type": "IDEOLOGY"}],
        },
        {
            "filename": "c.pdf",
            "file_hash": "hash-c",
            "chunk_id": "chunk-3",
            "chunk_text": "Something else.",
            "text": "Something else.",
            "page": 3,
            "entities": [{"text": "Migration", "type": "IDEOLOGY"}],
        },
    ]

    result = rag.run_entity_occurrence_query("Where is Remigration mentioned?")

    assert result["retrieval_mode"] == "entity_occurrence"
    assert result["coverage_unit"] == "entity_mentions"
    assert len(result["sources"]) == 2
    assert result["sources"][0]["matched_entity"]["text"] == "Remigration"
    assert result["sources"][0]["occurrence_count"] == 1
    assert "Found 2 occurrence(s) of 'Remigration'" in result["response"]


def test_run_entity_occurrence_query_reports_no_match() -> None:
    """Entity occurrence mode should return a clear no-match response."""
    rag = RAG(qdrant_collection="test")
    rag.ner_sources = [
        {
            "filename": "a.pdf",
            "chunk_id": "chunk-1",
            "chunk_text": "Acme is here.",
            "text": "Acme is here.",
            "entities": [{"text": "Acme", "type": "ORG"}],
        }
    ]

    result = rag.run_entity_occurrence_query("Remigration")

    assert result["sources"] == []
    assert "couldn't find a named-entity match" in result["response"]


def test_run_entity_occurrence_query_reports_ambiguity() -> None:
    """Single-entity occurrence mode should stop when top-rank matches tie."""
    rag = RAG(qdrant_collection="test")
    rag.ner_sources = [
        {
            "filename": "a.pdf",
            "chunk_id": "chunk-1",
            "chunk_text": "Acme the organization is here.",
            "text": "Acme the organization is here.",
            "entities": [{"text": "Acme", "type": "ORG"}],
        },
        {
            "filename": "b.pdf",
            "chunk_id": "chunk-2",
            "chunk_text": "Acme the product is here.",
            "text": "Acme the product is here.",
            "entities": [{"text": "Acme", "type": "PRODUCT"}],
        },
    ]

    result = rag.run_entity_occurrence_query("Acme")

    assert result["retrieval_mode"] == "entity_occurrence_ambiguous"
    assert result["sources"] == []
    assert len(result["entity_match_candidates"]) == 2
    assert "matches multiple entities equally well" in result["response"]


def test_run_multi_entity_occurrence_query_groups_strong_matches() -> None:
    """Multi-entity occurrence mode should group all equally strong matches."""
    rag = RAG(qdrant_collection="test")
    rag.ner_sources = [
        {
            "filename": "a.pdf",
            "file_hash": "hash-a",
            "chunk_id": "chunk-1",
            "chunk_text": "Acme the organization is here.",
            "text": "Acme the organization is here.",
            "page": 1,
            "entities": [{"text": "Acme", "type": "ORG"}],
        },
        {
            "filename": "b.pdf",
            "file_hash": "hash-b",
            "chunk_id": "chunk-2",
            "chunk_text": "Acme the product is here.",
            "text": "Acme the product is here.",
            "page": 2,
            "entities": [{"text": "Acme", "type": "PRODUCT"}],
        },
    ]

    result = rag.run_multi_entity_occurrence_query("Acme")

    assert result["retrieval_mode"] == "entity_occurrence_multi"
    assert len(result["entity_match_groups"]) == 2
    assert {group["entity"]["type"] for group in result["entity_match_groups"]} == {
        "ORG",
        "PRODUCT",
    }
    assert len(result["sources"]) == 2


def test_collection_ner_graph_and_neighbors() -> None:
    """Graph endpoints should expose relation and co-occurrence edges."""
    rag = RAG(qdrant_collection="test")
    sources = [
        {
            "filename": "a.pdf",
            "entities": [
                {"text": "Acme", "type": "ORG"},
                {"text": "Rivertown", "type": "LOC"},
            ],
            "relations": [{"head": "Acme", "label": "located_in", "tail": "Rivertown"}],
        },
        {
            "filename": "b.pdf",
            "entities": [
                {"text": "Acme", "type": "ORG"},
                {"text": "Widget", "type": "PRODUCT"},
            ],
        },
    ]
    rag.ner_sources = sources

    graph = rag.get_collection_ner_graph(top_k_nodes=10, min_edge_weight=1)
    assert graph["meta"]["node_count"] >= 3
    assert graph["meta"]["edge_count"] >= 2

    neighbors = rag.get_collection_ner_graph_neighbors(entity="Acme", hops=1)
    assert neighbors["center"] is not None
    assert neighbors["center"]["text"] == "Acme"
    assert neighbors["neighbors"]


def test_expand_query_with_graph_with_debug_applies_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graph expansion should return expanded query plus debug anchor/neighbor metadata.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    rag.graphrag_enabled = True
    rag.graphrag_max_neighbors = 3
    rag.graphrag_neighbor_hops = 1
    rag.graphrag_top_k_nodes = 100
    rag.graphrag_min_edge_weight = 1

    monkeypatch.setattr(
        RAG,
        "_get_collection_ner_aggregate",
        lambda self, refresh=False: {
            "entities": [
                {"text": "Acme", "mentions": 10},
                {"text": "Widget", "mentions": 3},
            ]
        },
    )
    monkeypatch.setattr(
        RAG,
        "get_collection_ner_graph_neighbors",
        lambda self, *, entity, hops, top_k_nodes, min_edge_weight, refresh=False: {
            "neighbors": [{"text": "Widget"}, {"text": "Rivertown"}]
        },
    )

    query = "What changed for Acme this quarter?"
    expanded, debug = rag.expand_query_with_graph_with_debug(query)

    assert expanded.endswith("Related entities for retrieval: Widget, Rivertown")
    assert debug["enabled"] is True
    assert debug["applied"] is True
    assert debug["original_query"] == query
    assert debug["expanded_query"] == expanded
    assert debug["anchor_entities"] == ["Acme"]
    assert debug["neighbor_entities"] == ["Widget", "Rivertown"]


def test_expand_query_with_graph_with_debug_matches_acronym_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graph expansion should anchor acronym mentions to the canonical entity.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    rag.graphrag_enabled = True
    rag.graphrag_max_neighbors = 3
    rag.graphrag_neighbor_hops = 1
    rag.graphrag_top_k_nodes = 100
    rag.graphrag_min_edge_weight = 1

    monkeypatch.setattr(
        RAG,
        "_get_collection_ner_aggregate",
        lambda self, refresh=False: {
            "entities": [
                {"text": "European Union", "mentions": 10},
                {"text": "Brussels", "mentions": 4},
            ]
        },
    )
    monkeypatch.setattr(
        RAG,
        "get_collection_ner_graph_neighbors",
        lambda self, *, entity, hops, top_k_nodes, min_edge_weight, refresh=False: {
            "neighbors": [{"text": "Brussels"}]
        },
    )

    query = "What is said about the EU?"
    expanded, debug = rag.expand_query_with_graph_with_debug(query)

    assert expanded.endswith("Related entities for retrieval: Brussels")
    assert debug["applied"] is True
    assert debug["anchor_entities"] == ["European Union"]
    assert debug["neighbor_entities"] == ["Brussels"]


def test_expand_query_with_graph_with_debug_reports_disabled_state() -> None:
    """Graph debug payload should report no-op reason when GraphRAG is disabled."""
    rag = RAG(qdrant_collection="test")
    rag.graphrag_enabled = False

    query = "Acme outlook?"
    expanded, debug = rag.expand_query_with_graph_with_debug(query)

    assert expanded == query
    assert debug["applied"] is False
    assert debug["reason"] == "graphrag_disabled"


def test_start_session_initializes_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that start_session initializes the chat memory and engine.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: The temporary path fixture.
    """
    rag = RAG(qdrant_collection="test")
    rag.init_session_store(f"sqlite:///{tmp_path / 'sessions.db'}")
    rag.index = cast(Any, object())
    rag.query_engine = cast(Any, object())
    rag._text_model = cast(Any, object())

    class FakeMemory:
        def __init__(self) -> None:
            self.messages: list[object] = []

        def put(self, message) -> None:
            self.messages.append(message)

    class FakeChatEngine:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        @classmethod
        def from_defaults(cls, **kwargs):
            return cls(**kwargs)

    monkeypatch.setattr(
        "docint.core.state.session_manager.ChatMemoryBuffer",
        types.SimpleNamespace(from_defaults=lambda **_: FakeMemory()),
    )
    monkeypatch.setattr(
        "docint.core.state.session_manager.CondenseQuestionChatEngine",
        types.SimpleNamespace(from_defaults=lambda **kwargs: FakeChatEngine(**kwargs)),
    )
    session_id = rag.start_session("abc")
    assert session_id == "abc"
    assert isinstance(rag.chat_engine, FakeChatEngine)


def test_chat_rejects_empty_prompt() -> None:
    """Test that chat rejects empty prompts."""
    rag = RAG(qdrant_collection="test")
    with pytest.raises(ValueError):
        rag.chat("   ")


def test_sparse_model_raises_import_error_when_fastembed_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the sparse_model property raises ImportError when
    SparseTextEmbedding.list_supported_models raises ImportError.

    Args:
        monkeypatch: The monkeypatch fixture.
    """

    def broken() -> list[dict[str, str]]:
        raise ImportError("missing")

    monkeypatch.setattr(
        rag_module.SparseTextEmbedding,
        "list_supported_models",
        staticmethod(broken),
    )

    rag = RAG(qdrant_collection="test")
    rag.enable_hybrid = True
    rag.sparse_model_id = "some-model"
    with pytest.raises(ImportError, match="fastembed is not installed"):
        rag.sparse_model


def test_filter_docs_skips_existing_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that _filter_docs_by_existing_hashes skips documents with existing hashes.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    pipeline = rag._build_ingestion_pipeline()
    existing_hash = "abc"
    fresh_hash = "def"
    docs = [
        Document(text="keep", metadata={"file_hash": fresh_hash, "file_name": "b.txt"}),
        Document(
            text="skip", metadata={"file_hash": existing_hash, "file_name": "a.txt"}
        ),
    ]

    filtered = pipeline._filter_docs_by_existing_hashes(docs, {existing_hash})

    assert len(filtered) == 1
    assert filtered[0].metadata.get("file_hash") == fresh_hash


def test_build_ingestion_pipeline_reuses_text_model_for_ner_and_hate_speech(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """LLM-NER and hate-speech should share one cached text model instance."""
    rag = RAG(qdrant_collection="test")
    rag.data_dir = tmp_path
    rag.ner_enabled = True
    rag.openai_model_provider = "openai"

    monkeypatch.setattr(
        rag_module,
        "load_hate_speech_env",
        lambda: types.SimpleNamespace(enabled=True),
    )

    created: list[object] = []

    def _fake_create_text_model(self: RAG) -> object:
        model = object()
        created.append(model)
        return model

    monkeypatch.setattr(RAG, "_create_text_model", _fake_create_text_model)

    pipeline = rag._build_ingestion_pipeline()

    assert len(created) == 1
    assert pipeline.ner_model is created[0]
    assert pipeline.hate_speech_model is created[0]


def test_build_ingestion_pipeline_non_openai_keeps_gliner_ner_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Non-OpenAI provider should not wire LLM NER model into the pipeline."""
    rag = RAG(qdrant_collection="test")
    rag.data_dir = tmp_path
    rag.ner_enabled = True
    rag.openai_model_provider = "llama.cpp"

    monkeypatch.setattr(
        rag_module,
        "load_hate_speech_env",
        lambda: types.SimpleNamespace(enabled=True),
    )

    created: list[object] = []

    def _fake_create_text_model(self: RAG) -> object:
        model = object()
        created.append(model)
        return model

    monkeypatch.setattr(RAG, "_create_text_model", _fake_create_text_model)

    pipeline = rag._build_ingestion_pipeline()

    assert pipeline.ner_model is None
    assert pipeline.hate_speech_model is created[0]
    assert len(created) == 1

    def test_sparse_model_uses_cached_path(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Ensure sparse_model resolves to a cached snapshot path when available.

        Args:
            monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
            tmp_path (Path): The temporary path fixture.
        """

        # Build a fake HF cache layout with refs/main -> snapshots/abc123
        cache_root = tmp_path / "hub"
        model_dir = cache_root / "models--Qdrant--all_miniLM_L6_v2_with_attentions"
        refs_dir = model_dir / "refs"
        snaps_dir = model_dir / "snapshots"
        snap = snaps_dir / "abc123"
        snap.mkdir(parents=True)
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / "main").write_text("abc123")

        # Stub supported models to match the configured sparse ID
        monkeypatch.setattr(
            rag_module.SparseTextEmbedding,
            "list_supported_models",
            staticmethod(
                lambda: [
                    {
                        "model": "Qdrant/all_miniLM_L6_v2_with_attentions",
                        "sources": {"hf": "Qdrant/all_miniLM_L6_v2_with_attentions"},
                    }
                ]
            ),
        )

        rag = RAG(qdrant_collection="test")
        rag.hf_hub_cache = cache_root
        rag.sparse_model_id = "Qdrant/all_miniLM_L6_v2_with_attentions"

        resolved = rag.sparse_model
        assert resolved == str(snap)


def test_reranker_passes_configured_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reranker should pass the configured fp16 setting to FlagEmbeddingReranker.

    Args:
        monkeypatch: The monkeypatch fixture.
    """

    captured: dict[str, object] = {}

    class FakeFlagReranker:
        def __init__(self, top_n: int, model: str, use_fp16: bool) -> None:
            captured["top_n"] = top_n
            captured["model"] = model
            captured["use_fp16"] = use_fp16
            self._model = types.SimpleNamespace(compute_score=lambda _: [0.0])

    monkeypatch.setattr(rag_module, "FlagEmbeddingReranker", FakeFlagReranker)
    monkeypatch.setattr(
        rag_module,
        "resolve_hf_cache_path",
        lambda cache_dir, repo_id: None,
    )

    rag = RAG(qdrant_collection="test")
    rag.openai_model_provider = "llama.cpp"
    rag.rerank_use_fp16 = True
    rag.rerank_top_n = 7

    _ = rag.reranker

    assert captured["top_n"] == 7
    assert captured["model"] == rag.rerank_model_id
    assert captured["use_fp16"] is True


def test_embed_model_uses_ollama_embedding_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama embeddings should use the OpenAI-compatible embedding client.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    captured: dict[str, object] = {}

    class FakeOpenAIEmbedding:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(rag_module, "OpenAIEmbedding", FakeOpenAIEmbedding)
    monkeypatch.delenv("OPENAI_DIMENSIONS", raising=False)

    rag = RAG(qdrant_collection="test")
    rag.embed_model_id = "bge-m3"

    _ = rag.embed_model

    assert captured["model_name"] == "bge-m3"
    assert captured["api_base"] == rag.openai_api_base
    assert "dimensions" not in captured


def test_embed_model_forwards_explicit_dimensions_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit embedding dimensions should only be sent when configured.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    captured: dict[str, object] = {}

    class FakeOpenAIEmbedding:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(rag_module, "OpenAIEmbedding", FakeOpenAIEmbedding)

    rag = RAG(qdrant_collection="test")
    rag.embed_model_id = "text-embedding-3-small"
    rag.openai_dimensions = 1024

    _ = rag.embed_model

    assert captured["dimensions"] == 1024


def test_reranker_falls_back_to_llm_on_meta_tensor_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reranker should fallback to LLMRerank when FlagEmbedding hits meta tensor errors.

    Args:
        monkeypatch: The monkeypatch fixture.
    """

    class FakeFlagReranker:
        def __init__(self, top_n: int, model: str, use_fp16: bool) -> None:
            _ = (top_n, model, use_fp16)

            def _raise(_pairs) -> list[float]:
                raise NotImplementedError("Cannot copy out of meta tensor; no data!")

            self._model = types.SimpleNamespace(compute_score=_raise)

    llm_reranker_obj = object()

    monkeypatch.setattr(rag_module, "FlagEmbeddingReranker", FakeFlagReranker)
    monkeypatch.setattr(
        rag_module,
        "LLMRerank",
        lambda top_n, llm: llm_reranker_obj,
    )
    monkeypatch.setattr(
        rag_module,
        "resolve_hf_cache_path",
        lambda cache_dir, repo_id: None,
    )

    rag = RAG(qdrant_collection="test")
    rag.openai_model_provider = "llama.cpp"
    rag._text_model = None

    assert rag.reranker is llm_reranker_obj


class _FakeCompletion:
    """A simple class to simulate LLM completion responses for testing purposes."""

    def __init__(self, text: str) -> None:
        """Initializes a _FakeCompletion with the given text.

        Args:
            text: The completion text.
        """
        self.text = text


class _FakeSummaryLLM:
    """A simple class to simulate an LLM for testing the RAG summarization logic."""

    def __init__(self, text: str) -> None:
        """Initializes a _FakeSummaryLLM with the given text.

        Args:
            text: The fixed text every completion returns.
        """
        self.text = text
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> _FakeCompletion:
        """Record *prompt* and return a fixed completion.

        Args:
            prompt: The input prompt to complete.

        Returns:
            A ``_FakeCompletion`` wrapping the pre-configured text.
        """
        self.prompts.append(prompt)
        return _FakeCompletion(self.text)


def _summary_node(
    *,
    text: str,
    filename: str,
    file_hash: str,
    page: int = 1,
    score: float = 0.9,
) -> DummyNodeWithScore:
    """Helper function to create a DummyNodeWithScore with the appropriate metadata for summarization tests.

    Args:
        text: The text content of the node.
        filename: The name of the source file.
        file_hash: The hash of the source file.
        page: The page number. Defaults to 1.
        score: The relevance score. Defaults to 0.9.

    Returns:
        A ``DummyNodeWithScore`` with the specified metadata.
    """
    node = DummyNode(
        text,
        {
            "origin": {
                "filename": filename,
                "mimetype": "application/pdf",
                "file_hash": file_hash,
            },
            "page_number": page,
            "source": "document",
        },
    )
    nws = DummyNodeWithScore(node)
    nws.score = score
    return nws


def _social_summary_node(
    *,
    text: str,
    filename: str,
    file_hash: str,
    text_id: str,
    author: str,
    author_id: str,
    timestamp: str,
    row: int,
    score: float = 0.9,
) -> DummyNodeWithScore:
    """Build a dummy row-level social source node for summary tests."""
    node = DummyNode(
        text,
        {
            "origin": {
                "filename": filename,
                "mimetype": "text/csv",
                "file_hash": file_hash,
            },
            "source": "table",
            "table": {"row_index": row, "n_rows": 10},
            "reference_metadata": {
                "network": "Telegram",
                "type": "comment",
                "timestamp": timestamp,
                "author": author,
                "author_id": author_id,
                "text_id": text_id,
            },
        },
    )
    nws = DummyNodeWithScore(node)
    nws.score = score
    return nws


def test_build_query_engine_uses_refine_prompts_for_social_table_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Social/table-heavy collections should use grounded refine synthesis."""
    rag = RAG(qdrant_collection="test")
    rag._text_model = cast(Any, object())
    rag._reranker = cast(Any, object())
    rag.index = types.SimpleNamespace(
        as_retriever=lambda **kwargs: {"retriever_kwargs": kwargs}
    )
    monkeypatch.setattr(
        RAG,
        "list_documents",
        lambda self: [{"filename": "social.csv", "max_rows": 500}],
    )
    monkeypatch.setattr(
        RAG,
        "_sample_collection_payloads",
        lambda self, limit=128: [
            {
                "source": "table",
                "reference_metadata": {"type": "comment", "text_id": "p1"},
            }
        ],
    )

    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_from_args(**kwargs: Any) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        rag_module.RetrieverQueryEngine,
        "from_args",
        staticmethod(fake_from_args),
    )

    engine = rag.build_query_engine()

    assert engine is sentinel
    assert captured["response_mode"] == rag_module.ResponseMode.REFINE
    postprocessors = captured["node_postprocessors"]
    assert len(postprocessors) == 2
    assert isinstance(postprocessors[1], rag_module.SocialSourceDiversityPostprocessor)
    assert "keep each post distinct" in captured["text_qa_template"].template.lower()


def test_build_query_engine_uses_hybrid_retrieval_by_default_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid-capable collections should default to actual hybrid retrieval."""
    rag = RAG(qdrant_collection="test", enable_hybrid=True)
    rag._text_model = cast(Any, object())
    rag._reranker = cast(Any, object())

    captured_retriever_kwargs: dict[str, Any] = {}
    rag.index = types.SimpleNamespace(
        docstore=object(),
        as_retriever=lambda **kwargs: captured_retriever_kwargs.update(kwargs)
        or {"retriever_kwargs": kwargs},
    )
    monkeypatch.setattr(RAG, "list_documents", lambda self: [])
    monkeypatch.setattr(RAG, "_sample_collection_payloads", lambda self, limit=128: [])
    monkeypatch.setattr(
        rag_module.RetrieverQueryEngine,
        "from_args",
        staticmethod(lambda **kwargs: kwargs),
    )

    rag.build_query_engine()

    assert (
        captured_retriever_kwargs["vector_store_query_mode"]
        == rag_module.VectorStoreQueryMode.HYBRID
    )
    assert captured_retriever_kwargs["alpha"] == pytest.approx(rag.hybrid_alpha)
    assert captured_retriever_kwargs["sparse_top_k"] == rag.sparse_top_k
    assert captured_retriever_kwargs["hybrid_top_k"] == rag.hybrid_top_k


def test_build_query_engine_adds_parent_context_postprocessor_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hierarchical collections should expand fine hits to parent context."""
    rag = RAG(qdrant_collection="test")
    rag._text_model = cast(Any, object())
    rag._reranker = cast(Any, object())
    rag.index = types.SimpleNamespace(
        docstore=object(),
        as_retriever=lambda **kwargs: {"retriever_kwargs": kwargs},
    )
    monkeypatch.setattr(RAG, "list_documents", lambda self: [])
    monkeypatch.setattr(
        RAG,
        "_sample_collection_payloads",
        lambda self, limit=128: [{"docint_hier_type": "fine", "hier.parent_id": "p1"}],
    )

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        rag_module.RetrieverQueryEngine,
        "from_args",
        staticmethod(lambda **kwargs: captured.update(kwargs) or kwargs),
    )

    rag.build_query_engine()

    postprocessors = captured["node_postprocessors"]
    assert any(
        isinstance(postprocessor, rag_module.ParentContextPostprocessor)
        for postprocessor in postprocessors
    )


def test_parent_context_postprocessor_promotes_parent_nodes() -> None:
    """Parent-context postprocessor should replace child hits with their parent node."""
    parent = TextNode(
        text="Parent context", id_="parent-1", metadata={"filename": "a.txt"}
    )
    child = TextNode(
        text="Child match",
        id_="child-1",
        metadata={"hier.parent_id": "parent-1", "docint_hier_type": "fine"},
    )
    child_hit = NodeWithScore(node=child, score=0.77)

    postprocessor = rag_module.ParentContextPostprocessor(
        docstore=types.SimpleNamespace(
            get_node=lambda node_id, raise_error=False: parent
            if node_id == "parent-1"
            else None
        )
    )

    processed = postprocessor._postprocess_nodes([child_hit])

    assert len(processed) == 1
    assert processed[0].node.get_content() == "Parent context"
    assert processed[0].score == pytest.approx(0.77)


def test_summarize_collection_reports_coverage_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that summarize_collection returns diagnostics about document coverage and uncovered documents.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    rag._text_model = _FakeSummaryLLM("Collection summary")  # type: ignore[assignment]

    docs = [
        {"filename": "a.pdf", "file_hash": "ha", "node_count": 9},
        {"filename": "b.pdf", "file_hash": "hb", "node_count": 7},
        {"filename": "c.pdf", "file_hash": "hc", "node_count": 5},
    ]
    nodes_by_file = {
        "a.pdf": [
            _summary_node(text="A key finding", filename="a.pdf", file_hash="ha")
        ],
        "b.pdf": [
            _summary_node(text="B key finding", filename="b.pdf", file_hash="hb")
        ],
        "c.pdf": [],
    }

    monkeypatch.setattr(RAG, "_summary_document_targets", lambda self: docs)
    monkeypatch.setattr(
        RAG,
        "_retrieve_summary_nodes_for_document",
        lambda self, **kwargs: nodes_by_file.get(str(kwargs["filename"]), []),
    )
    monkeypatch.setattr(
        RAG,
        "_summary_kv_store",
        lambda self, collection=None, allow_create=True: None,
    )

    summary = rag.summarize_collection()

    diagnostics = summary["summary_diagnostics"]
    assert summary["response"] == "Collection summary"
    assert diagnostics["total_documents"] == 3
    assert diagnostics["covered_documents"] == 2
    assert diagnostics["coverage_ratio"] == 0.6667
    assert diagnostics["uncovered_documents"] == ["c.pdf"]
    assert len(summary["sources"]) == 2


def test_summarize_collection_uses_post_coverage_for_social_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Social summaries should report post-level coverage and preserve source distinctions."""
    rag = RAG(qdrant_collection="test")
    rag._text_model = _FakeSummaryLLM("Social summary")  # type: ignore[assignment]
    rag.social_summary_diversity_limit = 1

    monkeypatch.setattr(
        RAG,
        "_summary_kv_store",
        lambda self, collection=None, allow_create=True: None,
    )
    monkeypatch.setattr(
        RAG,
        "_infer_collection_profile",
        lambda self: {"is_social_table": True, "coverage_unit": "posts"},
    )
    monkeypatch.setattr(
        RAG,
        "_retrieve_social_summary_nodes",
        lambda self: [
            _social_summary_node(
                text="Alice says the launch moved to Friday.",
                filename="social.csv",
                file_hash="hash-social",
                text_id="p1",
                author="Alice",
                author_id="a1",
                timestamp="2026-01-02T10:00:00Z",
                row=1,
            ),
            _social_summary_node(
                text="Duplicate of Alice launch post.",
                filename="social.csv",
                file_hash="hash-social",
                text_id="p1",
                author="Alice",
                author_id="a1",
                timestamp="2026-01-02T10:00:00Z",
                row=2,
            ),
            _social_summary_node(
                text="Bob says the launch is still Thursday.",
                filename="social.csv",
                file_hash="hash-social",
                text_id="p2",
                author="Bob",
                author_id="b1",
                timestamp="2026-01-02T11:00:00Z",
                row=3,
            ),
        ],
    )
    monkeypatch.setattr(RAG, "_count_social_coverage_units", lambda self, unit: 5)

    summary = rag.summarize_collection()

    diagnostics = summary["summary_diagnostics"]
    assert summary["response"] == "Social summary"
    assert diagnostics["coverage_unit"] == "posts"
    assert diagnostics["total_documents"] == 5
    assert diagnostics["covered_documents"] == 2
    assert diagnostics["coverage_ratio"] == 0.4
    assert diagnostics["candidate_count"] == 3
    assert diagnostics["deduped_count"] == 2
    assert diagnostics["sampled_count"] == 2
    assert len(summary["sources"]) == 2
    assert "author=Alice" in rag._text_model.prompts[0]
    assert "author=Bob" in rag._text_model.prompts[0]


def test_merge_summary_sources_deduplicates_and_preserves_doc_coverage() -> None:
    """Test that _merge_summary_sources deduplicates sources by filename while preserving coverage information.
    This ensures that when multiple nodes from the same document are included in the summary, they are correctly merged into a single source entry without losing track of how many nodes from that document contributed to the summary.
    """
    rag = RAG(qdrant_collection="test")
    rag.summary_final_source_cap = 5

    doc_a_source = {
        "filename": "a.pdf",
        "file_hash": "ha",
        "page": 1,
        "preview_text": "A finding",
    }
    duplicate_doc_a = dict(doc_a_source)
    doc_b_source = {
        "filename": "b.pdf",
        "file_hash": "hb",
        "page": 2,
        "preview_text": "B finding",
    }

    merged = rag._merge_summary_sources(
        {
            "a.pdf": [doc_a_source, duplicate_doc_a],
            "b.pdf": [doc_b_source],
        }
    )

    filenames = [str(item.get("filename")) for item in merged]
    assert filenames.count("a.pdf") == 1
    assert "b.pdf" in filenames


def test_summarize_collection_handles_no_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that summarize_collection returns an appropriate response and diagnostics when there are no documents to summarize.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    rag._text_model = _FakeSummaryLLM("unused")  # type: ignore[assignment]
    rag.summary_coverage_target = 0.7
    rag.summary_max_docs = 30

    monkeypatch.setattr(RAG, "_summary_document_targets", lambda self: [])
    monkeypatch.setattr(
        RAG,
        "_summary_kv_store",
        lambda self, collection=None, allow_create=True: None,
    )
    summary = rag.summarize_collection()

    assert summary["response"] == "No documents available in the selected collection."
    diagnostics = summary["summary_diagnostics"]
    assert diagnostics["total_documents"] == 0
    assert diagnostics["covered_documents"] == 0
    assert diagnostics["coverage_ratio"] == 0.0
    assert diagnostics["uncovered_documents"] == []


def test_summarize_collection_requires_collection() -> None:
    """Test that summarize_collection raises a ValueError if no collection is selected, ensuring
    that the method cannot be called without a valid collection context.
    """
    rag = RAG(qdrant_collection="")
    with pytest.raises(ValueError, match="No collection selected"):
        rag.summarize_collection()


def test_retrieve_summary_nodes_for_document_falls_back_to_payload_scroll() -> None:
    """Summary retrieval should still return evidence when embedding calls fail."""
    rag = RAG(qdrant_collection="test")
    rag.summary_per_doc_top_k = 2
    rag.index = cast(
        Any,
        types.SimpleNamespace(
            as_retriever=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("nan"))
        ),
    )

    point = types.SimpleNamespace(
        id="pt-1",
        payload={
            "origin": {
                "filename": "39816-pdf.pdf",
                "mimetype": "application/pdf",
                "file_hash": "hash-39816",
            },
            "file_hash": "hash-39816",
            "page_number": 1,
            "source": "document",
            "text": "A grounded finding from the PDF.",
        },
    )
    rag._qdrant_client = cast(
        Any,
        types.SimpleNamespace(scroll=lambda **kwargs: ([point], None)),
    )

    nodes = rag._retrieve_summary_nodes_for_document(
        filename="39816-pdf.pdf",
        file_hash="hash-39816",
    )

    assert len(nodes) == 1
    source = rag._source_from_node_with_score(nodes[0])
    assert source is not None
    assert source["filename"] == "39816-pdf.pdf"
    assert source["text"] == "A grounded finding from the PDF."


class _InMemorySummaryKVStore:
    """In-memory KV store used for summary-cache unit tests."""

    def __init__(self) -> None:
        """Initialise an empty store."""
        self._rows: dict[tuple[str, str], dict[str, Any]] = {}

    def put(self, key: str, val: dict[str, Any], collection: str) -> None:
        """Store *val* under (*collection*, *key*).

        Args:
            key: Payload key.
            val: Dictionary payload to store.
            collection: Namespace for the key.
        """
        self._rows[(collection, key)] = dict(val)

    def get(self, key: str, collection: str) -> dict[str, Any] | None:
        """Retrieve a stored payload, or ``None`` if absent.

        Args:
            key: Payload key.
            collection: Namespace for the key.

        Returns:
            A shallow copy of the stored dict, or ``None``.
        """
        row = self._rows.get((collection, key))
        return dict(row) if isinstance(row, dict) else None


def _patch_summary_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch summary retrieval to return one deterministic grounded source.

    Args:
        monkeypatch: The monkeypatch fixture used to stub RAG methods.
    """
    docs = [{"filename": "a.pdf", "file_hash": "ha", "node_count": 3}]
    nodes_by_file = {
        "a.pdf": [_summary_node(text="A key finding", filename="a.pdf", file_hash="ha")]
    }
    monkeypatch.setattr(RAG, "_summary_document_targets", lambda self: docs)
    monkeypatch.setattr(
        RAG,
        "_retrieve_summary_nodes_for_document",
        lambda self, **kwargs: nodes_by_file.get(str(kwargs["filename"]), []),
    )


def test_summarize_collection_cache_miss_then_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Summary cache should store on miss and serve on subsequent hit.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    llm = _FakeSummaryLLM("Collection summary")
    rag._text_model = llm  # type: ignore[assignment]
    _patch_summary_context(monkeypatch)

    kv_store = _InMemorySummaryKVStore()

    def _summary_kv_store(
        self: RAG,
        collection: str | None = None,
        *,
        allow_create: bool = True,
    ) -> _InMemorySummaryKVStore:
        _ = (self, collection, allow_create)
        return kv_store

    monkeypatch.setattr(RAG, "_summary_kv_store", _summary_kv_store)

    first = rag.summarize_collection()
    second = rag.summarize_collection()

    assert first["response"] == "Collection summary"
    assert second["response"] == "Collection summary"
    assert len(llm.prompts) == 1
    assert (
        kv_store.get(
            rag_module.SUMMARY_CACHE_PAYLOAD_KEY,
            collection=rag_module.SUMMARY_CACHE_NAMESPACE,
        )
        is not None
    )


def test_summarize_collection_refresh_bypasses_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """refresh=True should bypass a valid summary cache entry.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    llm = _FakeSummaryLLM("Collection summary")
    rag._text_model = llm  # type: ignore[assignment]
    _patch_summary_context(monkeypatch)

    kv_store = _InMemorySummaryKVStore()

    def _summary_kv_store(
        self: RAG,
        collection: str | None = None,
        *,
        allow_create: bool = True,
    ) -> _InMemorySummaryKVStore:
        _ = (self, collection, allow_create)
        return kv_store

    monkeypatch.setattr(RAG, "_summary_kv_store", _summary_kv_store)

    rag.summarize_collection()
    rag.summarize_collection(refresh=True)

    assert len(llm.prompts) == 2


def test_summarize_collection_prompt_fingerprint_change_forces_recompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing summary knobs should invalidate cached payloads by fingerprint.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    llm = _FakeSummaryLLM("Collection summary")
    rag._text_model = llm  # type: ignore[assignment]
    _patch_summary_context(monkeypatch)

    kv_store = _InMemorySummaryKVStore()

    def _summary_kv_store(
        self: RAG,
        collection: str | None = None,
        *,
        allow_create: bool = True,
    ) -> _InMemorySummaryKVStore:
        _ = (self, collection, allow_create)
        return kv_store

    monkeypatch.setattr(RAG, "_summary_kv_store", _summary_kv_store)

    rag.summarize_collection()
    rag.summary_per_doc_top_k += 1
    rag.summarize_collection()

    assert len(llm.prompts) == 2


def test_summarize_collection_revision_bump_invalidates_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Revision bumps should invalidate previously cached summaries.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    llm = _FakeSummaryLLM("Collection summary")
    rag._text_model = llm  # type: ignore[assignment]
    _patch_summary_context(monkeypatch)

    kv_store = _InMemorySummaryKVStore()

    def _summary_kv_store(
        self: RAG,
        collection: str | None = None,
        *,
        allow_create: bool = True,
    ) -> _InMemorySummaryKVStore:
        _ = (self, collection, allow_create)
        return kv_store

    monkeypatch.setattr(RAG, "_summary_kv_store", _summary_kv_store)

    rag.summarize_collection()
    rag._bump_summary_revision()
    rag.summarize_collection()

    assert len(llm.prompts) == 2


def test_summary_kv_store_passes_docstore_retry_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Summary KV store construction should include docstore retry settings.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    captured: dict[str, Any] = {}

    class FakeQdrantKVStore:
        """Capture kwargs passed to KV store constructor."""

        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    rag = RAG(qdrant_collection="test")
    rag._qdrant_client = MagicMock()
    rag.docstore_max_retries = 8
    rag.docstore_retry_backoff_seconds = 0.6
    rag.docstore_retry_backoff_max_seconds = 4.0

    monkeypatch.setattr(rag_module, "QdrantKVStore", FakeQdrantKVStore)

    kv_store = rag._summary_kv_store()

    assert kv_store is not None
    assert captured["max_retries"] == 8
    assert captured["retry_backoff_seconds"] == 0.6
    assert captured["retry_backoff_max_seconds"] == 4.0


class _FakeDocStore:
    """Minimal docstore stub for ingest invalidation tests."""

    def add_documents(self, nodes: list[Any], allow_update: bool = True) -> None:
        """No-op document insertion.

        Args:
            nodes: Documents to store (ignored).
            allow_update: Whether to allow overwrites (ignored).
        """
        _ = (nodes, allow_update)


class _FakeIndex:
    """Minimal index stub for ingest invalidation tests."""

    def __init__(self, **kwargs: Any) -> None:
        """Create a fake index with an in-memory docstore.

        Args:
            **kwargs: Ignored keyword arguments.
        """
        _ = kwargs
        self.docstore = _FakeDocStore()

    def insert_nodes(self, nodes: list[Any]) -> None:
        """No-op synchronous node insertion.

        Args:
            nodes: Nodes to insert (ignored).
        """
        _ = nodes

    async def ainsert_nodes(self, nodes: list[Any]) -> None:
        """No-op asynchronous node insertion.

        Args:
            nodes: Nodes to insert (ignored).
        """
        _ = nodes


class _FakePipeline:
    """Minimal pipeline stub for ingest invalidation tests."""

    def __init__(self) -> None:
        """Initialise with empty reader and extractor fields."""
        self.dir_reader = None
        self.entity_extractor = None
        self.ner_max_workers = 1

    def build(self, processed_hashes: set[str]) -> list[tuple[list[Any], list[Any]]]:
        """Return an empty batch list.

        Args:
            processed_hashes: Hashes already ingested (ignored).

        Returns:
            An empty list.
        """
        _ = processed_hashes
        return []


class _FakeCorePDFReader:
    """Minimal core reader stub for ingest invalidation tests."""

    def __init__(
        self,
        data_dir: Path,
        entity_extractor: Any = None,
        ner_max_workers: int = 1,
        source_collection: str | None = None,
        image_ingestion_service: Any = None,
    ) -> None:
        """Record constructor args without touching the filesystem.

        Args:
            data_dir: Root directory for PDF documents (ignored).
            entity_extractor: Optional NER extractor (ignored).
            ner_max_workers: Worker thread count (ignored).
            source_collection: Qdrant collection name (ignored).
            image_ingestion_service: Image ingestion service (ignored).
        """
        _ = (
            data_dir,
            entity_extractor,
            ner_max_workers,
            source_collection,
            image_ingestion_service,
        )
        self.discovered_hashes: set[str] = set()

    def build(
        self,
        existing_hashes: set[str],
        progress_callback: Any = None,
    ) -> list[tuple[list[Any], list[Any], str]]:
        """Return an empty batch list.

        Args:
            existing_hashes: Hashes already processed (ignored).
            progress_callback: Optional progress reporter (ignored).

        Returns:
            An empty list.
        """
        _ = (existing_hashes, progress_callback)
        return []


def _patch_ingest_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch heavy ingest dependencies with minimal in-memory stubs.

    Args:
        monkeypatch: The monkeypatch fixture used to stub RAG methods
            and module-level classes.
    """
    monkeypatch.setattr(RAG, "_prepare_sources_dir", lambda self, data_dir: data_dir)
    monkeypatch.setattr(RAG, "_vector_store", lambda self: object())
    monkeypatch.setattr(RAG, "_storage_context", lambda self, vector_store: object())
    monkeypatch.setattr(rag_module, "VectorStoreIndex", _FakeIndex)
    monkeypatch.setattr(
        RAG,
        "_build_ingestion_pipeline",
        lambda self, progress_callback=None: _FakePipeline(),
    )
    monkeypatch.setattr(RAG, "_get_existing_file_hashes", lambda self: set())
    monkeypatch.setattr(rag_module, "CorePDFPipelineReader", _FakeCorePDFReader)
    monkeypatch.setattr(RAG, "reset_session_state", lambda self: None)
    monkeypatch.setattr(RAG, "_invalidate_ner_cache", lambda self, collection: None)


def test_persist_node_batches_streams_micro_batches() -> None:
    """Node persistence should split ingest writes into micro-batches.

    Args:
        None.
    """

    class FakeDocStore:
        """Capture docstore write batch sizes."""

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def add_documents(self, nodes: list[Any], allow_update: bool = True) -> None:
            _ = allow_update
            self.batch_sizes.append(len(nodes))

    class FakeIndex:
        """Capture vector insert batch sizes."""

        def __init__(self) -> None:
            self.docstore = FakeDocStore()
            self.vector_batch_sizes: list[int] = []

        def insert_nodes(self, nodes: list[Any]) -> None:
            self.vector_batch_sizes.append(len(nodes))

    rag = RAG(qdrant_collection="test")
    rag.docstore_batch_size = 2
    rag.index = cast(Any, FakeIndex())

    nodes = [types.SimpleNamespace(metadata={}) for _ in range(5)]
    rag._persist_node_batches(cast(list[Any], nodes))

    assert rag.index.docstore.batch_sizes == [2, 2, 1]
    assert rag.index.vector_batch_sizes == [2, 2, 1]


def test_apersist_node_batches_streams_micro_batches() -> None:
    """Async node persistence should split ingest writes into micro-batches."""

    class FakeDocStore:
        """Capture docstore write batch sizes."""

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def add_documents(self, nodes: list[Any], allow_update: bool = True) -> None:
            _ = allow_update
            self.batch_sizes.append(len(nodes))

    class FakeIndex:
        """Capture async vector insert batch sizes."""

        def __init__(self) -> None:
            self.docstore = FakeDocStore()
            self.vector_batch_sizes: list[int] = []

        async def ainsert_nodes(self, nodes: list[Any]) -> None:
            self.vector_batch_sizes.append(len(nodes))

    rag = RAG(qdrant_collection="test")
    rag.docstore_batch_size = 3
    rag.index = cast(Any, FakeIndex())

    nodes = [types.SimpleNamespace(metadata={}) for _ in range(7)]
    asyncio.run(rag._apersist_node_batches(cast(list[Any], nodes)))

    assert rag.index.docstore.batch_sizes == [3, 3, 1]
    assert rag.index.vector_batch_sizes == [3, 3, 1]


def test_log_ingest_benchmark_summary_emits_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark helper should emit a structured ingest summary log.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    captured: list[str] = []

    monkeypatch.setattr(
        rag_module.logger,
        "info",
        lambda message, *args: captured.append(message.format(*args)),
    )

    rag._log_ingest_benchmark_summary(
        mode="sync",
        started_at=0.0,
        core_docs=2,
        core_nodes=10,
        streaming_docs=3,
        streaming_nodes=20,
        enrich_batches=4,
        persist_batches=5,
    )

    assert len(captured) == 1
    assert "Ingest benchmark (sync)" in captured[0]
    assert "nodes_per_s=" in captured[0]
    assert "ingestion_batch_size=" in captured[0]
    assert "docstore_batch_size=" in captured[0]


def test_ingest_docs_bumps_summary_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ingest_docs should bump summary revision after successful ingestion.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: The temporary path fixture.
    """
    rag = RAG(qdrant_collection="test")
    rag._embed_model = cast(Any, object())
    _patch_ingest_dependencies(monkeypatch)

    bumps: list[tuple[str | None, bool]] = []

    def _bump_summary_revision(
        self: RAG,
        collection: str | None = None,
        *,
        allow_create: bool = True,
    ) -> int:
        _ = self
        bumps.append((collection, allow_create))
        return len(bumps)

    monkeypatch.setattr(RAG, "_bump_summary_revision", _bump_summary_revision)

    rag.ingest_docs(tmp_path, build_query_engine=False)

    assert bumps == [("test", True)]


def test_asingest_docs_bumps_summary_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """asingest_docs should bump summary revision after successful ingestion.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: The temporary path fixture.
    """
    rag = RAG(qdrant_collection="test")
    rag._embed_model = cast(Any, object())
    _patch_ingest_dependencies(monkeypatch)

    bumps: list[tuple[str | None, bool]] = []

    def _bump_summary_revision(
        self: RAG,
        collection: str | None = None,
        *,
        allow_create: bool = True,
    ) -> int:
        _ = self
        bumps.append((collection, allow_create))
        return len(bumps)

    monkeypatch.setattr(RAG, "_bump_summary_revision", _bump_summary_revision)

    asyncio.run(rag.asingest_docs(tmp_path, build_query_engine=False))

    assert bumps == [("test", True)]


def test_delete_collection_attempts_summary_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """delete_collection should attempt summary revision bump before deletion.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="active")
    rag._qdrant_client = MagicMock()
    monkeypatch.setattr(RAG, "_invalidate_ner_cache", lambda self, collection: None)

    bumps: list[tuple[str | None, bool]] = []

    def _bump_summary_revision(
        self: RAG,
        collection: str | None = None,
        *,
        allow_create: bool = True,
    ) -> int:
        _ = self
        bumps.append((collection, allow_create))
        return 1

    monkeypatch.setattr(RAG, "_bump_summary_revision", _bump_summary_revision)

    rag.delete_collection("target")

    assert bumps == [("target", False)]
    deleted = [
        str(call.args[0])
        for call in rag._qdrant_client.delete_collection.call_args_list
    ]
    assert deleted == ["target", "target_images", "target_dockv"]


def test_delete_collection_companion_name_does_not_expand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting a companion collection directly should not expand to siblings."""
    rag = RAG(qdrant_collection="active")
    rag._qdrant_client = MagicMock()
    monkeypatch.setattr(RAG, "_invalidate_ner_cache", lambda self, collection: None)
    monkeypatch.setattr(
        RAG,
        "_bump_summary_revision",
        lambda self, collection=None, allow_create=True: 1,
    )

    rag.delete_collection("target_images")

    deleted = [
        str(call.args[0])
        for call in rag._qdrant_client.delete_collection.call_args_list
    ]
    assert deleted == ["target_images"]
