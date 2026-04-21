"""Unit tests for the RAG engine.

Covers response normalisation, source extraction, ingestion hashing,
session management, collection selection, NER/hate-speech helpers,
summarisation with caching and diagnostics, sparse model resolution,
reranker configuration, and summary revision invalidation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import types
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from llama_index.core import Document
from llama_index.core.schema import NodeWithScore, TextNode

from docint.core import rag as rag_module
from docint.core.rag import RAG, LazyRerankerPostprocessor
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


def test_create_text_model_disables_reasoning_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared RAG text model should omit reasoning by default.

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
        num_output=256,
        inference_provider="openai",
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

    assert captured["reasoning_effort"] is None


def test_post_retrieval_text_model_enables_reasoning_for_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-retrieval generation should use a reasoning-enabled model.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    calls: list[bool] = []

    class _FakeModel:
        """A minimal stand-in for the text model that captures whether reasoning is enabled."""

        def complete(self, prompt: str) -> _FakeCompletion:
            """Simulate the completion method of the text model.

            Args:
                prompt (str): The input prompt for the model.

            Returns:
                _FakeCompletion: A fake completion response.
            """
            _ = prompt
            return _FakeCompletion("summary")

    rag = RAG(qdrant_collection="test")
    rag.openai_config = OpenAIConfig(
        api_base="https://api.openai.com/v1",
        api_key="sk-test",
        ctx_window=200000,
        dimensions=1024,
        max_retries=2,
        num_output=256,
        inference_provider="openai",
        reuse_client=False,
        seed=42,
        temperature=0.0,
        thinking_effort="high",
        thinking_enabled=True,
        timeout=300.0,
        top_p=0.0,
    )

    def _fake_create_text_model(self: RAG, *, enable_reasoning: bool = False) -> Any:
        """Simulate the text model creation method, capturing whether reasoning is enabled.

        Args:
            self (RAG): The RAG instance for which the text model is being created.
            enable_reasoning (bool, optional): Whether reasoning is enabled for the text model. Defaults to False.

        Returns:
            Any: A fake text model instance.
        """
        _ = self
        calls.append(enable_reasoning)
        return _FakeModel()

    monkeypatch.setattr(RAG, "_create_text_model", _fake_create_text_model)

    _ = rag.text_model
    _ = rag.post_retrieval_text_model

    assert calls == [False, True]


def test_build_query_engine_uses_post_retrieval_text_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Query-engine answer generation should use the post-retrieval model.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    rag.openai_config = OpenAIConfig(
        api_base="https://api.openai.com/v1",
        api_key="sk-test",
        ctx_window=200000,
        dimensions=1024,
        max_retries=2,
        num_output=256,
        inference_provider="openai",
        reuse_client=False,
        seed=42,
        temperature=0.0,
        thinking_effort="high",
        thinking_enabled=True,
        timeout=300.0,
        top_p=0.0,
    )
    rag._post_retrieval_text_model = cast(Any, object())
    rag._reranker = cast(Any, object())
    rag.index = cast(
        Any,
        types.SimpleNamespace(
            docstore=object(),
            as_retriever=lambda **kwargs: {"retriever_kwargs": kwargs},
        ),
    )
    monkeypatch.setattr(RAG, "list_documents", lambda self: [])
    monkeypatch.setattr(RAG, "_sample_collection_payloads", lambda self, limit=128: [])

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        rag_module.RetrieverQueryEngine,
        "from_args",
        staticmethod(lambda **kwargs: captured.update(kwargs) or kwargs),
    )

    rag.build_query_engine()

    assert captured["llm"] is rag._post_retrieval_text_model


def test_build_query_engine_does_not_materialize_reranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build_query_engine`` must not dereference ``self.reranker``.

    The reranker property lazily loads bge-reranker-v2-m3 (~1 GB on CPU)
    and runs a healthcheck ``compute_score`` on first access. Plugging
    the bare property value into ``node_postprocessors`` at construction
    forced that cost even when the caller never intended to execute a
    query — the root cause of the OOM regression chain addressed by
    commits 18a47a6, 72e299e, and this commit. The ``LazyRerankerPostprocessor``
    wrapper must defer the dereference until first ``_postprocess_nodes``.
    This test defends against future reintroduction of an eager attach.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    rag.openai_config = OpenAIConfig(
        api_base="https://api.openai.com/v1",
        api_key="sk-test",
        ctx_window=200000,
        dimensions=1024,
        max_retries=2,
        num_output=256,
        inference_provider="openai",
        reuse_client=False,
        seed=42,
        temperature=0.0,
        thinking_effort="high",
        thinking_enabled=True,
        timeout=300.0,
        top_p=0.0,
    )
    rag._post_retrieval_text_model = cast(Any, object())
    # Intentionally leave rag._reranker unset — if build_query_engine reads
    # the property, the tripwire below fires.
    rag.index = cast(
        Any,
        types.SimpleNamespace(
            docstore=object(),
            as_retriever=lambda **kwargs: {"retriever_kwargs": kwargs},
        ),
    )
    monkeypatch.setattr(RAG, "list_documents", lambda self: [])
    monkeypatch.setattr(RAG, "_sample_collection_payloads", lambda self, limit=128: [])

    def _forbidden_reranker(_self: RAG) -> Any:
        """Tripwire property — must NOT be reached during query engine construction.

        Args:
            _self (RAG): The RAG instance (ignored).

        Raises:
            AssertionError: Always, indicating an illegal eager access.
        """
        raise AssertionError(
            "rag.reranker must NOT be accessed during build_query_engine; "
            "LazyRerankerPostprocessor should defer the load until first "
            "_postprocess_nodes call."
        )

    monkeypatch.setattr(RAG, "reranker", property(_forbidden_reranker))

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        rag_module.RetrieverQueryEngine,
        "from_args",
        staticmethod(lambda **kwargs: captured.update(kwargs) or kwargs),
    )

    # Would raise under the pre-refactor code path.
    rag.build_query_engine()

    postprocessors = captured["node_postprocessors"]
    assert any(isinstance(p, LazyRerankerPostprocessor) for p in postprocessors), (
        "LazyRerankerPostprocessor must be installed in node_postprocessors "
        "so the reranker load is deferred to query time."
    )


def test_unload_models_releases_audio_reader_and_dir_reader() -> None:
    """``unload_models`` must release the Whisper reference held on the audio reader.

    On CPU-only deployments ``torch.cuda.empty_cache`` is a no-op, so the
    only way Whisper weights (hundreds of MB to ~1.5 GB) return to the
    allocator is via Python ref-count drop. Previously
    ``RAG.unload_models`` nulled the embed / text / reranker / image
    services but left ``self.audio_reader`` and ``self.dir_reader``
    holding captured ingestion-pipeline state. This test locks in the
    new behaviour: the Whisper model field is nulled explicitly, and
    both reader handles are dropped so refcounting can reclaim them.

    Uses a throwaway dummy reader so no real Whisper binary is loaded.
    """
    rag = RAG(qdrant_collection="test")

    class DummyAudioReader:
        """Stand-in for an ``AudioReader`` with a Whisper-like model field."""

        def __init__(self) -> None:
            """Initialize with a non-None ``_model`` to exercise the null path."""
            self._model: Any = object()

    dummy_audio = DummyAudioReader()
    rag.audio_reader = cast(Any, dummy_audio)
    rag.dir_reader = cast(Any, object())

    rag.unload_models()

    # The Whisper reference on the captured reader must be cleared so the
    # underlying weights are GC-eligible even if something else outside
    # RAG is still holding the reader instance.
    assert dummy_audio._model is None, (
        "unload_models must null audio_reader._model so the Whisper weights "
        "become ref-count-zero on CPU-only deployments."
    )
    assert rag.audio_reader is None, (
        "unload_models must drop the audio_reader handle itself."
    )
    assert rag.dir_reader is None, (
        "unload_models must drop the dir_reader handle as well."
    )
    # Existing fields remain nulled (guards against regressions to
    # unload_models's original behaviour).
    assert rag._embed_model is None
    assert rag._text_model is None
    assert rag._post_retrieval_text_model is None, (
        "unload_models must null _post_retrieval_text_model; on OpenAI-compatible "
        "backends it holds a separate client that can pin memory/connections."
    )
    assert rag._reranker is None
    assert rag._image_ingestion_service is None


def test_lazy_reranker_postprocessor_delegates_on_call() -> None:
    """``LazyRerankerPostprocessor`` materializes and delegates lazily.

    Construction of the wrapper must NOT read ``rag.reranker``; the first
    ``_postprocess_nodes`` call must read the property exactly once and
    forward the ``nodes`` and ``query_bundle`` arguments unchanged.
    """
    accesses: list[str] = []
    forwarded: dict[str, Any] = {}

    class FakeReranker:
        """Minimal reranker stub recording delegated arguments."""

        def _postprocess_nodes(self, nodes: list[Any], query_bundle: Any) -> list[Any]:
            """Record the forwarded arguments and return nodes unchanged.

            Args:
                nodes (list[Any]): Nodes forwarded from the wrapper.
                query_bundle (Any): Query bundle forwarded from the wrapper.

            Returns:
                list[Any]: The input ``nodes`` unchanged.
            """
            forwarded["nodes"] = nodes
            forwarded["query_bundle"] = query_bundle
            return nodes

    class FakeRAG:
        """Minimal RAG stub with a reranker property access counter."""

        @property
        def reranker(self) -> FakeReranker:
            """Track access to the reranker property.

            Returns:
                FakeReranker: A fresh stub reranker each call.
            """
            accesses.append("reranker")
            return FakeReranker()

    rag = FakeRAG()
    wrapper = LazyRerankerPostprocessor(rag=rag)
    assert accesses == [], "construction must not touch rag.reranker"

    sentinel_nodes: list[Any] = [object(), object()]
    sentinel_bundle = object()
    result = wrapper._postprocess_nodes(sentinel_nodes, sentinel_bundle)

    assert accesses == ["reranker"], (
        "rag.reranker must be read exactly once on first delegated call"
    )
    assert forwarded["nodes"] is sentinel_nodes
    assert forwarded["query_bundle"] is sentinel_bundle
    assert result is sentinel_nodes


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
    """Normalization should hide upstream empty-response sentinels.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
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
    """Normalization should summarize matching sources instead of claiming no context.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
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
            """Initialize the DummyImageService with a flag to track if it was called."""
            self.called = False

        def _resolve_collection_name(self, source_collection: str | None = None) -> str:
            """Resolve the collection name for the given source collection.

            Args:
                source_collection: The name of the source collection, or None for default.
            """
            return f"{source_collection}_images"

        def query_similar_images_by_text(
            self,
            query_text: str,
            top_k: int = 3,
            *,
            source_collection: str | None = None,
        ) -> list[dict[str, Any]]:
            """Query for similar images based on the provided text and optional source collection.

            Args:
                query_text (str): The text query to find similar images for.
                top_k (int, optional): The maximum number of similar images to return. Defaults to 3.
                source_collection (str | None, optional): The name of the source collection to query. Defaults to None.

            Returns:
                list[dict[str, Any]]: The list of similar images.

            Raises:
                AssertionError: If the method is called unexpectedly.
            """
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
    rag.ner_aggregate_cache[("alpha", "orthographic")] = {"entities": []}
    rag.ner_graph_cache[("alpha", "orthographic", 100, 1)] = {
        "nodes": [],
        "edges": [],
        "meta": {},
    }
    monkeypatch.setattr(
        RAG,
        "list_collections",
        lambda self, prefer_api=True: ["alpha", "beta"],
    )

    rag.select_collection("beta")

    assert rag.ner_sources == []
    assert rag.ner_aggregate_cache.get(("alpha", "orthographic")) is None
    assert ("alpha", "orthographic", 100, 1) not in rag.ner_graph_cache


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
    assert stats["top_entities"][0]["variant_count"] == 2

    loc_stats = rag.get_collection_ner_stats(
        top_k=10, min_mentions=1, entity_type="loc", include_relations=False
    )
    assert loc_stats["totals"]["unique_entities"] == 1
    assert loc_stats["totals"]["unique_relations"] == 0

    results = rag.search_collection_ner_entities(q="ac", limit=10)
    assert results[0]["text"] == "Acme"
    assert results[0]["mentions"] == 3
    assert results[0]["variant_count"] == 2


def test_collection_ner_stats_condense_orthographic_variants() -> None:
    """Orthographic variants should condense into one canonical entity by default."""
    rag = RAG(qdrant_collection="test")
    rag.ner_sources = [
        {
            "filename": "a.pdf",
            "entities": [{"text": "Parteitag", "type": "EVENT", "score": 0.7}],
            "relations": [{"head": "Parteitag", "label": "in", "tail": "Berlin"}],
        },
        {
            "filename": "b.pdf",
            "entities": [{"text": "Partei Tag", "type": "EVENT", "score": 0.9}],
            "relations": [{"head": "Partei Tag", "label": "in", "tail": "Berlin"}],
        },
        {
            "filename": "c.pdf",
            "entities": [{"text": "Parteitag", "type": "EVENT", "score": 0.6}],
        },
    ]

    stats = rag.get_collection_ner_stats(top_k=10, min_mentions=1)
    assert stats["totals"]["unique_entities"] == 1
    assert stats["top_entities"][0]["text"] == "Parteitag"
    assert stats["top_entities"][0]["mentions"] == 3
    assert stats["top_entities"][0]["variant_count"] == 2
    assert {row["text"] for row in stats["top_entities"][0]["variants"]} == {
        "Parteitag",
        "Partei Tag",
    }

    results = rag.search_collection_ner_entities(q="partei tag", limit=10)
    assert len(results) == 1
    assert results[0]["text"] == "Parteitag"
    assert results[0]["mentions"] == 3


def test_collection_ner_stats_exact_mode_preserves_orthographic_variants() -> None:
    """Exact mode should keep orthographic variants split."""
    rag = RAG(qdrant_collection="test")
    rag.ner_sources = [
        {"filename": "a.pdf", "entities": [{"text": "Parteitag", "type": "EVENT"}]},
        {"filename": "b.pdf", "entities": [{"text": "Partei Tag", "type": "EVENT"}]},
    ]

    stats = rag.get_collection_ner_stats(
        top_k=10,
        min_mentions=1,
        entity_merge_mode="exact",
    )

    assert stats["totals"]["unique_entities"] == 2
    assert {row["text"] for row in stats["top_entities"]} == {
        "Parteitag",
        "Partei Tag",
    }


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


def test_run_entity_occurrence_query_includes_orthographic_variants() -> None:
    """Occurrence mode should include all mentions from condensed variants."""
    rag = RAG(qdrant_collection="test")
    rag.ner_sources = [
        {
            "filename": "a.pdf",
            "chunk_id": "chunk-1",
            "chunk_text": "Parteitag appears here.",
            "text": "Parteitag appears here.",
            "entities": [{"text": "Parteitag", "type": "EVENT", "score": 0.8}],
        },
        {
            "filename": "b.pdf",
            "chunk_id": "chunk-2",
            "chunk_text": "Partei Tag appears here.",
            "text": "Partei Tag appears here.",
            "entities": [{"text": "Partei Tag", "type": "EVENT", "score": 0.9}],
        },
    ]

    result = rag.run_entity_occurrence_query("Partei Tag")

    assert result["retrieval_mode"] == "entity_occurrence"
    assert len(result["sources"]) == 2
    assert result["sources"][0]["matched_entity"]["text"] == "Partei Tag"
    assert result["sources"][0]["matched_entity"]["variant_count"] == 2
    assert {
        mention["text"]
        for source in result["sources"]
        for mention in source["matched_mentions"]
    } == {
        "Parteitag",
        "Partei Tag",
    }


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


def test_collection_ner_graph_merges_orthographic_relation_nodes() -> None:
    """Graph construction should merge orthographic variants before building edges."""
    rag = RAG(qdrant_collection="test")
    rag.ner_sources = [
        {
            "filename": "a.pdf",
            "entities": [
                {"text": "Parteitag", "type": "EVENT"},
                {"text": "Berlin", "type": "LOC"},
            ],
            "relations": [{"head": "Parteitag", "label": "in", "tail": "Berlin"}],
        },
        {
            "filename": "b.pdf",
            "entities": [
                {"text": "Partei Tag", "type": "EVENT"},
                {"text": "Berlin", "type": "LOC"},
            ],
            "relations": [{"head": "Partei Tag", "label": "in", "tail": "Berlin"}],
        },
    ]

    graph = rag.get_collection_ner_graph(top_k_nodes=10, min_edge_weight=1)

    assert {node["text"] for node in graph["nodes"]} == {"Parteitag", "Berlin"}
    relation_edges = [edge for edge in graph["edges"] if edge["kind"] == "relation"]
    assert len(relation_edges) == 1
    assert relation_edges[0]["weight"] == 2


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
        lambda self, refresh=False, entity_merge_mode="orthographic": {
            "entities": [
                {"text": "Acme", "mentions": 10},
                {"text": "Widget", "mentions": 3},
            ]
        },
    )
    monkeypatch.setattr(
        RAG,
        "get_collection_ner_graph_neighbors",
        lambda self,
        *,
        entity,
        hops,
        top_k_nodes,
        min_edge_weight,
        refresh=False,
        entity_merge_mode="orthographic": {
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
        lambda self, refresh=False, entity_merge_mode="orthographic": {
            "entities": [
                {"text": "European Union", "mentions": 10},
                {"text": "Brussels", "mentions": 4},
            ]
        },
    )
    monkeypatch.setattr(
        RAG,
        "get_collection_ner_graph_neighbors",
        lambda self,
        *,
        entity,
        hops,
        top_k_nodes,
        min_edge_weight,
        refresh=False,
        entity_merge_mode="orthographic": {"neighbors": [{"text": "Brussels"}]},
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
        """A fake memory class for testing purposes."""

        def __init__(self) -> None:
            """Initialize the FakeMemory with an empty message list."""
            self.messages: list[object] = []

        def put(self, message) -> None:
            """Add a message to the FakeMemory."""
            self.messages.append(message)

    class FakeChatEngine:
        """A fake chat engine class for testing purposes."""

        def __init__(self, **kwargs) -> None:
            """Initialize the FakeChatEngine with the provided keyword arguments.

            Args:
                **kwargs: Arbitrary keyword arguments to store in the instance.
            """

            self.kwargs = kwargs

        @classmethod
        def from_defaults(cls, **kwargs) -> FakeChatEngine:
            """Create a FakeChatEngine instance from default settings.

            Args:
                **kwargs: Arbitrary keyword arguments to pass to the constructor.

            Returns:
                An instance of FakeChatEngine initialized with the provided keyword arguments.
            """
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
        """Simulate a broken SparseTextEmbedding.list_supported_models method.

        Raises:
            ImportError: Always raised to simulate the absence of fastembed.
        """
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
    """LLM-NER and hate-speech should share one cached text model instance.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: The temporary path fixture.
    """
    rag = RAG(qdrant_collection="test")
    rag.data_dir = tmp_path
    rag.ner_enabled = True
    rag.openai_inference_provider = "openai"

    monkeypatch.setattr(
        rag_module,
        "load_hate_speech_env",
        lambda: types.SimpleNamespace(enabled=True),
    )

    created: list[object] = []

    def _fake_create_text_model(self: RAG) -> object:
        """Create a fake text model instance for testing purposes.

        Args:
            self: The RAG instance.

        Returns:
            A fake text model instance.
        """

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
    """Non-OpenAI provider should not wire LLM NER model into the pipeline.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: The temporary path fixture.
    """
    rag = RAG(qdrant_collection="test")
    rag.data_dir = tmp_path
    rag.ner_enabled = True
    rag.openai_inference_provider = "ollama"

    monkeypatch.setattr(
        rag_module,
        "load_hate_speech_env",
        lambda: types.SimpleNamespace(enabled=True),
    )

    created: list[object] = []

    def _fake_create_text_model(self: RAG) -> object:
        """Create a fake text model instance for testing purposes.

        Args:
            self: The RAG instance.

        Returns:
            A fake text model instance.
        """

        model = object()
        created.append(model)
        return model

    monkeypatch.setattr(RAG, "_create_text_model", _fake_create_text_model)

    pipeline = rag._build_ingestion_pipeline()

    assert pipeline.ner_model is None
    assert pipeline.hate_speech_model is created[0]
    assert len(created) == 1


def test_device_uses_use_device_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RAG device selection should honor the explicit runtime override.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    monkeypatch.setenv("USE_DEVICE", "cpu")
    monkeypatch.setattr(rag_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rag_module.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(rag_module.torch.backends.mps, "is_built", lambda: False)

    rag = RAG(qdrant_collection="test")

    assert rag.device == "cpu"


def test_build_ingestion_pipeline_passes_configured_device_to_gliner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Non-OpenAI ingestion should forward the resolved device to GLiNER.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: The temporary path fixture.
    """
    import docint.core.ingest.ingestion_pipeline as pipeline_module

    monkeypatch.setenv("USE_DEVICE", "cpu")
    monkeypatch.setenv("NER_ENABLED", "true")

    captured: dict[str, Any] = {}

    def _fake_build_gliner_ner_extractor(
        labels: list[str] | None = None,
        threshold: float = 0.3,
        device: str | None = None,
    ) -> object:
        """Capture GLiNER device selection during pipeline construction.

        Args:
            labels: Requested GLiNER labels.
            threshold: Requested GLiNER threshold.
            device: Requested execution device.

        Returns:
            object: Placeholder extractor object.
        """
        del labels, threshold
        captured["device"] = device
        return object()

    rag = RAG(qdrant_collection="test")
    rag.data_dir = tmp_path
    rag.openai_inference_provider = "ollama"

    monkeypatch.setattr(
        rag_module,
        "load_hate_speech_env",
        lambda: types.SimpleNamespace(enabled=False),
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_gliner_ner_extractor",
        _fake_build_gliner_ner_extractor,
    )
    monkeypatch.setattr(
        rag_module,
        "ImageIngestionService",
        lambda device: types.SimpleNamespace(device=device),
    )

    pipeline = rag._build_ingestion_pipeline()

    assert pipeline.entity_extractor is not None
    assert captured["device"] == "cpu"
    assert getattr(rag._image_ingestion_service, "device", None) == "cpu"

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
        """A fake FlagEmbeddingReranker for testing purposes."""

        def __init__(self, top_n: int, model: str, use_fp16: bool) -> None:
            """Initialize the FakeFlagReranker and capture the initialization parameters.

            Args:
                top_n (int): The number of top results to rerank.
                model (str): The model identifier to use for reranking.
                use_fp16 (bool): Whether to use fp16 precision for the model.
            """
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
    rag.openai_inference_provider = "ollama"
    rag.rerank_use_fp16 = True
    rag.rerank_top_n = 7

    _ = rag.reranker

    assert captured["top_n"] == 7
    assert captured["model"] == rag.rerank_model_id
    assert captured["use_fp16"] is True


def test_openai_reranker_uses_flag_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI provider should use the local FlagEmbedding reranker by default.

    Args:
        monkeypatch: The monkeypatch fixture.
    """

    captured: dict[str, object] = {}

    class FakeFlagReranker:
        """A fake FlagEmbeddingReranker for testing purposes."""

        def __init__(self, top_n: int, model: str, use_fp16: bool) -> None:
            """Initialize the FakeFlagReranker and capture the initialization parameters.

            Args:
                top_n (int): The number of top results to rerank.
                model (str): The model identifier to use for reranking.
                use_fp16 (bool): Whether to use fp16 precision for the model.
            """
            captured["top_n"] = top_n
            captured["model"] = model
            captured["use_fp16"] = use_fp16
            self._model = types.SimpleNamespace(compute_score=lambda _: [0.0])

    def fail_llm_rerank(*args: object, **kwargs: object) -> object:
        """A fake LLMRerank that raises an error if used.

        Raises:
            AssertionError: Always raised to indicate that LLMRerank should not be used.
        """
        raise AssertionError("LLMRerank should not be used for the openai provider")

    monkeypatch.setattr(rag_module, "FlagEmbeddingReranker", FakeFlagReranker)
    monkeypatch.setattr(rag_module, "LLMRerank", fail_llm_rerank)
    monkeypatch.setattr(
        rag_module,
        "resolve_hf_cache_path",
        lambda cache_dir, repo_id: None,
    )

    rag = RAG(qdrant_collection="test")
    rag.openai_inference_provider = "openai"
    rag.rerank_top_n = 6

    _ = rag.reranker

    assert captured["top_n"] == 6
    assert captured["model"] == rag.rerank_model_id


def test_vllm_reranker_uses_remote_postprocessor() -> None:
    """vLLM provider should use the remote rerank endpoint postprocessor."""

    rag = RAG(qdrant_collection="test")
    rag.openai_inference_provider = "vllm"
    rag.openai_api_base = "http://router:8000/v1"
    rag.openai_api_key = "token-abc123"
    rag.rerank_top_n = 4

    reranker = rag.reranker

    assert isinstance(reranker, rag_module.VLLMRerankPostprocessor)
    assert reranker.api_base == "http://router:8000/v1"
    assert reranker.api_key == "token-abc123"
    assert reranker.model == rag.rerank_model_id
    assert reranker.top_n == 4


def test_vllm_reranker_reorders_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """The vLLM reranker should reorder nodes using returned result indices.

    Args:
        monkeypatch: The monkeypatch fixture.
    """

    captured: dict[str, object] = {}

    class FakeResponse:
        """A fake HTTP response object that simulates the expected output of the vLLM rerank endpoint."""

        def __enter__(self) -> "FakeResponse":
            """A context manager enter method that returns self.

            Returns:
                FakeResponse: The fake response object itself.
            """
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            """A context manager exit method that does nothing.

            Args:
                exc_type (object): The exception type, if any.
                exc (object): The exception instance, if any.
                tb (object): The traceback object, if any.
            """
            _ = (exc_type, exc, tb)

        def read(self) -> bytes:
            """Simulates reading the response body, returning a JSON-encoded byte string with relevance scores and indices.

            Returns:
                bytes: A JSON-encoded byte string containing the reranking results.
            """
            return json.dumps(
                {
                    "results": [
                        {"index": 2, "relevance_score": 0.91},
                        {"index": 0, "relevance_score": 0.44},
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        """A fake urlopen function that captures the request details and simulates a response from the vLLM rerank endpoint.

        Args:
            request (urllib.request.Request): The HTTP request object sent to the rerank endpoint.
            timeout (float): The timeout value for the request.

        Returns:
            FakeResponse: A fake response object containing the simulated reranking results.
        """
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["json"] = json.loads(
            request.data.decode("utf-8") if isinstance(request.data, bytes) else "{}"
        )
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(rag_module.urllib.request, "urlopen", fake_urlopen)

    reranker = rag_module.VLLMRerankPostprocessor(
        api_base="http://router:8000/v1",
        api_key="secret",
        model="BAAI/bge-reranker-v2-m3",
        timeout=12.5,
        top_n=2,
    )
    nodes = [
        NodeWithScore(node=TextNode(text="alpha"), score=0.1),
        NodeWithScore(node=TextNode(text="beta"), score=0.2),
        NodeWithScore(node=TextNode(text="gamma"), score=0.3),
    ]

    reranked = reranker.postprocess_nodes(
        nodes,
        query_bundle=rag_module.QueryBundle(query_str="which greek letter?"),
    )

    assert [node.node.get_content() for node in reranked] == ["gamma", "alpha"]
    assert [node.score for node in reranked] == [0.91, 0.44]
    assert captured["url"] == "http://router:8000/v1/rerank"
    assert captured["headers"] == {
        "Content-type": "application/json",
        "Authorization": "Bearer secret",
    }
    assert captured["json"] == {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "which greek letter?",
        "documents": ["alpha", "beta", "gamma"],
        "top_n": 2,
    }
    assert captured["timeout"] == 12.5


def test_vllm_reranker_falls_back_to_original_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vLLM reranker should degrade to original ordering when the call fails.

    Args:
        monkeypatch: The monkeypatch fixture.
    """

    def fake_urlopen(request: object, timeout: float) -> object:
        _ = (request, timeout)
        raise urllib.error.URLError("upstream down")

    monkeypatch.setattr(rag_module.urllib.request, "urlopen", fake_urlopen)

    reranker = rag_module.VLLMRerankPostprocessor(
        api_base="http://router:8000/v1",
        model="BAAI/bge-reranker-v2-m3",
        top_n=2,
    )
    nodes = [
        NodeWithScore(node=TextNode(text="alpha"), score=0.3),
        NodeWithScore(node=TextNode(text="beta"), score=0.2),
        NodeWithScore(node=TextNode(text="gamma"), score=0.1),
    ]

    reranked = reranker.postprocess_nodes(
        nodes,
        query_bundle=rag_module.QueryBundle(query_str="which greek letter?"),
    )

    assert [node.node.get_content() for node in reranked] == ["alpha", "beta"]


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

    monkeypatch.setattr(rag_module, "TruncatingOpenAIEmbedding", FakeOpenAIEmbedding)
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

    monkeypatch.setattr(rag_module, "TruncatingOpenAIEmbedding", FakeOpenAIEmbedding)

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
    rag.openai_inference_provider = "ollama"
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
    """Build a dummy row-level social source node for summary tests.

    Args:
        text: The text content of the node.
        filename: The name of the source file.
        file_hash: The hash of the source file.
        text_id: The unique identifier for the social media text (e.g., comment ID).
        author: The author of the social media text.
        author_id: The unique identifier for the author.
        timestamp: The timestamp of the social media text.
        row: The row index in the table.
        score: The relevance score. Defaults to 0.9.

    Returns:
        A ``DummyNodeWithScore`` with the specified metadata.
    """
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
    """Social/table-heavy collections should use grounded refine synthesis.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    rag._text_model = cast(Any, object())
    rag._reranker = cast(Any, object())
    rag.index = cast(
        Any,
        types.SimpleNamespace(
            as_retriever=lambda **kwargs: {"retriever_kwargs": kwargs}
        ),
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
        """Fake implementation of the from_args method.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            object: A sentinel object to simulate the return value.
        """
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
    """Hybrid-capable collections should default to actual hybrid retrieval.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test", enable_hybrid=True)
    rag._text_model = cast(Any, object())
    rag._reranker = cast(Any, object())

    captured_retriever_kwargs: dict[str, Any] = {}
    rag.index = cast(
        Any,
        types.SimpleNamespace(
            docstore=object(),
            as_retriever=lambda **kwargs: (
                captured_retriever_kwargs.update(kwargs) or {"retriever_kwargs": kwargs}
            ),
        ),
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
    """Hierarchical collections should expand fine hits to parent context.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    rag._text_model = cast(Any, object())
    rag._reranker = cast(Any, object())
    rag.index = cast(
        Any,
        types.SimpleNamespace(
            docstore=object(),
            as_retriever=lambda **kwargs: {"retriever_kwargs": kwargs},
        ),
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
            get_node=lambda node_id, raise_error=False: (
                parent if node_id == "parent-1" else None
            )
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
    rag._post_retrieval_text_model = _FakeSummaryLLM("Collection summary")  # type: ignore[assignment]

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

    monkeypatch.setattr(
        RAG,
        "_infer_collection_profile",
        lambda self: {"is_social_table": False, "coverage_unit": "documents"},
    )
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
    """Social summaries should report post-level coverage and preserve source distinctions.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    llm = _FakeSummaryLLM("Social summary")
    rag._post_retrieval_text_model = llm  # type: ignore[assignment]
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
    assert "author=Alice" in llm.prompts[0]
    assert "author=Bob" in llm.prompts[0]


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
    rag._post_retrieval_text_model = _FakeSummaryLLM("unused")  # type: ignore[assignment]
    rag.summary_coverage_target = 0.7
    rag.summary_max_docs = 30

    monkeypatch.setattr(
        RAG,
        "_infer_collection_profile",
        lambda self: {"is_social_table": False, "coverage_unit": "documents"},
    )
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
    rag._post_retrieval_text_model = llm  # type: ignore[assignment]
    _patch_summary_context(monkeypatch)

    kv_store = _InMemorySummaryKVStore()

    def _summary_kv_store(
        self: RAG,
        collection: str | None = None,
        *,
        allow_create: bool = True,
    ) -> _InMemorySummaryKVStore:
        """Return the in-memory summary KV store.

        Args:
            self (RAG): The RAG instance.
            collection (str | None, optional): The collection name. Defaults to None.
            allow_create (bool, optional): Whether to allow creation of a new store. Defaults to True.

        Returns:
            _InMemorySummaryKVStore: _description_
        """
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
    rag._post_retrieval_text_model = llm  # type: ignore[assignment]
    _patch_summary_context(monkeypatch)

    kv_store = _InMemorySummaryKVStore()

    def _summary_kv_store(
        self: RAG,
        collection: str | None = None,
        *,
        allow_create: bool = True,
    ) -> _InMemorySummaryKVStore:
        """Return the in-memory summary KV store.

        Args:
            self (RAG): The RAG instance.
            collection (str | None, optional): The collection name. Defaults to None.
            allow_create (bool, optional): Whether to allow creation of a new store. Defaults to True.

        Returns:
            _InMemorySummaryKVStore: The in-memory summary KV store.
        """
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
    rag._post_retrieval_text_model = llm  # type: ignore[assignment]
    _patch_summary_context(monkeypatch)

    kv_store = _InMemorySummaryKVStore()

    def _summary_kv_store(
        self: RAG,
        collection: str | None = None,
        *,
        allow_create: bool = True,
    ) -> _InMemorySummaryKVStore:
        """Return the in-memory summary KV store.

        Args:
            self (RAG): The RAG instance.
            collection (str | None, optional): The collection name. Defaults to None.
            allow_create (bool, optional): Whether to allow creation of a new store. Defaults to True.

        Returns:
            _InMemorySummaryKVStore: The in-memory summary KV store.
        """
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
    rag._post_retrieval_text_model = llm  # type: ignore[assignment]
    _patch_summary_context(monkeypatch)

    kv_store = _InMemorySummaryKVStore()

    def _summary_kv_store(
        self: RAG,
        collection: str | None = None,
        *,
        allow_create: bool = True,
    ) -> _InMemorySummaryKVStore:
        """Return the in-memory summary KV store.

        Args:
            self (RAG): The RAG instance.
            collection (str | None, optional): The collection name. Defaults to None.
            allow_create (bool, optional): Whether to allow creation of a new store. Defaults to True.

        Returns:
            _InMemorySummaryKVStore: The in-memory summary KV store.
        """
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

    Verifies that the SQLite KV store built by ``_summary_kv_store``
    receives the retry knobs configured on the :class:`RAG` instance.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    captured: dict[str, Any] = {}

    class FakeSQLiteKVStore:
        """Capture kwargs passed to the KV store constructor."""

        def __init__(self, **kwargs: Any) -> None:
            """Record constructor kwargs for later assertion.

            Args:
                **kwargs: Keyword arguments forwarded to
                    :class:`SQLiteKVStore`.
            """
            captured.update(kwargs)

    rag = RAG(qdrant_collection="test")
    rag._qdrant_client = MagicMock()
    rag.docstore_max_retries = 8
    rag.docstore_retry_backoff_seconds = 0.6
    rag.docstore_retry_backoff_max_seconds = 4.0

    monkeypatch.setattr(rag_module, "SQLiteKVStore", FakeSQLiteKVStore)

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

    Treats the active collection as already existing in Qdrant so that the
    empty-ingestion guard (which fires only when nothing was produced *and*
    the main collection does not yet exist) does not short-circuit tests
    that exercise the post-ingest success bookkeeping path. Tests that
    intentionally exercise the empty-ingestion cleanup path should
    re-stub ``qdrant_collection_exists`` to return ``False``.

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
    monkeypatch.setattr(
        rag_module,
        "qdrant_collection_exists",
        lambda client, collection_name: True,
    )


def test_persist_node_batches_streams_micro_batches() -> None:
    """Node persistence should split ingest writes into micro-batches.

    Args:
        None.
    """

    class FakeDocStore:
        """Capture docstore write batch sizes."""

        def __init__(self) -> None:
            """Initialise an empty list to capture batch sizes."""
            self.batch_sizes: list[int] = []

        def add_documents(self, nodes: list[Any], allow_update: bool = True) -> None:
            """Capture the size of each batch of documents added to the docstore.

            Args:
                nodes (list[Any]): The list of nodes being added to the docstore.
                allow_update (bool, optional): Whether to allow updating existing nodes. Defaults to True.
            """
            _ = allow_update
            self.batch_sizes.append(len(nodes))

    class FakeIndex:
        """Capture vector insert batch sizes."""

        def __init__(self) -> None:
            """Initialise an empty list to capture batch sizes."""
            self.docstore = FakeDocStore()
            self.vector_batch_sizes: list[int] = []

        def insert_nodes(self, nodes: list[Any]) -> None:
            """Capture the size of each batch of nodes inserted into the vector store.

            Args:
                nodes (list[Any]): The list of nodes being inserted into the vector store.
            """
            self.vector_batch_sizes.append(len(nodes))

    rag = RAG(qdrant_collection="test")
    rag.docstore_batch_size = 2
    rag.index = cast(Any, FakeIndex())

    nodes = [types.SimpleNamespace(metadata={}) for _ in range(5)]
    rag._persist_node_batches(cast(list[Any], nodes))
    index = rag.index

    assert index.docstore.batch_sizes == [2, 2, 1]
    assert index.vector_batch_sizes == [2, 2, 1]


def test_apersist_node_batches_streams_micro_batches() -> None:
    """Async node persistence should split ingest writes into micro-batches."""

    class FakeDocStore:
        """Capture docstore write batch sizes."""

        def __init__(self) -> None:
            """Initialise an empty list to capture batch sizes."""
            self.batch_sizes: list[int] = []

        def add_documents(self, nodes: list[Any], allow_update: bool = True) -> None:
            """Capture the size of each batch of documents added to the docstore.

            Args:
                nodes (list[Any]): The list of nodes being added to the docstore.
                allow_update (bool, optional): Whether to allow updating existing nodes. Defaults to True.
            """
            _ = allow_update
            self.batch_sizes.append(len(nodes))

    class FakeIndex:
        """Capture async vector insert batch sizes."""

        def __init__(self) -> None:
            """Initialise an empty list to capture batch sizes."""
            self.docstore = FakeDocStore()
            self.vector_batch_sizes: list[int] = []

        async def ainsert_nodes(self, nodes: list[Any]) -> None:
            """Capture the size of each batch of nodes inserted into the vector store.

            Args:
                nodes (list[Any]): The list of nodes being inserted into the vector store.
            """
            self.vector_batch_sizes.append(len(nodes))

    rag = RAG(qdrant_collection="test")
    rag.docstore_batch_size = 3
    rag.index = cast(Any, FakeIndex())

    nodes = [types.SimpleNamespace(metadata={}) for _ in range(7)]
    asyncio.run(rag._apersist_node_batches(cast(list[Any], nodes)))
    index = cast(Any, rag.index)

    assert index.docstore.batch_sizes == [3, 3, 1]
    assert index.vector_batch_sizes == [3, 3, 1]


def test_persist_node_batches_skips_unembeddable_chunks() -> None:
    """Persistence should drop chunks whose embeddings must be skipped."""

    class FakeDocStore:
        """Capture persisted nodes for verification."""

        def __init__(self) -> None:
            """Initialise an empty capture list."""
            self.persisted_nodes: list[list[TextNode]] = []

        def add_documents(
            self, nodes: list[TextNode], allow_update: bool = True
        ) -> None:
            """Record persisted nodes for each batch.

            Args:
                nodes: Nodes persisted to the docstore.
                allow_update: Whether overwrites are allowed.
            """
            _ = allow_update
            self.persisted_nodes.append(nodes)

    class FakeIndex:
        """Capture vector inserts for verification."""

        def __init__(self) -> None:
            """Initialise the fake index state."""
            self.docstore = FakeDocStore()
            self.vector_batches: list[list[TextNode]] = []

        def insert_nodes(self, nodes: list[TextNode]) -> None:
            """Record vector nodes for each batch.

            Args:
                nodes: Nodes inserted into the vector store.
            """
            self.vector_batches.append(nodes)

    class FakeEmbedModel:
        """Return one embedding and one skipped result."""

        def get_text_embeddings_with_skips(
            self, texts: list[str]
        ) -> list[list[float] | None]:
            """Return aligned embeddings with one skipped entry.

            Args:
                texts: Texts to embed.

            Returns:
                list[list[float] | None]: Embeddings aligned to ``texts``.
            """
            assert len(texts) == 2
            return [[0.1, 0.2], None]

    rag = RAG(qdrant_collection="test")
    rag.docstore_batch_size = 10
    rag.index = cast(Any, FakeIndex())
    rag._embed_model = cast(Any, FakeEmbedModel())

    ok_node = TextNode(text="ok text", metadata={"chunk_id": "chunk-ok"})
    bad_node = TextNode(text="bad text", metadata={"chunk_id": "chunk-bad"})

    rag._persist_node_batches([ok_node, bad_node])
    index = cast(Any, rag.index)

    assert len(index.docstore.persisted_nodes) == 1
    assert index.docstore.persisted_nodes[0] == [ok_node]
    assert len(index.vector_batches) == 1
    assert index.vector_batches[0] == [ok_node]
    assert ok_node.embedding == [0.1, 0.2]
    assert bad_node.embedding is None


def _fake_prepare_vector_nodes_for_insert(
    _self: RAG, vector_nodes: list[Any]
) -> tuple[list[Any], set[int]]:
    """Return vector nodes untouched with no skipped IDs.

    Used as a class-level monkeypatch so that the prepared-vector-nodes
    helper does not try to load an embedding model during unit tests.

    Args:
        _self: Unused ``RAG`` instance (bound method signature).
        vector_nodes: Incoming vector nodes.

    Returns:
        A ``(prepared_nodes, skipped_ids)`` tuple where no nodes are
        skipped and ``prepared_nodes`` mirrors the input list.
    """
    return (list(vector_nodes), set())


def _capture_loguru(caplog: pytest.LogCaptureFixture) -> Callable[[], None]:
    """Route ``loguru`` logs into pytest's ``caplog`` handler.

    The rag module uses ``loguru``, which does not propagate to the
    stdlib ``logging`` hierarchy by default.  This helper installs a
    sink on the loguru logger that forwards every record into caplog's
    captured handler and returns a cleanup callable.

    Args:
        caplog: Pytest log capture fixture.

    Returns:
        A zero-argument cleanup callable that removes the sink.
    """
    from loguru import logger as _loguru_logger

    sink_id = _loguru_logger.add(
        lambda message: caplog.records.append(
            logging.LogRecord(
                name="loguru",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg=str(message),
                args=None,
                exc_info=None,
            )
        ),
        level="ERROR",
        format="{message}",
    )

    def _cleanup() -> None:
        _loguru_logger.remove(sink_id)

    return _cleanup


def test_persist_node_batches_logs_orphaned_kv_nodes_on_vector_failure(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Log node IDs and re-raise when the vector write fails after a KV write.

    Simulates a transient Qdrant failure on ``insert_nodes`` after the
    docstore write has committed, and asserts that the ``orphaned_kv_nodes``
    marker appears in the logs with the affected node IDs so operators
    can diagnose what needs to be re-ingested.

    Args:
        caplog: Pytest log capture fixture.
        monkeypatch: Pytest monkeypatch fixture used to stub the
            embedding-prep helper at the class level (RAG uses
            ``@dataclass(slots=True)`` so instance-level assignment
            is not possible).
    """

    class FakeDocStore:
        """Record persisted node IDs."""

        def __init__(self) -> None:
            """Initialise empty capture state."""
            self.persisted: list[str] = []

        def add_documents(self, nodes: list[Any], allow_update: bool = True) -> None:
            """Record node IDs for each persisted batch.

            Args:
                nodes: Nodes being persisted to the docstore.
                allow_update: Whether overwrites are allowed.
            """
            _ = allow_update
            self.persisted.extend(n.node_id for n in nodes)

    class FakeIndex:
        """Fail on vector insert to simulate a Qdrant outage."""

        def __init__(self) -> None:
            """Initialise the fake index state."""
            self.docstore = FakeDocStore()

        def insert_nodes(self, nodes: list[Any]) -> None:
            """Raise a transient error on every vector insert.

            Args:
                nodes: Nodes the caller is trying to insert.

            Raises:
                RuntimeError: Always, to simulate a vector-store failure.
            """
            _ = nodes
            raise RuntimeError("qdrant down")

    monkeypatch.setattr(
        RAG,
        "_prepare_vector_nodes_for_insert",
        _fake_prepare_vector_nodes_for_insert,
    )

    rag = RAG(qdrant_collection="active")
    rag.docstore_batch_size = 10
    rag.index = cast(Any, FakeIndex())

    node = TextNode(text="hello world", metadata={}, id_="node-1")

    cleanup = _capture_loguru(caplog)
    try:
        with pytest.raises(RuntimeError, match="qdrant down"):
            rag._persist_node_batches([node])
    finally:
        cleanup()

    index = cast(Any, rag.index)
    # The docstore write committed before the vector insert failed.
    assert index.docstore.persisted == ["node-1"]
    # The structured marker and node id appear in the logs.
    combined = "\n".join(str(record.msg) for record in caplog.records)
    assert "orphaned_kv_nodes" in combined
    assert "node-1" in combined
    assert "'active'" in combined


def test_persist_node_batches_logs_failed_persist_on_docstore_failure(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Log node IDs and re-raise when the docstore write itself fails.

    Args:
        caplog: Pytest log capture fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """

    class FakeDocStore:
        """Raise on every persistence attempt."""

        def add_documents(self, nodes: list[Any], allow_update: bool = True) -> None:
            """Raise a disk-full error on every batch.

            Args:
                nodes: Nodes being persisted.
                allow_update: Ignored flag.

            Raises:
                OSError: Always, to simulate a disk-full condition.
            """
            _ = (nodes, allow_update)
            raise OSError("disk full")

    class FakeIndex:
        """Fake index that should never reach ``insert_nodes``."""

        def __init__(self) -> None:
            """Initialise the fake index state."""
            self.docstore = FakeDocStore()
            self.inserts: list[list[Any]] = []

        def insert_nodes(self, nodes: list[Any]) -> None:
            """Record any calls — should not be invoked in this test.

            Args:
                nodes: Vector nodes the caller is trying to insert.
            """
            self.inserts.append(nodes)

    monkeypatch.setattr(
        RAG,
        "_prepare_vector_nodes_for_insert",
        _fake_prepare_vector_nodes_for_insert,
    )

    rag = RAG(qdrant_collection="active")
    rag.docstore_batch_size = 10
    rag.index = cast(Any, FakeIndex())

    node = TextNode(text="hello", metadata={}, id_="node-x")

    cleanup = _capture_loguru(caplog)
    try:
        with pytest.raises(OSError, match="disk full"):
            rag._persist_node_batches([node])
    finally:
        cleanup()

    # Vector insert must not have been attempted once the KV write failed.
    assert cast(Any, rag.index).inserts == []
    combined = "\n".join(str(record.msg) for record in caplog.records)
    assert "failed_persist_nodes" in combined
    assert "node-x" in combined


def test_apersist_node_batches_skips_unembeddable_chunks() -> None:
    """Async persistence should drop chunks whose embeddings must be skipped."""

    class FakeDocStore:
        """Capture persisted nodes for verification."""

        def __init__(self) -> None:
            """Initialise an empty capture list."""
            self.persisted_nodes: list[list[TextNode]] = []

        def add_documents(
            self, nodes: list[TextNode], allow_update: bool = True
        ) -> None:
            """Record persisted nodes for each batch.

            Args:
                nodes: Nodes persisted to the docstore.
                allow_update: Whether overwrites are allowed.
            """
            _ = allow_update
            self.persisted_nodes.append(nodes)

    class FakeIndex:
        """Capture async vector inserts for verification."""

        def __init__(self) -> None:
            """Initialise the fake index state."""
            self.docstore = FakeDocStore()
            self.vector_batches: list[list[TextNode]] = []

        async def ainsert_nodes(self, nodes: list[TextNode]) -> None:
            """Record vector nodes for each batch.

            Args:
                nodes: Nodes inserted into the vector store.
            """
            self.vector_batches.append(nodes)

    class FakeEmbedModel:
        """Return one embedding and one skipped result asynchronously."""

        async def aget_text_embeddings_with_skips(
            self, texts: list[str]
        ) -> list[list[float] | None]:
            """Return aligned embeddings with one skipped entry.

            Args:
                texts: Texts to embed.

            Returns:
                list[list[float] | None]: Embeddings aligned to ``texts``.
            """
            assert len(texts) == 2
            return [[0.3, 0.4], None]

    rag = RAG(qdrant_collection="test")
    rag.docstore_batch_size = 10
    rag.index = cast(Any, FakeIndex())
    rag._embed_model = cast(Any, FakeEmbedModel())

    ok_node = TextNode(text="ok text", metadata={"chunk_id": "chunk-ok"})
    bad_node = TextNode(text="bad text", metadata={"chunk_id": "chunk-bad"})

    asyncio.run(rag._apersist_node_batches([ok_node, bad_node]))
    index = rag.index

    assert len(index.docstore.persisted_nodes) == 1
    assert index.docstore.persisted_nodes[0] == [ok_node]
    assert len(index.vector_batches) == 1
    assert index.vector_batches[0] == [ok_node]
    assert ok_node.embedding == [0.3, 0.4]
    assert bad_node.embedding is None


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
        """Bump the summary revision.

        Args:
            self (RAG): The RAG instance.
            collection (str | None, optional): The collection name. Defaults to None.
            allow_create (bool, optional): Whether to allow creation of a new store. Defaults to True.

        Returns:
            int: The new summary revision.
        """
        _ = self
        bumps.append((collection, allow_create))
        return len(bumps)

    monkeypatch.setattr(RAG, "_bump_summary_revision", _bump_summary_revision)

    rag.ingest_docs(tmp_path, build_query_engine=False)

    assert bumps == [("test", True)]


def test_ingest_docs_sets_and_clears_embedding_warning_callback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ingestion should attach and then clear embedding warning callbacks.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: The temporary path fixture.
    """
    rag = RAG(qdrant_collection="test")
    _patch_ingest_dependencies(monkeypatch)

    captured_callbacks: list[Any] = []

    class FakeEmbedModel:
        """Capture warning callback changes."""

        def set_warning_callback(self, callback: Any) -> None:
            """Record callback updates.

            Args:
                callback: Callback being registered or cleared.
            """
            captured_callbacks.append(callback)

    rag._embed_model = cast(Any, FakeEmbedModel())

    def progress_callback(message: str) -> None:
        """Handle progress updates during ingestion.

        Args:
            message (str): The progress message.
        """
        del message

    rag.ingest_docs(
        tmp_path, build_query_engine=False, progress_callback=progress_callback
    )

    assert captured_callbacks == [progress_callback, None]


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
        """Bump the summary revision.

        Args:
            self (RAG): The RAG instance.
            collection (str | None, optional): The collection name. Defaults to None.
            allow_create (bool, optional): Whether to allow creation of a new store. Defaults to True.

        Returns:
            int: The new summary revision.
        """
        _ = self
        bumps.append((collection, allow_create))
        return len(bumps)

    monkeypatch.setattr(RAG, "_bump_summary_revision", _bump_summary_revision)

    asyncio.run(rag.asingest_docs(tmp_path, build_query_engine=False))

    assert bumps == [("test", True)]


def _setup_empty_ingest_rag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    main_collection_exists: bool,
) -> tuple[RAG, MagicMock, list[tuple[str | None, bool]]]:
    """Build a ``RAG`` wired for empty-ingestion tests.

    The shared ingest stubs are installed, the embed model is replaced
    with a benign sentinel, the Qdrant client is mocked, and
    ``qdrant_collection_exists`` is overridden to control whether the
    main collection appears to exist on Qdrant. The summary-revision and
    NER-cache hooks capture call counts so callers can assert the empty
    branch skipped them.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: The temporary path used as the Qdrant source root.
        main_collection_exists: Whether ``qdrant_collection_exists``
            should return ``True`` for the main collection (re-ingest
            scenario) or ``False`` (fresh empty ingestion).

    Returns:
        Tuple of ``(rag, qdrant_mock, bumps)`` where ``qdrant_mock`` is
        the ``MagicMock`` substituted for ``rag._qdrant_client`` (returned
        explicitly so that mypy keeps the ``MagicMock`` typing across the
        call boundary instead of widening to ``QdrantClient | None``),
        and ``bumps`` records each ``_bump_summary_revision`` call as
        ``(collection, allow_create)``.
    """
    rag = RAG(qdrant_collection="silence-test")
    rag._embed_model = cast(Any, object())
    qdrant_mock = MagicMock()
    rag._qdrant_client = qdrant_mock
    rag._qdrant_src_dir = tmp_path
    _patch_ingest_dependencies(monkeypatch)
    monkeypatch.setattr(
        rag_module,
        "qdrant_collection_exists",
        lambda client, collection_name: main_collection_exists,
    )

    bumps: list[tuple[str | None, bool]] = []

    def _bump_summary_revision(
        self: RAG,
        collection: str | None = None,
        *,
        allow_create: bool = True,
    ) -> int:
        """Capture summary-revision bump arguments without doing real work.

        Args:
            self: The RAG instance.
            collection: Collection passed to the bump call.
            allow_create: ``allow_create`` kwarg passed to the bump call.

        Returns:
            The 1-based call count.
        """
        _ = self
        bumps.append((collection, allow_create))
        return len(bumps)

    monkeypatch.setattr(RAG, "_bump_summary_revision", _bump_summary_revision)
    return rag, qdrant_mock, bumps


def test_ingest_docs_empty_raises_and_cleans_kv_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Empty fresh ingestion should raise EmptyIngestionError and unlink the KV file.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: The Qdrant source root.
    """
    rag, qdrant_mock, bumps = _setup_empty_ingest_rag(
        monkeypatch, tmp_path, main_collection_exists=False
    )

    # Pre-create the KV file the way SQLiteKVStore.__init__ would.
    kv_dir = tmp_path / "silence-test"
    kv_dir.mkdir(parents=True, exist_ok=True)
    kv_file = kv_dir / "silence-test_kv.db"
    kv_file.write_bytes(b"SQLITE_SENTINEL")

    qdrant_mock.collection_exists.return_value = False

    with pytest.raises(rag_module.EmptyIngestionError) as excinfo:
        rag.ingest_docs(tmp_path, build_query_engine=False)

    assert excinfo.value.collection_name == "silence-test"
    assert not kv_file.exists()
    # Source directory itself must be retained (uploaded files would live here).
    assert kv_dir.exists()
    # Empty branch must skip the summary-revision bump.
    assert bumps == []
    # No companion images collection delete attempted (the existence guard
    # short-circuits because the mocked qdrant_collection_exists returns False).
    assert qdrant_mock.delete_collection.call_count == 0


def test_ingest_docs_empty_skips_cleanup_when_main_collection_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Re-ingest yielding zero new content must not destroy the prior KV store.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: The Qdrant source root.
    """
    rag, qdrant_mock, bumps = _setup_empty_ingest_rag(
        monkeypatch, tmp_path, main_collection_exists=True
    )

    kv_dir = tmp_path / "silence-test"
    kv_dir.mkdir(parents=True, exist_ok=True)
    kv_file = kv_dir / "silence-test_kv.db"
    kv_file.write_bytes(b"PRIOR_DATA")

    # Should NOT raise; should NOT delete the prior KV file.
    rag.ingest_docs(tmp_path, build_query_engine=False)

    assert kv_file.exists()
    assert kv_file.read_bytes() == b"PRIOR_DATA"
    # Standard post-success path was followed: summary revision was bumped.
    assert bumps == [("silence-test", True)]
    # No deletes performed via the Qdrant client.
    assert qdrant_mock.delete_collection.call_count == 0


def test_ingest_docs_empty_removes_wal_and_shm_siblings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Empty cleanup must remove the SQLite WAL/SHM sidecar files too.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: The Qdrant source root.
    """
    rag, qdrant_mock, _bumps = _setup_empty_ingest_rag(
        monkeypatch, tmp_path, main_collection_exists=False
    )

    kv_dir = tmp_path / "silence-test"
    kv_dir.mkdir(parents=True, exist_ok=True)
    kv_file = kv_dir / "silence-test_kv.db"
    wal_file = kv_dir / "silence-test_kv.db-wal"
    shm_file = kv_dir / "silence-test_kv.db-shm"
    for f in (kv_file, wal_file, shm_file):
        f.write_bytes(b"x")

    qdrant_mock.collection_exists.return_value = False

    with pytest.raises(rag_module.EmptyIngestionError):
        rag.ingest_docs(tmp_path, build_query_engine=False)

    assert not kv_file.exists()
    assert not wal_file.exists()
    assert not shm_file.exists()


def test_ingest_docs_empty_emits_warning_progress_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Empty cleanup must emit a ``warning:``-prefixed progress callback message.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: The Qdrant source root.
    """
    rag, qdrant_mock, _bumps = _setup_empty_ingest_rag(
        monkeypatch, tmp_path, main_collection_exists=False
    )
    qdrant_mock.collection_exists.return_value = False

    captured: list[str] = []

    def _progress(message: str) -> None:
        """Capture progress messages for assertion.

        Args:
            message: The progress message.
        """
        captured.append(message)

    with pytest.raises(rag_module.EmptyIngestionError):
        rag.ingest_docs(tmp_path, build_query_engine=False, progress_callback=_progress)

    warning_messages = [m for m in captured if m.lower().startswith("warning:")]
    assert warning_messages, f"expected warning: prefix in {captured!r}"
    assert "silence-test" in warning_messages[0]


def test_asingest_docs_empty_raises_and_cleans_kv_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Async empty fresh ingestion must raise and clean the orphan KV file.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: The Qdrant source root.
    """
    rag, qdrant_mock, bumps = _setup_empty_ingest_rag(
        monkeypatch, tmp_path, main_collection_exists=False
    )

    kv_dir = tmp_path / "silence-test"
    kv_dir.mkdir(parents=True, exist_ok=True)
    kv_file = kv_dir / "silence-test_kv.db"
    kv_file.write_bytes(b"SQLITE_SENTINEL")

    qdrant_mock.collection_exists.return_value = False

    with pytest.raises(rag_module.EmptyIngestionError) as excinfo:
        asyncio.run(rag.asingest_docs(tmp_path, build_query_engine=False))

    assert excinfo.value.collection_name == "silence-test"
    assert not kv_file.exists()
    assert bumps == []


def test_delete_collection_attempts_summary_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """delete_collection should attempt summary revision bump before deletion.

    Deletes the main and ``_images`` Qdrant collections (SQLite KV store
    lives under the source directory and is cleaned up via the source-dir
    rmtree pass).

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
        """Capture summary revision bump calls.

        Args:
            self: The RAG instance.
            collection: The collection name.
            allow_create: Whether to allow creation of a new store.

        Returns:
            The stub revision number.
        """
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
    assert deleted == ["target", "target_images"]


def test_delete_collection_fail_fast_on_primary_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """delete_collection must raise before touching KV files on primary failure.

    When the primary Qdrant collection delete raises, the SQLite KV file
    and source directory for that collection must remain untouched so
    that the operator can diagnose and retry without losing data.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: Pytest-provided temporary directory used to place a
            fake source directory and KV file on disk.
    """
    rag = RAG(qdrant_collection="active")
    rag._qdrant_client = MagicMock()
    rag._qdrant_src_dir = tmp_path
    monkeypatch.setattr(RAG, "_invalidate_ner_cache", lambda self, collection: None)
    monkeypatch.setattr(
        RAG,
        "_bump_summary_revision",
        lambda self, collection=None, allow_create=True: 1,
    )

    # Simulate the primary delete_collection call raising.
    rag._qdrant_client.delete_collection.side_effect = RuntimeError("qdrant boom")

    # Place a sentinel KV file under the source dir — it must survive.
    src_dir = tmp_path / "target"
    src_dir.mkdir()
    kv_file = src_dir / "target_kv.db"
    kv_file.write_bytes(b"SQLITE_SENTINEL")

    with pytest.raises(RuntimeError, match="qdrant boom"):
        rag.delete_collection("target")

    # Source directory and KV file must still exist.
    assert kv_file.exists()
    assert kv_file.read_bytes() == b"SQLITE_SENTINEL"
    # Only the primary delete was attempted — secondary and source cleanup
    # must have been skipped.
    deleted = [
        str(call.args[0])
        for call in rag._qdrant_client.delete_collection.call_args_list
    ]
    assert deleted == ["target"]


def test_verify_collection_reports_drift_and_parents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """verify_collection surfaces KV/Qdrant drift and broken parents.

    Builds a real on-disk SQLite KV store via :meth:`RAG._build_kv_store`
    with three node IDs (a fine node, a coarse parent, and a fine node
    whose parent is missing), mocks out Qdrant to report only one of
    them plus an extra Qdrant-only id, and asserts the resulting
    report.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest-provided temporary directory used as the
            Qdrant source root.
    """
    from llama_index.core.schema import TextNode as _TN
    from llama_index.core.storage.docstore.keyval_docstore import (
        KVDocumentStore as _KVDocumentStore,
    )

    rag = RAG(qdrant_collection="active")
    rag._qdrant_client = MagicMock()
    rag._qdrant_src_dir = tmp_path

    # Seed the real SQLite KV store with three nodes.
    kv_store = rag._build_kv_store(collection="active")
    doc_store = _KVDocumentStore(kvstore=kv_store, batch_size=10)
    fine_ok = _TN(text="fine ok", metadata={"docint_hier_type": "fine"}, id_="fine-1")
    coarse = _TN(
        text="coarse parent",
        metadata={"docint_hier_type": "coarse"},
        id_="coarse-1",
    )
    dangling = _TN(
        text="dangling child",
        metadata={
            "docint_hier_type": "fine",
            "hier.parent_id": "coarse-missing",
        },
        id_="fine-dangling",
    )
    doc_store.add_documents([fine_ok, coarse, dangling])

    # Mock Qdrant: claim the collection exists, report one of the KV ids
    # plus a phantom id that only lives in Qdrant.
    rag._qdrant_client.collection_exists.return_value = True

    class _Point:
        """Tiny stand-in for a Qdrant point returned by scroll."""

        def __init__(self, pid: str) -> None:
            """Store the point id.

            Args:
                pid: The point identifier.
            """
            self.id = pid

    def _scroll(**_kwargs: Any) -> tuple[list[_Point], Any]:
        """Return a single page of two points and no continuation offset.

        Args:
            **_kwargs: Ignored scroll keyword arguments.

        Returns:
            ``(points, None)`` — one KV-matching and one Qdrant-only id.
        """
        return ([_Point("fine-1"), _Point("qdrant-ghost")], None)

    rag._qdrant_client.scroll.side_effect = _scroll

    report = rag.verify_collection("active")

    assert report["collection"] == "active"
    assert report["qdrant_count"] == 2
    assert report["kv_count"] == 3
    # fine-dangling lives in KV only and is not coarse → orphan.
    assert report["kv_orphans"] == ["fine-dangling"]
    # The coarse parent is correctly absent from Qdrant — informational.
    assert report["expected_coarse_only"] == ["coarse-1"]
    # qdrant-ghost has no KV backing.
    assert report["qdrant_orphans"] == ["qdrant-ghost"]
    # fine-dangling references a parent that does not exist in KV.
    assert report["missing_parent_ids"] == ["coarse-missing"]
    # Without repair=True, nothing is deleted.
    assert report["repaired_ids"] == []
    assert doc_store.get_document("fine-dangling", raise_error=False) is not None


def test_verify_collection_repair_deletes_kv_orphans(
    tmp_path: Path,
) -> None:
    """verify_collection(repair=True) removes kv_orphans from the docstore.

    Args:
        tmp_path: Pytest-provided temporary directory used as the
            Qdrant source root.
    """
    from llama_index.core.schema import TextNode as _TN
    from llama_index.core.storage.docstore.keyval_docstore import (
        KVDocumentStore as _KVDocumentStore,
    )

    rag = RAG(qdrant_collection="active")
    rag._qdrant_client = MagicMock()
    rag._qdrant_src_dir = tmp_path

    kv_store = rag._build_kv_store(collection="active")
    doc_store = _KVDocumentStore(kvstore=kv_store, batch_size=10)
    orphan = _TN(text="orphan", metadata={"docint_hier_type": "fine"}, id_="orphan-1")
    doc_store.add_documents([orphan])

    rag._qdrant_client.collection_exists.return_value = True
    rag._qdrant_client.scroll.return_value = ([], None)

    report = rag.verify_collection("active", repair=True)

    assert report["kv_orphans"] == ["orphan-1"]
    assert report["repaired_ids"] == ["orphan-1"]
    # A fresh docstore on the same KV store must no longer see the orphan.
    post_store = _KVDocumentStore(
        kvstore=rag._build_kv_store(collection="active"), batch_size=10
    )
    assert post_store.get_document("orphan-1", raise_error=False) is None


def test_delete_collection_tolerates_secondary_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Failures deleting supplementary collections are swallowed and logged.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: Pytest-provided temporary directory.
    """
    rag = RAG(qdrant_collection="active")
    rag._qdrant_client = MagicMock()
    rag._qdrant_src_dir = tmp_path
    monkeypatch.setattr(RAG, "_invalidate_ner_cache", lambda self, collection: None)
    monkeypatch.setattr(
        RAG,
        "_bump_summary_revision",
        lambda self, collection=None, allow_create=True: 1,
    )

    def delete_side_effect(name: str) -> None:
        """Raise only for the supplementary ``_images`` collection.

        Args:
            name: The Qdrant collection being deleted.

        Raises:
            RuntimeError: When ``name`` ends with ``"_images"``.
        """
        if name.endswith("_images"):
            raise RuntimeError("secondary boom")

    rag._qdrant_client.delete_collection.side_effect = delete_side_effect

    # Does not raise: primary delete succeeds, secondary failure is logged.
    rag.delete_collection("target")

    deleted = [
        str(call.args[0])
        for call in rag._qdrant_client.delete_collection.call_args_list
    ]
    assert deleted == ["target", "target_images"]


def test_delete_collection_companion_name_does_not_expand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting a companion collection directly should not expand to siblings.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
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


def test_vllm_sparse_encoder_converts_pooling_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """vLLM sparse encoder should map token scores into Qdrant sparse vectors.

    Args:
        monkeypatch: The monkeypatch fixture.
    """

    responses: dict[str, dict[str, Any]] = {
        "http://vllm-router:9000/pooling": {
            "data": [{"data": [[0.0], [0.5], [0.3], [0.7], [0.0]]}]
        },
        "http://vllm-router:9000/tokenize": {"tokens": [101, 10, 20, 10, 102]},
    }

    class FakeResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(request, timeout: float = 300.0) -> FakeResponse:
        _ = timeout
        return FakeResponse(responses[request.full_url])

    monkeypatch.setattr(rag_module.urllib.request, "urlopen", fake_urlopen)

    encoder = rag_module.VLLMSparseEncoder(
        api_base="http://vllm-router:9000/v1",
        api_key="sk-no-key-required",
        model="BAAI/bge-m3",
        timeout=30.0,
    )

    indices, values = encoder.encode_texts(["hello world"])

    assert indices == [[10, 20]]
    assert values == [[0.7, 0.3]]


def test_vector_store_uses_vllm_sparse_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """vLLM hybrid retrieval should use custom sparse encoder callables.

    Args:
        monkeypatch: The monkeypatch fixture.
    """

    captured: dict[str, Any] = {}

    class FakeQdrantVectorStore:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(rag_module, "QdrantVectorStore", FakeQdrantVectorStore)

    rag = RAG(qdrant_collection="test")
    rag._qdrant_client = object()  # type: ignore[assignment]
    rag._qdrant_aclient = object()  # type: ignore[assignment]
    rag.openai_inference_provider = "vllm"
    rag.openai_api_base = "http://vllm-router:9000/v1"
    rag.openai_api_key = "sk-no-key-required"
    rag.sparse_model_id = "BAAI/bge-m3"

    rag._vector_store()

    assert "sparse_doc_fn" in captured
    assert "sparse_query_fn" in captured
    assert "fastembed_sparse_model" not in captured


# ---------------------------------------------------------------------------
# run_query / run_query_async – context window overflow
# ---------------------------------------------------------------------------


def test_run_query_wraps_context_window_overflow() -> None:
    """run_query should catch llama_index context-size ValueError and re-raise with guidance."""

    class _FakeEngine:
        def query(self, prompt: str) -> None:
            raise ValueError(
                "Calculated available context size -907 was not non-negative."
            )

    rag = RAG(qdrant_collection="test")
    rag.query_engine = _FakeEngine()  # type: ignore[assignment]
    rag.openai_ctx_window = 4096

    with pytest.raises(ValueError, match="OPENAI_CTX_WINDOW"):
        rag.run_query("What is the summary?")


def test_run_query_propagates_unrelated_value_error() -> None:
    """ValueError without 'context size' should propagate unchanged."""

    class _FakeEngine:
        def query(self, prompt: str) -> None:
            raise ValueError("Something else went wrong")

    rag = RAG(qdrant_collection="test")
    rag.query_engine = _FakeEngine()  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Something else"):
        rag.run_query("hello")


def test_run_query_async_wraps_context_window_overflow() -> None:
    """Async variant should also catch context-size overflow."""

    class _FakeEngine:
        async def aquery(self, prompt: str) -> None:
            raise ValueError(
                "Calculated available context size -907 was not non-negative."
            )

    rag = RAG(qdrant_collection="test")
    rag.query_engine = _FakeEngine()  # type: ignore[assignment]
    rag.openai_ctx_window = 4096

    with pytest.raises(ValueError, match="OPENAI_CTX_WINDOW"):
        asyncio.run(rag.run_query_async("What is the summary?"))
