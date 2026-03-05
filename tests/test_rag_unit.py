from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from llama_index.core import Document

from docint.core import rag as rag_module
from docint.core.rag import RAG
from docint.utils.hashing import compute_file_hash


class DummyNode:
    """A dummy node class to simulate LlamaIndex nodes."""

    def __init__(self, text: str, metadata: dict[str, object]) -> None:
        """Initializes a DummyNode with text and metadata.

        Args:
            text (str): The text content of the node.
            metadata (dict[str, object]): Metadata associated with the node.
        """
        self.text = text
        self.metadata = metadata


class DummyNodeWithScore:
    """A dummy node with score class to simulate LlamaIndex nodes with scores."""

    def __init__(self, node: DummyNode, score: float = 0.0) -> None:
        """Initializes a DummyNodeWithScore with a DummyNode.

        Args:
            node (DummyNode): The dummy node associated with this score.
            score (float): The relevance score for this node. Defaults to 0.0.
        """
        self.node = node
        self.score = score


class DummyResponse:
    def __init__(self, text: str, nodes: list[DummyNodeWithScore]):
        """Initializes a DummyResponse with text and source nodes.

        Args:
            text (str): The response text.
            nodes (list[DummyNodeWithScore]): The source nodes associated with the response.
        """
        self.response = text
        self.source_nodes = nodes


def test_normalize_response_data_extracts_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that _normalize_response_data correctly extracts source information."""
    rag = RAG(qdrant_collection="test")
    monkeypatch.setattr(
        RAG,
        "_retrieve_image_sources",
        lambda self, query, top_k=3: [],
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

    # ensure preview helpers are attached when file hashes are present
    assert first_source.get("preview_text") == "Example text"
    assert first_source.get("preview_url")
    assert first_source.get("document_url") == first_source.get("preview_url")


def test_normalize_response_data_appends_image_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image retrieval results should be appended to normalized sources.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    rag = RAG(qdrant_collection="test")
    monkeypatch.setattr(
        RAG,
        "_retrieve_image_sources",
        lambda self, query, top_k=3: [
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


def test_directory_ingestion_attaches_file_hash(tmp_path: Path) -> None:
    """Test that directory ingestion attaches file hashes to documents.

    Args:
        tmp_path (Path): The temporary path fixture.
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
        tmp_path (Path): The temporary path fixture.
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
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
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
    """Selecting a collection should clear stale IE caches."""
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
    """Refreshing collection IE should re-fetch data instead of returning stale cache."""
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


def test_start_session_initializes_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that start_session initializes the chat memory and engine.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): The temporary path fixture.
    """
    rag = RAG(qdrant_collection="test")
    rag.init_session_store(f"sqlite:///{tmp_path / 'sessions.db'}")
    rag.index = object()
    rag.query_engine = object()
    rag._text_model = object()

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
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
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
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
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
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
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
    rag.openai_model_provider = "ollama"
    rag.rerank_use_fp16 = True
    rag.rerank_top_n = 7

    _ = rag.reranker

    assert captured["top_n"] == 7
    assert captured["model"] == rag.rerank_model_id
    assert captured["use_fp16"] is True


def test_reranker_falls_back_to_llm_on_meta_tensor_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reranker should fallback to LLMRerank when FlagEmbedding hits meta tensor errors.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
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
    rag.openai_model_provider = "ollama"
    rag._text_model = None

    assert rag.reranker is llm_reranker_obj


class _FakeCompletion:
    """A simple class to simulate LLM completion responses for testing purposes."""

    def __init__(self, text: str) -> None:
        """Initializes a _FakeCompletion with the given text.

        Args:
            text (str): The text of the completion.
        """
        self.text = text


class _FakeSummaryLLM:
    """A simple class to simulate an LLM for testing the RAG summarization logic."""

    def __init__(self, text: str) -> None:
        """Initializes a _FakeSummaryLLM with the given text.

        Args:
            text (str): The text of the summary.
        """
        self.text = text
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> _FakeCompletion:
        """Simulates completing a prompt by recording the prompt and returning a fixed completion.
        Args:
            prompt (str): The input prompt to complete.

        Returns:
            _FakeCompletion: A fixed completion object.
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
        text (str): The text of the node.
        filename (str): The name of the file the node belongs to.
        file_hash (str): The hash of the file the node belongs to.
        page (int, optional): The page number of the node. Defaults to 1.
        score (float, optional): The score of the node. Defaults to 0.9.

    Returns:
        DummyNodeWithScore: A DummyNodeWithScore instance with the specified metadata.
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


def test_summarize_collection_reports_coverage_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that summarize_collection returns diagnostics about document coverage and uncovered documents.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for modifying attributes during the test.
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

    summary = rag.summarize_collection()

    diagnostics = summary["summary_diagnostics"]
    assert summary["response"] == "Collection summary"
    assert diagnostics["total_documents"] == 3
    assert diagnostics["covered_documents"] == 2
    assert diagnostics["coverage_ratio"] == 0.6667
    assert diagnostics["uncovered_documents"] == ["c.pdf"]
    assert len(summary["sources"]) == 2


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
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for modifying attributes during the test.
    """
    rag = RAG(qdrant_collection="test")
    rag._text_model = _FakeSummaryLLM("unused")  # type: ignore[assignment]
    rag.summary_coverage_target = 0.7
    rag.summary_max_docs = 30

    monkeypatch.setattr(RAG, "_summary_document_targets", lambda self: [])
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
