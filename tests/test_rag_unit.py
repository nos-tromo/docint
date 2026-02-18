from __future__ import annotations

import types
from pathlib import Path

import pytest
from llama_index.core import Document

from docint.core import rag as rag_module
from docint.core.rag import RAG
from docint.utils.hashing import compute_file_hash


class DummyNode:
    """
    A dummy node class to simulate LlamaIndex nodes.
    """

    def __init__(self, text: str, metadata: dict[str, object]) -> None:
        """
        Initializes a DummyNode with text and metadata.

        Args:
            text (str): The text content of the node.
            metadata (dict[str, object]): Metadata associated with the node.
        """
        self.text = text
        self.metadata = metadata


class DummyNodeWithScore:
    """
    A dummy node with score class to simulate LlamaIndex nodes with scores.
    """

    def __init__(self, node: DummyNode) -> None:
        """
        Initializes a DummyNodeWithScore with a DummyNode.

        Args:
            node (DummyNode): The dummy node associated with this score.
        """
        self.node = node


class DummyResponse:
    def __init__(self, text: str, nodes: list[DummyNodeWithScore]):
        """
        Initializes a DummyResponse with text and source nodes.

        Args:
            text (str): The response text.
            nodes (list[DummyNodeWithScore]): The source nodes associated with the response.
        """
        self.response = text
        self.source_nodes = nodes


def test_normalize_response_data_extracts_sources() -> None:
    """
    Test that _normalize_response_data correctly extracts source information.
    """
    rag = RAG(qdrant_collection="test")
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


def test_directory_ingestion_attaches_file_hash(tmp_path: Path) -> None:
    """
    Test that directory ingestion attaches file hashes to documents.

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
    """
    Test that start_session raises RuntimeError if query_engine is not initialized.

    Args:
        tmp_path (Path): The temporary path fixture.
    """
    rag = RAG(qdrant_collection="test")
    rag.init_session_store(f"sqlite:///{tmp_path / 'sessions.db'}")
    rag.query_engine = None
    with pytest.raises(RuntimeError):
        rag.start_session()


def test_start_session_initializes_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Test that start_session initializes the chat memory and engine.

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
        "docint.core.session_manager.ChatMemoryBuffer",
        types.SimpleNamespace(from_defaults=lambda **_: FakeMemory()),
    )
    monkeypatch.setattr(
        "docint.core.session_manager.CondenseQuestionChatEngine",
        types.SimpleNamespace(from_defaults=lambda **kwargs: FakeChatEngine(**kwargs)),
    )
    session_id = rag.start_session("abc")
    assert session_id == "abc"
    assert isinstance(rag.chat_engine, FakeChatEngine)


def test_chat_rejects_empty_prompt() -> None:
    """
    Test that chat rejects empty prompts.
    """
    rag = RAG(qdrant_collection="test")
    with pytest.raises(ValueError):
        rag.chat("   ")


def test_sparse_model_raises_import_error_when_fastembed_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Test that the sparse_model property raises ImportError when
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
    """
    Test that _filter_docs_by_existing_hashes skips documents with existing hashes.

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
        """
        Ensure sparse_model resolves to a cached snapshot path when available.

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
