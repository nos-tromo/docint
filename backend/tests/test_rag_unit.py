from __future__ import annotations

import types
from pathlib import Path

import pytest

from docint.core import rag as rag_module
from docint.core.rag import RAG


class DummyNode:
    def __init__(self, text: str, metadata: dict[str, object]) -> None:
        self.text = text
        self.metadata = metadata


class DummyNodeWithScore:
    def __init__(self, node: DummyNode) -> None:
        self.node = node


class DummyResponse:
    def __init__(self, text: str, nodes: list[DummyNodeWithScore]):
        self.response = text
        self.source_nodes = nodes


def test_normalize_response_data_extracts_sources() -> None:
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
    assert normalized["sources"] == [
        {
            "text": "Example text",
            "filename": "doc.pdf",
            "filetype": "application/pdf",
            "source": "document",
            "file_hash": "abc",
            "page": 3,
        }
    ]


def test_start_session_requires_query_engine(tmp_path: Path) -> None:
    rag = RAG(qdrant_collection="test")
    rag.init_session_store(f"sqlite:///{tmp_path/'sessions.db'}")
    rag.query_engine = None
    with pytest.raises(RuntimeError):
        rag.start_session()


def test_start_session_initializes_memory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rag = RAG(qdrant_collection="test")
    rag.init_session_store(f"sqlite:///{tmp_path/'sessions.db'}")
    rag.index = object()
    rag.query_engine = object()
    rag._gen_model = object()

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
        rag_module,
        "ChatMemoryBuffer",
        types.SimpleNamespace(from_defaults=lambda **_: FakeMemory()),
    )
    monkeypatch.setattr(
        rag_module,
        "CondenseQuestionChatEngine",
        types.SimpleNamespace(from_defaults=lambda **kwargs: FakeChatEngine(**kwargs)),
    )
    session_id = rag.start_session("abc")
    assert session_id == "abc"
    assert isinstance(rag.chat_engine, FakeChatEngine)


def test_chat_rejects_empty_prompt() -> None:
    rag = RAG(qdrant_collection="test")
    with pytest.raises(ValueError):
        rag.chat("   ")


def test_list_supported_sparse_models_handles_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> list[dict[str, str]]:
        raise ImportError("missing")

    monkeypatch.setattr(
        rag_module.SparseTextEmbedding,
        "list_supported_models",
        staticmethod(broken),
    )
    assert rag_module.RAG._list_supported_sparse_models() == []
