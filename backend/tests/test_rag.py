import json
import types

import pytest
from llama_index.core import Response

from docint.core.rag import RAG


def test_run_query_returns_sources() -> None:
    rag = RAG()

    node_metadata = {
        "origin": {"filename": "sample.csv", "filetype": "text/csv"},
        "source": "table",
        "table": {"row_index": 5, "n_rows": 10, "n_cols": 3},
    }
    node = types.SimpleNamespace(text="clean row", metadata=node_metadata)
    source = types.SimpleNamespace(node=node, score=0.42)

    class DummyQueryEngine:
        def query(self, prompt: str):
            return Response(response="stubbed answer", source_nodes=[source])

    rag.query_engine = DummyQueryEngine()

    result = rag.run_query("What is inside the table?")

    assert result["query"] == "What is inside the table?"
    assert result["response"] == "stubbed answer"
    assert result["sources"][0]["text"] == "clean row"
    assert result["sources"][0]["filename"] == "sample.csv"
    assert result["sources"][0]["source"] == "table"
    assert result["sources"][0]["table_info"] == {"n_rows": 10, "n_cols": 3}


def test_run_query_requires_prompt_and_engine() -> None:
    rag = RAG()

    with pytest.raises(ValueError):
        rag.run_query("  ")

    with pytest.raises(RuntimeError):
        rag.run_query("needs engine")


def test_select_collection_resets_cached_state(monkeypatch) -> None:
    rag = RAG()

    rag.docs = ["doc"]
    rag.nodes = ["node"]
    rag.index = object()
    rag.query_engine = object()
    rag.chat_engine = object()
    rag.chat_memory = object()
    rag.session_id = "session"

    monkeypatch.setattr(
        RAG,
        "list_collections",
        lambda self, prefer_api=True: ["alpha"],
    )

    rag.select_collection("alpha")

    assert rag.milvus_collection == "alpha"
    assert rag.docs == []
    assert rag.nodes == []
    assert rag.index is None
    assert rag.query_engine is None
    assert rag.chat_engine is None
    assert rag.chat_memory is None
    assert rag.session_id is None


def test_sparse_encoder_persists_vocabulary(tmp_path) -> None:
    vocab_path = tmp_path / "vocab.json"
    rag = RAG()
    rag.sparse_vocab_path = vocab_path

    sparse_fn = rag._get_sparse_embedding_function()
    assert sparse_fn is not None

    embedding = sparse_fn.encode_documents(["Alpha beta beta"])[0]
    assert embedding
    assert vocab_path.exists()

    payload = json.loads(vocab_path.read_text())
    assert payload["doc_count"] == 1
    assert "beta" in payload["token_to_id"]


def test_sparse_queries_ignore_unknown_tokens(tmp_path) -> None:
    vocab_path = tmp_path / "vocab.json"
    rag = RAG()
    rag.sparse_vocab_path = vocab_path

    sparse_fn = rag._get_sparse_embedding_function()
    sparse_fn.encode_documents(["seen token"])

    query_embedding = sparse_fn.encode_queries(["seen unseen"])[0]

    assert len(query_embedding) == 1
