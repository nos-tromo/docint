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

    assert rag.qdrant_collection == "alpha"
    assert rag.docs == []
    assert rag.nodes == []
    assert rag.index is None
    assert rag.query_engine is None
    assert rag.chat_engine is None
    assert rag.chat_memory is None
    assert rag.session_id is None


def test_sparse_model_accepts_local_path(tmp_path) -> None:
    rag = RAG()
    rag.sparse_model_path = tmp_path

    (tmp_path / "config.json").write_text("{}")

    assert rag.sparse_model == str(tmp_path)


def test_sparse_model_path_validation(tmp_path) -> None:
    rag = RAG()
    rag.sparse_model_path = tmp_path / "missing"

    with pytest.raises(ValueError):
        _ = rag.sparse_model


def test_sparse_model_path_requires_snapshot(tmp_path) -> None:
    rag = RAG()
    invalid_dir = tmp_path / "empty"
    invalid_dir.mkdir()

    rag.sparse_model_path = invalid_dir

    with pytest.raises(ValueError):
        _ = rag.sparse_model


def test_sparse_model_resolves_hf_cache_root(tmp_path) -> None:
    rag = RAG()
    cache_root = tmp_path / "huggingface" / "hub"
    snapshot_dir = (
        cache_root
        / f"models--{rag.sparse_model_id.replace('/', '--')}"
        / "snapshots"
        / "1234567890abcdef"
    )
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}")

    rag.sparse_model_path = cache_root

    assert rag.sparse_model == str(snapshot_dir)
