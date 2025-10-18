import types

import pytest
from llama_index.core import Response

from docint.modules.core import RAG


def _make_node(score: float | None):
    return types.SimpleNamespace(score=score)


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


def test_rerank_threshold_filters_low_scores() -> None:
    rag = RAG(rerank_score_threshold=0.5)
    reranker = rag.reranker

    high = _make_node(0.7)
    low = _make_node(0.3)
    missing = _make_node(None)

    filtered = reranker.postprocess_nodes([high, low, missing], query_bundle=None)

    assert high in filtered
    assert missing in filtered  # nodes without scores are preserved
    assert low not in filtered
