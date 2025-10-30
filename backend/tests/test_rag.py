import types
from pathlib import Path

import pytest
from llama_index.core import Document, Response

from docint.core.rag import RAG
from docint.utils.hashing import compute_file_hash


def test_run_query_returns_sources() -> None:
    """
    Tests that the run_query method returns the expected sources.
    """
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
    """
    Tests that the run_query method requires a prompt and a query engine.
    """
    rag = RAG()

    with pytest.raises(ValueError):
        rag.run_query("  ")

    with pytest.raises(RuntimeError):
        rag.run_query("needs engine")


def test_select_collection_resets_cached_state(monkeypatch) -> None:
    """
    Tests that the select_collection method resets the cached state.

    Args:
        monkeypatch (_type_): The monkeypatch fixture.
    """
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


def test_ensure_file_hash_metadata(tmp_path: Path) -> None:
    """
    Tests that the file hash metadata is correctly computed and stored.

    Args:
        tmp_path (Path): The temporary directory path.
    """
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world", encoding="utf-8")

    doc = Document(
        text="hello world",
        metadata={
            "file_path": str(file_path),
            "origin": {"filename": file_path.name},
        },
    )

    rag = RAG(data_dir=tmp_path)
    rag.docs = [doc]
    rag._ensure_file_hash_metadata()

    expected_hash = compute_file_hash(file_path)
    assert doc.metadata["file_hash"] == expected_hash
    assert doc.metadata["origin"]["file_hash"] == expected_hash


def test_filter_docs_by_existing_hashes(monkeypatch) -> None:
    """
    Tests that the _filter_docs_by_existing_hashes method removes documents
    that are already ingested.

    Args:
        monkeypatch (_type_): The monkeypatch fixture.
    """
    doc_new = Document(
        text="keep",
        metadata={
            "file_hash": "new-hash",
            "file_name": "fresh.txt",
            "origin": {"file_hash": "new-hash"},
        },
    )
    doc_existing = Document(
        text="skip",
        metadata={
            "file_hash": "existing-hash",
            "file_name": "old.txt",
            "origin": {"file_hash": "existing-hash"},
        },
    )

    rag = RAG()
    rag.docs = [doc_new, doc_existing]

    monkeypatch.setattr(
        RAG,
        "_get_existing_file_hashes",
        lambda self: {"existing-hash"},
    )

    rag._filter_docs_by_existing_hashes()

    assert rag.docs == [doc_new]
