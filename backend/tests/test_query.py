from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import docint.core.query as query


class DummyRAG:
    """
    A dummy Retrieval-Augmented Generation (RAG) model for testing purposes.
    """

    def __init__(self, qdrant_collection: str) -> None:
        self.qdrant_collection = qdrant_collection
        self.index_created = False
        self.engine_created = False
        self.run_calls: list[tuple[str, int]] = []

    def create_index(self) -> None:
        """
        Creates the index for the RAG model.
        """
        self.index_created = True

    def create_query_engine(self) -> None:
        """
        Creates the query engine for the RAG model.
        """
        self.engine_created = True

    def run_query(self, prompt: str) -> dict[str, str]:
        """
        Runs a query against the RAG model.

        Args:
            prompt (str): The query prompt.

        Returns:
            dict[str, str]: The query result.
        """
        self.run_calls.append((prompt, len(self.run_calls)))
        return {"query": prompt, "response": "answer"}


@pytest.mark.parametrize("output_subdir", [None, "custom", Path("custom")])
def test_store_output_writes_json(
    tmp_path: Path, output_subdir: str | Path | None
) -> None:
    """
    Tests that the output is written to the correct JSON file.

    Args:
        tmp_path (Path): The temporary directory path.
        output_subdir (str | Path | None): The subdirectory for output.
    """
    data = {"key": "value"}
    filename = "result"
    output_path = tmp_path / "results"

    if isinstance(output_subdir, (str, Path)):
        output_path = output_path / str(output_subdir)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    query._store_output(filename, data, output_path)

    expected_file = output_path / f"{filename}.json"
    assert expected_file.exists()
    assert json.loads(expected_file.read_text(encoding="utf-8")) == data


def test_store_output_serializes_nodes(tmp_path: Path) -> None:
    """
    Tests that the output is serialized correctly for nodes.

    Args:
        tmp_path (Path): The temporary directory path.
    """
    node = SimpleNamespace(to_dict=lambda: {"id": 1})
    data = [node, {"fallback": True}]
    filename = "nodes"

    query._store_output(filename, data, tmp_path)

    expected_file = tmp_path / f"{filename}.json"
    assert expected_file.exists()
    contents = json.loads(expected_file.read_text(encoding="utf-8"))
    assert contents == [{"id": 1}, str({"fallback": True})]


def test_load_queries_reads_existing_file(tmp_path: Path) -> None:
    """
    Tests that the existing queries file is read correctly.

    Args:
        tmp_path (Path): The temporary directory path.
    """
    queries_file = tmp_path / "queries.txt"
    queries_file.write_text("first\nsecond\n\n", encoding="utf-8")

    queries = query.load_queries(queries_file)

    assert queries == ["first", "second"]


def test_load_queries_creates_default_file(tmp_path: Path) -> None:
    """
    Tests that a default queries file is created when missing.

    Args:
        tmp_path (Path): The temporary directory path.
    """
    queries_file = tmp_path / "missing.txt"

    queries = query.load_queries(queries_file)

    assert queries == ["Summarize the content with a maximum of 15 sentences."]
    assert queries_file.exists()
    assert queries_file.read_text(encoding="utf-8").strip() == queries[0]


def test_rag_pipeline_initializes_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Tests that the RAG pipeline initializes the collection correctly.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setattr(query, "_get_col_name", lambda: "demo")
    monkeypatch.setattr(query, "RAG", DummyRAG)

    rag = query.rag_pipeline()

    assert isinstance(rag, DummyRAG)
    assert rag.qdrant_collection == "demo"
    assert rag.index_created is True
    assert rag.engine_created is True


def test_run_query_records_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Tests that the query results are recorded correctly.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): The temporary directory path.
    """
    rag = DummyRAG("col")

    captured: dict[str, object] = {}

    def fake_store(filename: str, data: dict, output_path: Path | None = None) -> None:
        captured["filename"] = filename
        captured["data"] = data
        captured["output_path"] = output_path

    monkeypatch.setattr(query, "_store_output", fake_store)
    monkeypatch.setattr(query, "time", lambda: 42)

    query.run_query(rag, "How many rows?", 3)

    assert rag.run_calls == [("How many rows?", 0)]
    assert captured["filename"] == "42_3_result"
    assert captured["data"] == {"query": "How many rows?", "response": "answer"}
    assert captured["output_path"] is None
