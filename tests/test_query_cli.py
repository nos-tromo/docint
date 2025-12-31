import json
from pathlib import Path
from typing import Any, cast

import docint.cli.query as query_cli
import pytest


def test_get_col_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that _get_col_name returns the user input.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setattr("builtins.input", lambda _: "demo")
    assert query_cli.get_col_name() == "demo"


def test_store_output_writes_json(tmp_path: Path) -> None:
    """
    Test that _store_output writes the payload to a JSON file.

    Args:
        tmp_path (Path): The temporary path fixture.
    """
    payload = {"answer": "hi"}
    query_cli._store_output("result", payload, output_path=tmp_path)
    content = json.loads((tmp_path / "result.json").read_text())
    assert content == payload


def test_store_output_handles_nodes(tmp_path: Path) -> None:
    """
    Test that _store_output correctly handles objects with to_dict methods.

    Args:
        tmp_path (Path): The temporary path fixture.
    """

    class Node:
        def __init__(self, text: str) -> None:
            self.text = text

        def to_dict(self) -> dict[str, str]:
            return {"text": self.text}

    class NodeWithScore:
        def __init__(self, text: str) -> None:
            self.node = Node(text)

        def to_dict(self) -> dict[str, str]:
            return {"text": self.node.text}

    query_cli._store_output(
        "nodes",
        [NodeWithScore("one"), NodeWithScore("two")],
        output_path=tmp_path,
    )
    stored = json.loads((tmp_path / "nodes.json").read_text())
    assert stored == [{"text": "one"}, {"text": "two"}]


def test_rag_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test the RAG pipeline execution.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """

    class DummyRAG:
        def __init__(self, qdrant_collection: str) -> None:
            self.qdrant_collection = qdrant_collection
            self.index_built = False
            self.engine_built = False

        def create_index(self) -> None:
            self.index_built = True

        def create_query_engine(self) -> None:
            if not self.index_built:
                raise AssertionError("index must be built first")
            self.engine_built = True

    monkeypatch.setattr(query_cli, "RAG", DummyRAG)
    monkeypatch.setattr(query_cli, "get_col_name", lambda: "demo")
    rag = query_cli.rag_pipeline(col_name="demo")
    assert isinstance(rag, DummyRAG)
    assert rag.index_built and rag.engine_built


def test_load_queries_existing_file(tmp_path: Path) -> None:
    file = tmp_path / "queries.txt"
    file.write_text("one\n\ntwo\n")
    result = query_cli.load_queries(file)
    assert result == ["one", "two"]


def test_load_queries_creates_default(tmp_path: Path) -> None:
    target = tmp_path / "missing.txt"
    result = query_cli.load_queries(target)
    assert result == ["Summarize the content with a maximum of 15 sentences."]
    assert target.exists()


def test_run_query_records_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class DummyRAG:
        def run_query(self, query: str) -> dict[str, str]:
            return {"response": query}

    calls: list[tuple[str, dict]] = []

    def fake_store(filename: str, data: dict, output_path: Path | None = None) -> None:
        calls.append((filename, data))

    monkeypatch.setattr(query_cli, "_store_output", fake_store)
    rag = DummyRAG()
    monkeypatch.setattr(query_cli, "time", lambda: 1700000000)
    query_cli.run_query(cast(Any, rag), "hello", index=3, output_path=tmp_path)
    assert calls
    name, data = calls[0]
    assert "_3_result" in name
    assert data == {"response": "hello"}


def test_main_executes_all(monkeypatch: pytest.MonkeyPatch) -> None:
    sequence: list[str] = []

    class DummyRAG:
        def run_query(self, q: str) -> dict[str, str]:
            sequence.append(f"query:{q}")
            return {"response": q}

        def unload_models(self) -> None:
            # No-op for test double
            return None

    def fake_setup() -> None:
        sequence.append("setup")

    def fake_pipeline(col_name: str | None = None) -> DummyRAG:
        sequence.append("pipeline")
        return DummyRAG()

    def fake_load(q_path: Path) -> list[str]:
        sequence.append("load")
        return ["one", "two"]

    def fake_run(rag, query: str, index: int, output_path: Path) -> None:
        sequence.append(f"run:{index}:{query}")

    class FakePathConfig:
        queries = Path("/tmp/queries.txt")
        results = Path("/tmp/results")

    def fake_load_path_env() -> FakePathConfig:
        sequence.append("env")
        return FakePathConfig()

    monkeypatch.setattr(query_cli, "setup_logging", fake_setup)
    monkeypatch.setattr(query_cli, "set_offline_env", lambda: None)
    monkeypatch.setattr(query_cli, "load_path_env", fake_load_path_env)
    monkeypatch.setattr(query_cli, "rag_pipeline", fake_pipeline)
    monkeypatch.setattr(query_cli, "load_queries", fake_load)
    monkeypatch.setattr(query_cli, "run_query", fake_run)
    monkeypatch.setattr(query_cli, "get_col_name", lambda: "alpha")

    query_cli.main()
    # Order: setup -> env -> pipeline -> load -> run...
    assert "setup" in sequence
    assert "env" in sequence
    assert "pipeline" in sequence
    assert "load" in sequence
    assert "run:1:one" in sequence
    assert "run:2:two" in sequence
