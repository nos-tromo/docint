"""Tests for the CLI query entry point."""

import json
from pathlib import Path
from typing import Any, cast

import docint.cli.query as query_cli
import pytest


def test_get_col_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that _get_col_name returns the user input.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setattr("builtins.input", lambda _: "demo")
    assert query_cli.get_col_name() == "demo"


def test_store_output_writes_json(tmp_path: Path) -> None:
    """Test that _store_output writes the payload to a JSON file.

    Args:
        tmp_path (Path): The temporary path fixture.
    """
    payload = {"answer": "hi"}
    query_cli._store_output("result", payload, output_path=tmp_path)
    content = json.loads((tmp_path / "result.json").read_text())
    assert content == payload


def test_store_output_handles_nodes(tmp_path: Path) -> None:
    """Test that _store_output correctly handles objects with to_dict methods.

    Args:
        tmp_path (Path): The temporary path fixture.
    """

    class Node:
        """Simple Node class for testing purposes."""

        def __init__(self, text: str) -> None:
            """Initialize the Node with text.

            Args:
                text (str): The text content of the node.
            """
            self.text = text

        def to_dict(self) -> dict[str, str]:
            """Convert the Node to a dictionary representation.

            Returns:
                dict[str, str]: Dictionary containing the node's text.
            """
            return {"text": self.text}

    class NodeWithScore:
        """Wrapper for Node that includes a score, used to test nested to_dict handling."""

        def __init__(self, text: str) -> None:
            """Initialize the NodeWithScore with text.

            Args:
                text (str): The text content of the node.
            """
            self.node = Node(text)

        def to_dict(self) -> dict[str, str]:
            """Convert the NodeWithScore to a dictionary by delegating to the contained Node's to_dict method.

            Returns:
                dict[str, str]: Dictionary containing the node's text from the contained Node.
            """
            return {"text": self.node.text}

    query_cli._store_output(
        "nodes",
        [NodeWithScore("one"), NodeWithScore("two")],
        output_path=tmp_path,
    )
    stored = json.loads((tmp_path / "nodes.json").read_text())
    assert stored == [{"text": "one"}, {"text": "two"}]


def test_rag_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the RAG pipeline execution.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """

    class DummyRAG:
        """Dummy RAG class for testing the pipeline creation."""

        def __init__(self, qdrant_collection: str) -> None:
            """Initialize the DummyRAG with a collection name.

            Args:
                qdrant_collection (str): The name of the Qdrant collection.
            """
            self.qdrant_collection = qdrant_collection
            self.index_built = False
            self.engine_built = False

        def create_index(self) -> None:
            """Simulate index creation by setting the index_built flag to True."""
            self.index_built = True

        def create_query_engine(self) -> None:
            """Simulate query engine creation by setting the engine_built flag to True.

            Raises:
                AssertionError: If the index has not been built yet.
            """
            if not self.index_built:
                raise AssertionError("index must be built first")
            self.engine_built = True

    monkeypatch.setattr(query_cli, "RAG", DummyRAG)
    monkeypatch.setattr(query_cli, "get_col_name", lambda: "demo")
    rag = query_cli.rag_pipeline(col_name="demo")
    assert isinstance(rag, DummyRAG)
    assert rag.index_built and rag.engine_built


def test_load_queries_existing_file(tmp_path: Path) -> None:
    """
    Test that load_queries reads queries from an existing file and ignores empty lines.

    Args:
        tmp_path (Path): The temporary path fixture.
    """
    file = tmp_path / "queries.txt"
    file.write_text("one\n\ntwo\n")
    result = query_cli.load_queries(file, prompts_path=tmp_path)
    assert result == ["one", "two"]


def test_load_queries_falls_back_to_summarize_prompt(tmp_path: Path) -> None:
    """
    Test that load_queries falls back to the summarize prompt if no queries file exists.

    Args:
        tmp_path (Path): The temporary path fixture.
    """
    target = tmp_path / "missing.txt"
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "summarize.txt").write_text("Summarize the content.")
    result = query_cli.load_queries(target, prompts_path=prompts)
    assert result == ["Summarize the content."]
    assert not target.exists()


def test_run_query_records_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Test that run_query executes a query and stores the results with validation metadata.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): The temporary path fixture.
    """

    class DummyRAG:
        """Dummy RAG class for testing the run_query function, simulating query expansion
        and execution with debug information.
        """

        def __init__(self) -> None:
            """Initialize the DummyRAG with an empty list of seen queries."""
            self.seen_queries: list[str] = []

        def expand_query_with_graph_with_debug(
            self, query: str
        ) -> tuple[str, dict[str, Any]]:
            """Simulate query expansion with graph-assisted retrieval by returning an expanded
            query and debug information.

            Args:
                query (str): The original query string to expand.

            Returns:
                tuple[str, dict[str, Any]]: A tuple containing the expanded query string and a
                dictionary with debug information about the expansion process.
            """
            return (
                f"{query}\n\nRelated entities for retrieval: Acme",
                {
                    "enabled": True,
                    "applied": True,
                    "original_query": query,
                    "expanded_query": f"{query}\n\nRelated entities for retrieval: Acme",
                    "anchor_entities": ["Acme"],
                    "neighbor_entities": ["Widget"],
                },
            )

        def run_query(self, query: str) -> dict[str, Any]:
            """Simulate running a query by returning the query in the response and recording seen queries.

            Args:
                query (str): The query string to run.

            Returns:
                dict[str, Any]: A dictionary containing the query result payload.
            """
            self.seen_queries.append(query)
            return {
                "response": query,
                "sources": [
                    {
                        "filename": "social.csv",
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
                    }
                ],
            }

    calls: list[tuple[str, dict]] = []

    def fake_store(filename: str, data: dict, output_path: Path | None = None) -> None:
        """
        Fake implementation of _store_output for testing purposes.

        Args:
            filename (str): The name of the output file (without extension).
            data (dict): The data to store.
            output_path (Path | None, optional): The directory to store the output file. Defaults to None.
        """
        calls.append((filename, data))

    monkeypatch.setattr(query_cli, "_store_output", fake_store)
    rag = DummyRAG()
    monkeypatch.setattr(query_cli, "time", lambda: 1700000000)
    query_cli.run_query(cast(Any, rag), "hello", index=3, output_path=tmp_path)
    assert calls
    name, data = calls[0]
    assert "_3_result" in name
    assert data["response"] == "hello\n\nRelated entities for retrieval: Acme"
    assert data["graph_debug"]["applied"] is True
    assert data["graph_debug"]["anchor_entities"] == ["Acme"]
    assert data["sources"][0]["reference_metadata"]["text_id"] == "c1"
    assert "validation_checked" in data
    assert "validation_mismatch" in data
    assert "validation_reason" in data


def test_main_executes_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that the main function executes the full pipeline in order.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    sequence: list[str] = []

    class DummyRAG:
        """Dummy RAG class for testing the main function execution."""

        def run_query(self, q: str) -> dict[str, str]:
            """Simulate running a query by returning the query in the response.

            Args:
                q (str): The query string to run.

            Returns:
                dict[str, str]: A dictionary containing the query as the response.
            """
            sequence.append(f"query:{q}")
            return {"response": q}

        def unload_models(self) -> None:
            """Simulate unloading models. No-op for testing purposes."""
            # No-op for test double
            return None

    def fake_setup() -> None:
        """Fake implementation of setup_logging for testing purposes."""
        sequence.append("setup")

    def fake_pipeline(col_name: str | None = None) -> DummyRAG:
        """Fake implementation of rag_pipeline for testing purposes.

        Args:
            col_name (str | None, optional): The name of the collection. Defaults to None.

        Returns:
            DummyRAG: A dummy RAG instance for testing purposes.
        """
        sequence.append("pipeline")
        return DummyRAG()

    def fake_load(queries_path: Path, prompts_path: Path) -> list[str]:
        """Fake implementation of load_queries for testing purposes.

        Args:
            queries_path (Path): The path to the queries file.
            prompts_path (Path): The path to the prompts directory.

        Returns:
            list[str]: A list of queries loaded from the queries file.
        """
        sequence.append("load")
        return ["one", "two"]

    def fake_run(rag, query: str, index: int, output_path: Path) -> None:
        """Fake implementation of run_query for testing purposes.

        Args:
            rag (DummyRAG): The dummy RAG instance to use for the query.
            query (str): The query string to run.
            index (int): The index of the query in the sequence.
            output_path (Path): The path to the output directory.
        """
        sequence.append(f"run:{index}:{query}")

    class FakePathConfig:
        """Fake path configuration for testing purposes, simulating environment variable loading."""

        queries = Path("/tmp/queries.txt")
        prompts = Path("/tmp/prompts")
        results = Path("/tmp/results")

    def fake_load_path_env() -> FakePathConfig:
        """Fake implementation of load_path_env for testing purposes.

        Returns:
            FakePathConfig: A fake path configuration instance with predefined paths.
        """
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
