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


def test_parse_args_defaults_to_no_explicit_actions() -> None:
    """CLI parser should accept an empty argument list."""
    args = query_cli.parse_args([])

    assert args.collection is None
    assert args.query is None
    assert args.summary is False
    assert args.entities is False
    assert args.hate_speech is False
    assert args.all is False


def test_parse_args_supports_collection_query_and_all_flags() -> None:
    """CLI parser should support collection selection, optional query path, and all export flags."""
    args = query_cli.parse_args(
        ["-c", "alpha", "-q", "queries-custom.txt", "-s", "-e", "-h8", "-a"]
    )

    assert args.collection == "alpha"
    assert args.query == "queries-custom.txt"
    assert args.summary is True
    assert args.entities is True
    assert args.hate_speech is True
    assert args.all is True


def test_parse_args_query_without_path_uses_default_sentinel() -> None:
    """Passing ``-q`` without a value should activate query mode with the default file."""
    args = query_cli.parse_args(["-q"])

    assert args.query == query_cli.DEFAULT_CHAT_SENTINEL


def test_resolve_collection_name_prefers_argument() -> None:
    """Collection selection should use the CLI argument when present."""
    args = query_cli.parse_args(["-c", "alpha"])

    assert query_cli._resolve_collection_name(args) == "alpha"


def test_build_run_output_path_uses_timestamp_and_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run outputs should be grouped in a timestamped collection subdirectory.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setattr(query_cli, "time", lambda: 1700000123)

    result = query_cli._build_run_output_path(
        Path("/tmp/results"), collection_name="Alpha Collection"
    )

    assert result == Path("/tmp/results/1700000123_Alpha_Collection")


def test_store_output_writes_json(tmp_path: Path) -> None:
    """Test that _store_output still writes structured payloads to a JSON file.

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


def test_load_queries_returns_empty_when_file_is_missing(tmp_path: Path) -> None:
    """Missing query files should no longer trigger the summarize prompt fallback."""
    target = tmp_path / "missing.txt"
    result = query_cli.load_queries(target, prompts_path=tmp_path)

    assert result == []
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

    calls: list[tuple[str, str]] = []

    def fake_store_text(
        filename: str, data: str, output_path: Path | None = None
    ) -> None:
        """
        Fake implementation of _store_text_output for testing purposes.

        Args:
            filename (str): The name of the output file (without extension).
            data (str): The text data to store.
            output_path (Path | None, optional): The directory to store the output file. Defaults to None.
        """
        calls.append((filename, data))

    monkeypatch.setattr(query_cli, "_store_text_output", fake_store_text)
    rag = DummyRAG()
    monkeypatch.setattr(query_cli, "time", lambda: 1700000000)
    query_cli.run_query(cast(Any, rag), "hello", index=3, output_path=tmp_path)
    assert calls
    name, data = calls[0]
    assert "_3_result" in name
    assert "Query: hello" in data
    assert "Answer:" in data
    assert "hello\n\nRelated entities for retrieval: Acme" in data
    assert '"applied": true' in data
    assert '"anchor_entities": [' in data
    assert "- Text ID: c1" in data
    assert "Validation:" in data


def test_export_summary_stores_frontend_like_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Summary export should store a text summary with diagnostics and validation fields.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): The temporary path fixture.
    """

    class DummyRAG:
        qdrant_collection = "alpha"
        summarize_prompt = "Summarize the collection."

        def summarize_collection(self) -> dict[str, Any]:
            return {
                "response": "Summary text.",
                "sources": [{"filename": "doc.pdf"}],
                "summary_diagnostics": {"coverage_ratio": 0.8},
            }

    captured: list[tuple[str, str]] = []

    def fake_store_text(filename: str, data: str, output_path: Path) -> None:
        captured.append((filename, data))

    monkeypatch.setattr(query_cli, "_store_text_output", fake_store_text)
    monkeypatch.setattr(
        query_cli,
        "_get_validation_payload",
        lambda *args, **kwargs: {
            "validation_checked": True,
            "validation_mismatch": False,
            "validation_reason": None,
        },
    )

    query_cli.export_summary(cast(Any, DummyRAG()), output_path=tmp_path)

    assert captured
    filename, data = captured[0]
    assert filename == "summary_alpha"
    assert "Collection summary: alpha" in data
    assert "Summary:" in data
    assert "Summary text." in data
    assert "Validation:" in data
    assert '"coverage_ratio": 0.8' in data
    assert "Sources:" in data
    assert "doc.pdf" in data


def test_export_entities_writes_top_entity_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Entity export should render mention counts into a text payload.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): The temporary path fixture.
    """

    class DummyRAG:
        qdrant_collection = "alpha"

        def get_collection_ner_stats(self, **kwargs: Any) -> dict[str, Any]:
            """Simulate retrieval of named entity recognition statistics for a collection, returning a list of top entities with their types and mention counts.

            Returns:
                dict[str, Any]: A dictionary containing a list of top entities, where each entity is represented as a dictionary with 'text', 'type', and 'mentions' keys.
            """
            assert kwargs["top_k"] == query_cli.DEFAULT_ENTITY_LIMIT
            assert kwargs["min_mentions"] == 1
            return {
                "top_entities": [
                    {"text": "Acme", "type": "ORG", "mentions": 12},
                    {"text": "Rivertown", "type": "LOC", "mentions": 4},
                ]
            }

    captured: list[tuple[str, str]] = []

    def fake_store_text(filename: str, data: str, output_path: Path) -> None:
        captured.append((filename, data))

    monkeypatch.setattr(query_cli, "_store_text_output", fake_store_text)

    query_cli.export_entities(cast(Any, DummyRAG()), output_path=tmp_path)

    assert captured
    filename, data = captured[0]
    assert filename == "entities_alpha"
    assert "1. Acme [ORG] - 12" in data
    assert "2. Rivertown [LOC] - 4" in data


def test_export_hate_speech_writes_frontend_text_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Hate-speech export should use the same text format as the frontend download.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): The temporary path fixture.
    """

    class DummyRAG:
        qdrant_collection = "alpha"

        def get_collection_hate_speech(self) -> list[dict[str, Any]]:
            """Simulate retrieval of hate speech analysis results for a collection, returning a
            list of flagged chunks with their associated metadata.

            Returns:
                list[dict[str, Any]]: A list of dictionaries, where each dictionary represents
                    a flagged chunk of text with metadata such as source reference, category,
                    confidence level, reason for flagging, and reference metadata including text ID.
            """
            return [
                {
                    "source_ref": "doc.csv",
                    "page": None,
                    "row": 5,
                    "chunk_id": "c9",
                    "category": "ethnicity",
                    "confidence": "high",
                    "reason": "Derogatory language",
                    "reference_metadata": {"text_id": "p1"},
                }
            ]

    captured: list[tuple[str, str]] = []

    def fake_store_text(filename: str, data: str, output_path: Path) -> None:
        """Fake implementation of _store_text_output for testing the export_hate_speech function,
            capturing the filename and data for assertions.

        Args:
            filename (str): The name of the file being written.
            data (str): The text data being written to the file.
            output_path (Path): The path where the file would be written.
        """
        captured.append((filename, data))

    monkeypatch.setattr(query_cli, "_store_text_output", fake_store_text)

    query_cli.export_hate_speech(cast(Any, DummyRAG()), output_path=tmp_path)

    assert captured
    filename, data = captured[0]
    assert filename == "hate_speech_alpha"
    assert "Flagged hate-speech chunks" in data
    assert "- category: ethnicity" in data
    assert "- Text ID: p1" in data


def test_main_runs_default_chat_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no flags, the CLI should default to chat mode using the default queries file.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    sequence: list[str] = []

    class DummyRAG:
        def unload_models(self) -> None:
            return None

    def fake_pipeline(col_name: str | None = None) -> DummyRAG:
        sequence.append("pipeline")
        return DummyRAG()

    class FakePathConfig:
        queries = Path("/tmp/queries.txt")
        prompts = Path("/tmp/prompts")
        results = Path("/tmp/results")

    def fake_load_path_env() -> FakePathConfig:
        sequence.append("env")
        return FakePathConfig()

    def fake_export_chat_queries(rag: Any, **kwargs: Any) -> None:
        _ = rag
        sequence.append(f"chat:{kwargs['queries_path']}")
        sequence.append(f"out:{kwargs['output_path']}")

    monkeypatch.setattr(query_cli, "setup_logging", lambda: sequence.append("setup"))
    monkeypatch.setattr(query_cli, "set_offline_env", lambda: None)
    monkeypatch.setattr(query_cli, "time", lambda: 1700000123)
    monkeypatch.setattr(query_cli, "load_path_env", fake_load_path_env)
    monkeypatch.setattr(query_cli, "rag_pipeline", fake_pipeline)
    monkeypatch.setattr(query_cli, "export_chat_queries", fake_export_chat_queries)
    monkeypatch.setattr(query_cli, "get_col_name", lambda: "alpha")

    query_cli.main([])

    assert "setup" in sequence
    assert "env" in sequence
    assert "pipeline" in sequence
    assert "chat:/tmp/queries.txt" in sequence
    assert "out:/tmp/results/1700000123_alpha" in sequence


def test_main_uses_collection_argument_without_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Providing ``-c`` should bypass the interactive collection prompt.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    sequence: list[str] = []

    class DummyRAG:
        """Dummy RAG class for testing the main function's handling of the collection argument,
        with a no-op unload_models method."""

        def unload_models(self) -> None:
            """Simulate unloading of models by appending 'unload' to the sequence list, allowing
            verification that model cleanup is performed after the main execution."""
            sequence.append("unload")

    class FakePathConfig:
        """Fake path configuration class for testing purposes, providing fixed paths for queries,
        prompts, and results, and allowing verification of environment loading in the main function."""

        queries = Path("/tmp/queries.txt")
        prompts = Path("/tmp/prompts")
        results = Path("/tmp/results")

    def fake_pipeline(col_name: str | None = None) -> DummyRAG:
        """Simulate the RAG pipeline creation by appending the collection name to the sequence
        list and returning a DummyRAG instance.

        Args:
            col_name (str | None, optional): The name of the collection. Defaults to None.

        Returns:
            DummyRAG: An instance of the DummyRAG class.
        """
        sequence.append(f"pipeline:{col_name}")
        return DummyRAG()

    monkeypatch.setattr(query_cli, "setup_logging", lambda: None)
    monkeypatch.setattr(query_cli, "set_offline_env", lambda: None)
    monkeypatch.setattr(query_cli, "time", lambda: 1700000123)
    monkeypatch.setattr(query_cli, "load_path_env", lambda: FakePathConfig())
    monkeypatch.setattr(query_cli, "rag_pipeline", fake_pipeline)
    monkeypatch.setattr(
        query_cli,
        "export_chat_queries",
        lambda rag, **kwargs: sequence.append(f"out:{kwargs['output_path']}"),
    )
    monkeypatch.setattr(
        query_cli,
        "get_col_name",
        lambda: (_ for _ in ()).throw(AssertionError("prompt should not be used")),
    )

    query_cli.main(["-c", "alpha", "-q"])

    assert sequence == [
        "pipeline:alpha",
        "out:/tmp/results/1700000123_alpha",
        "unload",
    ]


def test_main_runs_all_requested_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``--all`` flag should invoke chat, summary, entities, and hate-speech exports.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    sequence: list[str] = []

    class DummyRAG:
        def unload_models(self) -> None:
            sequence.append("unload")

    class FakePathConfig:
        queries = Path("/tmp/queries.txt")
        prompts = Path("/tmp/prompts")
        results = Path("/tmp/results")

    monkeypatch.setattr(query_cli, "setup_logging", lambda: None)
    monkeypatch.setattr(query_cli, "set_offline_env", lambda: None)
    monkeypatch.setattr(query_cli, "time", lambda: 1700000123)
    monkeypatch.setattr(query_cli, "load_path_env", lambda: FakePathConfig())
    monkeypatch.setattr(query_cli, "rag_pipeline", lambda col_name=None: DummyRAG())
    monkeypatch.setattr(query_cli, "get_col_name", lambda: "alpha")
    monkeypatch.setattr(
        query_cli,
        "export_chat_queries",
        lambda rag, **kwargs: sequence.append(f"chat:{kwargs['output_path']}"),
    )
    monkeypatch.setattr(
        query_cli,
        "export_summary",
        lambda rag, output_path: sequence.append(f"summary:{output_path}"),
    )
    monkeypatch.setattr(
        query_cli,
        "export_entities",
        lambda rag, output_path: sequence.append(f"entities:{output_path}"),
    )
    monkeypatch.setattr(
        query_cli,
        "export_hate_speech",
        lambda rag, output_path: sequence.append(f"hate:{output_path}"),
    )

    query_cli.main(["--all"])

    assert sequence == [
        "chat:/tmp/results/1700000123_alpha",
        "summary:/tmp/results/1700000123_alpha",
        "entities:/tmp/results/1700000123_alpha",
        "hate:/tmp/results/1700000123_alpha",
        "unload",
    ]
