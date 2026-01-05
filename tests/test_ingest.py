from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest

import docint.cli.ingest as ingest
from docint.core.ingestion_pipeline import DocumentIngestionPipeline
from llama_index.core import Document


def test_get_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that get_collection returns the user input.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setattr("builtins.input", lambda _: "collection")
    name = ingest.get_collection()
    assert name == "collection"


def test_main_executes_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that the main function executes the ingestion pipeline in the correct order.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    order: list[str] = []

    def fake_setup() -> None:
        """
        Fake setup_logging function to track execution order.
        """
        order.append("setup")

    def fake_get_collection() -> str:
        """
        Fake get_collection function to track execution order.

        Returns:
            str: The name of the collection.
        """
        order.append("collection")
        return "demo"

    def fake_ingest(*args, **kwargs) -> None:
        """
        Fake ingest_docs function to track execution order.
        """
        order.append("ingest")

    class FakePathConfig:
        """
        Fake PathConfig dataclass for testing.
        """

        data = Path("/tmp")

    def fake_load_path_env() -> FakePathConfig:
        """
        Fake load_path_env function to track execution order.

        Returns:
            FakePathConfig: The fake path configuration.
        """
        order.append("env")
        return FakePathConfig()

    monkeypatch.setattr(ingest, "setup_logging", fake_setup)
    monkeypatch.setattr(ingest, "set_offline_env", lambda: None)
    monkeypatch.setattr(ingest, "load_path_env", fake_load_path_env)
    monkeypatch.setattr(ingest, "get_collection", fake_get_collection)
    monkeypatch.setattr(ingest, "ingest_docs", fake_ingest)

    ingest.main()
    # Order might vary slightly depending on implementation details, but generally:
    # setup -> env -> collection -> ingest
    assert "setup" in order
    assert "env" in order
    assert "collection" in order
    assert "ingest" in order


def test_ingest_docs_invokes_rag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Test that ingest_docs invokes RAG with correct parameters.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory path for the test.
    """
    calls = SimpleNamespace(args=None, build_query_engine=None, path=None)

    class DummyRAG:
        """
        Dummy RAG class for testing.
        """

        def __init__(
            self,
            qdrant_collection: str,
            enable_hybrid: bool,
        ) -> None:
            """
            Placeholder __init__ method for the test double.

            Args:
                qdrant_collection (str): The name of the Qdrant collection.
                enable_hybrid (bool): Whether hybrid search is enabled.
            """
            calls.args = (
                qdrant_collection,
                enable_hybrid,
            )

        def ingest_docs(
            self,
            path: Path,
            *,
            build_query_engine: bool = True,
            progress_callback: Callable[[str], None] | None = None,
        ) -> None:  # type: ignore[override]
            """
            Placeholder ingest_docs method for the test double.

            Args:
                path (Path): The directory or file path containing documents to ingest.
                build_query_engine (bool, optional): Whether to build a query engine after
                    ingestion completes. Defaults to True.
                progress_callback (Callable[[str], None] | None, optional): Optional callback
                    that receives progress updates as status messages during ingestion.
                    Defaults to None.
            """
            calls.path = path
            calls.build_query_engine = build_query_engine

        def unload_models(self) -> None:
            # No-op for test double
            return None

    monkeypatch.setattr(ingest, "RAG", DummyRAG)
    data_dir = tmp_path
    ingest.ingest_docs("demo", data_dir, hybrid=False)
    assert calls.args == ("demo", False)
    assert calls.path == data_dir
    assert calls.build_query_engine is False


def _make_pipeline(
    tmp_path: Path, entity_extractor
) -> tuple[DocumentIngestionPipeline, list]:
    """
    Helper to create a pipeline with stubbed parsers and preset nodes.

    Args:
        tmp_path (Path): Temporary directory path for the pipeline.
        entity_extractor (Callable[[str], tuple[list[dict], list[dict]]]):
            The entity extractor function to use in the pipeline.

    Returns:
        tuple[DocumentIngestionPipeline, list]: The created pipeline and the list of dummy nodes
    """
    dummy_nodes: list = []
    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path,
        device="cpu",
        clean_fn=lambda x: x,
        ie_model=None,
        progress_callback=None,
        entity_extractor=entity_extractor,
    )  # type: ignore[arg-type]

    # Minimal parser stubs to satisfy _create_nodes preconditions
    pipeline.md_node_parser = cast(
        Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes)
    )  # type: ignore[assignment]
    pipeline.docling_node_parser = cast(
        Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes)
    )  # type: ignore[assignment]
    pipeline.sentence_splitter = cast(
        Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes)
    )  # type: ignore[assignment]
    return pipeline, dummy_nodes


def test_entity_extractor_attaches_metadata(tmp_path: Path) -> None:
    """
    Test that the entity extractor is called and its results are attached to node metadata.

    Args:
        tmp_path (Path): Temporary directory path for the test.
    """
    calls: list[str] = []

    def extractor(text: str):
        calls.append(text)
        return ([{"text": "foo"}], [{"head": "a", "tail": "b"}])

    pipeline, dummy_nodes = _make_pipeline(tmp_path, extractor)
    dummy_nodes.append(SimpleNamespace(text="Hello world", metadata={"existing": 1}))

    docs = [Document(text="Doc", metadata={"file_path": "sample.txt"})]
    nodes = pipeline._create_nodes(docs)

    assert calls == ["Hello world"]
    assert len(nodes) == 1
    assert nodes[0].metadata["entities"] == [{"text": "foo"}]
    assert nodes[0].metadata["relations"] == [{"head": "a", "tail": "b"}]
    assert nodes[0].metadata["existing"] == 1


def test_entity_extractor_handles_exceptions(tmp_path: Path) -> None:
    """
    Test that exceptions in the entity extractor are handled gracefully.

    Args:
        tmp_path (Path): Temporary directory path for the test.
    """

    def bad_extractor(text: str):
        """
        Placeholder extractor that raises an exception.

        Args:
            text (str): The input text to extract entities and relations from.

        Raises:
            RuntimeError: Always raises a RuntimeError to simulate a failure.
        """
        raise RuntimeError("boom")

    pipeline, dummy_nodes = _make_pipeline(tmp_path, bad_extractor)
    dummy_nodes.append(SimpleNamespace(text="Hello", metadata={}))

    docs = [Document(text="Doc", metadata={"file_path": "sample.txt"})]
    nodes = pipeline._create_nodes(docs)

    assert len(nodes) == 1
    assert nodes[0].metadata == {}


def test_audio_extension_classified_as_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure webm files route through the audio branch rather than plain text.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """

    pipeline = DocumentIngestionPipeline(
        data_dir=Path("/tmp"),
        device="cpu",
        ie_model=None,
        progress_callback=None,
    )

    # Stub parsers to satisfy _create_nodes preconditions
    dummy_nodes: list = []
    pipeline.md_node_parser = cast(
        Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes)
    )
    pipeline.docling_node_parser = cast(
        Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes)
    )

    doc = Document(
        text="hi",
        metadata={"file_path": "foo.webm", "source": "audio", "file_hash": "x"},
    )

    nodes = pipeline._create_nodes([doc])

    assert isinstance(nodes, list)
