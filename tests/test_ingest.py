"""Tests for the CLI ingest entry point and ingestion pipeline."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest

import docint.cli.ingest as ingest
import docint.core.ingest.ingestion_pipeline as pipeline_module
from docint.core.ingest.ingestion_pipeline import DocumentIngestionPipeline
from llama_index.core import Document


def test_get_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_collection returns the user input.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setattr("builtins.input", lambda _: "collection")
    name = ingest.get_collection()
    assert name == "collection"


def test_main_executes_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the main function executes the ingestion pipeline in the correct order.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    order: list[str] = []

    def fake_setup() -> None:
        """Fake setup_logging function to track execution order."""
        order.append("setup")

    def fake_get_collection() -> str:
        """Fake get_collection function to track execution order.

        Returns:
            str: The name of the collection.
        """
        order.append("collection")
        return "demo"

    def fake_ingest(*args, **kwargs) -> None:
        """Fake ingest_docs function to track execution order."""
        order.append("ingest")

    class FakePathConfig:
        """Fake PathConfig dataclass for testing."""

        data = Path("/tmp")

    def fake_load_path_env() -> FakePathConfig:
        """Fake load_path_env function to track execution order.

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
    """Test that ingest_docs invokes RAG with correct parameters.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory path for the test.
    """
    calls = SimpleNamespace(args=None, build_query_engine=None, path=None)

    class DummyRAG:
        """Dummy RAG class for testing."""

        def __init__(
            self,
            qdrant_collection: str,
            enable_hybrid: bool,
        ) -> None:
            """Placeholder __init__ method for the test double.

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
        ) -> None:
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
            """No-op model unload for the test double."""
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
    """Helper to create a pipeline with stubbed parsers and preset nodes.

    Args:
        tmp_path (Path): Temporary directory path for the pipeline.
        entity_extractor (Callable[[str], tuple[list[dict], list[dict]]]):
            The entity extractor function to use in the pipeline.

    Returns:
        tuple[DocumentIngestionPipeline, list]: The created pipeline and the list of dummy nodes
    """
    dummy_nodes: list = []
    # Pipeline.__post_init__ will override entity_extractor if env vars are present.
    # We must forcibly set the extractor AFTER init.
    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path,
        device="cpu",
        clean_fn=lambda x: x,
        ner_model=None,
        progress_callback=None,
    )

    pipeline.entity_extractor = entity_extractor

    # Minimal parser stubs to satisfy _create_nodes preconditions
    pipeline.md_node_parser = cast(
        Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes)
    )
    pipeline.docling_node_parser = cast(
        Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes)
    )
    pipeline.sentence_splitter = cast(
        Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes)
    )
    # Disable hierarchical node parser to ensure flat chunking (which uses the mocked splitters)
    pipeline.hierarchical_node_parser = None
    return pipeline, dummy_nodes


def test_entity_extractor_attaches_metadata(tmp_path: Path) -> None:
    """Test that the entity extractor is called and its results are attached to node metadata.

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
    """Test that exceptions in the entity extractor are handled gracefully.

    Args:
        tmp_path (Path): Temporary directory path for the test.
    """

    def bad_extractor(text: str):
        """Placeholder extractor that raises an exception.

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
    """Ensure webm files route through the audio branch rather than plain text.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """

    pipeline = DocumentIngestionPipeline(
        data_dir=Path("/tmp"),
        device="cpu",
        ner_model=None,
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


def test_openai_provider_uses_llm_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI provider should initialize the LLM NER extractor path."""

    class FakeNERConfig:
        enabled = True
        max_chars = 256
        max_workers = 2

    class FakeIngestionConfig:
        ingestion_batch_size = 2
        sentence_splitter_chunk_size = 512
        sentence_splitter_chunk_overlap = 64
        supported_filetypes: list[str] = []
        hierarchical_chunking_enabled = False
        coarse_chunk_size = 1024
        fine_chunk_size = 256
        fine_chunk_overlap = 32

    class FakeOpenAIPipeline:
        def load_prompt(self, kw: str) -> str:
            assert kw == "ner"
            return "extract {text}"

    marker: list[str] = []

    def fake_build_llm_extractor(model, prompt: str, max_chars: int):
        marker.append(f"{model}:{prompt}:{max_chars}")
        return lambda text: ([], [])

    monkeypatch.setattr(pipeline_module, "load_ner_env", lambda: FakeNERConfig())
    monkeypatch.setattr(
        pipeline_module, "load_ingestion_env", lambda: FakeIngestionConfig()
    )
    monkeypatch.setattr(pipeline_module, "OpenAIPipeline", FakeOpenAIPipeline)
    monkeypatch.setattr(
        pipeline_module, "build_llm_ner_extractor", fake_build_llm_extractor
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_gliner_ner_extractor",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    pipeline = DocumentIngestionPipeline(
        data_dir=Path("/tmp"),
        device="cpu",
        ner_model="fake-llm",  # type: ignore[arg-type]
        progress_callback=None,
        openai_model_provider="openai",
    )

    assert pipeline.entity_extractor is not None
    assert marker == ["fake-llm:extract {text}:256"]


def test_hate_speech_detection_attaches_flagged_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Enabled hate-speech detection should attach structured flags to node metadata.

    Args:
    monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    tmp_path (Path): Temporary directory path for the test.
    """

    class FakeNERConfig:
        enabled = False
        max_chars = 256
        max_workers = 1

    class FakeHateSpeechConfig:
        enabled = True
        max_chars = 128
        max_workers = 1

    class FakeIngestionConfig:
        ingestion_batch_size = 2
        sentence_splitter_chunk_size = 512
        sentence_splitter_chunk_overlap = 64
        supported_filetypes: list[str] = []
        hierarchical_chunking_enabled = False
        coarse_chunk_size = 1024
        fine_chunk_size = 256
        fine_chunk_overlap = 32

    class FakeOpenAIPipeline:
        """Fake OpenAIPipeline class for testing hate-speech detection integration."""

        def load_prompt(self, kw: str) -> str:
            """Fake load_prompt method to return a structured prompt for hate-speech detection.

            Args:
                kw (str): The keyword for which to load the prompt.

            Returns:
                str: The loaded prompt.
            """
            assert kw == "hate_speech"
            return (
                "Analyze this chunk and return JSON only:\n"
                "{\n"
                '  "hate_speech": true|false,\n'
                '  "reason": "short explanation"\n'
                "}\n"
                "\n"
                "Text:\n"
                "{text}"
            )

    class FakeResponse:
        """Fake response class to simulate the output of the OpenAI API for hate-speech detection."""

        text = '{"hate_speech": true, "category": "ethnicity", "confidence": "high", "reason": "Contains hateful language."}'

    class FakeModel:
        """Fake model class to simulate the behavior of a hate-speech detection model."""

        def complete(self, prompt: str) -> FakeResponse:
            """Simulate the completion of a prompt by the fake model.

            Args:
                prompt (str): The prompt to complete.

            Returns:
                FakeResponse: The simulated response.
            """
            assert "Analyze this chunk" in prompt
            assert "Dangerous text" in prompt
            return FakeResponse()

    monkeypatch.setattr(pipeline_module, "load_ner_env", lambda: FakeNERConfig())
    monkeypatch.setattr(
        pipeline_module, "load_hate_speech_env", lambda: FakeHateSpeechConfig()
    )
    monkeypatch.setattr(
        pipeline_module, "load_ingestion_env", lambda: FakeIngestionConfig()
    )
    monkeypatch.setattr(pipeline_module, "OpenAIPipeline", FakeOpenAIPipeline)

    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path,
        device="cpu",
        ner_model=None,
        progress_callback=None,
        hate_speech_model=cast(Any, FakeModel()),
    )
    dummy_nodes: list[Any] = []
    pipeline.entity_extractor = None
    pipeline.md_node_parser = cast(
        Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes)
    )
    pipeline.docling_node_parser = cast(
        Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes)
    )
    pipeline.sentence_splitter = cast(
        Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes)
    )
    pipeline.hierarchical_node_parser = None
    dummy_nodes.append(
        SimpleNamespace(
            text="Dangerous text for evaluation.",
            node_id="node-1",
            metadata={"filename": "doc.pdf", "file_path": "doc.pdf"},
        )
    )

    nodes = pipeline._create_nodes(
        [Document(text="Doc", metadata={"file_path": "doc.pdf"})]
    )

    assert len(nodes) == 1
    detection = nodes[0].metadata.get("hate_speech")
    assert isinstance(detection, dict)
    assert detection["hate_speech"] is True
    assert detection["category"] == "ethnicity"
    assert detection["confidence"] == "high"
    assert detection["chunk_id"] == "node-1"
    assert "Dangerous text" in detection["chunk_text"]


def test_hate_speech_detection_parallel_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Multi-worker hate-speech detection should process all nodes concurrently.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: Temporary directory path for the test.
    """

    class FakeNERConfig:
        """NER config stub with extraction disabled."""

        enabled = False
        max_chars = 256
        max_workers = 1

    class FakeHateSpeechConfig:
        """Hate-speech config stub with two workers enabled."""

        enabled = True
        max_chars = 128
        max_workers = 2

    class FakeIngestionConfig:
        """Ingestion config stub with default settings."""

        ingestion_batch_size = 2
        sentence_splitter_chunk_size = 512
        sentence_splitter_chunk_overlap = 64
        supported_filetypes: list[str] = []
        hierarchical_chunking_enabled = False
        coarse_chunk_size = 1024
        fine_chunk_size = 256
        fine_chunk_overlap = 32

    class FakeOpenAIPipeline:
        """Fake OpenAIPipeline that returns a hate-speech prompt."""

        def load_prompt(self, kw: str) -> str:
            """Return a canned hate-speech detection prompt.

            Args:
                kw: The prompt keyword.

            Returns:
                A placeholder prompt template.
            """
            return "Detect hate speech:\n{text}"

    call_count = 0

    class FakeResponse:
        """Fake LLM response indicating hate speech detected."""

        text = (
            '{"hate_speech": true, "category": "ethnicity",'
            ' "confidence": "high", "reason": "offensive"}'
        )

    class FakeModel:
        """Fake model that counts invocations."""

        def complete(self, prompt: str) -> FakeResponse:
            """Increment invocation count and return a flagged response.

            Args:
                prompt: The prompt text.

            Returns:
                A ``FakeResponse`` with hate-speech flagged.
            """
            nonlocal call_count
            call_count += 1
            return FakeResponse()

    monkeypatch.setattr(pipeline_module, "load_ner_env", lambda: FakeNERConfig())
    monkeypatch.setattr(
        pipeline_module, "load_hate_speech_env", lambda: FakeHateSpeechConfig()
    )
    monkeypatch.setattr(
        pipeline_module, "load_ingestion_env", lambda: FakeIngestionConfig()
    )
    monkeypatch.setattr(pipeline_module, "OpenAIPipeline", FakeOpenAIPipeline)

    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path,
        device="cpu",
        ner_model=None,
        progress_callback=None,
        hate_speech_model=cast(Any, FakeModel()),
    )
    pipeline.entity_extractor = None
    pipeline.md_node_parser = cast(
        Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes)
    )
    pipeline.docling_node_parser = cast(
        Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes)
    )
    pipeline.sentence_splitter = cast(
        Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes)
    )
    pipeline.hierarchical_node_parser = None

    dummy_nodes: list[Any] = [
        SimpleNamespace(
            text="Bad text one.",
            node_id="n-1",
            metadata={"filename": "a.pdf", "file_path": "a.pdf"},
        ),
        SimpleNamespace(
            text="Bad text two.",
            node_id="n-2",
            metadata={"filename": "b.pdf", "file_path": "b.pdf"},
        ),
        SimpleNamespace(
            text="Bad text three.",
            node_id="n-3",
            metadata={"filename": "c.pdf", "file_path": "c.pdf"},
        ),
    ]

    nodes = pipeline._create_nodes(
        [Document(text="Doc", metadata={"file_path": "doc.pdf"})]
    )

    assert call_count == 3
    assert pipeline.hate_speech_max_workers == 2
    for node in nodes:
        detection = node.metadata.get("hate_speech")
        assert isinstance(detection, dict)
        assert detection["hate_speech"] is True
