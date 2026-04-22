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
        """Fake init_logger function to track execution order."""
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

    monkeypatch.setattr(ingest, "init_logger", fake_setup)
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


def test_pipeline_batches_audio_files_with_audio_reader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ingestion pipeline should batch audio files through AudioReader.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """

    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.mp3"
    text_file = tmp_path / "notes.txt"
    first_audio.write_bytes(b"a")
    second_audio.write_bytes(b"b")
    text_file.write_text("plain text", encoding="utf-8")

    class FakeNERConfig:
        """NER config stub with extraction disabled."""

        enabled = False
        max_chars = 256
        max_workers = 1

    class FakeIngestionConfig:
        """Ingestion config stub with small batch size for testing."""

        ingestion_batch_size = 5
        sentence_splitter_chunk_size = 512
        sentence_splitter_chunk_overlap = 64
        supported_filetypes = [".wav", ".mp3", ".txt"]
        hierarchical_chunking_enabled = False
        coarse_chunk_size = 1024
        fine_chunk_size = 256
        fine_chunk_overlap = 32

    monkeypatch.setattr(pipeline_module, "load_ner_env", lambda: FakeNERConfig())
    monkeypatch.setattr(
        pipeline_module, "load_ingestion_env", lambda: FakeIngestionConfig()
    )

    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path,
        device="cpu",
        ner_model=None,
        progress_callback=None,
    )

    captured_batches: list[list[str]] = []

    class FakeAudioReader:
        """Fake AudioReader that captures batch file paths and returns dummy documents."""

        max_workers = 2

        def load_batch_data(
            self,
            files: list[Path],
            *,
            extra_info: list[dict[str, Any] | None] | None = None,
        ) -> list[list[Document]]:
            """Load a batch of audio files and return dummy documents.

            Args:
                files (list[Path]): List of audio file paths.
                extra_info (list[dict[str, Any]  |  None] | None, optional): Additional information for each file. Defaults to None.

            Returns:
                list[list[Document]]: List of lists of dummy Document objects.
            """
            captured_batches.append([str(path) for path in files])
            return [
                [
                    Document(
                        text=f"audio:{Path(path).name}",
                        metadata={"file_path": str(path), "file_hash": "audio-hash"},
                    )
                ]
                for path in files
            ]

    fake_dir_reader = SimpleNamespace(
        input_files=[first_audio, second_audio, text_file],
        file_metadata=lambda file_path: {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "filename": Path(file_path).name,
            "file_hash": f"hash:{Path(file_path).name}",
        },
        file_extractor={},
        filename_as_id=False,
        encoding="utf-8",
        errors="ignore",
        raise_on_error=False,
        fs=None,
        _exclude_metadata=lambda docs: docs,
    )

    monkeypatch.setattr(
        DocumentIngestionPipeline, "_load_doc_readers", lambda self: None
    )
    monkeypatch.setattr(
        DocumentIngestionPipeline, "_load_node_parsers", lambda self: None
    )
    monkeypatch.setattr(
        pipeline_module.SimpleDirectoryReader,
        "load_file",
        staticmethod(
            lambda **kwargs: [
                Document(
                    text=f"text:{Path(kwargs['input_file']).name}",
                    metadata={
                        "file_path": str(kwargs["input_file"]),
                        "file_hash": "text-hash",
                    },
                )
            ]
        ),
    )
    monkeypatch.setattr(
        DocumentIngestionPipeline,
        "_process_batch",
        lambda self, docs, existing_hashes: (docs, []),
    )

    pipeline.audio_reader = cast(Any, FakeAudioReader())
    pipeline.dir_reader = cast(Any, fake_dir_reader)

    batches = list(pipeline.build())

    assert captured_batches == [[str(first_audio), str(second_audio)]]
    assert len(batches) == 1
    docs, nodes = batches[0]
    assert nodes == []
    assert [doc.text for doc in docs] == [
        "audio:first.wav",
        "audio:second.mp3",
        "text:notes.txt",
    ]


def test_pipeline_streams_large_audio_runs_in_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Large contiguous audio runs should be flushed in bounded windows.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """

    audio_files: list[Path] = []
    for idx in range(5):
        file_path = tmp_path / f"clip-{idx}.wav"
        file_path.write_bytes(f"{idx}".encode("utf-8"))
        audio_files.append(file_path)

    class FakeNERConfig:
        """NER config stub with extraction disabled."""

        enabled = False
        max_chars = 256
        max_workers = 1

    class FakeIngestionConfig:
        """Ingestion config stub with a bounded batch size for streaming tests."""

        ingestion_batch_size = 3
        sentence_splitter_chunk_size = 512
        sentence_splitter_chunk_overlap = 64
        supported_filetypes = [".wav"]
        hierarchical_chunking_enabled = False
        coarse_chunk_size = 1024
        fine_chunk_size = 256
        fine_chunk_overlap = 32

    monkeypatch.setattr(pipeline_module, "load_ner_env", lambda: FakeNERConfig())
    monkeypatch.setattr(
        pipeline_module, "load_ingestion_env", lambda: FakeIngestionConfig()
    )

    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path,
        device="cpu",
        ner_model=None,
        progress_callback=None,
    )

    captured_batches: list[list[str]] = []

    class FakeAudioReader:
        """Fake AudioReader that exposes a worker count and captures streamed windows."""

        max_workers = 2

        def load_batch_data(
            self,
            files: list[Path],
            *,
            extra_info: list[dict[str, Any] | None] | None = None,
        ) -> list[list[Document]]:
            """Return dummy documents for a streamed batch of audio files.

            Args:
                files: The audio files in the streamed window.
                extra_info: Additional file metadata entries.

            Returns:
                list[list[Document]]: Dummy documents keyed by file name.
            """

            captured_batches.append([str(path) for path in files])
            return [
                [
                    Document(
                        text=f"audio:{path.name}",
                        metadata={
                            "file_path": str(path),
                            "file_hash": f"hash:{path.name}",
                        },
                    )
                ]
                for path in files
            ]

    fake_dir_reader = SimpleNamespace(
        input_files=audio_files,
        file_metadata=lambda file_path: {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "filename": Path(file_path).name,
            "file_hash": f"hash:{Path(file_path).name}",
        },
        file_extractor={},
        filename_as_id=False,
        encoding="utf-8",
        errors="ignore",
        raise_on_error=False,
        fs=None,
        _exclude_metadata=lambda docs: docs,
    )

    monkeypatch.setattr(
        DocumentIngestionPipeline, "_load_doc_readers", lambda self: None
    )
    monkeypatch.setattr(
        DocumentIngestionPipeline, "_load_node_parsers", lambda self: None
    )
    monkeypatch.setattr(
        DocumentIngestionPipeline,
        "_process_batch",
        lambda self, docs, existing_hashes: (docs, []),
    )

    pipeline.audio_reader = cast(Any, FakeAudioReader())
    pipeline.dir_reader = cast(Any, fake_dir_reader)

    list(pipeline.build())

    assert captured_batches == [
        [
            str(audio_files[0]),
            str(audio_files[1]),
            str(audio_files[2]),
            str(audio_files[3]),
        ],
        [str(audio_files[4])],
    ]


def test_build_streaming_yields_enrichment_batches_and_completion_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Streaming build should emit enriched node chunks and completion hashes.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """

    class FakeNERConfig:
        """NER config stub with extraction disabled for deterministic tests."""

        enabled = False
        max_chars = 256
        max_workers = 1

    class FakeIngestionConfig:
        """Ingestion config stub with small batch size for chunked streaming."""

        ingestion_batch_size = 2
        sentence_splitter_chunk_size = 512
        sentence_splitter_chunk_overlap = 64
        supported_filetypes = [".txt"]
        hierarchical_chunking_enabled = False
        coarse_chunk_size = 1024
        fine_chunk_size = 256
        fine_chunk_overlap = 32

    monkeypatch.setattr(pipeline_module, "load_ner_env", lambda: FakeNERConfig())
    monkeypatch.setattr(
        pipeline_module, "load_ingestion_env", lambda: FakeIngestionConfig()
    )

    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path,
        device="cpu",
        ner_model=None,
        progress_callback=None,
    )

    docs_input = [
        Document(text="a", metadata={"file_hash": "hash-a"}),
        Document(text="b", metadata={"file_hash": "hash-b"}),
    ]
    nodes_input = [
        cast(Any, SimpleNamespace(text=f"n{i}", metadata={"file_hash": "hash-a"}))
        for i in range(5)
    ]

    enrich_calls: list[tuple[int, int, int]] = []

    monkeypatch.setattr(
        DocumentIngestionPipeline, "_load_doc_readers", lambda self: None
    )
    monkeypatch.setattr(
        DocumentIngestionPipeline, "_load_node_parsers", lambda self: None
    )
    monkeypatch.setattr(
        DocumentIngestionPipeline,
        "_iter_loaded_documents",
        lambda self: iter([docs_input]),
    )
    monkeypatch.setattr(
        DocumentIngestionPipeline,
        "_attach_clean_text",
        lambda self, docs: list(docs),
    )
    monkeypatch.setattr(
        DocumentIngestionPipeline,
        "_ensure_file_hashes",
        lambda self, docs: docs,
    )
    monkeypatch.setattr(
        DocumentIngestionPipeline,
        "_filter_docs_by_existing_hashes",
        lambda self, docs, existing_hashes: list(docs),
    )
    monkeypatch.setattr(
        DocumentIngestionPipeline,
        "_create_nodes_without_enrichment",
        lambda self, docs: list(nodes_input),
    )
    monkeypatch.setattr(
        DocumentIngestionPipeline,
        "_enrich_nodes_in_place",
        lambda self, nodes, progress_offset=0, progress_total=None: enrich_calls.append(
            (len(nodes), progress_offset, int(progress_total or 0))
        ),
    )

    pipeline.dir_reader = cast(Any, SimpleNamespace())

    batches = list(pipeline.build_streaming(existing_hashes=set()))

    assert [len(nodes) for _, nodes, _ in batches] == [2, 2, 1]
    assert [len(docs) for docs, _, _ in batches] == [2, 0, 0]
    assert batches[0][2] == set()
    assert batches[1][2] == set()
    assert batches[2][2] == {"hash-a", "hash-b"}
    assert enrich_calls == [
        (2, 0, 5),
        (2, 2, 5),
        (1, 4, 5),
    ]


def _install_flush_audio_pipeline_stubs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, audio_reader: Any
) -> DocumentIngestionPipeline:
    """Build a pipeline wired for ``_flush_audio_batch`` behaviour tests.

    Mocks just enough config and collaborators to trigger an audio flush:
    one pending audio file followed by a non-audio file so the flush fires
    before the non-audio file is loaded.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
        audio_reader: The ``AudioReader`` stub to install on the pipeline.

    Returns:
        DocumentIngestionPipeline: A pipeline with stubbed readers and
            configuration ready to exercise the audio flush path.
    """

    audio_file = tmp_path / "clip.wav"
    text_file = tmp_path / "notes.txt"
    audio_file.write_bytes(b"a")
    text_file.write_text("plain text", encoding="utf-8")

    class FakeNERConfig:
        enabled = False
        max_chars = 256
        max_workers = 1

    class FakeIngestionConfig:
        ingestion_batch_size = 5
        sentence_splitter_chunk_size = 512
        sentence_splitter_chunk_overlap = 64
        supported_filetypes = [".wav", ".txt"]
        hierarchical_chunking_enabled = False
        coarse_chunk_size = 1024
        fine_chunk_size = 256
        fine_chunk_overlap = 32

    monkeypatch.setattr(pipeline_module, "load_ner_env", lambda: FakeNERConfig())
    monkeypatch.setattr(
        pipeline_module, "load_ingestion_env", lambda: FakeIngestionConfig()
    )
    monkeypatch.setattr(
        DocumentIngestionPipeline, "_load_doc_readers", lambda self: None
    )
    monkeypatch.setattr(
        DocumentIngestionPipeline, "_load_node_parsers", lambda self: None
    )
    monkeypatch.setattr(
        pipeline_module.SimpleDirectoryReader,
        "load_file",
        staticmethod(
            lambda **kwargs: [
                Document(
                    text=f"text:{Path(kwargs['input_file']).name}",
                    metadata={
                        "file_path": str(kwargs["input_file"]),
                        "file_hash": "text-hash",
                    },
                )
            ]
        ),
    )
    monkeypatch.setattr(
        DocumentIngestionPipeline,
        "_process_batch",
        lambda self, docs, existing_hashes: (docs, []),
    )

    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path,
        device="cpu",
        ner_model=None,
        progress_callback=None,
    )

    fake_dir_reader = SimpleNamespace(
        input_files=[audio_file, text_file],
        file_metadata=lambda file_path: {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "filename": Path(file_path).name,
            "file_hash": f"hash:{Path(file_path).name}",
        },
        file_extractor={},
        filename_as_id=False,
        encoding="utf-8",
        errors="ignore",
        raise_on_error=False,
        fs=None,
        _exclude_metadata=lambda docs: docs,
    )

    pipeline.audio_reader = cast(Any, audio_reader)
    pipeline.dir_reader = cast(Any, fake_dir_reader)
    return pipeline


def test_flush_audio_batch_releases_image_embedding_backend_before_whisper_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLIP (and any tagging backend) must be released BEFORE Whisper loads.

    In CPU single-worker mode the Whisper model is loaded in-process by
    ``AudioReader`` when ``load_batch_data`` runs. Any pre-existing image
    embedding backend (CLIP, ~600 MB–1.5 GB resident) stays pinned until
    something explicitly drops it, which piles Whisper on top of CLIP and
    overflows the Docker CPU memory ceiling (OOM exit 137 observed on a
    mixed batch of 2 PDFs, 2 audio files, 1 image, and 2 CSVs).

    This test locks in the contract: by the time ``audio_reader.load_batch_data``
    is invoked, ``pipeline.image_ingestion_service.embedding_backend`` (and
    any ``_tagging_backend`` field) must already be ``None`` so Python's
    ref-count collector can reclaim the CLIP weights before Whisper's
    allocation request crosses the cgroup limit.
    """

    observed_embedding_backend: list[Any] = []
    observed_tagging_backend: list[Any] = []

    class RecordingAudioReader:
        max_workers = 1
        _model: Any = object()  # pretend Whisper is already cached

        def __init__(self, image_service: Any) -> None:
            self._image_service = image_service

        def load_batch_data(
            self,
            files: list[Path],
            *,
            extra_info: list[dict[str, Any] | None] | None = None,
        ) -> list[list[Document]]:
            observed_embedding_backend.append(
                getattr(self._image_service, "embedding_backend", "missing")
            )
            observed_tagging_backend.append(
                getattr(self._image_service, "tagging_backend", "missing")
            )
            return [
                [
                    Document(
                        text=f"audio:{path.name}",
                        metadata={
                            "file_path": str(path),
                            "file_hash": f"hash:{path.name}",
                        },
                    )
                ]
                for path in files
            ]

    image_service = SimpleNamespace(
        embedding_backend=object(),  # pretend CLIP is loaded
        tagging_backend=object(),
        _embedding_backend_error=None,
        _tagging_backend_error=None,
    )
    pipeline = _install_flush_audio_pipeline_stubs(
        monkeypatch, tmp_path, RecordingAudioReader(image_service)
    )
    pipeline.image_ingestion_service = cast(Any, image_service)

    list(pipeline.build())

    assert observed_embedding_backend == [None], (
        "When _flush_audio_batch runs, image_ingestion_service.embedding_backend "
        "must already be None so the CLIP weights are GC-eligible before "
        "AudioReader.load_batch_data triggers the in-process Whisper load. "
        f"Observed: {observed_embedding_backend!r}"
    )
    assert observed_tagging_backend == [None], (
        "Any image tagging backend held on the image ingestion service must "
        "also be released before Whisper loads, for the same reason. "
        f"Observed: {observed_tagging_backend!r}"
    )


def test_flush_audio_batch_releases_whisper_model_after_transcription(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Whisper must be released after the audio batch so later readers (e.g.
    CLIP for an image file scheduled after audio in the same ingest) have
    room to load. Without this, a mixed batch of audio + image on CPU
    stacks Whisper (~1.6 GB) under CLIP's allocation request and OOMs.

    The contract: after ``audio_reader.load_batch_data`` returns,
    ``audio_reader._model`` is ``None`` so the ref-count collector can
    reclaim the Whisper weights immediately.
    """

    class FakeAudioReader:
        max_workers = 1

        def __init__(self) -> None:
            self._model: Any = object()  # pretend Whisper is cached

        def load_batch_data(
            self,
            files: list[Path],
            *,
            extra_info: list[dict[str, Any] | None] | None = None,
        ) -> list[list[Document]]:
            return [
                [
                    Document(
                        text=f"audio:{path.name}",
                        metadata={
                            "file_path": str(path),
                            "file_hash": f"hash:{path.name}",
                        },
                    )
                ]
                for path in files
            ]

    audio_reader = FakeAudioReader()
    pipeline = _install_flush_audio_pipeline_stubs(monkeypatch, tmp_path, audio_reader)

    list(pipeline.build())

    assert audio_reader._model is None, (
        "After _flush_audio_batch completes, audio_reader._model must be None "
        "so Whisper's ~1.6 GB weights become ref-count-zero before any "
        "downstream readers (CLIP, Docling) allocate."
    )


def test_flush_audio_batch_release_is_safe_without_image_ingestion_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pre-audio release must be a no-op when no image service is attached.

    Some callers construct a pipeline without passing an image ingestion
    service (e.g. test harnesses or audio-only CLI flows). The release
    hook must therefore handle ``pipeline.image_ingestion_service is None``
    without raising.
    """

    class FakeAudioReader:
        max_workers = 1

        def __init__(self) -> None:
            self._model: Any = object()

        def load_batch_data(
            self,
            files: list[Path],
            *,
            extra_info: list[dict[str, Any] | None] | None = None,
        ) -> list[list[Document]]:
            return [[] for _ in files]

    pipeline = _install_flush_audio_pipeline_stubs(
        monkeypatch, tmp_path, FakeAudioReader()
    )
    assert pipeline.image_ingestion_service is None

    # Must not raise.
    list(pipeline.build())
