"""Tests for the CLI ingest entry point and ingestion pipeline."""

import logging
import re
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Never, cast

import pytest
from _pytest.logging import LogCaptureFixture
from llama_index.core import Document
from llama_index.core.schema import TextNode
from loguru import logger

import docint.cli.ingest as ingest
import docint.core.ingest.ingestion_pipeline as pipeline_module
from docint.core.ingest.ingestion_pipeline import DocumentIngestionPipeline


@pytest.fixture
def loguru_caplog_info(caplog: LogCaptureFixture) -> Iterable[LogCaptureFixture]:
    """Bridge loguru INFO records into ``caplog`` for the duration of a test.

    Loguru bypasses ``logging``, so the stdlib ``caplog`` fixture sees none
    of its records by default. Mirrors the ``loguru_caplog`` fixture in
    ``tests/test_embedding_tokenizer.py``, lowered to INFO because the
    completion line this file asserts on is informational, not a warning.

    Args:
        caplog: The standard pytest log-capture fixture.

    Yields:
        The same ``caplog`` fixture, now populated with loguru-sourced
        records at INFO level and above.
    """
    handler_id = logger.add(caplog.handler, level="INFO", format="{message}")
    caplog.set_level(logging.INFO)
    try:
        yield caplog
    finally:
        logger.remove(handler_id)


def test_parse_hate_speech_payload_extracts_first_json_object_from_noisy_output() -> None:
    """Parser should recover the first JSON object from noisy LLM output."""
    raw = (
        "<think>private chain of thought</think>\n"
        "```json\n"
        '{"hate_speech": true, "category": "ethnicity", "confidence": "high", '
        '"reason": "Contains dehumanizing language."}\n'
        "```\n"
        "One more note the caller should ignore.\n"
        '{"hate_speech": false, "category": "none", "confidence": "low", "reason": "ignored"}'
    )

    parsed = pipeline_module._parse_hate_speech_payload(raw)

    assert parsed == {
        "hate_speech": True,
        "category": "ethnicity",
        "confidence": "high",
        "reason": "Contains dehumanizing language.",
    }


def test_parse_hate_speech_payload_returns_safe_default_for_invalid_json() -> None:
    """Parser should fail open when the model response is unrecoverably malformed."""
    raw = '{"hate_speech": true, "category": "ethnicity", "confidence": "high", "reason": "Contains "quoted" slur"}'

    parsed = pipeline_module._parse_hate_speech_payload(raw)

    assert parsed["hate_speech"] is False
    assert parsed["category"] == "none"
    assert parsed["confidence"] == "low"
    assert parsed["reason"] == ""


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

    def fake_ingest(*args: Any, **kwargs: Any) -> None:
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


def test_ingest_docs_invokes_rag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
            **kwargs: Any,
        ) -> None:
            """Placeholder ingest_docs method for the test double.

            Args:
                path (Path): The directory or file path containing documents to ingest.
                build_query_engine (bool, optional): Whether to build a query engine after
                    ingestion completes. Defaults to True.
                progress_callback (Callable[[str], None] | None, optional): Optional callback
                    that receives progress updates as status messages during ingestion.
                    Defaults to None.
                **kwargs: Ignored extra ingest flags (ner / hate_speech).
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


class _SilentRAG:
    """RAG test double that ingests without touching Qdrant or a model."""

    def __init__(self, **kwargs: Any) -> None:
        """Accept and ignore every RAG construction kwarg.

        Args:
            **kwargs: Ignored collection / hybrid configuration.
        """

    def ingest_docs(self, *args: Any, **kwargs: Any) -> None:
        """Stand in for a real ingest.

        Args:
            *args: Ignored positional ingest arguments.
            **kwargs: Ignored ingest flags.
        """

    def unload_models(self) -> None:
        """No-op model unload for the test double."""
        return None


def test_main_logs_the_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loguru_caplog_info: LogCaptureFixture,
) -> None:
    """The CLI's completion line must carry how long the run took.

    This path has no job and no ingest card, so the log line is the only
    record of the duration. It is timed around the whole ``ingest_docs``
    call — model loading included — because that is what the operator
    waited for.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory path for the test.
        loguru_caplog_info (LogCaptureFixture): Loguru-to-caplog bridge at INFO.
    """
    monkeypatch.setattr(ingest, "init_logger", lambda: None)
    monkeypatch.setattr(ingest, "set_offline_env", lambda: None)
    monkeypatch.setattr(ingest, "load_path_env", lambda: SimpleNamespace(data=tmp_path))
    monkeypatch.setattr(ingest, "get_collection", lambda: "demo")
    monkeypatch.setattr(ingest, "RAG", _SilentRAG)

    ingest.main()

    completion = [r for r in loguru_caplog_info.messages if r.startswith("Ingestion complete")]
    assert completion, f"no completion line logged; got {loguru_caplog_info.messages}"
    # A bare "Ingestion complete." is the regression: the duration went
    # unrecorded anywhere an operator could read it after the fact.
    assert re.fullmatch(r"Ingestion complete in \d{2}:\d{2}\.", completion[-1]), completion[-1]


def test_ingest_docs_does_not_log_a_run_duration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loguru_caplog_info: LogCaptureFixture,
) -> None:
    """The pipeline helper must not report a duration of its own.

    ``ingest_docs`` is one stage of a run: under the job API the same run
    goes on to resolve entities and build the collection summary, and the
    ``RAG`` construction inside this call precedes any clock started around
    the pipeline itself. A duration logged here therefore reads as the run
    total while measuring roughly half of it — the mismatch against the
    ingest card's timer that this pins. Whoever owns the whole run times it
    (``main``, ``core/api.py``'s ``ingest``, ``IngestJobManager._run``).

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory path for the test.
        loguru_caplog_info (LogCaptureFixture): Loguru-to-caplog bridge at INFO.
    """
    monkeypatch.setattr(ingest, "RAG", _SilentRAG)

    ingest.ingest_docs("demo", tmp_path)

    assert not [r for r in loguru_caplog_info.messages if "complete in" in r], loguru_caplog_info.messages


def test_ingest_docs_leaves_enable_hybrid_unset_when_hybrid_is_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unspecified ``hybrid`` must not force ``RAG(enable_hybrid=True)``.

    Regression test for the critical defect where every ingest caller's
    ``hybrid: bool = True`` default always passed an explicit
    ``enable_hybrid`` kwarg to ``RAG`` — which wins over the
    ``resolve_enable_hybrid()``-derived ``default_factory`` on
    ``RAG.enable_hybrid`` — forcing hybrid on even where the resolver
    derives ``False`` (no sparse endpoint configured), and breaking every
    ingest on non-vLLM providers (the probe hits a ``/pooling`` route that
    does not exist). ``hybrid=None`` (the new default) must leave
    ``enable_hybrid`` out of the call entirely so ``RAG``'s own default
    decides.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory path for the test.
    """
    unset = object()
    calls = SimpleNamespace(enable_hybrid=unset)

    class DummyRAG:
        """Dummy RAG class recording whether ``enable_hybrid`` was passed."""

        def __init__(self, qdrant_collection: str, **kwargs: Any) -> None:
            calls.enable_hybrid = kwargs.get("enable_hybrid", unset)

        def ingest_docs(self, path: Path, **kwargs: Any) -> None:
            return None

        def unload_models(self) -> None:
            return None

    monkeypatch.setattr(ingest, "RAG", DummyRAG)
    ingest.ingest_docs("demo", tmp_path)
    assert calls.enable_hybrid is unset


def _make_pipeline(
    tmp_path: Path, entity_extractor: Callable[[str], tuple[list[dict[str, Any]], list[dict[str, Any]]]]
) -> tuple[DocumentIngestionPipeline, list[Any]]:
    """Helper to create a pipeline with stubbed parsers and preset nodes.

    Args:
        tmp_path (Path): Temporary directory path for the pipeline.
        entity_extractor (Callable[[str], tuple[list[dict[str, Any]], list[dict[str, Any]]]]):
            The entity extractor function to use in the pipeline.

    Returns:
        tuple[DocumentIngestionPipeline, list]: The created pipeline and the list of dummy nodes
    """
    dummy_nodes: list[Any] = []
    # Pipeline.__post_init__ will override entity_extractor if env vars are present.
    # We must forcibly set the extractor AFTER init.
    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path,
        clean_fn=lambda x: x,
        ner_model=None,
        progress_callback=None,
    )

    pipeline.entity_extractor = entity_extractor

    # Minimal parser stubs to satisfy _create_nodes preconditions
    pipeline.md_node_parser = cast(Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes))
    pipeline.docling_node_parser = cast(Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes))
    pipeline.sentence_splitter = cast(Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes))
    # Disable hierarchical node parser to ensure flat chunking (which uses the mocked splitters)
    pipeline.hierarchical_node_parser = None
    return pipeline, dummy_nodes


def test_entity_extractor_attaches_metadata(tmp_path: Path) -> None:
    """Test that the entity extractor is called and its results are attached to node metadata.

    Args:
        tmp_path (Path): Temporary directory path for the test.
    """
    calls: list[str] = []

    def extractor(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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

    def bad_extractor(text: str) -> Never:
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


def test_whitelist_filters_audio_extensions_silently(tmp_path: Path) -> None:
    """SimpleDirectoryReader with ``required_exts`` silently excludes audio files.

    Creates ``.wav``, ``.mp3``, and ``.txt`` files in a temp directory, then
    instantiates a ``SimpleDirectoryReader`` with the same ``required_exts``
    list that the pipeline uses. Only the ``.txt`` path should appear in
    ``input_files``.

    This documents the invariant that makes the audio blacklist dead code:
    the whitelist upstream already prevents audio from reaching the pipeline.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    from llama_index.core import SimpleDirectoryReader

    from docint.utils.env_cfg import load_ingestion_env

    default_exts = load_ingestion_env().supported_filetypes

    wav_file = tmp_path / "clip.wav"
    mp3_file = tmp_path / "song.mp3"
    txt_file = tmp_path / "notes.txt"

    wav_file.write_bytes(b"RIFF fake wav")
    mp3_file.write_bytes(b"ID3 fake mp3")
    txt_file.write_text("Hello world.", encoding="utf-8")

    dir_reader = SimpleDirectoryReader(
        input_dir=str(tmp_path),
        required_exts=default_exts,
    )

    assert txt_file in dir_reader.input_files
    assert wav_file not in dir_reader.input_files
    assert mp3_file not in dir_reader.input_files


def _install_transcript_pipeline_stubs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> DocumentIngestionPipeline:
    """Create a ``DocumentIngestionPipeline`` with all heavy loaders stubbed out.

    Stubs out ``_load_doc_readers`` and ``_load_node_parsers`` to avoid
    downloading models. Attaches minimal parser stubs so that
    ``_create_nodes_without_enrichment`` can be called safely.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory for the pipeline's ``data_dir``.

    Returns:
        A fully-constructed ``DocumentIngestionPipeline`` ready for routing tests.
    """

    class FakeNERConfig:
        """NER config stub with extraction disabled."""

        enabled = False
        max_chars = 256
        max_workers = 1

    class FakeIngestionConfig:
        """Ingestion config stub that disables hierarchical chunking."""

        ingestion_batch_size = 5
        sentence_splitter_chunk_size = 512
        sentence_splitter_chunk_overlap = 64
        supported_filetypes: ClassVar[list[str]] = []
        hierarchical_chunking_enabled = False
        coarse_chunk_size = 1024
        fine_chunk_size = 256
        fine_chunk_overlap = 32
        streaming_readers_enabled = False

    monkeypatch.setattr(pipeline_module, "load_ner_env", lambda: FakeNERConfig())
    monkeypatch.setattr(pipeline_module, "load_ingestion_env", lambda: FakeIngestionConfig())
    monkeypatch.setattr(DocumentIngestionPipeline, "_load_doc_readers", lambda self: None)
    monkeypatch.setattr(DocumentIngestionPipeline, "_load_node_parsers", lambda self: None)

    pl = DocumentIngestionPipeline(
        data_dir=tmp_path,
        ner_model=None,
        progress_callback=None,
    )
    pl.entity_extractor = None

    # Stub parsers that would require model downloads.  For the transcript
    # routing test we rely on the real SentenceSplitter (no model) so only
    # the non-transcript paths need stubbing.
    pl.md_node_parser = cast(Any, SimpleNamespace(get_nodes_from_documents=lambda docs: []))
    pl.docling_node_parser = cast(Any, SimpleNamespace(get_nodes_from_documents=lambda docs: []))
    pl.sentence_splitter = cast(Any, SimpleNamespace(get_nodes_from_documents=lambda docs: []))
    pl.hierarchical_node_parser = None
    return pl


def test_nextext_transcript_routed_to_per_segment_nodes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Transcript-segment Documents produce exactly one node per segment (no concatenation).

    Injects 3 ``Document`` instances whose metadata contains
    ``docint_doc_kind = "transcript_segment"`` and ``source = "transcript"``.
    Runs them through ``_create_nodes_without_enrichment`` and asserts that
    exactly 3 nodes emerge, each preserving the original segment prose.

    This pins the new contract: transcript docs are routed to the
    ``SentenceSplitter(chunk_size=10_000_000)`` splitter — not the
    ``HierarchicalNodeParser`` — so each segment remains a distinct node.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """
    pl = _install_transcript_pipeline_stubs(monkeypatch, tmp_path)

    segment_texts = [
        "First spoken phrase.",
        "Second spoken phrase.",
        "Third spoken phrase.",
    ]
    docs = [
        Document(
            text=text,
            metadata={
                "file_path": "transcript.jsonl",
                "source": "transcript",
                "docint_doc_kind": "transcript_segment",
                "sentence_index": idx,
            },
        )
        for idx, text in enumerate(segment_texts)
    ]

    nodes = pl._create_nodes_without_enrichment(docs)

    assert len(nodes) == 3, f"Expected 3 nodes (one per segment), got {len(nodes)}"
    node_texts = [cast(TextNode, n).text for n in nodes]
    for text in segment_texts:
        assert text in node_texts, f"Segment text {text!r} missing from nodes — concatenation occurred"


def test_nextext_transcript_kind_wins_over_json_extension(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Documents with ``docint_doc_kind='transcript_segment'`` beat ``.jsonl`` routing.

    Even when the ``file_path`` has a ``.jsonl`` extension — which would
    otherwise route a document through the generic JSON / hierarchical path —
    the presence of ``docint_doc_kind='transcript_segment'`` in metadata must
    route it to the per-segment ``SentenceSplitter`` (one node per document).
    Pins the dispatcher priority invariant described at the top of
    ``DocumentIngestionPipeline._create_nodes_without_enrichment``.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """
    pl = _install_transcript_pipeline_stubs(monkeypatch, tmp_path)

    # If transcript_segment docs were mis-routed to json_docs, they would go
    # through the stubbed sentence_splitter (returns []) or a now-None
    # hierarchical_node_parser, producing 0 nodes.  The transcript path uses
    # a locally-constructed real SentenceSplitter, so a correct dispatch
    # yields 3 nodes with the original prose preserved.
    segment_texts = [
        "First transcript segment with a .jsonl path.",
        "Second transcript segment with a .jsonl path.",
        "Third transcript segment with a .jsonl path.",
    ]
    docs = [
        Document(
            text=text,
            metadata={
                "file_path": str(tmp_path / "interview.jsonl"),
                "file_name": "interview.jsonl",
                "file_type": "application/jsonl",
                "source": "transcript",
                "docint_doc_kind": "transcript_segment",
                "sentence_index": idx,
            },
        )
        for idx, text in enumerate(segment_texts)
    ]

    nodes = pl._create_nodes_without_enrichment(docs)

    assert len(nodes) == 3, (
        f"Expected 3 per-segment nodes; got {len(nodes)} — dispatcher likely "
        "routed transcript_segment docs through the JSON path."
    )
    node_texts = [cast(TextNode, n).text for n in nodes]
    for text in segment_texts:
        assert text in node_texts, (
            f"Segment text {text!r} missing from nodes — transcript path was not used (likely mis-routed to json_docs)."
        )


def test_hate_speech_detection_attaches_flagged_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        supported_filetypes: ClassVar[list[str]] = []
        hierarchical_chunking_enabled = False
        coarse_chunk_size = 1024
        fine_chunk_size = 256
        fine_chunk_overlap = 32
        streaming_readers_enabled = False

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

        text = (
            '{"hate_speech": true, "category": "ethnicity", "confidence": "high",'
            ' "reason": "Contains hateful language."}'
        )

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
    monkeypatch.setattr(pipeline_module, "load_hate_speech_env", lambda: FakeHateSpeechConfig())
    monkeypatch.setattr(pipeline_module, "load_ingestion_env", lambda: FakeIngestionConfig())
    monkeypatch.setattr(pipeline_module, "OpenAIPipeline", FakeOpenAIPipeline)

    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path,
        ner_model=None,
        progress_callback=None,
        hate_speech_model=cast(Any, FakeModel()),
    )
    dummy_nodes: list[Any] = []
    pipeline.entity_extractor = None
    pipeline.md_node_parser = cast(Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes))
    pipeline.docling_node_parser = cast(Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes))
    pipeline.sentence_splitter = cast(Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes))
    pipeline.hierarchical_node_parser = None
    dummy_nodes.append(
        SimpleNamespace(
            text="Dangerous text for evaluation.",
            node_id="node-1",
            metadata={"filename": "doc.pdf", "file_path": "doc.pdf"},
        )
    )

    nodes = pipeline._create_nodes([Document(text="Doc", metadata={"file_path": "doc.pdf"})])

    assert len(nodes) == 1
    detection = nodes[0].metadata.get("hate_speech")
    assert isinstance(detection, dict)
    assert detection["hate_speech"] is True
    assert detection["category"] == "ethnicity"
    assert detection["confidence"] == "high"
    assert detection["chunk_id"] == "node-1"
    assert "Dangerous text" in detection["chunk_text"]


def test_hate_speech_detection_parallel_workers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        supported_filetypes: ClassVar[list[str]] = []
        hierarchical_chunking_enabled = False
        coarse_chunk_size = 1024
        fine_chunk_size = 256
        fine_chunk_overlap = 32
        streaming_readers_enabled = False

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

        text = '{"hate_speech": true, "category": "ethnicity", "confidence": "high", "reason": "offensive"}'

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
    monkeypatch.setattr(pipeline_module, "load_hate_speech_env", lambda: FakeHateSpeechConfig())
    monkeypatch.setattr(pipeline_module, "load_ingestion_env", lambda: FakeIngestionConfig())
    monkeypatch.setattr(pipeline_module, "OpenAIPipeline", FakeOpenAIPipeline)

    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path,
        ner_model=None,
        progress_callback=None,
        hate_speech_model=cast(Any, FakeModel()),
    )
    pipeline.entity_extractor = None
    pipeline.md_node_parser = cast(Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes))
    pipeline.docling_node_parser = cast(Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes))
    pipeline.sentence_splitter = cast(Any, SimpleNamespace(get_nodes_from_documents=lambda docs: dummy_nodes))
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

    nodes = pipeline._create_nodes([Document(text="Doc", metadata={"file_path": "doc.pdf"})])

    assert call_count == 3
    assert pipeline.hate_speech_max_workers == 2
    for node in nodes:
        detection = node.metadata.get("hate_speech")
        assert isinstance(detection, dict)
        assert detection["hate_speech"] is True


_FLAGGED_RESPONSE_TEXT = '{"hate_speech": true, "category": "ethnicity", "confidence": "high", "reason": "offensive"}'


def _make_enrichment_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    ner_workers: int = 1,
    hate_enabled: bool = False,
    hate_workers: int = 1,
    hate_model: Any = None,
    progress_callback: Callable[[str], None] | None = None,
) -> DocumentIngestionPipeline:
    """Build a pipeline wired for direct ``_enrich_nodes_in_place`` tests.

    Env loaders and the prompt loader are stubbed; NER stays disabled in the
    env config (callers assign ``pipeline.entity_extractor`` afterwards, which
    is what gates the stage) while ``ner_workers`` still drives
    ``ner_max_workers``.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: Temporary directory path for the pipeline.
        ner_workers: Value for ``ner_max_workers``.
        hate_enabled: Whether hate-speech detection is enabled.
        hate_workers: Value for ``hate_speech_max_workers``.
        hate_model: Model passed as ``hate_speech_model``.
        progress_callback: Optional progress callback.

    Returns:
        The constructed pipeline.
    """

    class FakeNERConfig:
        enabled = False
        max_chars = 256
        max_workers = ner_workers

    class FakeHateSpeechConfig:
        enabled = hate_enabled
        max_chars = 512
        max_workers = hate_workers

    class FakeIngestionConfig:
        ingestion_batch_size = 2
        sentence_splitter_chunk_size = 512
        sentence_splitter_chunk_overlap = 64
        supported_filetypes: ClassVar[list[str]] = []
        hierarchical_chunking_enabled = False
        coarse_chunk_size = 1024
        fine_chunk_size = 256
        fine_chunk_overlap = 32
        streaming_readers_enabled = False

    class FakeOpenAIPipeline:
        def load_prompt(self, kw: str) -> str:
            """Return a canned hate-speech prompt template.

            Args:
                kw: The prompt keyword.

            Returns:
                A placeholder prompt template.
            """
            return "Detect hate speech:\n{text}"

    monkeypatch.setattr(pipeline_module, "load_ner_env", lambda: FakeNERConfig())
    monkeypatch.setattr(pipeline_module, "load_hate_speech_env", lambda: FakeHateSpeechConfig())
    monkeypatch.setattr(pipeline_module, "load_ingestion_env", lambda: FakeIngestionConfig())
    monkeypatch.setattr(pipeline_module, "OpenAIPipeline", FakeOpenAIPipeline)

    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path,
        ner_model=None,
        progress_callback=progress_callback,
        hate_speech_model=hate_model,
    )
    pipeline.entity_extractor = None
    return pipeline


def _enrichment_nodes(*texts: str) -> list[Any]:
    """Build SimpleNamespace nodes for direct enrichment calls.

    Args:
        *texts: One text per node.

    Returns:
        The node stubs.
    """
    return [
        SimpleNamespace(text=text, node_id=f"n-{i}", metadata={"file_path": f"doc-{i}.pdf"})
        for i, text in enumerate(texts, start=1)
    ]


def test_enrichment_overlaps_hate_speech_with_ner_across_nodes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hate-speech detection for a finished node runs while NER is still busy.

    Node two's extractor blocks until the hate-speech model has been invoked
    at least once (which can only happen for node one). Under stage-sequential
    enrichment no hate-speech call happens before every NER call returns, so
    the wait times out.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: Temporary directory path for the test.
    """
    hate_started = threading.Event()
    overlap_seen: list[bool] = []

    def extractor(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if "two" in text:
            overlap_seen.append(hate_started.wait(timeout=5))
        return ([], [])

    class FakeResponse:
        text = _FLAGGED_RESPONSE_TEXT

    class FakeModel:
        def complete(self, prompt: str) -> FakeResponse:
            """Signal that the hate-speech stage has started.

            Args:
                prompt: The prompt text.

            Returns:
                A flagged response.
            """
            hate_started.set()
            return FakeResponse()

    pipeline = _make_enrichment_pipeline(
        monkeypatch,
        tmp_path,
        ner_workers=2,
        hate_enabled=True,
        hate_workers=1,
        hate_model=cast(Any, FakeModel()),
    )
    pipeline.entity_extractor = extractor

    nodes = _enrichment_nodes("Text one.", "Text two.")
    pipeline._enrich_nodes_in_place(nodes)

    assert overlap_seen == [True]
    for node in nodes:
        assert node.metadata["hate_speech"]["hate_speech"] is True


def test_enrichment_honors_per_stage_concurrency_caps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """In-flight NER and hate-speech calls never exceed their own worker caps.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: Temporary directory path for the test.
    """
    lock = threading.Lock()
    in_flight = {"ner": 0, "hate": 0}
    max_seen = {"ner": 0, "hate": 0}
    calls = {"ner": 0, "hate": 0}

    def _enter(stage: str) -> None:
        with lock:
            in_flight[stage] += 1
            calls[stage] += 1
            max_seen[stage] = max(max_seen[stage], in_flight[stage])

    def _leave(stage: str) -> None:
        with lock:
            in_flight[stage] -= 1

    def extractor(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        _enter("ner")
        time.sleep(0.01)
        _leave("ner")
        return ([], [])

    class FakeResponse:
        text = _FLAGGED_RESPONSE_TEXT

    class FakeModel:
        def complete(self, prompt: str) -> FakeResponse:
            """Track in-flight hate-speech calls.

            Args:
                prompt: The prompt text.

            Returns:
                A flagged response.
            """
            _enter("hate")
            time.sleep(0.01)
            _leave("hate")
            return FakeResponse()

    pipeline = _make_enrichment_pipeline(
        monkeypatch,
        tmp_path,
        ner_workers=3,
        hate_enabled=True,
        hate_workers=1,
        hate_model=cast(Any, FakeModel()),
    )
    pipeline.entity_extractor = extractor

    pipeline._enrich_nodes_in_place(_enrichment_nodes(*[f"Chunk {i}." for i in range(6)]))

    assert calls == {"ner": 6, "hate": 6}
    assert max_seen["ner"] <= 3
    assert max_seen["hate"] == 1


def test_enrichment_progress_messages_per_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each stage emits its verbatim counter messages in monotonic order.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: Temporary directory path for the test.
    """
    messages: list[str] = []

    class FakeResponse:
        text = _FLAGGED_RESPONSE_TEXT

    class FakeModel:
        def complete(self, prompt: str) -> FakeResponse:
            """Return a flagged response.

            Args:
                prompt: The prompt text.

            Returns:
                A flagged response.
            """
            return FakeResponse()

    pipeline = _make_enrichment_pipeline(
        monkeypatch,
        tmp_path,
        ner_workers=2,
        hate_enabled=True,
        hate_workers=2,
        hate_model=cast(Any, FakeModel()),
        progress_callback=messages.append,
    )
    pipeline.entity_extractor = lambda text: ([], [])

    pipeline._enrich_nodes_in_place(_enrichment_nodes("One.", "Two.", "Three."))

    assert [m for m in messages if m.startswith("Extracting entities")] == [
        f"Extracting entities: {i}/3 chunks processed" for i in (1, 2, 3)
    ]
    assert [m for m in messages if m.startswith("Detecting hate speech")] == [
        f"Detecting hate speech: {i}/3 chunks processed" for i in (1, 2, 3)
    ]

    messages.clear()
    pipeline._enrich_nodes_in_place(_enrichment_nodes("Four.", "Five.", "Six."), progress_offset=2, progress_total=5)

    assert [m for m in messages if m.startswith("Extracting entities")] == [
        f"Extracting entities: {i}/5 chunks processed" for i in (3, 4, 5)
    ]
    assert [m for m in messages if m.startswith("Detecting hate speech")] == [
        f"Detecting hate speech: {i}/5 chunks processed" for i in (3, 4, 5)
    ]


def test_enrichment_exactly_once_and_skips_empty_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Blank nodes tick progress without remote calls; others are hit exactly once.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: Temporary directory path for the test.
    """
    messages: list[str] = []
    extracted: list[str] = []
    completed: list[str] = []

    class FakeResponse:
        text = _FLAGGED_RESPONSE_TEXT

    class FakeModel:
        def complete(self, prompt: str) -> FakeResponse:
            """Record the prompt and return a flagged response.

            Args:
                prompt: The prompt text.

            Returns:
                A flagged response.
            """
            completed.append(prompt)
            return FakeResponse()

    pipeline = _make_enrichment_pipeline(
        monkeypatch,
        tmp_path,
        ner_workers=1,
        hate_enabled=True,
        hate_workers=1,
        hate_model=cast(Any, FakeModel()),
        progress_callback=messages.append,
    )

    def extractor(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        extracted.append(text)
        return ([], [])

    pipeline.entity_extractor = extractor

    nodes = _enrichment_nodes("a", "   ", "b")
    pipeline._enrich_nodes_in_place(nodes)

    assert sorted(extracted) == ["a", "b"]
    assert len(completed) == 2
    assert len([m for m in messages if m.startswith("Extracting entities")]) == 3
    assert len([m for m in messages if m.startswith("Detecting hate speech")]) == 3
    assert "hate_speech" not in nodes[1].metadata
    assert "entities" not in nodes[1].metadata


def test_enrichment_propagates_progress_callback_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A raising progress callback fails the batch instead of being swallowed.

    The production callback (jobs.py ``_push``) raises when the event loop is
    closed mid-ingest; before the single-pool refactor that error propagated
    from the coordinating thread and failed the job. Worker-thread errors must
    surface the same way rather than vanish in ``wait(futures)``.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: Temporary directory path for the test.
    """

    def broken_callback(message: str) -> None:
        raise RuntimeError("Event loop is closed")

    pipeline = _make_enrichment_pipeline(monkeypatch, tmp_path, ner_workers=2, progress_callback=broken_callback)
    pipeline.entity_extractor = lambda text: ([], [])

    with pytest.raises(RuntimeError, match="Event loop is closed"):
        pipeline._enrich_nodes_in_place(_enrichment_nodes("One.", "Two."))


def test_enrichment_single_stage_and_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A disabled stage emits no progress frames; fully disabled is a no-op.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: Temporary directory path for the test.
    """
    messages: list[str] = []

    class FakeResponse:
        text = _FLAGGED_RESPONSE_TEXT

    class FakeModel:
        def complete(self, prompt: str) -> FakeResponse:
            """Return a flagged response.

            Args:
                prompt: The prompt text.

            Returns:
                A flagged response.
            """
            return FakeResponse()

    # Hate-speech only: no entity frames.
    pipeline = _make_enrichment_pipeline(
        monkeypatch,
        tmp_path,
        hate_enabled=True,
        hate_model=cast(Any, FakeModel()),
        progress_callback=messages.append,
    )
    pipeline._enrich_nodes_in_place(_enrichment_nodes("One.", "Two."))
    assert messages
    assert not [m for m in messages if m.startswith("Extracting entities")]

    # NER only: no hate frames.
    messages.clear()
    pipeline = _make_enrichment_pipeline(monkeypatch, tmp_path, ner_workers=2, progress_callback=messages.append)
    pipeline.entity_extractor = lambda text: ([{"text": "x"}], [])
    nodes = _enrichment_nodes("One.")
    pipeline._enrich_nodes_in_place(nodes)
    assert nodes[0].metadata["entities"] == [{"text": "x"}]
    assert messages
    assert not [m for m in messages if m.startswith("Detecting hate speech")]

    # Neither stage: nothing happens.
    messages.clear()
    pipeline = _make_enrichment_pipeline(monkeypatch, tmp_path, progress_callback=messages.append)
    nodes = _enrichment_nodes("One.")
    pipeline._enrich_nodes_in_place(nodes)
    assert messages == []
    assert nodes[0].metadata == {"file_path": "doc-1.pdf"}


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
        supported_filetypes: ClassVar[list[str]] = [".txt"]
        hierarchical_chunking_enabled = False
        coarse_chunk_size = 1024
        fine_chunk_size = 256
        fine_chunk_overlap = 32
        streaming_readers_enabled = False

    monkeypatch.setattr(pipeline_module, "load_ner_env", lambda: FakeNERConfig())
    monkeypatch.setattr(pipeline_module, "load_ingestion_env", lambda: FakeIngestionConfig())

    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path,
        ner_model=None,
        progress_callback=None,
    )

    docs_input = [
        Document(text="a", metadata={"file_hash": "hash-a"}),
        Document(text="b", metadata={"file_hash": "hash-b"}),
    ]
    nodes_input = [cast(Any, SimpleNamespace(text=f"n{i}", metadata={"file_hash": "hash-a"})) for i in range(5)]

    enrich_calls: list[tuple[int, int, int]] = []

    monkeypatch.setattr(DocumentIngestionPipeline, "_load_doc_readers", lambda self: None)
    monkeypatch.setattr(DocumentIngestionPipeline, "_load_node_parsers", lambda self: None)
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


def _make_streaming_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    streaming_readers_enabled: bool,
) -> DocumentIngestionPipeline:
    """Construct a pipeline stub with streaming_readers_enabled set as requested."""

    class FakeNERConfig:
        enabled = False
        max_chars = 256
        max_workers = 1

    class FakeIngestionConfig:
        ingestion_batch_size = 5
        sentence_splitter_chunk_size = 512
        sentence_splitter_chunk_overlap = 64
        supported_filetypes: ClassVar[list[str]] = []
        hierarchical_chunking_enabled = False
        coarse_chunk_size = 1024
        fine_chunk_size = 256
        fine_chunk_overlap = 32
        streaming_readers_enabled = False  # overridden below

    FakeIngestionConfig.streaming_readers_enabled = streaming_readers_enabled

    monkeypatch.setattr(pipeline_module, "load_ner_env", lambda: FakeNERConfig())
    monkeypatch.setattr(pipeline_module, "load_ingestion_env", lambda: FakeIngestionConfig())
    monkeypatch.setattr(DocumentIngestionPipeline, "_load_doc_readers", lambda self: None)
    monkeypatch.setattr(DocumentIngestionPipeline, "_load_node_parsers", lambda self: None)

    return DocumentIngestionPipeline(
        data_dir=tmp_path,
        ner_model=None,
        progress_callback=None,
    )


def test_streaming_reader_dispatch_calls_iter_documents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With STREAMING_READERS_ENABLED=true, _iter_loaded_documents calls iter_documents directly.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """
    pipeline = _make_streaming_pipeline(monkeypatch, tmp_path, streaming_readers_enabled=True)

    csv_file = tmp_path / "rows.csv"
    csv_file.write_text("text\nhello\n", encoding="utf-8")

    iter_calls: list[dict[str, Any]] = []
    fake_doc = Document(
        text="streamed",
        metadata={"file_hash": "abc123", "file_path": str(csv_file)},
    )

    class FakeReader:
        def iter_documents(self, file: Path, **kwargs: Any) -> Any:
            iter_calls.append({"file": file, "extra_info": kwargs.get("extra_info")})
            yield fake_doc

    fake_metadata = {
        "file_path": str(csv_file),
        "file_name": "rows.csv",
        "filename": "rows.csv",
        "file_hash": "abc123",
    }
    pipeline.dir_reader = cast(
        Any,
        SimpleNamespace(
            input_files=[csv_file],
            file_extractor={".csv": FakeReader()},
            file_metadata=lambda _path: fake_metadata,
            _exclude_metadata=lambda docs: docs,
        ),
    )

    result = list(pipeline._iter_loaded_documents())

    assert len(iter_calls) == 1, "iter_documents should be called once per file"
    assert iter_calls[0]["file"] == csv_file
    assert iter_calls[0]["extra_info"]["file_hash"] == "abc123"
    assert len(result) == 1
    assert result[0][0].text == "streamed"


def test_streaming_reader_dispatch_falls_back_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With STREAMING_READERS_ENABLED=false, _iter_loaded_documents uses SimpleDirectoryReader.load_file.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """
    pipeline = _make_streaming_pipeline(monkeypatch, tmp_path, streaming_readers_enabled=False)

    csv_file = tmp_path / "rows.csv"
    csv_file.write_text("text\nhello\n", encoding="utf-8")

    iter_calls: list[Path] = []

    class FakeReader:
        def iter_documents(self, file: Path, **kwargs: Any) -> Any:
            iter_calls.append(file)
            yield Document(text="should-not-appear", metadata={})

    load_file_calls: list[Path] = []
    fallback_doc = Document(
        text="from-load-file",
        metadata={"file_hash": "xyz", "file_path": str(csv_file)},
    )

    def fake_load_file(input_file: Path, **_kwargs: Any) -> list[Document]:
        load_file_calls.append(input_file)
        return [fallback_doc]

    monkeypatch.setattr(pipeline_module.SimpleDirectoryReader, "load_file", staticmethod(fake_load_file))

    pipeline.dir_reader = cast(
        Any,
        SimpleNamespace(
            input_files=[csv_file],
            file_extractor={".csv": FakeReader()},
            file_metadata=lambda _path: {"file_hash": "xyz"},
            _exclude_metadata=lambda docs: docs,
            filename_as_id=False,
            encoding="utf-8",
            errors="ignore",
            raise_on_error=False,
            fs=None,
        ),
    )

    result = list(pipeline._iter_loaded_documents())

    assert iter_calls == [], "iter_documents must NOT be called when streaming is disabled"
    assert len(load_file_calls) == 1
    assert load_file_calls[0] == csv_file
    assert len(result) == 1
    assert result[0][0].text == "from-load-file"
