"""Document ingestion pipeline: chunking and metadata extraction."""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, NotRequired, TypedDict, cast

from llama_index.core import Document, SimpleDirectoryReader

__all__ = ["DocumentIngestionPipeline", "NoSupportedFilesError", "SimpleDirectoryReader"]
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    NodeParser,
    SentenceSplitter,
)
from llama_index.core.schema import BaseNode
from llama_index.llms.openai import OpenAI
from llama_index.node_parser.docling import DoclingNodeParser
from loguru import logger

from docint.core.ingest.enrichment import set_enrichment
from docint.core.ingest.hate_speech import (
    CHARS_PER_TOKEN,
    CHUNK_CATEGORIES,
    CONFIDENCE_LEVELS,
    REASON_MAX_CHARS,
    TranscriptLine,
    TranscriptWindow,
    WindowFinding,
    chunk_response_format,
    is_structured_output_rejection,
    next_window,
    normalize_choice,
    normalize_stance,
    parse_window_reply,
    render_window_prompt,
    window_response_format,
)
from docint.core.ingest.images_service import ImageIngestionService
from docint.core.ingest.preprocess import IMAGE_EXTENSIONS, get_preprocess_pool, submit_file
from docint.core.ingest.standalone_media import StandaloneMediaIngestor
from docint.core.jobs import JobCancelled
from docint.core.readers.docx import DocxReader
from docint.core.readers.images import ImageReader
from docint.core.readers.json import CustomJSONReader
from docint.core.readers.rtf import RTFReader
from docint.core.readers.tables import TableReader
from docint.core.storage.hierarchical import HierarchicalNodeParser
from docint.utils.batching import chunk_nodes
from docint.utils.clean_text import basic_clean
from docint.utils.env_cfg import (
    load_hate_speech_env,
    load_ingestion_env,
    load_ner_env,
)
from docint.utils.hashing import compute_file_hash
from docint.utils.llm_sanitize import strip_reasoning
from docint.utils.ner_client import build_remote_ner_extractor
from docint.utils.openai_cfg import OpenAIPipeline

CleanFn = Callable[[str], str]

WINDOW_MIN_OUTPUT_TOKENS: int = 1024
"""Minimum output-token cap for one transcript-window request."""

WINDOW_OUTPUT_TOKENS_PER_SEGMENT: int = 80
"""Output tokens budgeted per core segment, so a window whose every segment is a candidate still fits."""

_TRANSCRIPT_WINDOW_FAILURE_LIMIT: int = 3
"""Consecutive failed window requests after which the pass stops calling a dead endpoint."""


class NoSupportedFilesError(RuntimeError):
    """Raised when a staged batch holds nothing the pipeline can ingest.

    No file matches the reader's ``required_exts`` whitelist and the Nextext
    media pre-passes claimed nothing either (e.g. an audio-only upload with
    Nextext unconfigured, or a genuinely empty directory). Callers surface
    this as a visible "no ingestable files" warning rather than letting the
    run complete as a silent success.
    """


class HateSpeechDetection(TypedDict):
    """Structured hate-speech detection payload."""

    hate_speech: bool
    category: str
    confidence: str
    reason: str
    chunk_id: NotRequired[str]
    chunk_text: NotRequired[str]
    source_ref: NotRequired[str]


def _verdict_payload(value: Any) -> dict[str, Any] | list[Any] | None:
    """Return ``value`` when it can carry a verdict: an object, or a list holding one.

    Args:
        value (Any): A decoded JSON value.

    Returns:
        dict[str, Any] | list[Any] | None: ``value``, or ``None`` for anything else.
    """
    if isinstance(value, dict) or (isinstance(value, list) and any(isinstance(item, dict) for item in value)):
        return value
    return None


def _extract_first_json_verdict(text: str) -> dict[str, Any] | list[Any] | None:
    """Return the first JSON object, or list of objects, embedded in *text*.

    A list is returned whole: scanning for ``{`` alone would read a fenced
    per-statement list as its first item and drop every later verdict.

    Args:
        text (str): Arbitrary model output that may contain prose, fences,
            or multiple JSON values.

    Returns:
        dict[str, Any] | list[Any] | None: The first decodable verdict
            payload, or ``None`` when the text holds none.
    """
    decoder = json.JSONDecoder()

    for idx, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        payload = _verdict_payload(parsed)
        if payload is not None:
            return payload

    return None


def _parse_hate_speech_reply(raw: str) -> HateSpeechDetection | None:
    """Parse a stance-aware hate-speech verdict, or report it as unparseable.

    The verdict comes from ``stance`` alone: only ``endorses`` is hate speech.
    No boolean is read, so neither a stray ``"hate_speech": true`` nor the
    string ``"false"`` can create a finding. Categories and confidences are
    normalised to their enums; an endorsed verdict never carries ``none``.
    A model that answers with one verdict per statement is read as the
    passage the prompt asked about: its first endorsing verdict, else its
    first verdict — never its first item alone, which silently drops an
    endorsement that is not listed first.

    Args:
        raw (str): The raw model output (reasoning, prose and fences tolerated).

    Returns:
        HateSpeechDetection | None: The verdict, or ``None`` when the reply
            holds no JSON object.
    """
    cleaned, captured = strip_reasoning(raw or "")
    if captured:
        logger.debug(
            "Stripped {} chars of reasoning from hate-speech response",
            len(captured),
        )

    parsed: Any
    try:
        parsed = _verdict_payload(json.loads(cleaned))
    except json.JSONDecodeError:
        parsed = None
    if parsed is None:
        parsed = _extract_first_json_verdict(cleaned)
    if isinstance(parsed, list):
        verdicts = [item for item in parsed if isinstance(item, dict)]
        parsed = next((v for v in verdicts if normalize_stance(v.get("stance")) == "endorses"), verdicts[0])
    if not isinstance(parsed, dict):
        return None

    endorsed = normalize_stance(parsed.get("stance")) == "endorses"
    category = normalize_choice(parsed.get("category"), CHUNK_CATEGORIES, "other")
    if not endorsed:
        category = "none"
    elif category == "none":
        category = "other"
    return {
        "hate_speech": endorsed,
        "category": category,
        "confidence": normalize_choice(parsed.get("confidence"), CONFIDENCE_LEVELS, "low"),
        "reason": str(parsed.get("reason") or "").strip()[:REASON_MAX_CHARS],
    }


def _node_id(node: Any) -> str:
    """Return a node's id (``node_id``, else ``id_``), or ``""``.

    Args:
        node (Any): The node.

    Returns:
        str: The id.
    """
    return str(getattr(node, "node_id", "") or getattr(node, "id_", "") or "")


def _sentence_index(node: Any) -> int:
    """Return a transcript segment's ``sentence_index`` for ordering (unknown sorts last).

    Args:
        node (Any): The segment node.

    Returns:
        int: The index.
    """
    value = (getattr(node, "metadata", {}) or {}).get("sentence_index")
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return 2**31
    try:
        return int(value)
    except ValueError:
        return 2**31


def _attach_hate_speech_finding(node: BaseNode, finding: WindowFinding) -> None:
    """Store an endorsed-hate finding on a transcript segment in the per-chunk shape.

    Args:
        node (BaseNode): The segment node.
        finding (WindowFinding): Its finding.
    """
    metadata = dict(getattr(node, "metadata", {}) or {})
    detection: HateSpeechDetection = {
        "hate_speech": True,
        "category": finding["category"],
        "confidence": finding["confidence"],
        "reason": finding["reason"],
        "chunk_id": _node_id(node),
        "chunk_text": getattr(node, "text", "") or "",
        "source_ref": str(
            metadata.get("file_path")
            or metadata.get("filename")
            or metadata.get("file_name")
            or metadata.get("source")
            or ""
        ),
    }
    set_enrichment(node, {"hate_speech": detection})


def _transcript_key(metadata: dict[str, Any]) -> tuple[str, ...]:
    """Identify the transcript a segment belongs to.

    The file hash alone is not enough: the same clip attached to two postings
    (or served twice from the transcript cache) yields two transcripts whose
    segments must not share a window.

    Args:
        metadata (dict[str, Any]): Segment metadata.

    Returns:
        tuple[str, ...]: ``(file_hash, file_path, posting_uuid, media_id)``.
    """
    return tuple(str(metadata.get(key) or "") for key in ("file_hash", "file_path", "posting_uuid", "media_id"))


def _finish_reason(response: Any) -> str | None:
    """Return why generation stopped, from a llama-index completion's raw provider response.

    Args:
        response (Any): The completion response.

    Returns:
        str | None: ``"stop"``, ``"length"``, ..., or ``None`` when not reported.
    """
    raw = getattr(response, "raw", None)
    choices = raw.get("choices") if isinstance(raw, dict) else getattr(raw, "choices", None)
    if not choices:
        return None
    first = choices[0]
    reason = first.get("finish_reason") if isinstance(first, dict) else getattr(first, "finish_reason", None)
    return reason if isinstance(reason, str) else None


def _response_text(response: Any) -> str:
    """Return the text of a llama-index completion response.

    Args:
        response (Any): The completion response.

    Returns:
        str: Its text.
    """
    return str(response.text if hasattr(response, "text") else response)


@dataclass(slots=True)
class DocumentIngestionPipeline:
    """Encapsulates document loading, cleaning, and node construction."""

    # --- Constructor args ---
    data_dir: Path
    ner_model: OpenAI | None
    progress_callback: Callable[[str], None] | None
    hate_speech_model: OpenAI | None = None
    # Per-request enrichment overrides: None keeps the deployment env default
    # (NER_ENABLED / ENABLE_HATE_SPEECH_DETECTION); an explicit bool wins for
    # this run only.
    ner_override: bool | None = None
    hate_speech_override: bool | None = None

    # --- Cleaning config ---
    clean_fn: CleanFn = basic_clean

    # --- Directory reader config ---
    reader_errors: str = "ignore"
    reader_recursive: bool = True
    reader_encoding: str = "utf-8"
    reader_required_exts: list[str] = field(default_factory=list, init=False)

    # --- Ingestion config ---
    ingestion_batch_size: int = field(default=5, init=False)
    streaming_readers_enabled: bool = field(default=False, init=False)

    # --- OpenAI config (for LLM-based NER) ---
    openai_inference_provider: str = field(default="ollama")

    # --- Table reader config ---
    table_text_cols: list[str] | None = None
    table_metadata_cols: list[str] | str | None = None
    table_id_col: str | None = None
    table_excel_sheet: str | int | None = None
    target_collection: str | None = None
    image_ingestion_service: ImageIngestionService | None = None

    # --- Named entity recognition (NER) ---
    ner_max_workers: int = field(default=4, init=False)
    entity_extractor: Callable[[str], tuple[list[dict[str, Any]], list[dict[str, Any]]]] | None = None
    hate_speech_enabled: bool = field(default=False, init=False)
    hate_speech_max_chars: int = field(default=1500, init=False)
    hate_speech_max_workers: int = field(default=1, init=False)
    hate_speech_prompt: str | None = field(default=None, init=False)
    # Cleared once a provider rejects ``response_format`` (or its constrained
    # replies prove unparseable) so the rest of the run stops sending it.
    hate_speech_structured: bool = field(default=True, init=False)
    # Nextext transcript segments are classified in context windows instead of
    # one isolated sentence per request; ``None`` keeps the per-chunk path.
    hate_speech_transcript_prompt: str | None = field(default=None, init=False)
    hate_speech_window_tokens: int = field(default=1000, init=False)
    hate_speech_context_tokens: int = field(default=300, init=False)
    hate_speech_windowed_ids: set[str] = field(default_factory=set, init=False)

    # None when the batch holds no reader-supported files at all (e.g. an
    # audio/video-only upload): the generic sweep is skipped but the Nextext
    # pre-passes still run, so a media-only batch is not an error.
    dir_reader: SimpleDirectoryReader | None = field(default=None, init=False)
    social_link_consumed: set[Path] = field(default_factory=set, init=False)
    social_link_documents: list[Document] = field(default_factory=list, init=False)
    md_node_parser: MarkdownNodeParser | None = field(default=None, init=False)
    docling_node_parser: DoclingNodeParser | None = field(default=None, init=False)
    sentence_splitter: SentenceSplitter = field(default_factory=SentenceSplitter, init=False)
    hierarchical_node_parser: HierarchicalNodeParser | None = field(default=None, init=False)
    docs: list[Document] = field(default_factory=list, init=False)
    nodes: list[BaseNode] = field(default_factory=list, init=False)
    file_hash_cache: dict[str, str] = field(default_factory=dict, init=False)
    # Files this pipeline declined because the collection already holds them.
    # Read by ``RAG.ingest_docs`` for the run summary's ``files_skipped``,
    # which had no source of truth at all before: the two gates counted
    # locally and logged, and nothing aggregated them. Kept as two fields
    # because the gates run at different stages — ``prefilter_skipped``
    # before any file is loaded, ``skipped_hashes`` after. They are disjoint:
    # the pre-filter removes files from ``dir_reader.input_files``, so the
    # post-load gate never sees them.
    prefilter_skipped: int = field(default=0, init=False)
    skipped_hashes: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        """Post-initialization to load configurations and set up components."""
        # --- Named Entity Recognition (NER) config ---
        ner_cfg = load_ner_env()
        ner_enabled = ner_cfg.enabled if self.ner_override is None else self.ner_override
        self.ner_max_workers = ner_cfg.max_workers

        if ner_enabled:
            logger.info("Initializing remote NER extractor")
            try:
                self.entity_extractor = build_remote_ner_extractor()
            except Exception:
                logger.warning("Remote NER client init failed - continuing without NER")
                self.entity_extractor = None

        hate_speech_cfg = load_hate_speech_env()
        self.hate_speech_enabled = (
            hate_speech_cfg.enabled if self.hate_speech_override is None else self.hate_speech_override
        )
        self.hate_speech_max_chars = hate_speech_cfg.max_chars
        self.hate_speech_max_workers = hate_speech_cfg.max_workers
        self.hate_speech_window_tokens = hate_speech_cfg.window_tokens
        self.hate_speech_context_tokens = hate_speech_cfg.context_tokens
        if self.hate_speech_enabled and self.hate_speech_model is not None:
            try:
                self.hate_speech_prompt = OpenAIPipeline().load_prompt(kw="hate_speech")
            except Exception as exc:
                logger.warning(
                    "Hate-speech prompt unavailable - disabling detector: {}",
                    exc,
                )
                self.hate_speech_enabled = False
            if self.hate_speech_enabled:
                try:
                    self.hate_speech_transcript_prompt = OpenAIPipeline().load_prompt(kw="hate_speech_transcript")
                except Exception as exc:
                    logger.warning(
                        "Transcript hate-speech prompt unavailable - classifying transcript segments one by one: {}",
                        exc,
                    )

        # --- Ingestion config ---
        ingestion_cfg = load_ingestion_env()
        self.ingestion_batch_size = ingestion_cfg.ingestion_batch_size
        self.streaming_readers_enabled = ingestion_cfg.streaming_readers_enabled
        sentence_splitter_chunk_size = ingestion_cfg.sentence_splitter_chunk_size
        sentence_splitter_chunk_overlap = ingestion_cfg.sentence_splitter_chunk_overlap
        self.sentence_splitter = SentenceSplitter(
            chunk_size=sentence_splitter_chunk_size,
            chunk_overlap=sentence_splitter_chunk_overlap,
            paragraph_separator="\n\n",
        )
        self.reader_required_exts = ingestion_cfg.supported_filetypes
        if ingestion_cfg.hierarchical_chunking_enabled:
            logger.info("Hierarchical chunking is ENABLED.")
            self.hierarchical_node_parser = HierarchicalNodeParser(
                coarse_chunk_size=ingestion_cfg.coarse_chunk_size,
                fine_chunk_size=ingestion_cfg.fine_chunk_size,
                fine_chunk_overlap=ingestion_cfg.fine_chunk_overlap,
            )
        else:
            self.hierarchical_node_parser = None

    def build(self, existing_hashes: set[str] | None = None) -> Iterable[tuple[list[Document], list[BaseNode]]]:
        """Execute the full ingestion pipeline and yield batches of cleaned docs + nodes.

        Args:
            existing_hashes (set[str] | None): Document hashes to filter out (already-processed
                docs).

        Yields:
            tuple[list[Document], list[BaseNode]]: Batches of cleaned documents and their nodes.

        Raises:
            NoSupportedFilesError: If the batch holds nothing ingestable.
        """
        self._load_doc_readers()
        self._load_node_parsers()

        # Pre-filter files based on existing hashes to avoid unnecessary processing
        if existing_hashes:
            self._filter_input_files(existing_hashes)

        # Process in batches
        current_docs: list[Document] = []
        files_processed = 0

        for file_docs in self._iter_loaded_documents():
            current_docs.extend(file_docs)
            files_processed += 1

            if files_processed >= self.ingestion_batch_size:
                yield self._process_batch(current_docs, existing_hashes)
                current_docs = []
                files_processed = 0

        if current_docs:
            yield self._process_batch(current_docs, existing_hashes)

    def build_streaming(
        self, existing_hashes: set[str] | None = None
    ) -> Iterable[tuple[list[Document], list[BaseNode], set[str]]]:
        """Execute ingestion and yield node batches as enrichment progresses.

        Args:
            existing_hashes (set[str] | None): Optional set of already ingested file hashes.

        Yields:
            tuple[list[Document], list[BaseNode], set[str]]:
                - docs: Processed documents for the current source batch. Only
                  present on the first yielded node batch for that source batch.
                - nodes: Incrementally enriched nodes ready for persistence.
                - completed_hashes: File hashes that are complete for this source
                  batch and safe to mark as processed.
        """
        self._load_doc_readers()
        self._load_node_parsers()

        if existing_hashes:
            self._filter_input_files(existing_hashes)
        self._prefetch_images()

        current_docs: list[Document] = []
        files_processed = 0

        for file_docs in self._iter_loaded_documents():
            current_docs.extend(file_docs)
            files_processed += 1

            if files_processed >= self.ingestion_batch_size:
                yield from self._stream_processed_batch(current_docs, existing_hashes)
                current_docs = []
                files_processed = 0

        if current_docs:
            yield from self._stream_processed_batch(current_docs, existing_hashes)

    @staticmethod
    def _extract_doc_file_hashes(docs: list[Document]) -> set[str]:
        """Collect unique file hashes from processed documents.

        Args:
            docs (list[Document]): Processed document batch.

        Returns:
            set[str]: Extracted file hashes.
        """
        hashes: set[str] = set()
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            value = metadata.get("file_hash")
            if isinstance(value, str) and value.strip():
                hashes.add(value.strip())
        return hashes

    def _stream_processed_batch(
        self,
        docs: list[Document],
        existing_hashes: set[str] | None,
    ) -> Iterable[tuple[list[Document], list[BaseNode], set[str]]]:
        """Yield incrementally enriched node batches for one source-doc batch."""
        docs = self._attach_clean_text(docs)
        docs = self._ensure_file_hashes(docs)
        docs = self._filter_docs_by_existing_hashes(docs, existing_hashes)
        file_hashes = self._extract_doc_file_hashes(docs)
        nodes = self._create_nodes_without_enrichment(docs)

        self.docs = docs
        self.nodes = nodes

        if not nodes:
            yield docs, [], file_hashes
            return

        # Before batching: a transcript's segments are spread over many small
        # enrichment batches, and a window needs its neighbours.
        self._detect_transcript_hate_speech(nodes)
        total_nodes = len(nodes)
        processed_nodes = 0
        node_batches = chunk_nodes(nodes, self.ingestion_batch_size)
        for batch_idx, node_batch in enumerate(node_batches):
            self._enrich_nodes_in_place(
                node_batch,
                progress_offset=processed_nodes,
                progress_total=total_nodes,
            )
            processed_nodes += len(node_batch)
            completed_hashes = file_hashes if batch_idx == len(node_batches) - 1 else set()
            yield docs if batch_idx == 0 else [], node_batch, completed_hashes

    def _prefetch_images(self, pool: Any = None) -> int:
        """Submit every image the sweep will read to the preprocessing pool.

        The sweep reads files one at a time; submitting the images first lets
        their caption/OCR/CLIP calls run across files while it does, so each
        ``ImageReader`` call becomes a join. Images the social linker claimed
        are its own to link and are skipped.

        Args:
            pool (Any): Pool to submit to; the shared one when ``None``.

        Returns:
            int: How many images were submitted.
        """
        if self.dir_reader is None or not self.target_collection:
            return 0
        pool = pool or get_preprocess_pool()
        service = self.image_ingestion_service
        submitted = 0
        for input_file in self.dir_reader.input_files:
            path = Path(input_file)
            if path.suffix.lower() not in IMAGE_EXTENSIONS or path in self.social_link_consumed:
                continue
            file_hash = self.file_hash_cache.get(str(path))
            if submit_file(path, self.target_collection, pool=pool, file_hash=file_hash, image_service=service):
                submitted += 1
        return submitted

    def _iter_loaded_documents(self) -> Iterable[list[Document]]:
        """Yield loaded documents from the configured directory reader.

        The ``required_exts`` whitelist configured on
        :class:`SimpleDirectoryReader` already filters out audio/video files
        upstream (docint no longer transcribes media locally — Nextext
        produces the ``.jsonl`` transcripts consumed here instead), so this
        method simply iterates the reader's accepted input files.

        Reports one counter per file read. This sweep is where an image batch
        spends its hours — every picture is a caption, an OCR read and a CLIP
        embedding — and it emitted nothing at all, so the card said "Working…"
        for the whole run. The message deliberately carries **no filename**:
        the log throttle keys on the digit-masked message, so a varying name
        would defeat it and log a line per file.

        Yields:
            list[Document]: The loaded documents for each processed file.
        """
        # dir_reader is None on a media-only batch: nothing for the generic
        # sweep, but the Nextext pre-passes may still have produced transcript
        # Documents, yielded through the shared tail below.
        if self.dir_reader is not None:
            dir_reader = self.dir_reader
            pending = [f for f in dir_reader.input_files if Path(f) not in self.social_link_consumed]
            total_files = len(pending)
            for read, input_file in enumerate(pending, start=1):
                ext = input_file.suffix.lower()
                reader = dir_reader.file_extractor.get(ext) if dir_reader.file_extractor else None
                if self.streaming_readers_enabled and reader is not None and hasattr(reader, "iter_documents"):
                    extra_info = dir_reader.file_metadata(str(input_file))
                    docs = list(reader.iter_documents(input_file, extra_info=extra_info))
                else:
                    docs = SimpleDirectoryReader.load_file(
                        input_file=input_file,
                        file_metadata=dir_reader.file_metadata,
                        file_extractor=dir_reader.file_extractor,
                        filename_as_id=dir_reader.filename_as_id,
                        encoding=dir_reader.encoding,
                        errors=dir_reader.errors,
                        raise_on_error=dir_reader.raise_on_error,
                        fs=dir_reader.fs,
                    )
                if self.progress_callback:
                    self.progress_callback(f"Reading files: {read}/{total_files} files read")
                if docs:
                    yield dir_reader._exclude_metadata(docs)
        if self.social_link_documents:
            yield self.social_link_documents

    def _process_batch(
        self, docs: list[Document], existing_hashes: set[str] | None
    ) -> tuple[list[Document], list[BaseNode]]:
        """Process a batch of documents through cleaning, hashing, filtering, and node creation.

        Args:
            docs (list[Document]): The list of documents to process.
            existing_hashes (set[str] | None): Document hashes to filter out (already-processed
                docs).

        Returns:
            tuple[list[Document], list[BaseNode]]: The processed documents and their nodes.
        """
        docs = self._attach_clean_text(docs)
        docs = self._ensure_file_hashes(docs)
        # We still keep this filter as a safety net, though pre-filtering should catch most
        docs = self._filter_docs_by_existing_hashes(docs, existing_hashes)
        nodes = self._create_nodes(docs)

        # Update internal state (optional, but useful for debugging last batch)
        self.docs = docs
        self.nodes = nodes

        return docs, nodes

    def enrich_nodes(
        self,
        nodes: list[BaseNode],
        *,
        ner: bool = True,
        hate_speech: bool = True,
        progress_offset: int = 0,
        progress_total: int | None = None,
    ) -> None:
        """Apply this run's NER and hate-speech enrichment to nodes built outside the generic lane.

        The core PDF lane and the image companion build their nodes outside
        this pipeline; routing them through the same pass keeps every lane on
        every stage. ``ner`` / ``hate_speech`` narrow the stages for this call,
        so one text can be read for entities while another is judged for hate
        speech; a stage the run has switched off stays off either way.

        Args:
            nodes (list[BaseNode]): Nodes to enrich in place.
            ner (bool): Run entity extraction on these nodes.
            hate_speech (bool): Run hate-speech detection on these nodes.
            progress_offset (int): Processed count offset for cumulative progress.
            progress_total (int | None): Total count for progress display.
        """
        self._enrich_nodes_in_place(
            nodes,
            progress_offset=progress_offset,
            progress_total=progress_total,
            ner=ner,
            hate_speech=hate_speech,
        )

    def _enrich_nodes_in_place(
        self,
        nodes: list[BaseNode],
        *,
        progress_offset: int = 0,
        progress_total: int | None = None,
        ner: bool = True,
        hate_speech: bool = True,
    ) -> None:
        """Apply NER and hate-speech enrichment to *nodes* in-place.

        Both stages run in one thread pool: each per-node task performs NER
        then hate-speech detection sequentially for its node, so the two
        remote backends (GLiNER, chat LLM) are kept busy concurrently across
        nodes while a node's metadata is only ever written from one thread.
        Per-stage semaphores keep in-flight calls within ``ner_max_workers``
        and ``hate_speech_max_workers``.

        Args:
            nodes (list[BaseNode]): Nodes to enrich.
            progress_offset (int): Processed node count offset for cumulative progress.
            progress_total (int | None): Total node count for progress display.
            ner (bool): Whether this call runs entity extraction (when the run has it enabled).
            hate_speech (bool): Whether this call runs hate-speech detection (when the run has it enabled).
        """
        if not nodes:
            return

        ner_enabled = ner and self.entity_extractor is not None
        hate_enabled = hate_speech and bool(
            self.hate_speech_enabled and self.hate_speech_prompt and self.hate_speech_model is not None
        )
        if not ner_enabled and not hate_enabled:
            return

        total_nodes = progress_total or (progress_offset + len(nodes))
        ner_sem = threading.Semaphore(max(1, self.ner_max_workers))
        hate_sem = threading.Semaphore(max(1, self.hate_speech_max_workers))
        progress_lock = threading.Lock()
        counters = {"ner": progress_offset, "hate": progress_offset}

        def _tick(stage: str, label: str) -> None:
            """Advance one stage's counter and report cumulative progress."""
            with progress_lock:
                counters[stage] += 1
                if self.progress_callback:
                    self.progress_callback(f"{label}: {counters[stage]}/{total_nodes} chunks processed")

        def _extract_entities(idx: int, node: BaseNode, text_value: str) -> None:
            """Run NER extraction on ``node`` and merge entities/relations into metadata."""
            try:
                if self.entity_extractor:
                    ents, rels = self.entity_extractor(text_value)
                    set_enrichment(node, {"entities": ents, "relations": rels})
            except Exception as exc:
                logger.warning("Entity extractor failed on chunk {}: {}", idx, exc)

        def _detect_hate_speech(idx: int, node: BaseNode, text_value: str) -> None:
            """Run hate-speech detection on ``node`` and annotate metadata when positive."""
            try:
                prompt = self.hate_speech_prompt.replace(  # type: ignore[union-attr]
                    "{text}", text_value[: self.hate_speech_max_chars]
                )
                raw, structured, _ = self._complete_hate_speech(prompt, chunk_response_format())
                parsed = _parse_hate_speech_reply(raw)
                if parsed is None and structured:
                    raw, _, _ = self._complete_hate_speech(prompt, None)
                    parsed = _parse_hate_speech_reply(raw)
                    if parsed is not None and self.hate_speech_structured:
                        self.hate_speech_structured = False
                        logger.warning("Constrained hate-speech replies were unparseable; continuing without it")
                if parsed is None:
                    logger.warning("Hate-speech reply for chunk {} was unparseable ({} chars)", idx, len(raw))
                    return
                if parsed["hate_speech"]:
                    meta = dict(getattr(node, "metadata", {}) or {})
                    chunk_id = str(getattr(node, "node_id", "") or getattr(node, "id_", "") or "")
                    source_ref = str(
                        meta.get("file_path")
                        or meta.get("filename")
                        or meta.get("file_name")
                        or meta.get("source")
                        or ""
                    )
                    parsed["chunk_id"] = chunk_id
                    parsed["chunk_text"] = text_value
                    parsed["source_ref"] = source_ref
                    set_enrichment(node, {"hate_speech": parsed})
            except Exception as exc:
                logger.warning("Hate-speech detection failed on chunk {}: {}", idx, exc)

        def _process_node(idx: int, node: BaseNode) -> None:
            """Enrich one node: NER then hate-speech, ticking each stage's progress."""
            text_value = getattr(node, "text", "") or ""
            has_text = bool(text_value.strip())
            if ner_enabled:
                if has_text:
                    with ner_sem:
                        _extract_entities(idx, node, text_value)
                _tick("ner", "Extracting entities")
            if hate_enabled:
                if has_text and self._wants_hate_speech(node):
                    with hate_sem:
                        _detect_hate_speech(idx, node, text_value)
                _tick("hate", "Detecting hate speech")

        pool_size = (self.ner_max_workers if ner_enabled else 0) + (self.hate_speech_max_workers if hate_enabled else 0)
        with ThreadPoolExecutor(max_workers=max(1, pool_size)) as executor:
            futures = [executor.submit(_process_node, progress_offset + i, node) for i, node in enumerate(nodes)]
            wait(futures)
        # Stage errors are swallowed per node above; anything stored on a
        # future is a coordination failure (e.g. the progress callback raised)
        # and must fail the batch like it did pre-pooling, not vanish.
        for future in futures:
            future.result()

    def _complete_hate_speech(
        self,
        prompt: str,
        response_format: dict[str, Any] | None,
        *,
        max_tokens: int | None = None,
    ) -> tuple[str, bool, str | None]:
        """Send one hate-speech request, constrained by ``response_format`` while the provider allows it.

        A provider rejecting the schema (HTTP 400/422 that is not a context
        overflow) gets the same prompt again unconstrained, and the rest of the
        run stops sending the schema.

        Args:
            prompt (str): The rendered prompt.
            response_format (dict[str, Any] | None): The JSON-schema constraint,
                or ``None`` for an unconstrained request.
            max_tokens (int | None): Output-token cap for this request, or
                ``None`` for the model's default.

        Returns:
            tuple[str, bool, str | None]: The reply text, whether it was
                constrained, and the provider's finish reason when reported.
        """
        model = cast(OpenAI, self.hate_speech_model)
        extra: dict[str, Any] = {"max_tokens": max_tokens} if max_tokens is not None else {}
        if response_format is not None and self.hate_speech_structured:
            try:
                response = model.complete(prompt, response_format=response_format, **extra)
                return _response_text(response), True, _finish_reason(response)
            except Exception as exc:
                if not is_structured_output_rejection(exc):
                    raise
                self.hate_speech_structured = False
                logger.warning(
                    "The inference endpoint rejected response_format (HTTP {}); continuing hate-speech detection "
                    "without it",
                    getattr(exc, "status_code", "?"),
                )
        response = model.complete(prompt, **extra)
        return _response_text(response), False, _finish_reason(response)

    def _wants_hate_speech(self, node: BaseNode) -> bool:
        """Report whether the per-chunk detector should classify ``node``.

        Coarse hierarchical parents are never stored as vectors, so their
        verdicts would never surface — classifying them only costs requests.

        Args:
            node (BaseNode): The node.

        Returns:
            bool: ``False`` for coarse parents.
        """
        metadata = getattr(node, "metadata", {}) or {}
        if metadata.get("docint_hier_type") == "coarse":
            return False
        return _node_id(node) not in self.hate_speech_windowed_ids

    def _detect_transcript_hate_speech(self, nodes: list[BaseNode]) -> None:
        """Classify Nextext transcript segments in context windows, annotating endorsed hate.

        Segments are grouped per source file and ordered by ``sentence_index``;
        each window labels a core of segments while showing its neighbours as
        read-only context, so a sentence such as "Das ist antisemitisch." is
        judged as the condemnation it is. Runs on the full node list of a source
        batch — before enrichment splits it into small node batches. Windowed
        segments are skipped by the per-chunk detector afterwards; a window whose
        request fails or whose reply is unparseable yields no finding.

        Args:
            nodes (list[BaseNode]): Nodes of one source batch.
        """
        if not (self.hate_speech_enabled and self.hate_speech_transcript_prompt and self.hate_speech_model is not None):
            return
        groups: dict[tuple[str, ...], list[BaseNode]] = {}
        for node in nodes:
            metadata = getattr(node, "metadata", {}) or {}
            if metadata.get("docint_doc_kind") != "transcript_segment" or not _node_id(node):
                continue
            if not (getattr(node, "text", "") or "").strip():
                continue
            groups.setdefault(_transcript_key(metadata), []).append(node)
        if not groups:
            return

        core_chars = max(1, int(self.hate_speech_window_tokens * CHARS_PER_TOKEN))
        context_chars = max(0, int(self.hate_speech_context_tokens * CHARS_PER_TOKEN))
        jobs: list[tuple[list[BaseNode], list[TranscriptLine], TranscriptWindow, str]] = []
        for group in groups.values():
            ordered = sorted(group, key=_sentence_index)
            lines = [
                TranscriptLine(
                    index=position,
                    text=(getattr(node, "text", "") or "").strip(),
                    speaker=str((getattr(node, "metadata", {}) or {}).get("speaker") or "").strip() or None,
                )
                for position, node in enumerate(ordered)
            ]
            language = str((getattr(ordered[0], "metadata", {}) or {}).get("whisper_language") or "") or "—"
            position = 0
            while position < len(lines):
                window = next_window(lines, position, core_chars, context_chars)
                jobs.append((ordered, lines, window, language))
                position = window.core_end

        # Every windowed segment is settled here, even when its window fails or
        # the pass stops early: re-judging it alone per chunk would bring back
        # the context-free verdicts this pass exists to avoid.
        for ordered, _, window, _ in jobs:
            for position in range(window.core_start, window.core_end):
                self.hate_speech_windowed_ids.add(_node_id(ordered[position]))

        total = sum(len(group) for group in groups.values())
        done = 0
        failures_in_a_row = 0
        workers = max(1, self.hate_speech_max_workers)
        queued = iter(jobs)
        in_flight: deque[tuple[tuple[list[BaseNode], list[TranscriptLine], TranscriptWindow, str], Future[Any]]]
        in_flight = deque()
        with ThreadPoolExecutor(max_workers=workers) as executor:

            def submit_next() -> None:
                """Start the next queued window, if any."""
                job = next(queued, None)
                if job is not None:
                    _, lines, window, language = job
                    in_flight.append(
                        (
                            job,
                            executor.submit(
                                self._classify_transcript_window, lines, window, language, core_chars, context_chars
                            ),
                        )
                    )

            for _ in range(workers):
                submit_next()
            try:
                while in_flight:
                    (ordered, _, window, _), future = in_flight.popleft()
                    findings, failed = future.result()
                    for finding in findings:
                        _attach_hate_speech_finding(ordered[finding["index"]], finding)
                    failures_in_a_row = failures_in_a_row + 1 if failed else 0
                    done += window.core_end - window.core_start
                    if failures_in_a_row >= _TRANSCRIPT_WINDOW_FAILURE_LIMIT:
                        logger.warning(
                            "Hate-speech detection gave up on transcript windows after {} consecutive failures; "
                            "{} segment(s) left unclassified",
                            failures_in_a_row,
                            total - done,
                        )
                        break
                    submit_next()
                    if self.progress_callback:
                        self.progress_callback(f"Detecting hate speech: {done}/{total} chunks processed")
            finally:
                for _, pending in in_flight:
                    pending.cancel()

    def _classify_transcript_window(
        self,
        lines: list[TranscriptLine],
        window: TranscriptWindow,
        language: str,
        core_chars: int,
        context_chars: int,
    ) -> tuple[list[WindowFinding], bool]:
        """Classify one transcript window, fail-soft.

        The reply's output cap grows with the core (see
        :data:`WINDOW_OUTPUT_TOKENS_PER_SEGMENT`). A reply stopped at that cap is
        not trusted: the window's segments are asked again one per request, each
        with its own context margins.

        Args:
            lines (list[TranscriptLine]): The transcript's segments, in order.
            window (TranscriptWindow): The window to classify.
            language (str): Transcript language label.
            core_chars (int): Core clip limit.
            context_chars (int): Context clip limit.

        Returns:
            tuple[list[WindowFinding], bool]: Endorsed-hate findings (``[]`` when
                the reply is unparseable), and whether the request itself failed.
        """
        indices = [line.index for line in lines[window.core_start : window.core_end]]
        prompt = render_window_prompt(
            cast(str, self.hate_speech_transcript_prompt),
            lines,
            window,
            core_chars=core_chars,
            context_chars=context_chars,
            language=language,
        )
        max_tokens = max(WINDOW_MIN_OUTPUT_TOKENS, WINDOW_OUTPUT_TOKENS_PER_SEGMENT * len(indices))
        try:
            raw, structured, finish_reason = self._complete_hate_speech(
                prompt, window_response_format(indices), max_tokens=max_tokens
            )
            items = parse_window_reply(raw, indices)
            if items is None and structured and finish_reason != "length":
                raw, _, finish_reason = self._complete_hate_speech(prompt, None, max_tokens=max_tokens)
                items = parse_window_reply(raw, indices)
                if items is not None and self.hate_speech_structured:
                    self.hate_speech_structured = False
                    logger.warning("Constrained hate-speech replies were unparseable; continuing without it")
        except Exception as exc:
            logger.warning(
                "Hate-speech detection failed on a transcript window of {} segment(s): {}", len(indices), exc
            )
            return [], True
        if finish_reason == "length" and len(indices) > 1:
            logger.warning(
                "A hate-speech reply for a transcript window of {} segment(s) stopped at the output cap; "
                "asking its segments one by one",
                len(indices),
            )
            findings: list[WindowFinding] = []
            failed = False
            for position in range(window.core_start, window.core_end):
                single = next_window(lines, position, 1, context_chars)
                sub_findings, sub_failed = self._classify_transcript_window(
                    lines, single, language, core_chars, context_chars
                )
                findings.extend(sub_findings)
                failed = failed or sub_failed
            return findings, failed
        if items is None:
            logger.warning(
                "Hate-speech reply for a transcript window of {} segment(s) was unparseable ({} chars)",
                len(indices),
                len(raw),
            )
            return [], False
        return items, False

    def _create_nodes_without_enrichment(self, docs: list[Document]) -> list[BaseNode]:
        """Create nodes from documents without applying enrichment stages.

        Args:
            docs (list[Document]): Documents to parse.

        Returns:
            list[BaseNode]: Parsed nodes before NER/hate-speech enrichment.

        Raises:
            RuntimeError: If node parsers are not initialized.
        """
        if self.md_node_parser is None or self.docling_node_parser is None:
            raise RuntimeError("Node parsers are not initialized.")

        document_docs: list[Document] = []
        img_docs: list[Document] = []
        json_docs: list[Document] = []
        table_docs: list[Document] = []
        text_docs: list[Document] = []
        transcript_docs: list[Document] = []
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            file_type = (meta.get("file_type") or "").lower()
            source_kind = meta.get("source", "") or ""
            file_path = str(meta.get("file_path") or meta.get("file_name") or "")
            ext = file_path.lower().rsplit(".", 1)[-1] if "." in file_path else ""

            # Dispatcher: check the per-segment key first so generic JSON/JSONL
            # documents continue to flow through the normal JSON path below.
            if meta.get("docint_doc_kind") == "transcript_segment":
                transcript_docs.append(d)
            elif source_kind == "image" or ext in {"gif", "jpeg", "jpg", "png"}:
                img_docs.append(d)
            elif source_kind == "table" or ext in {"csv", "tsv"}:
                table_docs.append(d)
            elif file_type.endswith(("json", "jsonl")) or ext in {"json", "jsonl"}:
                json_docs.append(d)
            elif file_type.endswith(("docx", "pdf")) or ext in {"docx", "pdf"}:
                document_docs.append(d)
            elif file_type.startswith("text/") or ext in {"txt", "md", "rst", "rtf"}:
                text_docs.append(d)
            else:
                logger.warning(
                    "Unrecognized document type for file '{}'; treating as plain text.",
                    file_path,
                )
                text_docs.append(d)

        nodes: list[BaseNode] = []

        if img_docs:
            logger.info(
                "Parsing {} image documents with SentenceSplitter",
                len(img_docs),
            )
            nodes.extend(self._process_docs_hierarchical(img_docs))

        if json_docs:
            logger.info(
                "Parsing {} JSON documents with SentenceSplitter",
                len(json_docs),
            )
            nodes.extend(self._process_docs_hierarchical(json_docs))

        if document_docs:

            def _is_docling_json(doc: Document) -> bool:
                """Return True when ``doc`` text is a JSON-encoded Docling payload."""
                try:
                    json.loads(getattr(doc, "text", "") or "")
                    return True
                except Exception:
                    return False

            pdf_docs_docling = [d for d in document_docs if _is_docling_json(d)]
            pdf_docs_md = [d for d in document_docs if not _is_docling_json(d)]

            if pdf_docs_docling:
                logger.info(
                    "Parsing {} Docling JSON PDFs with DoclingNodeParser",
                    len(pdf_docs_docling),
                )
                nodes.extend(self._process_docs_hierarchical(pdf_docs_docling, self.docling_node_parser))
            if pdf_docs_md:
                logger.info(
                    "Parsing {} Markdown PDFs with MarkdownNodeParser",
                    len(pdf_docs_md),
                )
                nodes.extend(self._process_docs_hierarchical(pdf_docs_md, self.md_node_parser))

        if table_docs:
            logger.info(
                "Parsing {} table documents with SentenceSplitter (one node per document)",
                len(table_docs),
            )
            table_splitter = SentenceSplitter(chunk_size=10_000_000, chunk_overlap=0)
            nodes.extend(table_splitter.get_nodes_from_documents(table_docs))

        if transcript_docs:
            logger.info(
                "Parsing {} transcript segment documents with SentenceSplitter (one node per segment)",
                len(transcript_docs),
            )
            transcript_splitter = SentenceSplitter(chunk_size=10_000_000, chunk_overlap=0)
            nodes.extend(transcript_splitter.get_nodes_from_documents(transcript_docs))

        if text_docs:
            markdown_docs = [
                d
                for d in text_docs
                if str(d.metadata.get("file_path", "")).endswith((".md", ".markdown", ".rst"))
                or (d.text.strip().startswith("#"))
            ]
            plain_docs = [d for d in text_docs if d not in markdown_docs]

            if markdown_docs:
                logger.info(
                    "Parsing {} markdown documents with MarkdownNodeParser",
                    len(markdown_docs),
                )
                nodes.extend(self._process_docs_hierarchical(markdown_docs, self.md_node_parser))
            if plain_docs:
                logger.info(
                    "Parsing {} plain text documents with SentenceSplitter",
                    len(plain_docs),
                )
                nodes.extend(self._process_docs_hierarchical(plain_docs))

        return nodes

    def _filter_input_files(self, existing_hashes: set[str]) -> None:
        """Filter self.dir_reader.input_files based on existing hashes. Populates self.file_hash_cache.

        Args:
            existing_hashes (set[str]): A set of existing document hashes to filter out already processed documents.
        """
        if not self.dir_reader or not self.dir_reader.input_files:
            return

        filtered_files: list[Path | PurePosixPath] = []
        skipped_count = 0

        for file_path in self.dir_reader.input_files:
            path_obj: Path = Path(file_path)
            if path_obj in self.social_link_consumed:
                # The linker's own files: the sweep never reads them, and a known
                # dossier is already counted skipped through its rows.
                continue
            path_str = str(path_obj)
            try:
                # Compute hash (or get from cache if we ever re-run)
                if path_str in self.file_hash_cache:
                    f_hash = self.file_hash_cache[path_str]
                else:
                    f_hash = compute_file_hash(path_obj)
                    self.file_hash_cache[path_str] = f_hash

                if f_hash in existing_hashes:
                    skipped_count += 1
                    continue

                filtered_files.append(path_obj)
            except Exception as e:
                logger.warning(f"Failed to compute hash for {file_path}, skipping pre-filter: {e}")
                filtered_files.append(path_obj)

        self.prefilter_skipped = skipped_count
        if skipped_count > 0:
            logger.info("Skipping {} files that already exist in the collection.", skipped_count)
            self.dir_reader.input_files = filtered_files

    def _ensure_file_hashes(self, docs: list[Document]) -> list[Document]:
        """Ensure every document has a file_hash in its metadata. Computes it from the file path if missing.

        Args:
            docs (list[Document]): The list of documents to process.

        Returns:
            list[Document]: The list of documents with ensured file_hash metadata.

        Raises:
            RuntimeError: If the file hash computation fails.
        """
        # Cache hashes by path to avoid re-reading the same file multiple times
        path_hash_map: dict[str, str] = {}

        for doc in docs:
            if doc.metadata.get("file_hash"):
                continue

            # Try to find the file path
            file_path = doc.metadata.get("file_path") or doc.metadata.get("path") or doc.metadata.get("filename")

            if not file_path:
                continue

            file_path_str = str(file_path)

            # Use cached hash if available
            if file_path_str in path_hash_map:
                doc.metadata["file_hash"] = path_hash_map[file_path_str]
                continue

            # Compute and cache
            try:
                # Only compute if file exists
                p = Path(file_path_str)
                if p.is_file():
                    f_hash = compute_file_hash(p)
                    doc.metadata["file_hash"] = f_hash
                    path_hash_map[file_path_str] = f_hash
            except Exception as e:
                logger.warning("Could not compute hash for {}: {}", file_path_str, e)
                raise RuntimeError(f"Failed to compute file hash for {file_path}") from e

        return docs

    def _attach_clean_text(self, docs: Iterable[Document]) -> list[Document]:
        """Attach cleaned text to each document.

        Args:
            docs (Iterable[Document]): The documents to process.

        Returns:
            list[Document]: The documents with cleaned text.
        """
        cleaned: list[Document] = []
        for doc in docs:
            if hasattr(doc, "text") and isinstance(doc.text, str):
                cleaned.append(Document(text=self.clean_fn(doc.text), metadata=doc.metadata))
            else:
                cleaned.append(doc)
        return cleaned

    def _has_reader_supported_files(self) -> bool:
        """Return whether the batch tree holds any reader-supported file.

        Mirrors :class:`SimpleDirectoryReader`'s defaults: matches against
        ``reader_required_exts``, honors ``reader_recursive``, and excludes
        hidden files and files under hidden directories.

        Returns:
            bool: True when at least one supported file exists.
        """
        exts = {ext.lower() for ext in self.reader_required_exts}
        candidates = self.data_dir.rglob("*") if self.reader_recursive else self.data_dir.glob("*")
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            relative = path.relative_to(self.data_dir)
            if any(part.startswith(".") for part in relative.parts):
                continue
            return True
        return False

    def _load_doc_readers(self) -> None:
        """Load document readers, then run the Nextext media pre-passes.

        The generic reader is only constructed when the batch tree actually
        holds reader-supported files (pre-scanned here — a media-only batch,
        whose audio/video ``required_exts`` deliberately excludes, is served
        by the Nextext pre-passes instead). ``dir_reader is None`` therefore
        means "no reader-supported files", not a construction failure.

        Raises:
            NoSupportedFilesError: When no file matches ``required_exts`` and
                the pre-passes claimed nothing either — the batch holds
                nothing ingestable and the run must not complete silently.
        """
        self.dir_reader = self._build_dir_reader() if self._has_reader_supported_files() else None
        self._run_social_linker()
        self._run_standalone_media()
        if self.dir_reader is None and not self.social_link_documents and not self.social_link_consumed:
            raise NoSupportedFilesError(f"No ingestable files in batch directory {self.data_dir}.")

    def _build_dir_reader(self) -> SimpleDirectoryReader:
        """Construct the generic directory reader over the batch tree.

        Only called when :meth:`_has_reader_supported_files` found at least
        one matching file, so the reader's own "No files found" refusal cannot
        trigger here.

        Returns:
            SimpleDirectoryReader: The reader restricted to
                ``reader_required_exts``, with docint's custom per-extension
                extractors registered.
        """
        image_reader = ImageReader(
            image_ingestion_service=(self.image_ingestion_service or ImageIngestionService()),
            source_collection=self.target_collection,
            pool=get_preprocess_pool(),
        )
        table_reader = TableReader(
            text_cols=self.table_text_cols,
            metadata_cols=self.table_metadata_cols if self.table_metadata_cols else None,
            id_col=self.table_id_col,
            excel_sheet=self.table_excel_sheet,
        )

        def _metadata(path: str | Path) -> dict[str, str]:
            """Get metadata for a file.

            Args:
                path (str | Path): The path to the file.

            Returns:
                dict[str, str]: Metadata including file path, name, and hash.
            """
            resolved = path if isinstance(path, Path) else Path(path)
            path_str = str(resolved)

            if path_str in self.file_hash_cache:
                file_hash = self.file_hash_cache[path_str]
            else:
                file_hash = compute_file_hash(resolved)
                self.file_hash_cache[path_str] = file_hash

            filename = resolved.name
            return {
                "file_path": path_str,
                "file_name": filename,
                "filename": filename,
                "file_hash": file_hash,
            }

        return SimpleDirectoryReader(
            input_dir=self.data_dir,
            errors=self.reader_errors,
            recursive=self.reader_recursive,
            encoding=self.reader_encoding,
            required_exts=self.reader_required_exts,
            file_metadata=_metadata,
            file_extractor={
                ".json": CustomJSONReader(),
                ".jsonl": CustomJSONReader(is_jsonl=True),
                ".ndjson": CustomJSONReader(is_jsonl=True),
                ".gif": image_reader,
                ".jpeg": image_reader,
                ".jpg": image_reader,
                ".png": image_reader,
                ".csv": table_reader,
                ".parquet": TableReader(
                    text_cols=self.table_text_cols or ["text"],
                    metadata_cols=set(self.table_metadata_cols) if self.table_metadata_cols else None,
                    id_col=self.table_id_col,
                ),
                ".tsv": TableReader(
                    csv_sep="\t",
                    text_cols=self.table_text_cols,
                    metadata_cols=set(self.table_metadata_cols) if self.table_metadata_cols else None,
                    id_col=self.table_id_col,
                ),
                ".xls": table_reader,
                ".xlsx": table_reader,
                ".docx": DocxReader(),
                ".rtf": RTFReader(),
            },
        )

    def _open_ingest_manifest(self) -> Any:
        """Open the ingest manifest for this collection, or a no-op stub.

        Returns:
            IngestManifest | NullIngestManifest: A manifest keyed by
            ``(collection, file_hash)`` when enabled and a sources root + target
            collection are configured; otherwise a no-op stub. Callers must
            ``close()`` the returned object.
        """
        from docint.core.storage.ingest_manifest import open_ingest_manifest

        return open_ingest_manifest(self.target_collection)

    def _run_social_linker(self) -> None:
        """Run the social linker; record consumed paths + its posting, comment and transcript Documents.

        No-op (and fail-soft) unless the batch holds ``me-dossier/1`` dossiers.
        """
        from docint.core.ingest.social_linker import SocialLinker
        from docint.utils.env_cfg import load_ingestion_env, load_nextext_env
        from docint.utils.nextext_client import NextextClient

        manifest = self._open_ingest_manifest()
        try:
            nextext_cfg = load_nextext_env()
            ingestion_cfg = load_ingestion_env()
            result = SocialLinker(
                image_service=self.image_ingestion_service or ImageIngestionService(),
                nextext_client=NextextClient(nextext_cfg),
                target_collection=self.target_collection,
                manifest=manifest,
                keyframe_dedup_cosine=nextext_cfg.keyframe_dedup_cosine,
                nextext_max_concurrency=nextext_cfg.nextext_max_concurrency,
                album_link_enabled=ingestion_cfg.social_album_link_enabled,
                album_tolerance_s=ingestion_cfg.social_album_tolerance_s,
                pool=get_preprocess_pool(),
                progress_callback=self.progress_callback,
            ).run(self.data_dir)
        except JobCancelled:
            # The run was abandoned; fail-soft is for a bad export, not for
            # an abort, which must not be reported as a skipped pre-pass.
            raise
        except Exception as exc:  # pragma: no cover - fail-soft guard
            logger.warning("Social linker skipped due to error: {}", exc)
            return
        finally:
            manifest.close()
        self.social_link_consumed = result.consumed_paths
        self.social_link_documents = result.documents

    def _run_standalone_media(self) -> None:
        """Transcribe loose audio/video files the social linker did not claim.

        Runs after :meth:`_run_social_linker`; merges its consumed paths +
        transcript Documents into the shared pre-pass accumulators
        (``social_link_consumed`` / ``social_link_documents``) that the generic
        sweep skips / yields. Fail-soft: any error logs a warning and is swallowed
        so ingestion of the rest of the batch proceeds.
        """
        from docint.core.ingest.media_transcribe import MediaTranscriber
        from docint.utils.env_cfg import load_ingestion_env, load_nextext_env
        from docint.utils.nextext_client import NextextClient

        manifest = self._open_ingest_manifest()
        try:
            nextext_cfg = load_nextext_env()
            pool = get_preprocess_pool()
            transcriber = MediaTranscriber(
                image_service=self.image_ingestion_service or ImageIngestionService(),
                nextext_client=NextextClient(nextext_cfg),
                target_collection=self.target_collection,
                manifest=manifest,
                keyframe_dedup_cosine=nextext_cfg.keyframe_dedup_cosine,
                nextext_max_concurrency=nextext_cfg.nextext_max_concurrency,
                pool=pool,
                progress_callback=self.progress_callback,
                preprocess_progress=pool.progress,
            )
            result = StandaloneMediaIngestor(
                transcriber,
                media_filetypes=set(load_ingestion_env().media_filetypes),
                nextext_enabled=nextext_cfg.enabled,
            ).run(self.data_dir, self.social_link_consumed)
        except JobCancelled:
            raise
        except Exception as exc:  # pragma: no cover - fail-soft guard
            logger.warning("Standalone media ingestion skipped due to error: {}", exc)
            return
        finally:
            manifest.close()
        self.social_link_consumed = self.social_link_consumed | result.consumed_paths
        self.social_link_documents = [*self.social_link_documents, *result.transcript_documents]

    def _load_node_parsers(self) -> None:
        """Load document parsers for various file types."""
        self.md_node_parser = MarkdownNodeParser()
        self.docling_node_parser = DoclingNodeParser()

    @staticmethod
    def _extract_file_hash(data: dict[str, Any] | None) -> str | None:
        """Extract the file hash from the given data.

        Args:
            data (dict | None): The input data.

        Returns:
            str | None: The extracted file hash, or None if not found.
        """
        if not isinstance(data, dict):
            return None
        candidate = data.get("file_hash")
        if isinstance(candidate, str) and candidate:
            return candidate
        origin = data.get("origin")
        if isinstance(origin, dict):
            candidate = origin.get("file_hash")
            if isinstance(candidate, str) and candidate:
                return candidate
        for key in ("metadata", "meta", "extra_info"):
            nested = data.get(key)
            if isinstance(nested, dict):
                nested_hash = DocumentIngestionPipeline._extract_file_hash(nested)
                if nested_hash:
                    return nested_hash
        for value in data.values():
            if isinstance(value, dict):
                nested_hash = DocumentIngestionPipeline._extract_file_hash(value)
                if nested_hash:
                    return nested_hash
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        nested_hash = DocumentIngestionPipeline._extract_file_hash(item)
                        if nested_hash:
                            return nested_hash
        return None

    def _filter_docs_by_existing_hashes(
        self, docs: Iterable[Document], existing_hashes: set[str] | None
    ) -> list[Document]:
        """Filter documents by their existing hashes.

        Args:
            docs (Iterable[Document]): The documents to filter.
            existing_hashes (set[str] | None): The set of existing document hashes.

        Returns:
            list[Document]: The filtered list of documents.
        """
        if not existing_hashes:
            return list(docs)

        filtered: list[Document] = []
        skipped: dict[str, str] = {}
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            file_hash = metadata.get("file_hash") or self._extract_file_hash(metadata)
            if not file_hash or file_hash not in existing_hashes:
                filtered.append(doc)
                continue

            filename = (
                metadata.get("file_name")
                or metadata.get("filename")
                or metadata.get("file_path")
                or metadata.get("path")
                or metadata.get("source")
                or ""
            )
            if not filename:
                origin = metadata.get("origin")
                if isinstance(origin, dict):
                    filename = origin.get("filename") or origin.get("file_path") or origin.get("path") or ""
            skipped[file_hash] = filename

        self.skipped_hashes.update(skipped)
        if skipped:
            display = [name or h[:12] for h, name in skipped.items()]
            logger.info(
                "Skipping {} file(s) already ingested: {}",
                len(skipped),
                ", ".join(sorted(display)),
            )
        return filtered

    def _process_docs_hierarchical(self, docs: list[Document], base_parser: NodeParser | None = None) -> list[BaseNode]:
        """Process documents possibly using hierarchical chunking.

        Args:
            docs (list[Document]): Documents to process.
            base_parser (NodeParser | None): Base parser for coarse chunking. If None, uses the
                internal HierarchicalNodeParser logic (defaulting to SentenceSplitter).

        Returns:
            list[BaseNode]: The processed nodes.
        """
        if not docs:
            return []

        if self.hierarchical_node_parser:
            # If a specific base parser is provided (and it's not the default sentence splitter),
            # use it to generate Level 1 chunks, then refine.
            if base_parser and base_parser != self.sentence_splitter:
                # Use base parser for Coarse L1
                # Note: get_nodes_from_documents returns list[BaseNode]
                coarse = base_parser.get_nodes_from_documents(docs)
                # Refine to Level 2
                return self.hierarchical_node_parser._parse_nodes(coarse)
            else:
                # Use hierarchical parser from scratch (L0 -> L1 -> L2)
                # This uses SentenceSplitter internally for L1
                return self.hierarchical_node_parser.get_nodes_from_documents(docs)

        parser = base_parser or self.sentence_splitter
        return parser.get_nodes_from_documents(docs)

    def _create_nodes(self, docs: list[Document]) -> list[BaseNode]:
        """Create nodes from the provided documents.

        Args:
            docs (list[Document]): The documents to process.

        Returns:
            list[BaseNode]: The created nodes.

        Raises:
            RuntimeError: If node parsers are not initialized.
        """
        nodes = self._create_nodes_without_enrichment(docs)
        self._detect_transcript_hate_speech(nodes)
        self._enrich_nodes_in_place(nodes)
        return nodes
