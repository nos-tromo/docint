"""Custom JSON / JSONL reader supporting Nextext transcript schema detection."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.json import JSONReader
from loguru import logger

from docint.utils.hashing import compute_file_hash, ensure_file_hash
from docint.utils.mimetype import get_mimetype

# Permissive Nextext detection: either timing pair is sufficient.
NEXTEXT_TS_KEYS: frozenset[str] = frozenset({"start_ts", "end_ts"})
NEXTEXT_SECONDS_KEYS: frozenset[str] = frozenset({"start_seconds", "end_seconds"})
JSONL_SUFFIXES: frozenset[str] = frozenset({".jsonl", ".ndjson"})

# Safety cap on the number of Nextext segments materialized from a single
# file. A pathological or adversarial JSONL could otherwise exhaust memory by
# emitting one Document per line for arbitrarily many lines. 50 000 is well
# above any realistic transcription (roughly 40+ hours at sub-second cadence),
# so legitimate inputs are never truncated; anything beyond that is treated as
# a safety stop with a warning log rather than a hard failure.
NEXTEXT_MAX_SEGMENTS: int = 50_000

# Number of non-blank leading lines probed when sniffing a JSONL for the
# Nextext schema. A single malformed leading line should not mask a
# well-formed transcript further down the file, so the detector keeps looking
# until it finds a valid Nextext-shaped payload or exhausts the probe budget.
NEXTEXT_DETECTION_PROBE_LINES: int = 8


def _is_nextext_segment(payload: Any) -> bool:
    """Return ``True`` when ``payload`` looks like a Nextext transcript segment.

    A segment qualifies when it is a mapping, carries a ``"text"`` key, and
    carries either the ``start_ts`` / ``end_ts`` pair or the
    ``start_seconds`` / ``end_seconds`` pair. Either timing representation is
    accepted — Nextext may emit only strings, only seconds, or both.

    Args:
        payload: Parsed JSON value.

    Returns:
        ``True`` when the mapping has the required keys for a transcript
        segment; ``False`` otherwise.
    """
    if not isinstance(payload, Mapping) or "text" not in payload:
        return False
    keys = set(payload.keys())
    return NEXTEXT_TS_KEYS.issubset(keys) or NEXTEXT_SECONDS_KEYS.issubset(keys)


def _format_timestamp(seconds: float | int) -> str:
    """Format seconds into zero-padded ``HH:MM:SS``.

    Uses :func:`round` (banker's rounding) on the input before splitting into
    hours, minutes, and seconds. This avoids up-to-one-second drift that
    ``int()`` truncation introduces when Nextext emits fractional seconds
    (for example ``3.9`` becomes ``"00:00:04"`` rather than ``"00:00:03"``).

    Non-finite (``NaN`` / ``inf``) or negative inputs are clamped to
    ``"00:00:00"`` and a warning is logged — the formatter never raises so
    that a single malformed segment cannot abort an otherwise healthy
    transcript ingest.

    Args:
        seconds: Number of seconds to format. Fractional values are rounded
            to the nearest whole second.

    Returns:
        Zero-padded ``HH:MM:SS`` string.
    """
    seconds_float = float(seconds)
    if not math.isfinite(seconds_float) or seconds_float < 0:
        logger.warning(
            "Refusing to format non-finite or negative timestamp: {!r}; "
            "returning '00:00:00'",
            seconds,
        )
        return "00:00:00"
    total_seconds = round(seconds_float)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _parse_timestamp(value: str) -> float | None:
    """Parse a ``HH:MM:SS`` (or ``MM:SS``) string into fractional seconds.

    Validates that each component is a non-negative integer and that the
    ``MM`` / ``SS`` components fall in ``[0, 60)``. The ``HH`` component has
    no upper bound because stitched or concatenated audio can legitimately
    exceed 24 hours, so clamping hours would corrupt legal inputs.

    Args:
        value: Timestamp string.

    Returns:
        The total number of seconds as a ``float``, or ``None`` when
        ``value`` is not a string, is malformed, has the wrong number of
        components, or carries an out-of-range / negative field.
    """
    if not isinstance(value, str):
        return None
    parts = value.strip().split(":")
    try:
        parts_i = [int(p) for p in parts]
    except ValueError:
        return None
    if any(part < 0 for part in parts_i):
        return None
    if len(parts_i) == 3:
        h, m, s = parts_i
        if m >= 60 or s >= 60:
            return None
        return h * 3600.0 + m * 60.0 + s
    if len(parts_i) == 2:
        m, s = parts_i
        if s >= 60:
            return None
        return m * 60.0 + s
    if len(parts_i) == 1:
        return float(parts_i[0])
    return None


class CustomJSONReader(BaseReader):
    """Custom JSON reader with optional Nextext transcript schema handling.

    Nextext transcripts are detected permissively: a file is treated as a
    transcript when a parsed object carries a ``"text"`` key together with
    either the ``start_ts`` / ``end_ts`` pair or the
    ``start_seconds`` / ``end_seconds`` pair (both are accepted). Each line of
    the JSONL stream (or each element of a ``.json`` array) is emitted as one
    :class:`llama_index.core.Document` whose text is the segment's prose and
    whose metadata carries both a flat backward-compatible block (matching
    what the retired audio reader produced) and a ``reference_metadata`` dict
    that surfaces timing / speaker / language / source fields in the UI.

    The ingestion pipeline routes segments through the one-node-per-document
    :class:`SentenceSplitter` override — the same pattern used for social
    tables in :mod:`docint.core.readers.tables` — so each transcript segment
    remains a distinct retrievable node rather than being re-chunked.

    For any other ``.json`` payload, the reader falls back to the generic
    ``JSONReader`` path so existing JSON ingestion is not affected.
    """

    def __init__(
        self,
        levels_back: int | None = 0,
        collapse_length: int | None = None,
        ensure_ascii: bool = False,
        is_jsonl: bool = False,
        clean_json: bool = True,
        schema_sample_size: int = 200,
        list_sample_size: int = 50,
    ) -> None:
        """Initialize the CustomJSONReader.

        Args:
            levels_back: Number of levels to traverse back in nested JSON.
                Forwarded to the generic ``JSONReader``.
            collapse_length: Maximum length of collapsed text for generic JSON.
            ensure_ascii: Whether to force ASCII output in the generic path.
            is_jsonl: Whether the input should always be treated as JSON Lines.
            clean_json: Whether to apply the generic reader's cleaning pass.
            schema_sample_size: Maximum number of JSON objects sampled when
                inferring schema keys.
            list_sample_size: Maximum number of list elements sampled when
                inferring schema keys.
        """
        self.json_reader = JSONReader(
            levels_back=levels_back,
            collapse_length=collapse_length,
            ensure_ascii=ensure_ascii,
            is_jsonl=is_jsonl,
            clean_json=clean_json,
        )
        self.is_jsonl = is_jsonl
        self.schema_sample_size = max(schema_sample_size, 0)
        self.list_sample_size = max(list_sample_size, 0)

    # ------------------------------------------------------------------
    # Helpers for generic JSON schema inference (unchanged semantics)
    # ------------------------------------------------------------------

    def _sample_list_items(self, values: Sequence[Any]) -> Iterable[Any]:
        """Sample a subset of items from a list.

        Args:
            values: Source list.

        Returns:
            An iterable yielding the sampled items in input order.
        """
        if self.list_sample_size == 0:
            return []
        total = len(values)
        if total <= self.list_sample_size:
            return values
        if self.list_sample_size == 1:
            return [values[0]]
        step = total / float(self.list_sample_size)
        sampled: list[Any] = []
        position = 0.0
        for _ in range(self.list_sample_size):
            sampled.append(values[int(position)])
            position += step
            if position >= total:
                break
        return sampled

    def _collect_nested_keys(self, data: Any, prefix: str = "") -> set[str]:
        """Collect nested keys from a JSON-like structure.

        Args:
            data: JSON-like structure to inspect.
            prefix: Prefix for nested keys.

        Returns:
            A set of all dot-joined key paths found in ``data``.
        """
        keys: set[str] = set()
        if isinstance(data, dict):
            for key, value in data.items():
                path = f"{prefix}.{key}" if prefix else key
                keys.add(path)
                keys.update(self._collect_nested_keys(value, path))
        elif isinstance(data, list):
            for item in self._sample_list_items(data):
                keys.update(self._collect_nested_keys(item, prefix))
        return keys

    def _infer_schema(self, file_path: Path, is_jsonl: bool) -> dict[str, list[str]]:
        """Infer the schema of the JSON data.

        Args:
            file_path: Path to the JSON / JSONL file.
            is_jsonl: Whether the file is JSON Lines.

        Returns:
            A dict whose ``nested_keys`` entry lists every key path observed.
        """
        nested_keys: set[str] = set()
        try:
            if is_jsonl:
                rng = random.Random(file_path.stat().st_size)
                reservoir: list[Any] = []
                with file_path.open("r", encoding="utf-8") as handle:
                    for idx, line in enumerate(handle):
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if self.schema_sample_size <= 0:
                            break
                        if len(reservoir) < self.schema_sample_size:
                            reservoir.append(payload)
                        else:
                            j = rng.randint(0, idx)
                            if j < self.schema_sample_size:
                                reservoir[j] = payload
                for sample in reservoir:
                    nested_keys.update(self._collect_nested_keys(sample))
            else:
                with file_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                    nested_keys.update(self._collect_nested_keys(payload))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Unable to infer JSON schema for {}: {}", file_path, exc)
        return {"nested_keys": sorted(nested_keys)}

    # ------------------------------------------------------------------
    # Nextext transcript detection and emission
    # ------------------------------------------------------------------

    @staticmethod
    def _is_nextext_payload(payload: Any) -> bool:
        """Return ``True`` when ``payload`` looks like a Nextext segment.

        Thin wrapper around :func:`_is_nextext_segment` retained for backward
        compatibility with callers that imported the method from this class.

        Args:
            payload: Parsed JSON value.

        Returns:
            ``True`` if ``payload`` is a Nextext-shaped segment.
        """
        return _is_nextext_segment(payload)

    @classmethod
    def _detect_nextext_transcript(cls, file_path: Path) -> bool:
        """Check whether ``file_path`` holds a Nextext transcript stream.

        Uses :func:`_is_nextext_segment` so either the ``start_ts`` / ``end_ts``
        pair or the ``start_seconds`` / ``end_seconds`` pair (combined with a
        ``"text"`` key) is sufficient.

        For JSONL / NDJSON inputs, the detector probes up to
        :data:`NEXTEXT_DETECTION_PROBE_LINES` non-blank leading lines. A
        malformed or non-Nextext leading line no longer masks a well-formed
        transcript further down the file — the detector skips malformed lines
        and any valid payload that simply fails the Nextext shape check,
        stopping early on the first Nextext-shaped line. If no line within
        the probe budget matches, ``False`` is returned and the file falls
        through to the generic JSON path.

        Args:
            file_path: JSON or JSONL candidate file.

        Returns:
            ``True`` when a Nextext-shaped payload is found within the probe
            budget (for JSONL / NDJSON) or when a ``.json`` payload is a
            Nextext-shaped dict / list of dicts. ``False`` otherwise.
        """
        suffix = file_path.suffix.lower()
        try:
            if suffix in JSONL_SUFFIXES:
                with file_path.open("r", encoding="utf-8") as handle:
                    probed = 0
                    for line in handle:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if probed >= NEXTEXT_DETECTION_PROBE_LINES:
                            break
                        probed += 1
                        try:
                            payload = json.loads(stripped)
                        except (json.JSONDecodeError, ValueError):
                            # Malformed leading line: skip and keep probing.
                            continue
                        if _is_nextext_segment(payload):
                            return True
                return False

            if suffix == ".json":
                with file_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, list):
                    for item in payload:
                        if isinstance(item, Mapping):
                            return _is_nextext_segment(item)
                    return False
                return _is_nextext_segment(payload)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Unable to sniff Nextext transcript schema for {}: {}",
                file_path,
                exc,
            )
        return False

    @staticmethod
    def _iter_segments(file_path: Path) -> Iterable[dict[str, Any]]:
        """Yield parsed segments from a Nextext transcript file.

        Args:
            file_path: JSONL / NDJSON file (or JSON array / object).

        Yields:
            Parsed segment dictionaries in input order. Malformed lines or
            non-dict entries are skipped with a warning.
        """
        suffix = file_path.suffix.lower()
        if suffix in JSONL_SUFFIXES:
            with file_path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "Skipping malformed Nextext line {} in {}: {}",
                            line_no,
                            file_path,
                            exc,
                        )
                        continue
                    if isinstance(payload, dict):
                        yield payload
            return

        # ``.json`` – either a single object or a list of objects.
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            for entry in payload:
                if isinstance(entry, dict):
                    yield entry
        elif isinstance(payload, dict):
            yield payload

    @staticmethod
    def _build_segment_metadata(
        segment: dict[str, Any],
        *,
        base: dict[str, Any],
        sentence_index: int,
        segment_text: str,
    ) -> dict[str, Any]:
        """Compose segment-level metadata + ``reference_metadata`` block.

        Produces the flat backward-compatible keys (``whisper_task``,
        ``whisper_language``, ``sentence_index``, ``start_seconds``,
        ``end_seconds``, ``start_ts``, ``end_ts``, ``speaker``,
        ``source_file``, ``source_file_hash``) and a ``reference_metadata``
        dict that surfaces timing / speaker / language / source fields through
        the existing citation UI. Mirrors the specialized-schema pattern used
        by :mod:`docint.core.readers.tables`.

        Args:
            segment: Parsed Nextext line.
            base: Shared file-level metadata (filename, hash, ...).
            sentence_index: Zero-based segment index in input order.
            segment_text: The cleaned segment prose (used verbatim in
                ``reference_metadata["text"]``).

        Returns:
            A metadata dict populated with timing, speaker, source fields, and
            a nested ``reference_metadata`` sub-dict.
        """
        metadata = dict(base)
        metadata["sentence_index"] = sentence_index
        # Dispatcher key consumed by the ingestion pipeline to route this
        # Document to the one-node-per-segment SentenceSplitter.
        metadata["docint_doc_kind"] = "transcript_segment"

        task = segment.get("task")
        if isinstance(task, str) and task:
            metadata["whisper_task"] = task

        language = segment.get("language")
        language_str: str | None = None
        if isinstance(language, str) and language:
            metadata["whisper_language"] = language
            language_str = language

        start = segment.get("start_seconds")
        end = segment.get("end_seconds")
        start_seconds_f: float | None = None
        end_seconds_f: float | None = None
        if isinstance(start, (int, float)):
            start_seconds_f = float(start)
            metadata["start_seconds"] = start_seconds_f
        if isinstance(end, (int, float)):
            end_seconds_f = float(end)
            metadata["end_seconds"] = end_seconds_f

        raw_start_ts = segment.get("start_ts")
        raw_end_ts = segment.get("end_ts")
        start_ts_str: str | None = None
        end_ts_str: str | None = None

        if isinstance(raw_start_ts, str) and raw_start_ts:
            start_ts_str = raw_start_ts
        elif start_seconds_f is not None:
            start_ts_str = _format_timestamp(start_seconds_f)
        if start_ts_str is not None:
            metadata["start_ts"] = start_ts_str

        if isinstance(raw_end_ts, str) and raw_end_ts:
            end_ts_str = raw_end_ts
        elif end_seconds_f is not None:
            end_ts_str = _format_timestamp(end_seconds_f)
        if end_ts_str is not None:
            metadata["end_ts"] = end_ts_str

        # Back-fill seconds from timestamp strings when only the string pair
        # is given, so downstream filters that key off *_seconds still work.
        if start_seconds_f is None and isinstance(raw_start_ts, str):
            parsed = _parse_timestamp(raw_start_ts)
            if parsed is not None:
                metadata["start_seconds"] = parsed
        if end_seconds_f is None and isinstance(raw_end_ts, str):
            parsed = _parse_timestamp(raw_end_ts)
            if parsed is not None:
                metadata["end_seconds"] = parsed

        speaker = segment.get("speaker")
        speaker_str: str | None = None
        if isinstance(speaker, str) and speaker:
            metadata["speaker"] = speaker
            speaker_str = speaker

        source_file = segment.get("source_file")
        source_file_str: str | None = None
        if isinstance(source_file, str) and source_file:
            metadata["source_file"] = source_file
            source_file_str = source_file

        source_file_hash = segment.get("source_file_hash")
        if isinstance(source_file_hash, str) and source_file_hash:
            metadata["source_file_hash"] = source_file_hash

        # Build the reference_metadata block consumed by the citation UI.
        # When no ``speaker`` field is present, ``author`` falls back to an
        # empty string rather than the source filename — the citation UI
        # (``docint/ui/components.py``) skips empty / falsy values, so an
        # empty string simply omits the Author row. Using the source filename
        # here used to misleadingly render ``interview.mp3`` where a human
        # speaker name belongs.
        reference_metadata: dict[str, Any] = {
            "type": "transcript_segment",
            "network": "nextext",
            "author": speaker_str if speaker_str is not None else "",
            "text": segment_text,
            "text_id": f"{source_file_str or ''}:{sentence_index}",
        }
        if start_ts_str is not None:
            reference_metadata["timestamp"] = start_ts_str
            reference_metadata["start_ts"] = start_ts_str
        if end_ts_str is not None:
            reference_metadata["end_ts"] = end_ts_str
        if language_str is not None:
            reference_metadata["language"] = language_str
        if source_file_str is not None:
            reference_metadata["source_file"] = source_file_str
        if speaker_str is not None:
            reference_metadata["speaker"] = speaker_str

        metadata["reference_metadata"] = reference_metadata
        return metadata

    def _iter_nextext_transcript(
        self,
        file_path: Path,
        *,
        file_hash: str,
        base_extra_info: dict[str, Any],
    ) -> Iterator[Document]:
        """Yield one ``Document`` per Nextext transcript segment.

        Streaming variant of :meth:`_load_nextext_transcript` introduced
        for Phase 2 of the ingestion-streaming generalisation. The
        underlying :meth:`_iter_segments` is already a generator over
        the JSONL file, so this routes its output straight to the
        ingestion pipeline without materialising the full list — large
        transcripts no longer hold every segment in memory before any
        node persistence happens.

        Iteration is capped at :data:`NEXTEXT_MAX_SEGMENTS` documents
        to bound pathological inputs. When the cap is hit, a warning
        is logged and iteration ends; the reader never raises so
        partial ingest still succeeds.

        Args:
            file_path: Path to the transcript file.
            file_hash: Precomputed hash of the transcript file itself.
            base_extra_info: Ingestion-provided metadata (filename,
                mimetype, ``file_hash`` of the transcript, etc.).

        Yields:
            Document: One per non-empty transcript segment, in input
                order.
        """
        seen_segments = 0
        truncated = False
        for segment in self._iter_segments(file_path):
            text = segment.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            if seen_segments >= NEXTEXT_MAX_SEGMENTS:
                truncated = True
                break
            cleaned_text = text.strip()
            metadata = self._build_segment_metadata(
                segment,
                base=base_extra_info,
                sentence_index=seen_segments,
                segment_text=cleaned_text,
            )
            ensure_file_hash(metadata, file_hash=file_hash, path=file_path)
            yield Document(text=cleaned_text, metadata=metadata)
            seen_segments += 1

        if truncated:
            logger.warning(
                "[CustomJSONReader] Nextext segment cap reached: truncated "
                "{} at {} segments (file={})",
                file_path,
                NEXTEXT_MAX_SEGMENTS,
                file_path,
            )

        logger.info(
            "[CustomJSONReader] Loaded {} Nextext segment(s) from {}",
            seen_segments,
            file_path,
        )

    def _load_nextext_transcript(
        self,
        file_path: Path,
        *,
        file_hash: str,
        base_extra_info: dict[str, Any],
    ) -> list[Document]:
        """Materialise the streaming Nextext iterator into a list.

        Kept for backward compatibility with code paths that need an
        eager list (notably the existing :meth:`load_data` shim). New
        callers should prefer :meth:`_iter_nextext_transcript`.

        Args:
            file_path: Path to the transcript file.
            file_hash: Precomputed hash of the transcript file itself.
            base_extra_info: Ingestion-provided metadata.

        Returns:
            Ordered list of segment documents. Empty segments are
            skipped.
        """
        return list(
            self._iter_nextext_transcript(
                file_path,
                file_hash=file_hash,
                base_extra_info=base_extra_info,
            )
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def iter_documents(self, file: str | Path, **kwargs: Any) -> Iterator[Document]:
        """Yield ``Document`` objects for a JSON / JSONL / Nextext file.

        Streaming variant of :meth:`load_data` introduced in Phase 2 of
        the ingestion-streaming generalisation. The Nextext transcript
        path streams one document per segment without materialising
        the full list; the generic JSON path falls back to
        :class:`JSONReader.load_data` (which has no streaming API) and
        yields its output sequentially.

        Args:
            file: Path to the JSON or JSONL file.
            **kwargs: Accepts ``extra_info`` (``dict``) following the
                LlamaIndex convention. When provided, an existing
                ``file_hash`` is reused instead of being recomputed.

        Yields:
            Document: One per Nextext segment for transcript files,
                otherwise the shape follows the underlying
                ``JSONReader`` behaviour.

        Raises:
            FileNotFoundError: If ``file`` does not exist.
            ValueError: If the suffix is not ``.json`` / ``.jsonl``
                / ``.ndjson``.
        """
        file_path = Path(file) if not isinstance(file, Path) else file

        if not file_path.exists():
            logger.error("FileNotFoundError: File not found: {}", file_path)
            raise FileNotFoundError(f"File not found: {file_path}")
        suffix = file_path.suffix.lower()
        if suffix not in {".json", ".jsonl", ".ndjson"}:
            logger.error(
                "ValueError: Expected a .json / .jsonl / .ndjson file but got: {}",
                file_path.suffix,
            )
            raise ValueError(
                f"Expected a .json, .jsonl, or .ndjson file but got: {file_path.suffix}"
            )

        provided_info = kwargs.get("extra_info", {})
        file_hash = (
            provided_info.get("file_hash") if isinstance(provided_info, dict) else None
        )
        if file_hash is None:
            file_hash = compute_file_hash(file_path)

        file_path_str = str(file_path)
        filename = file_path.name
        mimetype = get_mimetype(file_path)

        logger.info("[CustomJSONReader] Loading JSON file: {}", file_path_str)

        is_nextext = self._detect_nextext_transcript(file_path)
        is_jsonl = self.is_jsonl or suffix in JSONL_SUFFIXES

        base_extra_info: dict[str, Any] = {
            "file_path": file_path_str,
            "file_name": filename,
            "filename": filename,
            "file_type": mimetype,
            "mimetype": mimetype,
            "source": "transcript" if is_nextext else "json",
            "origin": {
                "filename": filename,
                "mimetype": mimetype,
            },
        }
        if isinstance(provided_info, dict):
            base_extra_info.update(provided_info)
        ensure_file_hash(base_extra_info, file_hash=file_hash, path=file_path)

        if is_nextext:
            yield from self._iter_nextext_transcript(
                file_path,
                file_hash=file_hash,
                base_extra_info=base_extra_info,
            )
            return

        schema_info = self._infer_schema(file_path, is_jsonl)
        base_extra_info["schema"] = schema_info

        # JSONReader has no streaming API; consume its eager list and
        # yield sequentially so callers see the same lazy interface.
        for doc in self.json_reader.load_data(
            input_file=file_path_str,
            extra_info=base_extra_info,
        ):
            yield doc

    def load_data(self, file: str | Path, **kwargs: Any) -> list[Document]:
        """Eager-list shim over :meth:`iter_documents` for legacy callers.

        Args:
            file: Path to the JSON or JSONL file.
            **kwargs: Forwarded to :meth:`iter_documents`.

        Returns:
            A list of ``Document`` objects in input order.

        Raises:
            FileNotFoundError: Propagated from :meth:`iter_documents`.
            ValueError: Propagated from :meth:`iter_documents`.
        """
        return list(self.iter_documents(file, **kwargs))
