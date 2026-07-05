"""Streaming CSV helpers shared by FastAPI exports and the CLI.

Column schemas are kept in lockstep with ``frontend/src/lib/exports.ts`` and the
CLI's existing output so all three paths (UI download, HTTP stream, CLI batch)
produce identical CSV files.
"""

from __future__ import annotations

import csv
import io
import traceback
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from loguru import logger

UTF8_BOM = b"\xef\xbb\xbf"
DEFAULT_ROWS_PER_CHUNK = 64


ENTITY_STATS_COLUMNS: tuple[str, ...] = ("rank", "entity", "type", "mentions")

NER_SOURCE_COLUMNS: tuple[str, ...] = (
    "entity",
    "source",
    "page",
    "row",
    "chunk_id",
    "chunk_text",
    "network",
    "ref_type",
    "uuid",
    "timestamp",
    "author",
    "author_id",
    "vanity",
    "text_id",
    "anchor_text",
    "parent_text",
    "translation",
)

HATE_SPEECH_COLUMNS: tuple[str, ...] = (
    "source",
    "page",
    "row",
    "chunk_id",
    "category",
    "confidence",
    "reason",
    "chunk_text",
    "network",
    "ref_type",
    "uuid",
    "timestamp",
    "author",
    "author_id",
    "vanity",
    "text_id",
    "anchor_text",
    "parent_text",
    "translation",
)

DOCUMENT_COLUMNS: tuple[str, ...] = (
    "filename",
    "mimetype",
    "file_hash",
    "node_count",
    "page_count",
    "max_rows",
    "max_duration",
    "entity_types",
)


def _coerce_cell(value: Any) -> str:
    """Coerce an arbitrary cell value to its CSV string form."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, frozenset)):
        return ";".join(str(v) for v in value if v is not None)
    return str(value)


def stream_csv(
    rows: Iterable[dict[str, Any]],
    columns: Sequence[str],
    *,
    rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK,
) -> Iterator[bytes]:
    """Stream a CSV body row-by-row.

    The first yielded chunk contains a UTF-8 byte-order mark followed by the
    header row. Subsequent chunks bundle up to ``rows_per_chunk`` rows each.
    Exceptions raised by the underlying iterator are caught, logged with a
    traceback, and surfaced as a synthetic final row whose first cell is
    ``# ERROR: <reason>`` so partial downloads remain recognizable.

    Args:
        rows (Iterable[dict[str, Any]]): Source rows. Missing column keys
            render as empty cells.
        columns (Sequence[str]): Ordered column names. Become both the header
            and the per-row field order.
        rows_per_chunk (int): Number of rows to buffer between flushes.

    Yields:
        bytes: UTF-8 encoded chunks of CSV text suitable for an HTTP response.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, dialect="excel")

    writer.writerow(list(columns))
    yield UTF8_BOM + buffer.getvalue().encode("utf-8")
    buffer.seek(0)
    buffer.truncate(0)

    rows_in_chunk = 0
    iterator = iter(rows)
    while True:
        try:
            row = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            logger.error("CSV stream aborted mid-flight: {}\n{}", exc, traceback.format_exc())
            sentinel = [f"# ERROR: {exc.__class__.__name__}: {exc}"] + [""] * (len(columns) - 1)
            writer.writerow(sentinel)
            break

        writer.writerow([_coerce_cell(row.get(col)) for col in columns])
        rows_in_chunk += 1
        if rows_in_chunk >= rows_per_chunk:
            yield buffer.getvalue().encode("utf-8")
            buffer.seek(0)
            buffer.truncate(0)
            rows_in_chunk = 0

    if buffer.tell():
        yield buffer.getvalue().encode("utf-8")


def _reference_field(metadata: Any, key: str) -> Any:
    """Look up a reference_metadata field, mapping ``ref_type`` to ``type``."""
    if not isinstance(metadata, dict):
        return ""
    lookup = "type" if key == "ref_type" else key
    return metadata.get(lookup) or ""


def entity_stats_row(entity: dict[str, Any], *, rank: int) -> dict[str, Any]:
    """Build one CSV row for the entity-frequency export.

    Args:
        entity (dict[str, Any]): Ranked entity row from collection NER stats.
        rank (int): 1-based rank to embed in the row.

    Returns:
        dict[str, Any]: Row keyed by :data:`ENTITY_STATS_COLUMNS`.
    """
    return {
        "rank": rank,
        "entity": str(entity.get("text") or "Unknown"),
        "type": str(entity.get("type") or "Unlabeled"),
        "mentions": int(entity.get("mentions", entity.get("count", 0)) or 0),
    }


def _source_label(chunk: dict[str, Any]) -> str:
    """Pick the canonical 'source' field for a chunk row."""
    return str(chunk.get("filename") or chunk.get("source_ref") or "")


def ner_source_row(chunk: dict[str, Any], *, entity_label: str) -> dict[str, Any]:
    """Build one CSV row for the entity-findings export.

    Mirrors ``entityFindingsToCsv`` in ``frontend/src/lib/exports.ts``.
    """
    ref = chunk.get("reference_metadata") or {}
    return {
        "entity": entity_label,
        "source": _source_label(chunk),
        "page": chunk.get("page") or "",
        "row": chunk.get("row") or "",
        "chunk_id": chunk.get("chunk_id") or "",
        "chunk_text": chunk.get("chunk_text") or chunk.get("text") or "",
        "network": _reference_field(ref, "network"),
        "ref_type": _reference_field(ref, "ref_type"),
        "uuid": _reference_field(ref, "uuid"),
        "timestamp": _reference_field(ref, "timestamp"),
        "author": _reference_field(ref, "author"),
        "author_id": _reference_field(ref, "author_id"),
        "vanity": _reference_field(ref, "vanity"),
        "text_id": _reference_field(ref, "text_id"),
        "anchor_text": _reference_field(ref, "anchor_text"),
        "parent_text": _reference_field(ref, "parent_text"),
        "translation": (chunk.get("translation") or {}).get("text") or "",
    }


def hate_speech_row(chunk: dict[str, Any]) -> dict[str, Any]:
    """Build one CSV row for the hate-speech export.

    Mirrors ``hateSpeechToCsv`` in ``frontend/src/lib/exports.ts``.
    """
    ref = chunk.get("reference_metadata") or {}
    return {
        "source": _source_label(chunk),
        "page": chunk.get("page"),
        "row": chunk.get("row"),
        "chunk_id": chunk.get("chunk_id"),
        "category": chunk.get("category"),
        "confidence": chunk.get("confidence"),
        "reason": chunk.get("reason"),
        "chunk_text": chunk.get("chunk_text") or chunk.get("text"),
        "network": _reference_field(ref, "network"),
        "ref_type": _reference_field(ref, "ref_type"),
        "uuid": _reference_field(ref, "uuid"),
        "timestamp": _reference_field(ref, "timestamp"),
        "author": _reference_field(ref, "author"),
        "author_id": _reference_field(ref, "author_id"),
        "vanity": _reference_field(ref, "vanity"),
        "text_id": _reference_field(ref, "text_id"),
        "anchor_text": _reference_field(ref, "anchor_text"),
        "parent_text": _reference_field(ref, "parent_text"),
        "translation": (chunk.get("translation") or {}).get("text") or "",
    }


def document_row(doc: dict[str, Any]) -> dict[str, Any]:
    """Build one CSV row for the documents-list export."""
    return {
        "filename": doc.get("filename") or "",
        "mimetype": doc.get("mimetype") or "",
        "file_hash": doc.get("file_hash") or "",
        "node_count": doc.get("node_count") or 0,
        "page_count": doc.get("page_count") or 0,
        "max_rows": doc.get("max_rows") or "",
        "max_duration": doc.get("max_duration") or "",
        "entity_types": doc.get("entity_types") or [],
    }
