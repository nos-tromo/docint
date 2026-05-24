"""Tests for the streaming CSV helpers."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from typing import Any

import pytest

from docint.utils.csv_stream import (
    DOCUMENT_COLUMNS,
    ENTITY_STATS_COLUMNS,
    HATE_SPEECH_COLUMNS,
    NER_SOURCE_COLUMNS,
    UTF8_BOM,
    document_row,
    entity_stats_row,
    hate_speech_row,
    ner_source_row,
    stream_csv,
)


def _collect(stream: Iterator[bytes]) -> bytes:
    """Drain a byte iterator into a single payload."""
    return b"".join(stream)


def _parse(body: bytes) -> list[list[str]]:
    """Parse a CSV byte payload back into rows (stripping the BOM)."""
    text = body.decode("utf-8-sig")
    return list(csv.reader(io.StringIO(text)))


def test_stream_csv_emits_bom_and_header() -> None:
    chunks = list(stream_csv([], ["a", "b"]))
    assert chunks
    assert chunks[0].startswith(UTF8_BOM)
    parsed = _parse(_collect(iter(chunks)))
    assert parsed == [["a", "b"]]


def test_stream_csv_writes_rows_in_column_order() -> None:
    rows = [{"b": "2", "a": "1"}, {"a": "x", "b": "y"}]
    parsed = _parse(_collect(stream_csv(rows, ["a", "b"])))
    assert parsed == [["a", "b"], ["1", "2"], ["x", "y"]]


def test_stream_csv_rfc4180_quoting() -> None:
    rows = [
        {"a": 'has "quotes"', "b": "has,comma"},
        {"a": "newline\nhere", "b": "ok"},
    ]
    parsed = _parse(_collect(stream_csv(rows, ["a", "b"])))
    assert parsed[1] == ['has "quotes"', "has,comma"]
    assert parsed[2] == ["newline\nhere", "ok"]


def test_stream_csv_coerces_none_to_empty_string() -> None:
    parsed = _parse(_collect(stream_csv([{"a": None, "b": "x"}], ["a", "b"])))
    assert parsed[1] == ["", "x"]


def test_stream_csv_coerces_list_with_semicolon_join() -> None:
    parsed = _parse(_collect(stream_csv([{"types": ["PERSON", "ORG"]}], ["types"])))
    assert parsed[1] == ["PERSON;ORG"]


def test_stream_csv_emits_chunks_at_boundary() -> None:
    rows = [{"a": str(i), "b": str(i * 2)} for i in range(130)]
    chunks = list(stream_csv(rows, ["a", "b"], rows_per_chunk=64))
    # 1 chunk for BOM+header, 2 chunks for 64+64 rows, 1 trailing chunk for 2.
    assert len(chunks) == 4
    parsed = _parse(_collect(iter(chunks)))
    assert len(parsed) == 131
    assert parsed[1] == ["0", "0"]
    assert parsed[-1] == ["129", "258"]


def test_stream_csv_first_chunk_arrives_before_full_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    yielded: list[int] = []

    def slow_rows() -> Iterator[dict[str, Any]]:
        for i in range(5):
            yielded.append(i)
            yield {"a": i}

    stream = stream_csv(slow_rows(), ["a"], rows_per_chunk=2)
    first = next(stream)
    assert first.startswith(UTF8_BOM)
    # The header must land before the generator has fully iterated.
    assert len(yielded) == 0
    list(stream)
    assert len(yielded) == 5


def test_stream_csv_emits_error_row_on_iterator_failure() -> None:
    def failing_rows() -> Iterator[dict[str, Any]]:
        yield {"a": 1, "b": 2}
        raise RuntimeError("boom")

    parsed = _parse(_collect(stream_csv(failing_rows(), ["a", "b"])))
    assert parsed[0] == ["a", "b"]
    assert parsed[1] == ["1", "2"]
    assert parsed[2][0].startswith("# ERROR:")
    assert parsed[2][1] == ""


def test_entity_stats_row_handles_missing_fields() -> None:
    row = entity_stats_row({"text": "Alice", "type": "PERSON", "mentions": 5}, rank=1)
    assert row == {"rank": 1, "entity": "Alice", "type": "PERSON", "mentions": 5}

    fallback = entity_stats_row({}, rank=7)
    assert fallback == {"rank": 7, "entity": "Unknown", "type": "Unlabeled", "mentions": 0}


def test_entity_stats_row_accepts_count_alias() -> None:
    row = entity_stats_row({"text": "X", "type": "Y", "count": 11}, rank=2)
    assert row["mentions"] == 11


def test_ner_source_row_pulls_reference_metadata() -> None:
    chunk = {
        "filename": "doc.pdf",
        "page": 3,
        "chunk_id": "c1",
        "chunk_text": "hello",
        "reference_metadata": {
            "network": "telegram",
            "type": "post",
            "author": "alice",
        },
    }
    row = ner_source_row(chunk, entity_label="Alice [PERSON]")
    assert row["entity"] == "Alice [PERSON]"
    assert row["source"] == "doc.pdf"
    assert row["network"] == "telegram"
    assert row["ref_type"] == "post"
    assert row["author"] == "alice"
    assert row["chunk_text"] == "hello"


def test_ner_source_row_falls_back_to_source_ref_and_text() -> None:
    chunk = {"source_ref": "transcript.json", "text": "fallback text"}
    row = ner_source_row(chunk, entity_label="X")
    assert row["source"] == "transcript.json"
    assert row["chunk_text"] == "fallback text"


def test_hate_speech_row_full_payload() -> None:
    chunk = {
        "filename": "doc.pdf",
        "page": 3,
        "chunk_id": "c1",
        "category": "X",
        "confidence": 0.8,
        "reason": "...",
        "chunk_text": "bad",
        "reference_metadata": {"network": "telegram", "type": "post"},
    }
    row = hate_speech_row(chunk)
    assert row["source"] == "doc.pdf"
    assert row["page"] == 3
    assert row["network"] == "telegram"
    assert row["ref_type"] == "post"


def test_document_row_preserves_entity_types_list() -> None:
    doc = {
        "filename": "a.pdf",
        "mimetype": "application/pdf",
        "file_hash": "deadbeef",
        "node_count": 12,
        "page_count": 4,
        "entity_types": ["PERSON", "ORG"],
    }
    row = document_row(doc)
    assert row["filename"] == "a.pdf"
    assert row["node_count"] == 12
    assert row["entity_types"] == ["PERSON", "ORG"]


def test_column_constants_match_documented_schemas() -> None:
    # Guard against accidental column drift that would break the CLI <-> HTTP
    # parity contract.
    assert ENTITY_STATS_COLUMNS == ("rank", "entity", "type", "mentions")
    assert NER_SOURCE_COLUMNS[:6] == ("entity", "source", "page", "row", "chunk_id", "chunk_text")
    assert HATE_SPEECH_COLUMNS[:7] == ("source", "page", "row", "chunk_id", "category", "confidence", "reason")
    assert DOCUMENT_COLUMNS[:5] == ("filename", "mimetype", "file_hash", "node_count", "page_count")
