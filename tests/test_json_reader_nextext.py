"""Tests for the Nextext transcript schema handling inside CustomJSONReader."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable

import pytest
from _pytest.logging import LogCaptureFixture
from llama_index.core import Document
from loguru import logger

from docint.core.readers.json import CustomJSONReader


@pytest.fixture
def loguru_caplog(caplog: LogCaptureFixture) -> Iterable[LogCaptureFixture]:
    """Bridge loguru records into ``caplog`` for the duration of a test.

    Loguru bypasses ``logging`` so the stdlib ``caplog`` fixture does not see
    its records out of the box. This fixture adds a temporary loguru sink
    that re-emits every record through the stdlib handler ``caplog``
    attaches to the root logger, then tears the sink down on exit.

    Args:
        caplog: The standard pytest log-capture fixture.

    Yields:
        The same ``caplog`` fixture, now populated with loguru-sourced
        records at WARNING level and above.
    """
    handler_id = logger.add(
        caplog.handler,
        level="WARNING",
        format="{message}",
    )
    caplog.set_level(logging.WARNING)
    try:
        yield caplog
    finally:
        logger.remove(handler_id)


def _write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    """Write one JSON object per line to ``path``.

    Args:
        path: Target JSONL file.
        entries: Ordered list of payload dictionaries.
    """
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def _nextext_segment(**overrides: Any) -> dict[str, Any]:
    """Build a Nextext-shaped JSONL line with sensible defaults.

    Args:
        **overrides: Fields to override or add.

    Returns:
        A dict that includes the required Nextext keys.
    """
    base = {
        "source_file": "interview.mp3",
        "source_file_hash": "sha256:abc",
        "language": "de",
        "task": "transcribe",
        "sentence_index": 0,
        "start_seconds": 12.3,
        "end_seconds": 18.7,
        "start_ts": "00:00:12",
        "end_ts": "00:00:18",
        "speaker": "SPEAKER_01",
        "text": "Hallo Welt.",
    }
    base.update(overrides)
    return base


def test_jsonl_reader_detects_nextext_schema(tmp_path: Path) -> None:
    """A one-line Nextext JSONL produces one document with rich metadata.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    jsonl = tmp_path / "transcript.jsonl"
    _write_jsonl(jsonl, [_nextext_segment()])

    reader = CustomJSONReader(is_jsonl=True)
    documents = reader.load_data(jsonl)

    assert len(documents) == 1
    doc = documents[0]
    assert isinstance(doc, Document)
    assert doc.text == "Hallo Welt."

    meta = doc.metadata
    assert meta["source"] == "transcript"
    assert meta["filename"] == "transcript.jsonl"
    assert meta["whisper_task"] == "transcribe"
    assert meta["whisper_language"] == "de"
    assert meta["sentence_index"] == 0
    assert meta["start_ts"] == "00:00:12"
    assert meta["end_ts"] == "00:00:18"
    assert meta["start_seconds"] == pytest.approx(12.3)
    assert meta["end_seconds"] == pytest.approx(18.7)
    assert meta["speaker"] == "SPEAKER_01"
    assert meta["source_file"] == "interview.mp3"
    assert meta["source_file_hash"] == "sha256:abc"
    assert meta["file_hash"]  # populated by ensure_file_hash


def test_jsonl_reader_generic_json_unchanged(tmp_path: Path) -> None:
    """A plain JSON payload (no Nextext markers) keeps the generic path.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    plain = tmp_path / "records.json"
    plain.write_text(
        json.dumps({"title": "Report", "body": "Nothing to see here."}),
        encoding="utf-8",
    )

    reader = CustomJSONReader()
    documents = reader.load_data(plain)

    assert len(documents) >= 1
    # Generic path uses ``source == "json"`` and populates schema info.
    meta = documents[0].metadata
    assert meta["source"] == "json"
    assert "schema" in meta
    assert "sentence_index" not in meta


def test_jsonl_reader_multi_segment_preserves_order(tmp_path: Path) -> None:
    """Three JSONL lines yield three documents in order with monotonic indices.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    jsonl = tmp_path / "multi.jsonl"
    entries = [
        _nextext_segment(
            sentence_index=0,
            start_seconds=0.0,
            end_seconds=1.5,
            start_ts="00:00:00",
            end_ts="00:00:01",
            text="First segment.",
        ),
        _nextext_segment(
            sentence_index=1,
            start_seconds=1.6,
            end_seconds=4.2,
            start_ts="00:00:01",
            end_ts="00:00:04",
            text="Second segment.",
        ),
        _nextext_segment(
            sentence_index=2,
            start_seconds=4.3,
            end_seconds=8.0,
            start_ts="00:00:04",
            end_ts="00:00:08",
            text="Third segment.",
        ),
    ]
    _write_jsonl(jsonl, entries)

    reader = CustomJSONReader(is_jsonl=True)
    documents = reader.load_data(jsonl)

    assert [doc.text for doc in documents] == [
        "First segment.",
        "Second segment.",
        "Third segment.",
    ]
    indices = [doc.metadata["sentence_index"] for doc in documents]
    assert indices == [0, 1, 2]
    # All segments carry the shared transcript-file hash.
    hashes = {doc.metadata["file_hash"] for doc in documents}
    assert len(hashes) == 1


def test_jsonl_reader_nextext_without_optional_fields(tmp_path: Path) -> None:
    """Missing optional fields (speaker, hash) must not break segment ingestion.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    jsonl = tmp_path / "partial.jsonl"
    entry = {
        "source_file": "call.wav",
        "task": "transcribe",
        "sentence_index": 0,
        "start_seconds": 0.0,
        "end_seconds": 2.0,
        "text": "Hello there.",
    }
    _write_jsonl(jsonl, [entry])

    reader = CustomJSONReader(is_jsonl=True)
    documents = reader.load_data(jsonl)

    assert len(documents) == 1
    meta = documents[0].metadata
    assert "speaker" not in meta
    assert "source_file_hash" not in meta
    assert meta["start_ts"] == "00:00:00"
    assert meta["end_ts"] == "00:00:02"


# ---------------------------------------------------------------------------
# New TDD tests — these pin the NEW contract and intentionally fail against
# the current implementation until the implementation agent lands its changes.
# ---------------------------------------------------------------------------


def _nextext_ts_only_segment(**overrides: Any) -> dict[str, Any]:
    """Build a Nextext segment that has ONLY the ``start_ts`` / ``end_ts`` pair.

    No ``start_seconds`` / ``end_seconds`` keys are present. This is a legal
    Nextext payload under the permissive detection rule introduced in the
    audio-removal branch.

    Args:
        **overrides: Fields to override or add on top of the base dict.

    Returns:
        A dict matching the permissive Nextext schema (ts-keys only).
    """
    base: dict[str, Any] = {
        "source_file": "interview.mp3",
        "language": "de",
        "task": "transcribe",
        "sentence_index": 0,
        "start_ts": "00:00:12",
        "end_ts": "00:00:18",
        "speaker": "SPEAKER_01",
        "text": "Hallo Welt.",
    }
    base.update(overrides)
    return base


def _nextext_seconds_only_segment(**overrides: Any) -> dict[str, Any]:
    """Build a Nextext segment that has ONLY the ``start_seconds`` / ``end_seconds`` pair.

    No ``start_ts`` / ``end_ts`` keys are present.

    Args:
        **overrides: Fields to override or add on top of the base dict.

    Returns:
        A dict matching the permissive Nextext schema (seconds-keys only).
    """
    base: dict[str, Any] = {
        "source_file": "interview.mp3",
        "language": "de",
        "task": "transcribe",
        "sentence_index": 0,
        "start_seconds": 12.3,
        "end_seconds": 18.7,
        "speaker": "SPEAKER_01",
        "text": "Hallo Welt.",
    }
    base.update(overrides)
    return base


def test_nextext_detection_accepts_ts_keys_only(tmp_path: Path) -> None:
    """A JSONL with ``start_ts`` / ``end_ts`` but no ``*_seconds`` keys is Nextext.

    Under the permissive detection rule, either timing pair is sufficient.
    This test fails on the legacy ``NEXTEXT_REQUIRED_KEYS`` check which demands
    ``start_seconds`` and ``end_seconds``.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    jsonl = tmp_path / "ts_only.jsonl"
    _write_jsonl(jsonl, [_nextext_ts_only_segment()])

    from docint.core.readers.json import CustomJSONReader as _CJR

    is_nextext = _CJR._detect_nextext_transcript(jsonl)
    assert is_nextext, (
        "_detect_nextext_transcript should accept start_ts/end_ts without "
        "start_seconds/end_seconds"
    )


def test_nextext_detection_accepts_seconds_keys_only(tmp_path: Path) -> None:
    """A JSONL with only ``start_seconds`` / ``end_seconds`` (no ts strings) is Nextext.

    This mirrors ``test_nextext_detection_accepts_ts_keys_only`` for the legacy
    timing-key variant.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    jsonl = tmp_path / "seconds_only.jsonl"
    _write_jsonl(jsonl, [_nextext_seconds_only_segment()])

    from docint.core.readers.json import CustomJSONReader as _CJR

    is_nextext = _CJR._detect_nextext_transcript(jsonl)
    assert is_nextext, (
        "_detect_nextext_transcript should accept start_seconds/end_seconds"
    )


def test_nextext_detection_rejects_missing_text(tmp_path: Path) -> None:
    """A JSONL that has both timing pairs but no ``text`` key is NOT Nextext.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    jsonl = tmp_path / "no_text.jsonl"
    segment: dict[str, Any] = {
        "source_file": "interview.mp3",
        "start_ts": "00:00:12",
        "end_ts": "00:00:18",
        "start_seconds": 12.0,
        "end_seconds": 18.0,
        # deliberately no "text" key
    }
    _write_jsonl(jsonl, [segment])

    from docint.core.readers.json import CustomJSONReader as _CJR

    is_nextext = _CJR._detect_nextext_transcript(jsonl)
    assert not is_nextext, (
        "_detect_nextext_transcript must reject payloads that lack a 'text' key"
    )


def test_nextext_segment_emits_one_document_per_line(tmp_path: Path) -> None:
    """Three JSONL lines yield exactly 3 Documents with prose text and monotonic indices.

    Each ``document.text`` must be the prose string from ``segment["text"]``, not
    raw JSON. ``metadata["sentence_index"]`` must be 0, 1, 2 in order.
    ``metadata["docint_doc_kind"]`` must equal ``"transcript_segment"``.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    jsonl = tmp_path / "three_segs.jsonl"
    segments = [
        _nextext_segment(
            sentence_index=0,
            start_seconds=0.0,
            end_seconds=1.0,
            start_ts="00:00:00",
            end_ts="00:00:01",
            text="Äpfel 中 first segment.",
            speaker="SPEAKER_01",
        ),
        _nextext_segment(
            sentence_index=1,
            start_seconds=1.1,
            end_seconds=2.5,
            start_ts="00:00:01",
            end_ts="00:00:02",
            text="Second segment text.",
            task="translate",
        ),
        _nextext_segment(
            sentence_index=2,
            start_seconds=2.6,
            end_seconds=4.0,
            start_ts="00:00:02",
            end_ts="00:00:04",
            text="Third and final segment.",
        ),
    ]
    _write_jsonl(jsonl, segments)

    reader = CustomJSONReader(is_jsonl=True)
    documents = reader.load_data(jsonl)

    assert len(documents) == 3, f"Expected 3 documents, got {len(documents)}"

    expected_texts = [
        "Äpfel 中 first segment.",
        "Second segment text.",
        "Third and final segment.",
    ]
    for idx, (doc, expected_text) in enumerate(zip(documents, expected_texts)):
        assert doc.text == expected_text, (
            f"Document {idx}: expected prose text {expected_text!r}, got {doc.text!r}"
        )
        assert doc.metadata["sentence_index"] == idx
        assert doc.metadata["docint_doc_kind"] == "transcript_segment", (
            f"Document {idx}: missing docint_doc_kind == 'transcript_segment'"
        )


def test_nextext_reference_metadata_populated(tmp_path: Path) -> None:
    """Each emitted Document carries a ``reference_metadata`` dict with required keys.

    Required keys: ``type``, ``network``, ``text``, ``text_id``,
    ``start_ts``, ``end_ts``, ``language``, ``source_file``.
    ``author`` must equal ``speaker`` when a speaker is present.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    source_file = "interview.mp3"
    jsonl = tmp_path / "refmeta.jsonl"
    _write_jsonl(
        jsonl,
        [
            _nextext_segment(
                source_file=source_file,
                speaker="SPEAKER_01",
                language="de",
                start_ts="00:00:12",
                end_ts="00:00:18",
                text="Hallo Welt.",
                sentence_index=0,
            )
        ],
    )

    reader = CustomJSONReader(is_jsonl=True)
    documents = reader.load_data(jsonl)

    assert len(documents) == 1
    ref = documents[0].metadata.get("reference_metadata")
    assert isinstance(ref, dict), "reference_metadata must be a dict"

    assert ref["type"] == "transcript_segment"
    assert ref["network"] == "nextext"
    assert ref["text"] == "Hallo Welt."
    assert ref["text_id"] == f"{source_file}:0"
    assert ref["start_ts"] == "00:00:12"
    assert ref["end_ts"] == "00:00:18"
    assert ref["language"] == "de"
    assert ref["source_file"] == source_file
    assert ref["author"] == "SPEAKER_01"


def test_nextext_segment_without_speaker_omits_key(tmp_path: Path) -> None:
    """A segment with no ``speaker`` field omits the key from ``reference_metadata``.

    The key must be absent — not set to ``None``.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    jsonl = tmp_path / "no_speaker.jsonl"
    seg: dict[str, Any] = {
        "source_file": "call.mp3",
        "language": "en",
        "task": "transcribe",
        "sentence_index": 0,
        "start_seconds": 0.0,
        "end_seconds": 1.5,
        "start_ts": "00:00:00",
        "end_ts": "00:00:01",
        "text": "No speaker here.",
    }
    _write_jsonl(jsonl, [seg])

    reader = CustomJSONReader(is_jsonl=True)
    documents = reader.load_data(jsonl)

    assert len(documents) == 1
    ref = documents[0].metadata.get("reference_metadata")
    assert isinstance(ref, dict), "reference_metadata must be a dict"
    assert "speaker" not in ref, (
        "speaker key must be absent from reference_metadata when not in segment"
    )


@pytest.mark.parametrize(
    "start_seconds,end_seconds,expected_start_ts,expected_end_ts",
    [
        (3725.0, 3726.0, "01:02:05", "01:02:06"),
        (0.0, 59.0, "00:00:00", "00:00:59"),
        (3600.0, 7199.0, "01:00:00", "01:59:59"),
    ],
)
def test_nextext_derives_missing_timestamp_strings(
    tmp_path: Path,
    start_seconds: float,
    end_seconds: float,
    expected_start_ts: str,
    expected_end_ts: str,
) -> None:
    """Segments with only ``*_seconds`` produce derived ``start_ts`` / ``end_ts``.

    The derivation must follow ``HH:MM:SS`` zero-padded format.

    Args:
        tmp_path: Temporary directory provided by pytest.
        start_seconds: Input start time in fractional seconds.
        end_seconds: Input end time in fractional seconds.
        expected_start_ts: Expected derived ``start_ts`` string.
        expected_end_ts: Expected derived ``end_ts`` string.
    """
    jsonl = tmp_path / "derive_ts.jsonl"
    seg: dict[str, Any] = {
        "source_file": "audio.mp3",
        "language": "en",
        "task": "transcribe",
        "sentence_index": 0,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "text": "Some spoken text.",
        # deliberately no start_ts / end_ts
    }
    _write_jsonl(jsonl, [seg])

    reader = CustomJSONReader(is_jsonl=True)
    documents = reader.load_data(jsonl)

    assert len(documents) == 1
    ref = documents[0].metadata.get("reference_metadata")
    assert isinstance(ref, dict), "reference_metadata must be a dict"
    assert ref["start_ts"] == expected_start_ts, (
        f"Expected start_ts={expected_start_ts!r}, got {ref.get('start_ts')!r}"
    )
    assert ref["end_ts"] == expected_end_ts, (
        f"Expected end_ts={expected_end_ts!r}, got {ref.get('end_ts')!r}"
    )


# ---------------------------------------------------------------------------
# Regression coverage for the hardening review follow-ups.
# ---------------------------------------------------------------------------


def test_nextext_detection_rejects_cross_pair_incomplete(tmp_path: Path) -> None:
    """A cross-pair segment (``start_ts`` + ``end_seconds``) is NOT Nextext.

    Either the ``start_ts`` / ``end_ts`` pair or the
    ``start_seconds`` / ``end_seconds`` pair must be fully present. A cross
    pair with one key from each variant does not satisfy the permissive
    detection rule.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    jsonl = tmp_path / "cross_pair.jsonl"
    _write_jsonl(
        jsonl,
        [{"text": "hi", "start_ts": "00:00:01", "end_seconds": 5.0}],
    )

    assert CustomJSONReader._detect_nextext_transcript(jsonl) is False


def test_nextext_detection_probes_past_malformed_leading_line(tmp_path: Path) -> None:
    """A JSONL whose first line is malformed but second line is Nextext IS detected.

    Pins the multi-line probe behavior introduced to prevent a corrupt leading
    byte from routing an otherwise valid Nextext file through the generic
    JSON path.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    jsonl = tmp_path / "malformed_leading.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        handle.write("{not json\n")
        handle.write(json.dumps(_nextext_segment()) + "\n")

    assert CustomJSONReader._detect_nextext_transcript(jsonl) is True


def test_nextext_max_segments_cap_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loguru_caplog: LogCaptureFixture,
) -> None:
    """Segment iteration stops at ``NEXTEXT_MAX_SEGMENTS`` with a warning.

    Writes three Nextext-shaped lines and monkeypatches the cap down to 2, so
    only 2 documents are emitted and a truncation warning is logged.

    Args:
        tmp_path: Temporary directory provided by pytest.
        monkeypatch: The pytest monkeypatch fixture.
        loguru_caplog: Loguru-aware caplog bridge fixture.
    """
    import docint.core.readers.json as json_reader_mod

    monkeypatch.setattr(json_reader_mod, "NEXTEXT_MAX_SEGMENTS", 2)

    jsonl = tmp_path / "capped.jsonl"
    _write_jsonl(
        jsonl,
        [
            _nextext_segment(sentence_index=0, text="first"),
            _nextext_segment(sentence_index=1, text="second"),
            _nextext_segment(sentence_index=2, text="third"),
        ],
    )

    reader = CustomJSONReader(is_jsonl=True)
    documents = reader.load_data(jsonl)

    assert len(documents) == 2
    assert [doc.text for doc in documents] == ["first", "second"]
    assert any(
        "segment cap reached" in record.getMessage() for record in loguru_caplog.records
    ), "expected a truncation warning when the cap is hit"


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (3.4, "00:00:03"),
        (3.5, "00:00:04"),
        (3.9, "00:00:04"),
        (3725.9, "01:02:06"),
    ],
)
def test_format_timestamp_rounds_fractional_seconds(
    seconds: float, expected: str
) -> None:
    """Fractional-second inputs round (not truncate) to the nearest whole second.

    Args:
        seconds: Fractional-seconds input to format.
        expected: Expected zero-padded ``HH:MM:SS`` output.
    """
    from docint.core.readers.json import _format_timestamp

    assert _format_timestamp(seconds) == expected


@pytest.mark.parametrize("bad_value", [math.inf, -math.inf, math.nan, -1.0])
def test_format_timestamp_handles_non_finite_and_negative(
    bad_value: float, loguru_caplog: LogCaptureFixture
) -> None:
    """Non-finite and negative inputs return ``"00:00:00"`` and emit a warning.

    Args:
        bad_value: The pathological input to exercise.
        loguru_caplog: Loguru-aware caplog bridge fixture.
    """
    from docint.core.readers.json import _format_timestamp

    assert _format_timestamp(bad_value) == "00:00:00"
    assert any(
        "non-finite or negative timestamp" in record.getMessage()
        for record in loguru_caplog.records
    ), f"expected a warning for {bad_value!r}"


@pytest.mark.parametrize(
    "raw",
    ["99:99:99", "-1:0:0", "0:60:0", "0:0:60"],
)
def test_parse_timestamp_rejects_out_of_range(raw: str) -> None:
    """Components out of the valid HH / MM / SS ranges yield ``None``.

    Args:
        raw: Malformed or out-of-range timestamp string.
    """
    from docint.core.readers.json import _parse_timestamp

    assert _parse_timestamp(raw) is None
