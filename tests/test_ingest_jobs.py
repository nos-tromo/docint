"""Unit tests for the in-memory ingest job registry."""

from __future__ import annotations

import json
from pathlib import Path

from docint.core.jobs import IngestJobState, JobStatus, format_sse


def _state() -> IngestJobState:
    return IngestJobState(
        job_id="job-1",
        owner="alice",
        logical_name="mydocs",
        physical="u000000000000__mydocs",
        batch_dir=Path("/nonexistent/batch"),
        hybrid=True,
        ner=None,
        hate_speech=None,
        resolve=False,
    )


def _events(frames: list[str]) -> list[str]:
    return [line[len("event: ") :] for f in frames for line in f.splitlines() if line.startswith("event: ")]


def test_history_keeps_only_the_latest_progress_frame() -> None:
    """Test that history collapses progress frames to the latest only."""
    state = _state()
    state.record("ingestion_started", format_sse("ingestion_started", {"collection": "mydocs"}))
    for i in range(1, 501):
        state.record("ingestion_progress", format_sse("ingestion_progress", {"message": f"chunk {i}"}))

    assert _events(state.history()) == ["ingestion_started", "ingestion_progress"]
    assert "chunk 500" in state.history()[-1]


def test_history_keeps_every_warning() -> None:
    """Test that all warning frames are retained in history."""
    state = _state()
    state.record("ingestion_started", format_sse("ingestion_started", {"collection": "mydocs"}))
    state.record("warning", format_sse("warning", {"message": "first"}))
    state.record("ingestion_progress", format_sse("ingestion_progress", {"message": "working"}))
    state.record("warning", format_sse("warning", {"message": "second"}))

    assert _events(state.history()) == [
        "ingestion_started",
        "warning",
        "warning",
        "ingestion_progress",
    ]


def test_history_ends_with_the_terminal_frame() -> None:
    """Test that terminal frames always appear at the end of history."""
    state = _state()
    state.record("ingestion_started", format_sse("ingestion_started", {"collection": "mydocs"}))
    state.record("ingestion_progress", format_sse("ingestion_progress", {"message": "working"}))
    state.record("ingestion_complete", format_sse("ingestion_complete", {"collection": "mydocs"}))

    assert _events(state.history())[-1] == "ingestion_complete"


def test_record_tracks_latest_message() -> None:
    """Test that record method extracts and updates message from frames."""
    state = _state()
    state.record("ingestion_progress", format_sse("ingestion_progress", {"message": "working"}))
    assert state.message == "working"


def test_snapshot_is_json_serializable() -> None:
    """Test that snapshot produces JSON-serializable output without physical name."""
    state = _state()
    state.status = JobStatus.RUNNING
    payload = json.loads(json.dumps(state.snapshot()))
    assert payload["job_id"] == "job-1"
    assert payload["collection"] == "mydocs"
    assert payload["status"] == "running"
    # The physical (owner-namespaced) name is internal and must never be echoed.
    assert "u000000000000__mydocs" not in json.dumps(payload)
