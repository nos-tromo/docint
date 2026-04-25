"""Tests for the typed ingestion events module."""

from __future__ import annotations

from docint.core.ingest.events import (
    IngestionEvent,
    format_progress_message,
)


def test_format_with_explicit_message_passthrough() -> None:
    """If a message is provided, it should pass through verbatim."""
    event: IngestionEvent = {
        "stage": "summary",
        "message": "All done.",
    }
    assert format_progress_message(event) == "All done."


def test_format_persist_event_with_batch_counters() -> None:
    """Persist events should render stage + batch counters + node count."""
    event: IngestionEvent = {
        "stage": "persist",
        "batch": 3,
        "of": 12,
        "nodes": 64,
        "file_hash": "abc1234567890def",
    }
    rendered = format_progress_message(event)
    assert "[persist]" in rendered
    assert "batch 3/12" in rendered
    assert "64 node(s)" in rendered
    assert "abc12345" in rendered  # truncated file_hash


def test_format_embed_event_omits_unset_fields() -> None:
    """Optional payload should not surface when absent."""
    event: IngestionEvent = {"stage": "embed", "batch": 2}
    rendered = format_progress_message(event)
    assert "[embed]" in rendered
    assert "batch 2" in rendered
    # No "of" was provided, so no slash form.
    assert "/" not in rendered.split("batch 2")[1]
    assert "node(s)" not in rendered
    assert "file_hash" not in rendered


def test_format_attempt_only_surfaces_when_above_one() -> None:
    """Attempt count should only render on retries, not on first attempts."""
    first: IngestionEvent = {"stage": "persist", "attempts": 1}
    retried: IngestionEvent = {"stage": "persist", "attempts": 3}
    assert "attempt" not in format_progress_message(first)
    assert "attempt 3" in format_progress_message(retried)


def test_format_unknown_stage_still_renders_safely() -> None:
    """Even an empty event should produce a non-empty fallback string."""
    event: IngestionEvent = {}
    rendered = format_progress_message(event)
    # Falls back to "[ingest]" tag; should not crash.
    assert "[" in rendered and "]" in rendered


def test_typed_event_payloads_preserved() -> None:
    """The TypedDict structure should round-trip through dict semantics."""
    event: IngestionEvent = {
        "stage": "manifest",
        "file_hash": "deadbeef",
        "nodes": 42,
    }
    # Direct dict access (TypedDict is just a dict at runtime).
    assert event["stage"] == "manifest"
    assert event["file_hash"] == "deadbeef"
    assert event["nodes"] == 42
