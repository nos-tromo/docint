"""Tests for ingest UI formatting and event de-noising helpers."""

from docint.ui.ingest import (
    _append_event,
    _normalize_file_status,
    _prettify_progress_message,
    _render_event_list,
)


def test_prettify_progress_message_formats_ner_updates() -> None:
    """NER progress messages should be shortened for readability."""
    raw = "Extracting entities: 6/12 chunks processed"
    assert _prettify_progress_message(raw) == "NER extraction 6/12 chunks"


def test_append_event_collapses_consecutive_progress_events() -> None:
    """Consecutive ingestion-progress events should be replaced, not appended."""
    events: list[dict[str, str]] = []
    _append_event(events, stage="ingestion_progress", message="step 1")
    _append_event(events, stage="ingestion_progress", message="step 2")
    _append_event(events, stage="ingestion_progress", message="step 3")
    assert len(events) == 1
    assert events[0]["message"] == "step 3"


def test_render_event_list_formats_bullet_timeline() -> None:
    """Event timeline should render as markdown bullets with icons."""
    events = [
        {"stage": "start", "message": "Upload started", "filename": ""},
        {
            "stage": "file_saved",
            "message": "Saved upload",
            "filename": "paper.pdf",
        },
    ]
    timeline = _render_event_list(events, limit=5)
    assert "- 🚀 Upload started" in timeline
    assert "- 💾 `paper.pdf` · Saved upload" in timeline


def test_normalize_file_status_maps_common_stage_names() -> None:
    """Raw stage names should map to stable, user-facing statuses."""
    assert _normalize_file_status("file_saved") == "Uploading"
    assert _normalize_file_status("ingestion_started") == "Processing"
    assert _normalize_file_status("done") == "Done"
    assert _normalize_file_status("error") == "Error"
