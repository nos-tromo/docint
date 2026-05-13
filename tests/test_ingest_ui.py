"""Tests for ingest UI formatting and event de-noising helpers."""

from typing import Any

import pytest

from docint.ui import ingest as ingest_module
from docint.ui.ingest import (
    _append_event,
    _normalize_file_status,
    _prettify_progress_message,
    _render_event_list,
    _sync_backend_collection,
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


def test_render_event_list_formats_warning_events() -> None:
    """Warning events should render with a dedicated warning icon."""
    events = [
        {
            "stage": "warning",
            "message": "Warning: truncated oversized embedding input",
            "filename": "",
        }
    ]

    timeline = _render_event_list(events, limit=5)

    assert "- ⚠️ Warning: truncated oversized embedding input" in timeline


def test_normalize_file_status_maps_common_stage_names() -> None:
    """Raw stage names should map to stable, user-facing statuses."""
    assert _normalize_file_status("file_saved") == "Uploading"
    assert _normalize_file_status("ingestion_started") == "Processing"
    assert _normalize_file_status("done") == "Done"
    assert _normalize_file_status("error") == "Error"


class _FakeResponse:
    """Minimal stand-in for ``requests.Response`` used in tests."""

    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _ErrorRecorder:
    """Capture ``st.error`` calls without touching Streamlit's runtime."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def __call__(self, message: str) -> None:
        self.messages.append(message)


def test_sync_backend_collection_posts_select_and_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for the post-ingest navigation bug.

    After a successful ingest, the UI must POST to ``/collections/select``
    so the API's module-level ``RAG`` singleton's ``qdrant_collection``
    is synchronized with the new collection. Previously the ingest page
    only mutated ``st.session_state.selected_collection``, which left
    the sidebar's switch guard short-circuited (because the session-state
    key already matched the dropdown value), so the select POST never
    fired and the Analysis / Chat pages saw "No collection selected".
    """
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        captured["timeout"] = kwargs.get("timeout")
        return _FakeResponse(200)

    monkeypatch.setattr(ingest_module.requests, "post", fake_post)
    monkeypatch.setattr(ingest_module.st, "error", _ErrorRecorder())

    assert _sync_backend_collection("brand-new-coll") is True
    assert captured["url"].endswith("/collections/select")
    assert captured["json"] == {"name": "brand-new-coll"}
    assert captured["timeout"] is not None, (
        "timeout must be forwarded to requests.post to prevent UI hangs"
        " when the backend is unreachable"
    )


def test_sync_backend_collection_returns_false_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-200 responses must return False so session state stays untouched."""
    monkeypatch.setattr(
        ingest_module.requests,
        "post",
        lambda url, **kwargs: _FakeResponse(404, "missing"),
    )
    recorder = _ErrorRecorder()
    monkeypatch.setattr(ingest_module.st, "error", recorder)

    assert _sync_backend_collection("missing-coll") is False
    assert any("missing-coll" in msg for msg in recorder.messages)


def test_sync_backend_collection_returns_false_on_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport errors must be reported and return False."""

    def boom(url: str, **kwargs: Any) -> _FakeResponse:
        raise ConnectionError("backend offline")

    monkeypatch.setattr(ingest_module.requests, "post", boom)
    recorder = _ErrorRecorder()
    monkeypatch.setattr(ingest_module.st, "error", recorder)

    assert _sync_backend_collection("any-coll") is False
    assert any("backend offline" in msg for msg in recorder.messages)
