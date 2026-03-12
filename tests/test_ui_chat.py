"""Tests for chat UI helper functions."""

from contextlib import nullcontext

import pytest

from docint.ui import chat as chat_module
from docint.ui.chat import _format_graph_debug_summary


def test_format_graph_debug_summary_includes_core_fields() -> None:
    """Graph debug formatter should render key GraphRAG fields in one line."""
    summary = _format_graph_debug_summary(
        {
            "enabled": True,
            "applied": False,
            "reason": "no_anchor_entities",
            "anchor_entities": [],
            "neighbor_entities": [],
        }
    )
    assert summary is not None
    assert "enabled=True" in summary
    assert "applied=False" in summary
    assert "reason=no_anchor_entities" in summary


def test_format_graph_debug_summary_none_returns_none() -> None:
    """Formatter should gracefully return None when no payload is provided."""
    assert _format_graph_debug_summary(None) is None


def test_render_sources_panel_uses_popover_with_scrolling_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chat source browser should avoid page growth by using a fixed-height panel."""
    events: list[tuple[str, object]] = []

    def _popover(label: str):
        events.append(("popover", label))
        return nullcontext()

    def _container(*, height: int):
        events.append(("container", height))
        return nullcontext()

    monkeypatch.setattr(chat_module.st, "popover", _popover)
    monkeypatch.setattr(chat_module.st, "container", _container)
    monkeypatch.setattr(
        chat_module.st, "markdown", lambda text: events.append(("markdown", text))
    )
    monkeypatch.setattr(
        chat_module,
        "render_source_item",
        lambda src, collection: events.append(
            ("source", f"{src.get('filename')}::{collection}")
        ),
    )
    monkeypatch.setattr(
        chat_module,
        "render_ner_overview",
        lambda sources: events.append(("ner", len(sources))),
    )

    chat_module._render_sources_panel(
        [
            {"filename": "alpha.pdf"},
            {"filename": "beta.pdf"},
        ],
        "collection-a",
    )

    assert ("popover", "View Sources") in events
    assert ("container", chat_module.CHAT_SOURCES_CONTAINER_HEIGHT) in events
    assert ("source", "alpha.pdf::collection-a") in events
    assert ("source", "beta.pdf::collection-a") in events
    assert ("markdown", "**Information Extraction Overview**") in events
    assert ("ner", 2) in events
