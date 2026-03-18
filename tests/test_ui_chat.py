"""Tests for chat UI helper functions."""

from contextlib import nullcontext

import pytest

from docint.ui import chat as chat_module
from docint.ui.chat import (
    _entity_candidate_label,
    _format_graph_debug_summary,
    _query_mode_badge_label,
    _retrieval_mode_badge_label,
    build_chat_metadata_filters,
)


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


def test_retrieval_mode_badge_label_defaults_to_stateless() -> None:
    """Badge label helper should default unknown values to stateless label."""
    assert _retrieval_mode_badge_label(None) == "Stateless Retrieval"
    assert _retrieval_mode_badge_label("unexpected") == "Stateless Retrieval"


def test_retrieval_mode_badge_label_session() -> None:
    """Badge label helper should render session mode label."""
    assert _retrieval_mode_badge_label("session") == "Session Retrieval"


def test_query_mode_badge_label_defaults_to_answer() -> None:
    """Query mode badge should default unknown values to answer mode."""
    assert _query_mode_badge_label(None) == "Answer Mode"
    assert _query_mode_badge_label("unexpected") == "Answer Mode"


def test_query_mode_badge_label_occurrence() -> None:
    """Query mode badge should render entity occurrence mode clearly."""
    assert _query_mode_badge_label("entity_occurrence") == "Entity Occurrence Mode"


def test_query_mode_badge_label_multi_occurrence() -> None:
    """Query mode badge should render the multi-entity occurrence label."""
    assert (
        _query_mode_badge_label("entity_occurrence_multi")
        == "Multi-Entity Occurrence Mode"
    )


def test_entity_candidate_label_includes_type_and_mentions() -> None:
    """Candidate labels should help users disambiguate tied entities quickly."""
    label = _entity_candidate_label({"text": "Acme", "type": "ORG", "mentions": 3})
    assert label == "Acme [ORG] · 3 mention(s)"


def test_build_chat_metadata_filters_combines_scope_date_and_custom_rules() -> None:
    """Chat helper should serialize enabled UI filters into API payload rules."""
    filters = build_chat_metadata_filters(
        {
            "enabled": True,
            "scope": "Images only",
            "mime_pattern": "",
            "date_field": "reference_metadata.timestamp",
            "start_date": "2026-01-01",
            "end_date": "2026-01-31",
            "hate_speech_only": True,
        },
        custom_rules=[
            {
                "field": "reference_metadata.author_id",
                "operator": "eq",
                "value": "alice",
            }
        ],
    )

    assert filters == [
        {"field": "mimetype", "operator": "mime_match", "value": "image/*"},
        {
            "field": "reference_metadata.timestamp",
            "operator": "date_on_or_after",
            "value": "2026-01-01",
        },
        {
            "field": "reference_metadata.timestamp",
            "operator": "date_on_or_before",
            "value": "2026-01-31",
        },
        {
            "field": "hate_speech.hate_speech",
            "operator": "eq",
            "value": True,
        },
        {
            "field": "reference_metadata.author_id",
            "operator": "eq",
            "value": "alice",
        },
    ]


def test_build_chat_metadata_filters_returns_empty_when_disabled() -> None:
    """Disabled chat filters should not send metadata filter payloads."""
    assert (
        build_chat_metadata_filters(
            {
                "enabled": False,
                "scope": "Images only",
                "date_field": "timestamp",
                "start_date": "2026-01-01",
            }
        )
        == []
    )


def test_render_sources_panel_uses_popover_with_scrolling_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chat source browser should avoid page growth by using a fixed-height panel.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
    """
    events: list[tuple[str, object]] = []

    def _popover(label: str):
        events.append(("popover", label))
        return nullcontext()

    def _container(*, height: int):
        events.append(("container", height))
        return nullcontext()

    monkeypatch.setattr(chat_module.st, "popover", _popover)
    monkeypatch.setattr(chat_module.st, "container", _container)
    monkeypatch.setattr(chat_module.st, "caption", lambda text: None)
    monkeypatch.setattr(
        chat_module.st,
        "markdown",
        lambda text: events.append(("markdown", text)),
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

    assert ("popover", "Sources") in events
    assert ("container", chat_module.CHAT_SOURCES_CONTAINER_HEIGHT) in events
    assert ("source", "alpha.pdf::collection-a") in events
    assert ("source", "beta.pdf::collection-a") in events
    assert ("markdown", "**Information Extraction Overview**") in events
    assert ("ner", 2) in events


def test_render_graph_debug_panel_uses_popover_with_scrolling_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GraphRAG debug details should render in a fixed-height popover.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
    """
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
        chat_module.st,
        "json",
        lambda payload: events.append(("json", payload)),
    )

    chat_module._render_graph_debug_panel({"enabled": True, "applied": False})

    assert ("popover", "GraphRAG Debug") in events
    assert ("container", chat_module.CHAT_DEBUG_CONTAINER_HEIGHT) in events
    assert ("json", {"enabled": True, "applied": False}) in events


def test_render_answer_tool_panels_aligns_buttons_horizontally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graph debug and source controls should share one horizontal row.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
    """
    events: list[tuple[str, object]] = []

    def _columns(count: int):
        events.append(("columns", count))
        return [nullcontext() for _ in range(count)]

    monkeypatch.setattr(chat_module.st, "columns", _columns)
    monkeypatch.setattr(
        chat_module,
        "_render_graph_debug_panel",
        lambda payload: events.append(("graph_debug", dict(payload))),
    )
    monkeypatch.setattr(
        chat_module,
        "_render_sources_panel",
        lambda sources, collection: events.append(
            ("sources", (len(sources), collection))
        ),
    )

    chat_module._render_answer_tool_panels(
        graph_debug={"enabled": True},
        sources=[{"filename": "alpha.pdf"}],
        collection="collection-a",
    )

    assert ("columns", 2) in events
    assert ("graph_debug", {"enabled": True}) in events
    assert ("sources", (1, "collection-a")) in events
