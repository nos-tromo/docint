"""Tests for chat UI helper functions."""

from contextlib import nullcontext

import pytest

from docint.ui import chat as chat_module
from docint.ui.chat import (
    ChatStreamState,
    _entity_candidate_label,
    _format_graph_debug_summary,
    _iter_chat_stream_text,
    _parse_sse_payload,
    _prime_chat_stream,
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


def test_query_mode_badge_label_graph_modes() -> None:
    """Query mode badge should render graph analysis labels."""

    assert _query_mode_badge_label("graph_lookup") == "Graph Lookup Mode"
    assert _query_mode_badge_label("graph_path") == "Graph Path Mode"


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


def test_parse_sse_payload_reads_data_lines() -> None:
    """SSE parser should decode JSON payloads from backend data lines."""
    assert _parse_sse_payload(b'data: {"token":"Hello"}') == {"token": "Hello"}


def test_parse_sse_payload_ignores_invalid_lines() -> None:
    """SSE parser should ignore blanks, non-data lines, and bad JSON."""
    assert _parse_sse_payload(b"") is None
    assert _parse_sse_payload(b"event: token") is None
    assert _parse_sse_payload(b"data: not-json") is None


def test_chat_stream_state_accumulates_text_and_metadata() -> None:
    """Stream state should retain rendered answer text and final metadata."""
    state = ChatStreamState()

    assert state.apply_payload({"token": "Hello"}) == "Hello"
    assert state.apply_payload({"token": " world"}) == " world"
    assert (
        state.apply_payload(
            {
                "session_id": "sess-1",
                "sources": [{"filename": "alpha.pdf"}],
                "reasoning": "Grounded in sources.",
                "validation_checked": True,
                "validation_mismatch": False,
                "validation_reason": None,
                "graph_debug": {"enabled": True},
                "retrieval_query": "rewritten::hello",
                "coverage_unit": "documents",
                "retrieval_mode": "rewrite_compact_graph",
                "entity_match_candidates": [{"text": "Acme"}],
                "entity_match_groups": [{"entity": {"text": "Acme"}}],
            }
        )
        is None
    )

    assert state.full_answer == "Hello world"
    assert state.session_id == "sess-1"
    assert state.sources == [{"filename": "alpha.pdf"}]
    assert state.reasoning == "Grounded in sources."
    assert state.validation_checked is True
    assert state.validation_mismatch is False
    assert state.graph_debug == {"enabled": True}
    assert state.retrieval_query == "rewritten::hello"
    assert state.coverage_unit == "documents"
    assert state.retrieval_mode == "rewrite_compact_graph"
    assert state.entity_match_candidates == [{"text": "Acme"}]
    assert state.entity_match_groups == [{"entity": {"text": "Acme"}}]


def test_iter_chat_stream_text_yields_tokens_and_invokes_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stream iterator should emit incremental text and surface state updates.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
    """
    monkeypatch.setattr(chat_module.st, "session_state", {"chat_running": True})
    state = ChatStreamState()
    snapshots: list[tuple[str, str | None]] = []

    tokens = list(
        _iter_chat_stream_text(
            [
                b'data: {"token":"Hello"}',
                b'data: {"token":" world"}',
                b'data: {"session_id":"sess-2"}',
            ],
            state,
            on_update=lambda current: snapshots.append(
                (current.full_answer, current.session_id)
            ),
        )
    )

    assert tokens == ["Hello", " world"]
    assert state.full_answer == "Hello world"
    assert state.session_id == "sess-2"
    assert snapshots == [
        ("Hello", None),
        ("Hello world", None),
        ("Hello world", "sess-2"),
    ]


def test_iter_chat_stream_text_uses_final_answer_when_no_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stream iterator should still render answers carried only in the final payload.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
    """
    monkeypatch.setattr(chat_module.st, "session_state", {"chat_running": True})
    state = ChatStreamState()

    tokens = list(
        _iter_chat_stream_text(
            [b'data: {"answer":"Full answer","session_id":"sess-3"}'],
            state,
        )
    )

    assert tokens == ["Full answer"]
    assert state.full_answer == "Full answer"
    assert state.session_id == "sess-3"


def test_prime_chat_stream_consumes_metadata_before_first_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Priming should keep the spinner up until a renderable chunk arrives."""
    monkeypatch.setattr(chat_module.st, "session_state", {"chat_running": True})
    state = ChatStreamState()
    snapshots: list[tuple[str, str | None]] = []

    first_chunk, remaining = _prime_chat_stream(
        [
            b'data: {"session_id":"sess-4"}',
            b'data: {"token":"Hello"}',
            b'data: {"token":" world"}',
        ],
        state,
        on_update=lambda current: snapshots.append(
            (current.full_answer, current.session_id)
        ),
    )

    assert first_chunk == "Hello"
    assert state.full_answer == "Hello"
    assert state.session_id == "sess-4"
    assert list(remaining) == [b'data: {"token":" world"}']
    assert snapshots == [("", "sess-4"), ("Hello", "sess-4")]
