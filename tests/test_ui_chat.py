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
