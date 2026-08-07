"""Unit tests for the request-scoped metadata filter translator."""

from __future__ import annotations

from docint.core.retrieval_filters import _coerce_rule


def test_coerce_rule_wraps_a_single_field_into_the_fields_list() -> None:
    """A legacy single-field rule must expose a one-element ``fields`` list."""
    rule = _coerce_rule({"field": "mimetype", "operator": "eq", "value": "text/plain"})

    assert rule is not None
    assert rule["fields"] == ["mimetype"]
    assert rule["field"] == "mimetype"


def test_coerce_rule_accepts_a_fields_list_without_a_field() -> None:
    """A multi-field rule needs no scalar ``field`` to be valid."""
    rule = _coerce_rule(
        {
            "fields": ["reference_metadata.timestamp", "reference_metadata.posting_timestamp"],
            "operator": "date_on_or_after",
            "value": "2026-01-01",
        }
    )

    assert rule is not None
    assert rule["fields"] == [
        "reference_metadata.timestamp",
        "reference_metadata.posting_timestamp",
    ]
    assert rule["field"] == "reference_metadata.timestamp"


def test_coerce_rule_drops_blank_entries_from_the_fields_list() -> None:
    """Blank and whitespace-only field names must not become filter keys."""
    rule = _coerce_rule({"fields": ["a.b", "", "   ", "c.d"], "operator": "eq", "value": "x"})

    assert rule is not None
    assert rule["fields"] == ["a.b", "c.d"]


def test_coerce_rule_rejects_a_rule_with_no_usable_target() -> None:
    """A rule naming neither a field nor any fields is not a rule."""
    assert _coerce_rule({"fields": ["", "  "], "operator": "eq", "value": "x"}) is None
    assert _coerce_rule({"operator": "eq", "value": "x"}) is None
