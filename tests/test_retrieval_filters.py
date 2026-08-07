"""Unit tests for the request-scoped metadata filter translator."""

from __future__ import annotations

from typing import Any, cast

from llama_index.vector_stores.qdrant.base import QdrantVectorStore

from docint.core.retrieval_filters import (
    _coerce_rule,
    build_metadata_filters,
    build_qdrant_filter,
)


def _compiles_for_qdrant(rules: list[dict[str, Any]]) -> bool:
    """Return whether the LlamaIndex filters for ``rules`` survive the Qdrant vector store.

    ``QdrantVectorStore._build_subfilter`` is what turns a ``MetadataFilters``
    tree into a native Qdrant filter at query time. It is an instance method
    that never touches ``self``, so it is called unbound here to avoid
    constructing a client.

    Args:
        rules (list[dict[str, Any]]): Raw wire-format filter rules.

    Returns:
        bool: ``True`` when the compiled filters are acceptable to the vector
            store, including the vacuous case where nothing was compiled.
    """
    compiled = build_metadata_filters(rules)
    if compiled is None:
        return True
    QdrantVectorStore._build_subfilter(cast(Any, None), compiled)
    return True


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


def test_date_rules_emit_no_llama_index_filter() -> None:
    """Date bounds are only expressible natively; Range() rejects ISO strings."""
    rules = [
        {
            "field": "reference_metadata.timestamp",
            "operator": "date_on_or_after",
            "value": "2026-01-01",
        }
    ]

    assert build_metadata_filters(rules) is None
    assert _compiles_for_qdrant(rules)
    # The native path still carries the rule.
    assert build_qdrant_filter(rules) is not None


def test_contains_rules_emit_no_llama_index_filter() -> None:
    """QdrantVectorStore raises NotImplementedError for FilterOperator.CONTAINS."""
    rules = [{"field": "filename", "operator": "contains", "value": "quarterly"}]

    assert build_metadata_filters(rules) is None
    assert _compiles_for_qdrant(rules)
    assert build_qdrant_filter(rules) is not None


def test_non_numeric_range_rules_emit_no_llama_index_filter() -> None:
    """Qdrant Range bounds are floats, so a text bound cannot be compiled."""
    rules = [{"field": "section_path", "operator": "gte", "value": "chapter-two"}]

    assert build_metadata_filters(rules) is None
    assert _compiles_for_qdrant(rules)


def test_numeric_range_rules_still_compile() -> None:
    """A numeric bound is expressible and must survive."""
    rules = [{"field": "page_number", "operator": "gte", "value": 3}]

    compiled = build_metadata_filters(rules)

    assert compiled is not None
    assert _compiles_for_qdrant(rules)


def test_mime_rules_still_compile() -> None:
    """MIME matching is unaffected by the hardening."""
    rules = [{"field": "mimetype", "operator": "mime_match", "value": "image/*"}]

    compiled = build_metadata_filters(rules)

    assert compiled is not None
    assert _compiles_for_qdrant(rules)
