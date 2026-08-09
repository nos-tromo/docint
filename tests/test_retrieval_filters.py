"""Unit tests for the request-scoped metadata filter translator."""

from __future__ import annotations

from typing import Any, cast

from llama_index.vector_stores.qdrant.base import QdrantVectorStore
from qdrant_client import models

from docint.core.retrieval_filters import (
    _coerce_rule,
    build_metadata_filters,
    build_qdrant_filter,
    matches_metadata_filters,
    merge_qdrant_filters,
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


def _condition_keys(conditions: Any) -> list[str]:
    """Return the payload keys of a Qdrant condition list.

    ``must``/``must_not``/``should`` are typed as unions covering condition
    shapes that have no ``key`` (nested filters, has-id, is-empty), so the
    narrowing is done once here rather than at every assertion.

    Args:
        conditions (Any): A condition list from a ``models.Filter``, or ``None``.

    Returns:
        list[str]: Each condition's ``key``, in order.
    """
    return [cast(Any, condition).key for condition in (conditions or [])]


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


def test_multi_field_rule_matches_when_any_field_matches() -> None:
    """A media artifact carries posting_timestamp, not timestamp."""
    artifact = {"reference_metadata": {"posting_timestamp": "2026-03-10T09:00:00Z"}}
    chunk = {"reference_metadata": {"timestamp": "2026-03-10T09:00:00Z"}}
    rules = [
        {
            "fields": ["reference_metadata.timestamp", "reference_metadata.posting_timestamp"],
            "operator": "date_on_or_after",
            "value": "2026-03-01",
        }
    ]

    assert matches_metadata_filters(artifact, rules)
    assert matches_metadata_filters(chunk, rules)


def test_multi_field_rule_fails_when_no_field_matches() -> None:
    """Neither timestamp key satisfying the bound means no match."""
    payload = {"reference_metadata": {"timestamp": "2026-01-05T09:00:00Z"}}
    rules = [
        {
            "fields": ["reference_metadata.timestamp", "reference_metadata.posting_timestamp"],
            "operator": "date_on_or_after",
            "value": "2026-03-01",
        }
    ]

    assert not matches_metadata_filters(payload, rules)


def test_neq_on_a_missing_field_includes_the_payload() -> None:
    """``neq`` must mirror Qdrant's must_not, which passes an absent field.

    Qdrant routes ``neq`` into ``must_not``; a point lacking the key does not
    match the inner condition, so ``must_not`` passes and the point is kept.
    The in-memory predicate has to agree, or the image lane drops sources the
    text lane keeps.
    """
    assert matches_metadata_filters({"other": "x"}, [{"field": "absent", "operator": "neq", "value": "x"}])


def test_neq_across_fields_requires_every_field_to_differ() -> None:
    """Negation mirrors ``must_not=[Filter(should=[...])]`` — NOT (a OR b)."""
    rules = [{"fields": ["a", "b"], "operator": "neq", "value": "x"}]

    assert matches_metadata_filters({"a": "y", "b": "z"}, rules)
    assert not matches_metadata_filters({"a": "x", "b": "z"}, rules)


def test_multi_field_date_rule_compiles_to_a_nested_should_filter() -> None:
    """One rule over two timestamp keys becomes an OR group inside ``must``."""
    compiled = build_qdrant_filter(
        [
            {
                "fields": [
                    "reference_metadata.timestamp",
                    "reference_metadata.posting_timestamp",
                ],
                "operator": "date_on_or_after",
                "value": "2026-01-01",
            }
        ]
    )

    assert compiled is not None
    must: list[Any] = list(compiled.must or [])
    assert len(must) == 1
    group = must[0]
    assert isinstance(group, models.Filter)
    assert _condition_keys(group.should) == [
        "reference_metadata.timestamp",
        "reference_metadata.posting_timestamp",
    ]


def test_multi_field_negated_rule_wraps_the_group_in_must_not() -> None:
    """NOT (a OR b) is how "neither field holds this value" is expressed."""
    compiled = build_qdrant_filter([{"fields": ["a", "b"], "operator": "neq", "value": "x"}])

    assert compiled is not None
    assert not list(compiled.must or [])
    must_not: list[Any] = list(compiled.must_not or [])
    assert len(must_not) == 1
    assert isinstance(must_not[0], models.Filter)
    assert len(_condition_keys(cast(Any, must_not[0]).should)) == 2


def test_single_field_rule_stays_a_bare_condition() -> None:
    """One field must not gain a pointless nesting level."""
    compiled = build_qdrant_filter([{"field": "mimetype", "operator": "eq", "value": "text/plain"}])

    assert compiled is not None
    must: list[Any] = list(compiled.must or [])
    assert len(must) == 1
    assert isinstance(must[0], models.FieldCondition)


def test_merge_qdrant_filters_appends_to_an_existing_must() -> None:
    """Internal conditions must survive alongside a user filter."""
    base = build_qdrant_filter([{"field": "mimetype", "operator": "eq", "value": "text/plain"}])
    extra = [models.FieldCondition(key="docint_hier_type", match=models.MatchValue(value="fine"))]

    merged = merge_qdrant_filters(base, extra)

    assert merged is not None
    assert _condition_keys(merged.must) == ["mimetype", "docint_hier_type"]


def test_merge_qdrant_filters_preserves_must_not() -> None:
    """Merging must not discard negated user conditions."""
    base = build_qdrant_filter([{"field": "a", "operator": "neq", "value": "x"}])
    extra = [models.FieldCondition(key="docint_hier_type", match=models.MatchValue(value="fine"))]

    merged = merge_qdrant_filters(base, extra)

    assert merged is not None
    assert len(list(merged.must_not or [])) == 1
    assert _condition_keys(merged.must) == ["docint_hier_type"]


def test_merge_qdrant_filters_builds_a_filter_from_extras_alone() -> None:
    """With no user filter the internal conditions still need a filter."""
    merged = merge_qdrant_filters(
        None,
        [models.FieldCondition(key="docint_hier_type", match=models.MatchValue(value="fine"))],
    )

    assert merged is not None
    assert _condition_keys(merged.must) == ["docint_hier_type"]


def test_merge_qdrant_filters_returns_the_base_when_there_are_no_extras() -> None:
    """No internal conditions means nothing to merge."""
    base = build_qdrant_filter([{"field": "mimetype", "operator": "eq", "value": "text/plain"}])

    assert merge_qdrant_filters(base, []) is base
    assert merge_qdrant_filters(None, []) is None


def test_numeric_string_range_bounds_compile_on_every_path() -> None:
    """A range bound typed into a text input arrives as a string.

    The SPA's custom-rule value field is a plain text input, so "3" is what
    reaches the API. Dropping the rule would run the query unfiltered — the
    user narrowed and silently got everything.
    """
    rules = [{"field": "page_number", "operator": "gte", "value": "3"}]

    native = build_qdrant_filter(rules)
    assert native is not None
    condition = cast(Any, next(iter(native.must or [])))
    assert condition.range.gte == 3

    assert build_metadata_filters(rules) is not None
    assert _compiles_for_qdrant(rules)
    assert matches_metadata_filters({"page_number": 5}, rules)
    assert not matches_metadata_filters({"page_number": 1}, rules)


def test_non_numeric_range_bounds_compile_on_no_path() -> None:
    """Qdrant has no string range, so such a rule cannot be honoured at all.

    Nothing can express it, which is exactly why the API rejects it rather
    than letting it reach here and silently widen the result set.
    """
    rules = [{"field": "section_path", "operator": "gte", "value": "chapter-two"}]

    assert build_metadata_filters(rules) is None
    assert build_qdrant_filter(rules) is None
    assert _compiles_for_qdrant(rules)


def test_neq_emits_no_llama_index_filter() -> None:
    """Negation is carried natively so multi-field semantics stay consistent.

    ``_metadata_filter_for`` ORs a rule's fields, which is right for positive
    operators but inverts negation: ``NE(a) OR NE(b)`` is not
    ``NOT(a == x OR b == x)``. The native filter expresses the latter with
    ``must_not=[Filter(should=[...])]``, so ``neq`` is left to it.
    """
    rules = [{"fields": ["a", "b"], "operator": "neq", "value": "x"}]

    assert build_metadata_filters(rules) is None
    assert build_qdrant_filter(rules) is not None

    single = [{"field": "a", "operator": "neq", "value": "x"}]
    assert build_metadata_filters(single) is None
    assert build_qdrant_filter(single) is not None
