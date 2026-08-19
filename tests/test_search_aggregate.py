"""Unit tests for the exhaustive (grouped) search lane."""

from __future__ import annotations

from typing import Any, cast

import pytest
from qdrant_client import models

from docint.core.search.aggregate import (
    GROUP_BY_FIELDS,
    FacetGroup,
    UnknownGroupFieldError,
    build_group_filter,
    ensure_group_indexes,
    facet_groups,
    group_payload_key,
    member_filter,
)
from docint.core.search.fulltext import SEARCH_TEXT_FIELD, not_coarse_condition


class _FakeClient:
    """Records index and facet calls without a server."""

    def __init__(
        self,
        *,
        facet_hits: list[tuple[bool | int | str, int]] | None = None,
        fail: bool = False,
    ) -> None:
        self.fail = fail
        self.index_calls: list[dict[str, Any]] = []
        self.facet_calls: list[dict[str, Any]] = []
        self._facet_hits = facet_hits or []

    def create_payload_index(self, **kwargs: Any) -> None:
        if self.fail:
            raise RuntimeError("qdrant unreachable")
        self.index_calls.append(kwargs)

    def facet(self, **kwargs: Any) -> Any:
        if self.fail:
            raise RuntimeError("qdrant unreachable")
        self.facet_calls.append(kwargs)
        hits = [models.FacetValueHit(value=v, count=c) for v, c in self._facet_hits]
        return models.FacetResponse(hits=hits)


def _must_conditions(f: models.Filter) -> list[Any]:
    """Return a filter's ``must`` conditions as a plain list for assertions.

    ``models.Filter.must`` is typed as ``list[Condition] | Condition | None``
    on the wire model, so tests that index or measure it need a narrowed view
    rather than raw attribute access.
    """
    return cast(list[Any], f.must or [])


def test_group_payload_key_maps_short_names_to_payload_paths() -> None:
    """Each short name resolves to its dotted payload path."""
    assert group_payload_key("author") == "reference_metadata.author"
    assert group_payload_key("network") == "reference_metadata.network"
    assert group_payload_key("file_name") == "file_name"


def test_group_payload_key_rejects_unknown_names() -> None:
    """A free-form key would let a caller facet arbitrary payload fields."""
    with pytest.raises(UnknownGroupFieldError):
        group_payload_key("reference_metadata.author")


def test_build_group_filter_with_keywords_matches_search_filter_semantics() -> None:
    """With keywords, the group filter is exactly the keyword lane's filter."""
    f = build_group_filter(["election"], base_filter=None)
    keys = [c.key for c in _must_conditions(f) if isinstance(c, models.FieldCondition)]
    assert keys == [SEARCH_TEXT_FIELD]
    assert any(isinstance(c, models.Filter) and c.must_not for c in _must_conditions(f))


def test_build_group_filter_without_keywords_still_excludes_coarse_parents() -> None:
    """A keyword-less aggregate is a facet over the filtered collection, not a scan.

    Coarse parents would otherwise double-count every hierarchical chunk.
    """
    f = build_group_filter([], base_filter=None)
    assert f.must == [not_coarse_condition()]


def test_build_group_filter_keeps_the_callers_metadata_filter() -> None:
    """A keyword-less aggregate still respects the caller's metadata filter."""
    base = models.Filter(
        must=[models.FieldCondition(key="reference_metadata.network", match=models.MatchValue(value="Instagram"))]
    )
    f = build_group_filter([], base_filter=base)
    assert _must_conditions(f)[0] == _must_conditions(base)[0]


def test_ensure_group_indexes_creates_one_keyword_index_per_field() -> None:
    """One KEYWORD index is created per groupable field, waited on."""
    client = _FakeClient()
    assert ensure_group_indexes(client, "col") is True
    created = {c["field_name"] for c in client.index_calls}
    assert created == set(GROUP_BY_FIELDS.values())
    assert all(c["field_schema"] == models.PayloadSchemaType.KEYWORD for c in client.index_calls)
    assert all(c["wait"] is True for c in client.index_calls)


def test_ensure_group_indexes_is_fail_soft() -> None:
    """A Qdrant outage degrades to False rather than raising."""
    assert ensure_group_indexes(_FakeClient(fail=True), "col") is False


def test_facet_groups_returns_groups_sorted_by_count_then_value() -> None:
    """Groups sort by count descending, then value ascending on ties."""
    client = _FakeClient(facet_hits=[("b_news", 5), ("a_news", 5), ("c_news", 9)])
    groups = facet_groups(client, "col", "reference_metadata.author", group_filter=None, limit=10)
    assert groups == [FacetGroup("c_news", 9), FacetGroup("a_news", 5), FacetGroup("b_news", 5)]
    call = client.facet_calls[0]
    assert call["collection_name"] == "col"
    assert call["key"] == "reference_metadata.author"
    assert call["exact"] is True
    assert call["limit"] == 10


def test_facet_groups_stringifies_non_string_values() -> None:
    """A non-string payload value (e.g. a year) still yields a str value."""
    client = _FakeClient(facet_hits=[(2024, 3)])
    assert facet_groups(client, "col", "reference_metadata.type", group_filter=None, limit=5) == [FacetGroup("2024", 3)]


def test_member_filter_ands_the_group_value_onto_the_base() -> None:
    """Selecting a group ANDs its value onto the group filter."""
    base = build_group_filter(["election"], base_filter=None)
    f = member_filter("reference_metadata.author", "acme_news", base)
    last = _must_conditions(f)[-1]
    assert isinstance(last, models.FieldCondition)
    assert last.key == "reference_metadata.author"
    assert last.match == models.MatchValue(value="acme_news")
    assert len(_must_conditions(f)) == len(_must_conditions(base)) + 1
