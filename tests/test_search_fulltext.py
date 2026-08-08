"""Unit tests for full-text keyword parsing and filter construction."""

from __future__ import annotations

from typing import Any, cast

import pytest
from qdrant_client import models

from docint.core.retrieval_filters import build_qdrant_filter
from docint.core.search.fulltext import (
    KeywordTooShortError,
    build_search_filter,
    parse_keywords,
)
from docint.core.search.index import SEARCH_TEXT_FIELD


def test_parse_keywords_splits_on_whitespace() -> None:
    """Two keywords means two AND conditions, order-independent."""
    assert parse_keywords("  Berlin   Konferenz ") == ["Berlin", "Konferenz"]


def test_parse_keywords_deduplicates_case_insensitively() -> None:
    """The index lowercases, so a repeated keyword adds nothing but cost."""
    assert parse_keywords("Berlin berlin BERLIN") == ["Berlin"]


def test_parse_keywords_rejects_a_keyword_below_the_index_minimum() -> None:
    """A one-character keyword is unindexable and can never match.

    Accepting it would contribute a condition that matches nothing, so a
    two-keyword search would silently return zero hits.
    """
    with pytest.raises(KeywordTooShortError) as excinfo:
        parse_keywords("Berlin a")

    assert "a" in str(excinfo.value)


def test_parse_keywords_returns_nothing_for_blank_input() -> None:
    """An empty query is not an error — it is simply no search."""
    assert parse_keywords("   ") == []


def _keys(conditions: Any) -> list[str]:
    """Return the payload key of each condition in a filter clause."""
    return [cast(Any, condition).key for condition in (conditions or [])]


def test_build_search_filter_ands_one_condition_per_keyword() -> None:
    """Every keyword must match the same chunk."""
    compiled = build_search_filter(["berlin", "konferenz"])

    assert compiled is not None
    text_conditions = [c for c in (compiled.must or []) if isinstance(c, models.FieldCondition)]
    assert _keys(text_conditions) == [SEARCH_TEXT_FIELD, SEARCH_TEXT_FIELD]


def test_build_search_filter_excludes_coarse_parent_chunks() -> None:
    """A coarse parent and its fine child both contain the keyword.

    Excluding ``coarse`` rather than requiring ``fine`` is deliberate: a
    collection ingested without hierarchical chunking tags nothing, and
    requiring ``fine`` would return zero hits there.
    """
    compiled = build_search_filter(["berlin"])

    assert compiled is not None
    nested = [c for c in (compiled.must or []) if isinstance(c, models.Filter)]
    assert len(nested) == 1
    assert _keys(nested[0].must_not) == ["docint_hier_type"]


def test_build_search_filter_merges_the_callers_metadata_filter() -> None:
    """Panel filters must constrain the search, not be ignored by it."""
    base = build_qdrant_filter([{"field": "mimetype", "operator": "eq", "value": "text/plain"}])

    compiled = build_search_filter(["berlin"], base_filter=base)

    assert compiled is not None
    keys = _keys([c for c in (compiled.must or []) if isinstance(c, models.FieldCondition)])
    assert "mimetype" in keys
    assert SEARCH_TEXT_FIELD in keys


def test_build_search_filter_returns_none_without_keywords() -> None:
    """No keywords means no search — never an unfiltered scan of everything."""
    assert build_search_filter([]) is None
