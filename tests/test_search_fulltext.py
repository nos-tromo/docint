"""Unit tests for full-text keyword parsing and filter construction."""

from __future__ import annotations

from typing import Any, cast

from qdrant_client import models

from docint.core.retrieval_filters import build_qdrant_filter
from docint.core.search.fulltext import (
    build_scan_filter,
    build_search_filter,
    matches_phrase,
    not_coarse_condition,
    parse_keywords,
)
from docint.core.search.index import SEARCH_TEXT_FIELD


def test_parse_keywords_splits_on_whitespace() -> None:
    """Two keywords means two AND conditions, order-independent."""
    assert parse_keywords("  Berlin   Konferenz ") == ["Berlin", "Konferenz"]


def test_parse_keywords_deduplicates_case_insensitively() -> None:
    """The index lowercases, so a repeated keyword adds nothing but cost."""
    assert parse_keywords("Berlin berlin BERLIN") == ["Berlin"]


def test_parse_keywords_drops_short_words_silently() -> None:
    """A one-character keyword is unindexable but valid inside a phrase.

    Short words are dropped from the keyword list (Qdrant pre-filter) so
    they don't contribute a condition that can never match. The phrase
    post-filter still checks the full query text.
    """
    assert parse_keywords("Berlin a") == ["Berlin"]


def test_parse_keywords_returns_nothing_when_all_keywords_are_too_short() -> None:
    """Every word below the index minimum yields an empty list."""
    assert parse_keywords("a b c") == []


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


# ---------- matches_phrase ----------


def test_matches_phrase_single_keyword_always_matches() -> None:
    """A single keyword has no adjacency to check."""
    assert matches_phrase("anything at all", ["Berlin"])


def test_matches_phrase_matches_contiguous_text() -> None:
    """Adjacent keywords in the text are a phrase match."""
    assert matches_phrase("This is about machine learning today", ["machine", "learning"])


def test_matches_phrase_rejects_non_contiguous() -> None:
    """Words present but separated by other text are not a phrase."""
    assert not matches_phrase("The machine stopped learning", ["machine", "learning"])


def test_matches_phrase_is_case_insensitive() -> None:
    """Mixed case in the text must not prevent a match."""
    assert matches_phrase("Machine Learning is great", ["machine", "learning"])


def test_matches_phrase_normalizes_whitespace() -> None:
    """Extra whitespace between words should not break phrase matching."""
    assert matches_phrase("about  machine\tlearning  today", ["machine", "learning"])


def test_matches_phrase_empty_keywords_always_matches() -> None:
    """No keywords means no phrase constraint."""
    assert matches_phrase("anything", [])


def test_matches_phrase_requires_keyword_order() -> None:
    """Reversed keyword order must not match."""
    assert not matches_phrase("learning machine", ["machine", "learning"])


# ---------- build_search_filter field_key parameter ----------


def test_build_search_filter_targets_the_given_field_key() -> None:
    """A field search puts the MatchText conditions on that key, not on search_text."""
    f = build_search_filter(["mar"], field_key="reference_metadata.author")
    assert f is not None
    conditions = [c for c in cast(list[Any], f.must or []) if isinstance(c, models.FieldCondition)]
    assert [c.key for c in conditions] == ["reference_metadata.author"]
    assert conditions[0].match == models.MatchText(text="mar")


def test_build_search_filter_defaults_to_search_text() -> None:
    """Callers that pass no key keep searching the chunk text."""
    f = build_search_filter(["election"])
    assert f is not None
    conditions = [c for c in cast(list[Any], f.must or []) if isinstance(c, models.FieldCondition)]
    assert [c.key for c in conditions] == [SEARCH_TEXT_FIELD]


# ---------- build_scan_filter ----------


def test_build_scan_filter_excludes_coarse_parents_without_keywords() -> None:
    """A blank-query export scans the collection, minus the coarse parents."""
    f = build_scan_filter(None)
    assert f.must == [not_coarse_condition()]


def test_build_scan_filter_keeps_the_callers_metadata_filter() -> None:
    """The metadata filter still narrows a keyword-less scan."""
    base = models.Filter(
        must=[models.FieldCondition(key="reference_metadata.network", match=models.MatchValue(value="Instagram"))]
    )
    f = build_scan_filter(base)
    assert cast(list[Any], f.must)[0] == cast(list[Any], base.must)[0]
    assert not_coarse_condition() in cast(list[Any], f.must)
