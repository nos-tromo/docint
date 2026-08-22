"""Unit tests for full-text keyword parsing and filter construction."""

from __future__ import annotations

from typing import Any, cast

from qdrant_client import models

from docint.core.retrieval_filters import build_qdrant_filter
from docint.core.search.fulltext import (
    build_scan_filter,
    build_search_filter,
    matches_any_phrase,
    matches_phrase,
    not_coarse_condition,
    parse_keywords,
    value_match_forms,
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


def test_build_search_filter_targets_the_given_text_keys() -> None:
    """A field search puts the MatchText conditions on that key, not on search_text."""
    f = build_search_filter(["mar"], text_keys=("reference_metadata.author",))
    assert f is not None
    conditions = [c for c in cast(list[Any], f.must or []) if isinstance(c, models.FieldCondition)]
    assert [c.key for c in conditions] == ["reference_metadata.author"]
    assert conditions[0].match == models.MatchText(text="mar")


def test_build_search_filter_defaults_to_search_text() -> None:
    """Callers that pass no keys keep searching the chunk text."""
    f = build_search_filter(["election"])
    assert f is not None
    conditions = [c for c in cast(list[Any], f.must or []) if isinstance(c, models.FieldCondition)]
    assert [c.key for c in conditions] == [SEARCH_TEXT_FIELD]


# ---------- several keys behind one picker option ----------


def _alternatives(compiled: models.Filter) -> list[Any]:
    """Return the OR-branches of a multi-key filter."""
    nested = [c for c in cast(list[Any], compiled.must or []) if isinstance(c, models.Filter) and c.should]
    assert len(nested) == 1, "expected exactly one should-clause"
    return cast(list[Any], nested[0].should)


def _first_condition(branch: Any) -> Any:
    """Return a branch's first ``must`` condition, narrowed for the type checker."""
    return cast(list[Any], branch.must)[0]


def test_several_text_keys_become_alternatives_not_requirements() -> None:
    """A hit needs the query in ONE of the keys, not in all of them."""
    compiled = build_search_filter(
        ["krieger"],
        text_keys=("reference_metadata.author", "reference_metadata.vanity"),
    )
    assert compiled is not None
    branches = _alternatives(compiled)
    assert [_first_condition(b).key for b in branches] == [
        "reference_metadata.author",
        "reference_metadata.vanity",
    ]


def test_every_keyword_stays_bound_to_a_single_key() -> None:
    """The whole query must land in one key, or a name splits across two fields.

    Otherwise "Wolfgang Krieger" could match a chunk whose author is Wolfgang
    and whose vanity handle happens to contain Krieger — two different people
    reported as one hit.
    """
    compiled = build_search_filter(
        ["wolfgang", "krieger"],
        text_keys=("reference_metadata.author", "reference_metadata.vanity"),
    )
    assert compiled is not None
    for branch in _alternatives(compiled):
        conditions = cast(list[Any], branch.must)
        assert len(conditions) == 2
        assert len({c.key for c in conditions}) == 1


def test_value_keys_match_exactly_in_both_number_and_string_form() -> None:
    """Author ids are ints in Qdrant but strings in other collections; try both."""
    compiled = build_search_filter(
        ["100007940942252"],
        text_keys=("reference_metadata.author",),
        value_keys=("reference_metadata.author_id",),
        value_forms=value_match_forms("100007940942252"),
    )
    assert compiled is not None
    matches = [
        _first_condition(b).match for b in _alternatives(compiled) if _first_condition(b).key.endswith("author_id")
    ]
    assert models.MatchValue(value=100007940942252) in matches
    assert models.MatchValue(value="100007940942252") in matches


def test_value_keys_are_skipped_when_the_query_cannot_be_an_id() -> None:
    """A multi-word query is a name, never an identifier."""
    compiled = build_search_filter(
        ["wolfgang", "krieger"],
        text_keys=("reference_metadata.author",),
        value_keys=("reference_metadata.author_id",),
        value_forms=value_match_forms("wolfgang krieger"),
    )
    assert compiled is not None
    # With the id keys gone there is only one alternative left, so the filter
    # compiles flat — assert on what it targets, not on its shape.
    assert "author_id" not in repr(compiled)


def test_multi_key_filter_still_excludes_coarse_parents_and_keeps_base_filter() -> None:
    """The OR-clause narrows what matches; it must not drop the other guards."""
    base = build_qdrant_filter([{"field": "mimetype", "operator": "eq", "value": "text/plain"}])
    compiled = build_search_filter(
        ["krieger"],
        text_keys=("reference_metadata.author", "reference_metadata.vanity"),
        base_filter=base,
    )
    assert compiled is not None
    must = cast(list[Any], compiled.must)
    assert any(isinstance(c, models.Filter) and c.must_not for c in must)
    assert "mimetype" in [c.key for c in must if isinstance(c, models.FieldCondition)]


# ---------- value_match_forms ----------


def test_value_match_forms_offers_the_numeric_and_string_reading() -> None:
    """A digit-only query could be stored either way."""
    assert value_match_forms("100007940942252") == ["100007940942252", 100007940942252]


def test_value_match_forms_keeps_a_non_numeric_handle_as_a_string() -> None:
    """Not every network numbers its accounts."""
    assert value_match_forms("krieger.advokat") == ["krieger.advokat"]


def test_value_match_forms_rejects_anything_with_a_space() -> None:
    """An identifier is one token; a phrase is a name."""
    assert value_match_forms("wolfgang krieger") == []
    assert value_match_forms("   ") == []


# ---------- matches_any_phrase ----------


def test_matches_any_phrase_passes_when_one_value_holds_the_phrase() -> None:
    """The phrase has to be contiguous in a single field's value."""
    assert matches_any_phrase(["Wolfgang Krieger", "krieger.advokat"], ["wolfgang", "krieger"]) is True


def test_matches_any_phrase_rejects_a_phrase_split_across_values() -> None:
    """Half the name in one field and half in another is not a match."""
    assert matches_any_phrase(["Wolfgang Berger", "krieger.advokat"], ["wolfgang", "krieger"]) is False


def test_matches_any_phrase_ignores_missing_values() -> None:
    """Most points carry only some of a field's keys."""
    assert matches_any_phrase(["", "Wolfgang Krieger"], ["wolfgang", "krieger"]) is True


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
