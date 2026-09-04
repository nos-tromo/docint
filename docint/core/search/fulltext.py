"""Keyword parsing and native filter construction for full-text search."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from qdrant_client import models

from docint.core.retrieval_filters import merge_qdrant_filters
from docint.core.search.index import SEARCH_MIN_TOKEN_LEN, SEARCH_TEXT_FIELD

#: Payload key marking a node's place in the coarse/fine hierarchy.
_HIER_TYPE_FIELD = "docint_hier_type"

#: Hierarchy value for an oversize parent chunk kept for context expansion.
_HIER_COARSE = "coarse"


class KeywordTooShortError(ValueError):
    """Raised when a keyword is shorter than the index can tokenize."""


def parse_keywords(raw: str) -> list[str]:
    """Split a raw query into the keywords that must all match.

    Args:
        raw (str): Raw query text as typed.

    Returns:
        list[str]: Keywords in input order, deduplicated case-insensitively
            because the index lowercases anyway.

    Raises:
        KeywordTooShortError: Legacy — no longer raised since short words are
            silently dropped (they are valid inside a phrase even when the
            index cannot tokenize them on their own).
    """
    keywords: list[str] = []
    seen: set[str] = set()
    for token in str(raw or "").split():
        keyword = token.strip()
        if not keyword or len(keyword) < SEARCH_MIN_TOKEN_LEN:
            continue
        folded = keyword.lower()
        if folded in seen:
            continue
        seen.add(folded)
        keywords.append(keyword)
    return keywords


def matches_phrase(text: str, keywords: list[str]) -> bool:
    """Check whether *keywords* appear as a contiguous phrase in *text*.

    Args:
        text (str): Haystack — typically the ``search_text`` payload.
        keywords (list[str]): Keywords in query order.

    Returns:
        bool: ``True`` when the phrase occurs (case-insensitive,
            whitespace-normalized) or when there are fewer than two keywords
            (no phrase to check).
    """
    if len(keywords) <= 1:
        return True
    phrase = " ".join(keywords).lower()
    normalized = " ".join(text.lower().split())
    return phrase in normalized


def not_coarse_condition() -> models.Filter:
    """Return the condition that excludes coarse parent chunks.

    Expressed as "not coarse" rather than "is fine": a collection ingested
    without hierarchical chunking tags nothing, and requiring ``fine`` would
    return zero hits there. Shared by keyword search and the blank-query scan
    so both count a logical hit once.

    Returns:
        models.Filter: A ``must_not`` filter on the hierarchy tag.
    """
    return models.Filter(
        must_not=[
            models.FieldCondition(
                key=_HIER_TYPE_FIELD,
                match=models.MatchValue(value=_HIER_COARSE),
            )
        ]
    )


def build_scan_filter(base_filter: models.Filter | None) -> models.Filter:
    """Compile the filter a keyword-less scan runs under.

    The caller's metadata filter plus the coarse-parent exclusion, never
    ``None``: a blank-query export legitimately covers the whole (filtered)
    collection, but a hierarchical collection would otherwise yield each
    logical chunk twice, through its parent and its child.

    Args:
        base_filter (models.Filter | None): The caller's metadata filter.

    Returns:
        models.Filter: The compiled filter.
    """
    merged = merge_qdrant_filters(base_filter, [not_coarse_condition()])
    return merged if merged is not None else models.Filter(must=[not_coarse_condition()])


def value_match_forms(raw: str) -> list[Any]:
    """Return the exact values an identifier query should be tried as.

    Identifiers are not prose. The same logical author id is stored as an
    ``int`` in one collection and a ``str`` in another — a Facebook profile
    id arrives numeric, a handle-style id does not — and Qdrant's
    ``MatchValue`` is type-strict, so ``"123"`` will not match ``123``.
    Offering both readings is what lets one query work against either.

    A query containing whitespace is never an identifier, so it yields no
    forms at all and the id keys drop out of the compiled filter entirely.

    Args:
        raw (str): Raw query text as typed.

    Returns:
        list[Any]: The string form first, plus the integer form when the
            query is all digits; empty when the query is blank or is a
            multi-word phrase.
    """
    text = " ".join(str(raw or "").split())
    if not text or " " in text:
        return []
    forms: list[Any] = [text]
    if text.isdigit():
        forms.append(int(text))
    return forms


_UUID_HEX_LEN = 32


def uuid_match_forms(raw: str) -> list[Any]:
    """Return the exact values a posting-uuid query should be tried as.

    The CSV ``UUID`` column is copied into the payload verbatim, and the
    exports seen so far write it undashed — but a user may paste either
    style, and another source may store it dashed. Rather than guess which,
    try the raw paste and its dash-normalised twin: the stripped form when
    dashes are present, the canonical 8-4-4-4-12 form when it is 32 hex
    characters. Anything else is an opaque identifier and is tried as-is.

    A query containing whitespace is never an identifier and yields nothing,
    so a value-only field compiles to no filter at all for it.

    Args:
        raw (str): Raw query text as typed.

    Returns:
        list[Any]: The pasted form first, then its twin when one exists.
    """
    text = " ".join(str(raw or "").split())
    if not text or " " in text:
        return []
    forms: list[Any] = [text]
    if "-" in text:
        stripped = text.replace("-", "")
        if stripped and stripped != text:
            forms.append(stripped)
    elif len(text) == _UUID_HEX_LEN and all(c in "0123456789abcdefABCDEF" for c in text):
        forms.append(f"{text[:8]}-{text[8:12]}-{text[12:16]}-{text[16:20]}-{text[20:]}")
    return forms


def matches_any_phrase(values: Iterable[str], keywords: list[str]) -> bool:
    """Check whether *keywords* occur as a phrase in at least one value.

    The counterpart of the compiled filter's OR over a field's keys: a hit is
    legitimate when one key holds the whole phrase, and a phrase split across
    two keys — half a name in ``author``, half in ``vanity`` — is not a hit at
    all.

    Args:
        values (Iterable[str]): The point's value for each of the field's
            text keys; missing keys pass through as empty strings.
        keywords (list[str]): Keywords in query order.

    Returns:
        bool: ``True`` when any value contains the contiguous phrase.
    """
    return any(matches_phrase(value, keywords) for value in values)


def build_search_filter(
    keywords: Sequence[str],
    *,
    text_keys: Sequence[str] = (SEARCH_TEXT_FIELD,),
    value_keys: Sequence[str] = (),
    value_forms: Sequence[Any] = (),
    base_filter: models.Filter | None = None,
) -> models.Filter | None:
    """Compile keywords into a native Qdrant filter over one field's keys.

    A picker option can cover several payload keys, and the query has to be
    satisfied by **one** of them: every keyword ANDed within a key, the keys
    ORed against each other. Binding the whole query to a single key is what
    stops "Wolfgang Krieger" matching a chunk whose ``author`` is Wolfgang
    and whose unrelated ``vanity`` contains Krieger.

    ``text_keys`` are matched with ``MatchText`` (prefix, case-insensitive);
    ``value_keys`` hold identifiers and are matched exactly, once per form in
    ``value_forms`` — a text matcher cannot touch them, because they are
    numbers in Qdrant and a text index over a number indexes nothing.

    Coarse parent chunks are excluded so a hierarchical collection does not
    return both a parent and its child for one logical hit; the exclusion is
    expressed as "not coarse" rather than "is fine" because a collection
    ingested without hierarchical chunking tags nothing at all, and requiring
    ``fine`` would return nothing there.

    Args:
        keywords (Sequence[str]): Keywords that must all match.
        text_keys (Sequence[str]): Payload keys matched with ``MatchText``.
        value_keys (Sequence[str]): Payload keys matched exactly.
        value_forms (Sequence[Any]): Values to try against ``value_keys``,
            from :func:`value_match_forms`. Empty means the query cannot be
            an identifier, so those keys contribute nothing.
        base_filter (models.Filter | None): The caller's metadata filter, ANDed
            with the keyword conditions so panel filters constrain the search.

    Returns:
        models.Filter | None: The compiled filter, or ``None`` when there are no
            keywords — a keyword-less search must never degrade into an
            unfiltered scan of the whole collection.
    """
    if not keywords:
        return None

    alternatives: list[models.Filter] = [
        models.Filter(
            must=[models.FieldCondition(key=key, match=models.MatchText(text=keyword)) for keyword in keywords]
        )
        for key in text_keys
    ]
    alternatives.extend(
        models.Filter(must=[models.FieldCondition(key=key, match=models.MatchValue(value=form))])
        for key in value_keys
        for form in value_forms
    )
    if not alternatives:
        return None

    # One alternative needs no OR around it. Splicing its conditions in flat
    # keeps the common single-key search — every Text query — compiling to
    # exactly the filter it did before this field could span several keys.
    conditions: list[Any] = (
        list(alternatives[0].must or []) if len(alternatives) == 1 else [models.Filter(should=alternatives)]
    )
    conditions.append(not_coarse_condition())
    return merge_qdrant_filters(base_filter, conditions)


def build_any_keyword_filter(
    keywords: Sequence[str],
    *,
    text_keys: Sequence[str] = (SEARCH_TEXT_FIELD,),
    min_match: int,
    base_filter: models.Filter | None = None,
) -> models.Filter | None:
    """Compile keywords into a filter satisfied by *some* of them.

    :func:`build_search_filter` ANDs every keyword, which is right for a
    search box: the user typed the terms they want. It is wrong for a chat
    question, which is a sentence — demanding that a one-line caption contain
    every content word of "which vehicles appear at the roadblock" matches
    nothing at all.

    So the conditions go into ``should`` under a ``min_should`` count, and the
    caller ranks the survivors by how many keywords they actually matched. A
    point matching all of them still ranks first; a point matching half is a
    candidate rather than a silent miss.

    Args:
        keywords (Sequence[str]): Keywords to match, in query order.
        text_keys (Sequence[str]): Payload keys matched with ``MatchText``.
        min_match (int): How many conditions a point must satisfy. Clamped to
            at least one and at most the number of conditions built.
        base_filter (models.Filter | None): Filter ANDed with the keyword
            conditions, so a request's own filters still constrain the lane.

    Returns:
        models.Filter | None: The compiled filter, or ``None`` when there are
            no keywords or no keys to match them against — a keyword-less
            lane must never degrade into an unfiltered scan.
    """
    if not keywords or not text_keys:
        return None

    conditions: list[Any] = [
        models.FieldCondition(key=key, match=models.MatchText(text=keyword))
        for key in text_keys
        for keyword in keywords
    ]
    if not conditions:
        return None

    # ``min_should`` is a nested filter rather than a flat condition:
    # ``merge_qdrant_filters`` composes ``must``/``must_not`` clauses and
    # would drop a ``min_should`` spliced in beside them.
    keyword_filter = models.Filter(
        min_should=models.MinShould(
            conditions=conditions,
            min_count=max(1, min(int(min_match), len(conditions))),
        )
    )
    return merge_qdrant_filters(base_filter, [keyword_filter])
