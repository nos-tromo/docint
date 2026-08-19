"""Keyword parsing and native filter construction for full-text search."""

from __future__ import annotations

from collections.abc import Sequence
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
    return zero hits there. Shared by keyword search and the grouped lane so
    both count a logical hit once.

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


def build_search_filter(
    keywords: Sequence[str],
    *,
    base_filter: models.Filter | None = None,
) -> models.Filter | None:
    """Compile keywords into a native Qdrant filter.

    One ``MatchText`` condition per keyword, all in ``must`` — so a chunk has
    to contain every keyword, in any order. Coarse parent chunks are excluded
    so a hierarchical collection does not return both a parent and its child
    for one logical hit; the exclusion is expressed as "not coarse" rather than
    "is fine" because a collection ingested without hierarchical chunking tags
    nothing at all, and requiring ``fine`` would return nothing there.

    Args:
        keywords (Sequence[str]): Keywords that must all match.
        base_filter (models.Filter | None): The caller's metadata filter, ANDed
            with the keyword conditions so panel filters constrain the search.

    Returns:
        models.Filter | None: The compiled filter, or ``None`` when there are no
            keywords — a keyword-less search must never degrade into an
            unfiltered scan of the whole collection.
    """
    if not keywords:
        return None

    conditions: list[Any] = [
        models.FieldCondition(key=SEARCH_TEXT_FIELD, match=models.MatchText(text=keyword)) for keyword in keywords
    ]
    conditions.append(not_coarse_condition())
    return merge_qdrant_filters(base_filter, conditions)
