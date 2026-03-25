"""Deterministic helpers for canonical graph identity resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


def canonicalize_value(value: Any) -> str:
    """Normalize an arbitrary value into a stable lowercase identifier fragment.
    
    Args:
        value: The input value to normalize.

    Returns:
        str: The normalized lowercase identifier fragment.
    """

    normalized = re.sub(r"\s+", " ", str(value or "").strip()).casefold()
    normalized = re.sub(r"[^a-z0-9:/#@._ -]+", "", normalized)
    return normalized.strip(" -")


def extract_domain(url: str | None) -> str | None:
    """Extract a lowercase domain from a URL string.
    
    Args:
        url: The input URL string to extract the domain from.

    Returns:
        str | None: The extracted lowercase domain, or None if the URL is invalid.
    """

    if not url:
        return None
    parsed = urlparse(str(url))
    domain = parsed.netloc or parsed.path
    domain = domain.split("@")[-1].split(":")[0].strip().casefold()
    return domain or None


def parse_tag_values(raw: Any) -> list[str]:
    """Parse tags from scalar or list-like inputs.
    
    Args:
        raw: The raw input value, which can be a string, list, or other scalar type.

    Returns:
        list[str]: A list of canonicalized tag strings.
    """

    if raw is None:
        return []
    if isinstance(raw, list):
        values = raw
    else:
        values = re.split(r"[;,|]", str(raw))
    tags = [canonicalize_value(value).lstrip("#") for value in values]
    return [tag for tag in tags if tag]


@dataclass(frozen=True, slots=True)
class ResolvedIdentity:
    """A deterministic or deferred canonical identity resolution decision."""

    node_id: str
    canonical_key: str
    canonical_name: str
    scope: str
    auto_merged: bool
    candidate_key: str | None = None


def resolve_entity_identity(
    *,
    text: str,
    entity_type: str,
    collection: str,
    score: float | None,
    min_confidence: float,
) -> ResolvedIdentity:
    """Resolve an entity mention into a canonical or collection-scoped identity.
    
    Args:
        text: The raw text of the entity mention.
        entity_type: The type or category of the entity, if available.
        collection: The active collection name for scoping non-merged identities.
        score: The confidence score of the entity match, if available.
        min_confidence: The minimum confidence threshold for auto-merging entities into the global scope.

    Returns:
        ResolvedIdentity: A resolved identity object containing the canonical key, name, scope, and auto-merge status of the entity.
    """

    normalized_text = canonicalize_value(text)
    normalized_type = canonicalize_value(entity_type or "unlabeled") or "unlabeled"
    confidence = float(score if score is not None else 1.0)
    auto_merged = bool(normalized_text) and (
        confidence >= min_confidence and len(normalized_text) >= 4
    )
    scope = "global" if auto_merged else canonicalize_value(collection) or "collection"
    canonical_key = f"{normalized_type}:{scope}:{normalized_text}"
    candidate_key = None
    if not auto_merged and normalized_text:
        candidate_key = f"{normalized_type}:global:{normalized_text}"
    return ResolvedIdentity(
        node_id=canonical_key,
        canonical_key=canonical_key,
        canonical_name=str(text or "").strip(),
        scope=scope,
        auto_merged=auto_merged,
        candidate_key=candidate_key,
    )


def resolve_author_identity(
    *,
    author: str | None,
    author_id: str | None,
    collection: str,
) -> ResolvedIdentity | None:
    """Resolve an author/account identity with deterministic preference for author IDs.
    
    Args:
        author: The display name of the author or account, if available.
        author_id: The unique identifier of the author or account, if available.
        collection: The active collection name for scoping non-merged identities.

    Returns:
        ResolvedIdentity | None: A resolved identity object if a valid author or ID is provided, otherwise None.
    """

    raw_value = author_id or author
    normalized = canonicalize_value(raw_value)
    if not normalized:
        return None
    auto_merged = bool(author_id)
    scope = "global" if auto_merged else canonicalize_value(collection) or "collection"
    canonical_key = f"author:{scope}:{normalized}"
    candidate_key = None if auto_merged else f"author:global:{normalized}"
    return ResolvedIdentity(
        node_id=canonical_key,
        canonical_key=canonical_key,
        canonical_name=str(raw_value).strip(),
        scope=scope,
        auto_merged=auto_merged,
        candidate_key=candidate_key,
    )
