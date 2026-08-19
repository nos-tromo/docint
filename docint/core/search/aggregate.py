"""Exhaustive, grouped search over chunk payload — the "find all" lane.

Vector retrieval returns the *best* few matches; an investigation often needs
*every* match, grouped: which authors, on which network, in which files. This
module compiles the same keyword + metadata filter the keyword lane uses and
asks Qdrant to facet it on a whitelisted payload key — a count, not a scan,
and no embedding or inference anywhere in the path.

Facet needs a KEYWORD payload index on the key. ``ensure_group_indexes``
creates them idempotently; the RAG layer calls it where the ``search_text``
index is ensured and once per collection before the first grouped call.

Like ``fulltext.py`` and ``index.py`` this module imports nothing from
``core/rag.py``; the client is injected.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger
from qdrant_client import models

from docint.core.retrieval_filters import merge_qdrant_filters
from docint.core.search.fulltext import build_search_filter, not_coarse_condition

#: Short group-by names → payload keys. A closed whitelist: faceting an
#: arbitrary payload path would let a caller enumerate any field, and every
#: entry here also gets a payload index, which is not free.
GROUP_BY_FIELDS: dict[str, str] = {
    "author": "reference_metadata.author",
    "author_id": "reference_metadata.author_id",
    "network": "reference_metadata.network",
    "posting_author": "reference_metadata.posting_author",
    "type": "reference_metadata.type",
    "speaker": "reference_metadata.speaker",
    "language": "reference_metadata.language",
    "file_name": "file_name",
}

DEFAULT_GROUP_LIMIT = 100
MAX_GROUP_LIMIT = 500
MAX_SAMPLES_PER_GROUP = 5


class UnknownGroupFieldError(ValueError):
    """Raised when a group-by name is not in :data:`GROUP_BY_FIELDS`."""


@dataclass(frozen=True)
class FacetGroup:
    """One facet bucket.

    Attributes:
        value (str): The payload value, stringified.
        count (int): Matching chunks carrying that value.
    """

    value: str
    count: int


def group_payload_key(name: str) -> str:
    """Resolve a short group-by name to its payload key.

    Args:
        name (str): A key of :data:`GROUP_BY_FIELDS`.

    Returns:
        str: The Qdrant payload path to facet on.

    Raises:
        UnknownGroupFieldError: When ``name`` is not whitelisted.
    """
    try:
        return GROUP_BY_FIELDS[name]
    except KeyError as exc:
        raise UnknownGroupFieldError(name) from exc


def build_group_filter(
    keywords: Sequence[str],
    *,
    base_filter: models.Filter | None,
) -> models.Filter:
    """Compile the filter every group and member query runs under.

    With keywords this is exactly the keyword lane's filter. Without them it is
    the caller's metadata filter plus the coarse-parent exclusion — a facet is
    a count, so a keyword-less call is legitimate here, but a hierarchical
    collection would otherwise count each logical hit twice.

    Args:
        keywords (Sequence[str]): Keywords that must all match; may be empty.
        base_filter (models.Filter | None): The caller's metadata filter.

    Returns:
        models.Filter: The compiled filter (never ``None``).
    """
    if keywords:
        compiled = build_search_filter(keywords, base_filter=base_filter)
        if compiled is not None:
            return compiled
    merged = merge_qdrant_filters(base_filter, [not_coarse_condition()])
    return merged if merged is not None else models.Filter(must=[not_coarse_condition()])


def ensure_group_indexes(client: Any, collection: str) -> bool:
    """Create the KEYWORD payload index every groupable key needs.

    Idempotent (Qdrant ignores a duplicate) and fail-soft. ``wait=True`` so a
    facet issued right after does not race the index build.

    Args:
        client (Any): Qdrant client exposing ``create_payload_index``.
        collection (str): Physical collection name.

    Returns:
        bool: ``True`` when every index call succeeded.
    """
    ok = True
    for key in GROUP_BY_FIELDS.values():
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=key,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception as exc:
            ok = False
            logger.debug("Group index on '{}' for {} skipped: {}", collection, key, exc)
    return ok


def facet_groups(
    client: Any,
    collection: str,
    key: str,
    *,
    group_filter: models.Filter | None,
    limit: int,
) -> list[FacetGroup]:
    """Count matching chunks per distinct value of ``key``.

    Args:
        client (Any): Qdrant client exposing ``facet``.
        collection (str): Physical collection name.
        key (str): Payload key (from :func:`group_payload_key`).
        group_filter (models.Filter | None): Filter the counts run under.
        limit (int): Maximum number of groups.

    Returns:
        list[FacetGroup]: Groups by count desc, then value asc.
    """
    response = client.facet(
        collection_name=collection,
        key=key,
        facet_filter=group_filter,
        limit=limit,
        exact=True,
    )
    groups = [FacetGroup(value=str(hit.value), count=int(hit.count)) for hit in response.hits]
    groups.sort(key=lambda g: (-g.count, g.value))
    return groups


def member_filter(key: str, value: str, base: models.Filter) -> models.Filter:
    """AND one group's value onto the group filter to fetch its members.

    Args:
        key (str): Payload key the groups were faceted on.
        value (str): The group's value.
        base (models.Filter): The compiled group filter.

    Returns:
        models.Filter: ``base`` plus a ``MatchValue`` on ``key``.
    """
    merged = merge_qdrant_filters(base, [models.FieldCondition(key=key, match=models.MatchValue(value=value))])
    assert merged is not None  # base is never None here
    return merged
