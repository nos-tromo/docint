"""The search field whitelist and the payload indexes field search needs.

The panel's "Search in" picker chooses which payload field a query matches:
the chunk text by default, or one of a few reference-metadata keys (author,
author id, network, …). The list is a closed whitelist — an arbitrary payload
path would let a caller probe any field, and every entry here costs a payload
index.

Matching a metadata field the way the text is matched (case-insensitive,
prefix-based) needs the same ``TextIndexParams`` the ``search_text`` index
uses. Qdrant holds one index per field, and these keys used to carry KEYWORD
indexes for the old facet lane, so ``ensure_field_indexes`` replaces a
keyword index rather than failing on it.

Like ``fulltext.py`` and ``index.py`` this module imports nothing from
``core/rag.py``; the client is injected.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from docint.core.search.index import SEARCH_TEXT_FIELD, search_index_params

#: Picker name → payload key. ``"text"`` is first and is the default.
SEARCH_FIELDS: dict[str, str] = {
    "text": SEARCH_TEXT_FIELD,
    "author": "reference_metadata.author",
    "author_id": "reference_metadata.author_id",
    "network": "reference_metadata.network",
    "posting_author": "reference_metadata.posting_author",
    "type": "reference_metadata.type",
    "speaker": "reference_metadata.speaker",
    "language": "reference_metadata.language",
    "file_name": "file_name",
}

DEFAULT_SEARCH_FIELD = "text"

#: Fields whose key an ``_images`` companion point can carry. An image point
#: has a caption in ``search_text`` and, for social media, the parent
#: posting's ``reference_metadata.posting_*`` fields and its ``type``; it has
#: no speaker, language, document author or ``file_name`` key, so searching
#: those fields runs the text lane only.
IMAGE_LANE_FIELDS: frozenset[str] = frozenset({"text", "posting_author", "type"})


class UnknownSearchFieldError(ValueError):
    """Raised when a field name is not in :data:`SEARCH_FIELDS`."""


def search_payload_key(name: str) -> str:
    """Resolve a picker name to its payload key.

    Args:
        name (str): A key of :data:`SEARCH_FIELDS`.

    Returns:
        str: The Qdrant payload path the query matches against.

    Raises:
        UnknownSearchFieldError: When ``name`` is not whitelisted.
    """
    try:
        return SEARCH_FIELDS[name]
    except KeyError as exc:
        raise UnknownSearchFieldError(name) from exc


def field_index_kind(client: Any, collection: str, key: str) -> str | None:
    """Report which payload index, if any, a key carries.

    Args:
        client (Any): Qdrant client exposing ``get_collection``.
        collection (str): Physical collection name.
        key (str): Payload key.

    Returns:
        str | None: The schema's data type lowercased (``"keyword"``,
            ``"text"``, …), or ``None`` when the key is unindexed or Qdrant
            is unreachable — an unknown state must never read as indexed.
    """
    try:
        info = client.get_collection(collection_name=collection)
    except Exception as exc:
        logger.debug("Payload index status unavailable for {} on {}: {}", key, collection, exc)
        return None
    schema = getattr(info, "payload_schema", None) or {}
    entry = schema.get(key)
    if entry is None:
        return None
    data_type = getattr(entry, "data_type", None)
    value = getattr(data_type, "value", data_type)
    return None if value is None else str(value).lower()


def ensure_field_indexes(client: Any, collection: str) -> bool:
    """Ensure every metadata search field carries a prefix/lowercase TEXT index.

    Idempotent — an existing TEXT index is left alone — and fail-soft. A
    KEYWORD index left behind by the old grouped lane is deleted first, since
    Qdrant holds one index per field. ``wait=True`` so a search issued right
    after does not race the rebuild. ``search_text`` itself is owned by
    :func:`docint.core.search.index.ensure_search_index` and is not touched.

    Args:
        client (Any): Qdrant client exposing ``get_collection``,
            ``create_payload_index`` and ``delete_payload_index``.
        collection (str): Physical collection name.

    Returns:
        bool: ``True`` when every field ends up with a TEXT index.
    """
    ok = True
    for name, key in SEARCH_FIELDS.items():
        if name == DEFAULT_SEARCH_FIELD:
            continue
        try:
            kind = field_index_kind(client, collection, key)
            if kind == "text":
                continue
            if kind is not None:
                client.delete_payload_index(collection_name=collection, field_name=key, wait=True)
            client.create_payload_index(
                collection_name=collection,
                field_name=key,
                field_schema=search_index_params(),
                wait=True,
            )
        except Exception as exc:
            ok = False
            logger.debug("Field index on '{}' for {} skipped: {}", collection, key, exc)
    return ok
