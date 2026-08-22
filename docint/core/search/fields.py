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


def _index_kind_from_schema(schema: Any, key: str) -> str | None:
    """Extract one payload key's indexed data type from a schema mapping.

    Factored out of :func:`field_index_kind` so :func:`ensure_field_indexes`
    can read a collection's schema once and check every key against it,
    rather than paying a ``get_collection`` round-trip per key.

    Args:
        schema (Any): The ``payload_schema`` mapping from ``get_collection``,
            or a falsy value when it could not be read.
        key (str): Payload key.

    Returns:
        str | None: The schema's data type lowercased (``"keyword"``,
            ``"text"``, …), or ``None`` when the key is unindexed or the
            schema is unavailable — an unknown state must never read as
            indexed.
    """
    entry = (schema or {}).get(key)
    if entry is None:
        return None
    data_type = getattr(entry, "data_type", None)
    value = getattr(data_type, "value", data_type)
    return None if value is None else str(value).lower()


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
    return _index_kind_from_schema(schema, key)


def ensure_field_indexes(client: Any, collection: str) -> bool:
    """Ensure every metadata search field carries a prefix/lowercase TEXT index.

    Idempotent — an existing TEXT index is left alone — and fail-soft. A
    KEYWORD index left behind by the old grouped lane is deleted first, since
    Qdrant holds one index per field. ``wait=True`` so a search issued right
    after does not race the rebuild. ``search_text`` itself is owned by
    :func:`docint.core.search.index.ensure_search_index` and is not touched.

    The collection's schema is read once, up front, and reused for every
    key — checking it via :func:`field_index_kind` inside the loop would cost
    one ``get_collection`` round-trip per key instead of one per call. A
    schema that cannot be read is treated exactly like an empty one, so every
    key is still attempted rather than skipped as if already indexed.

    Args:
        client (Any): Qdrant client exposing ``get_collection``,
            ``create_payload_index`` and ``delete_payload_index``.
        collection (str): Physical collection name.

    Returns:
        bool: ``True`` when every field ends up with a TEXT index.
    """
    try:
        info = client.get_collection(collection_name=collection)
        schema: dict[str, Any] = getattr(info, "payload_schema", None) or {}
    except Exception as exc:
        logger.debug("Payload index status unavailable for {}: {}", collection, exc)
        schema = {}

    ok = True
    for name, key in SEARCH_FIELDS.items():
        if name == DEFAULT_SEARCH_FIELD:
            continue
        try:
            kind = _index_kind_from_schema(schema, key)
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
