"""The search field whitelist and the payload indexes field search needs.

The panel's "Search in" picker chooses which payload field a query matches:
the chunk text by default, or one of a few reference-metadata keys. The list
is a closed whitelist — an arbitrary payload path would let a caller probe any
field, and every entry here costs payload indexes.

**One option can cover several keys, because one option is one question.**
"Everything this person posted" is a single investigative ask, but the answer
is spread over the posting's own author, their vanity handle, and — for an
image or transcript hanging off that posting — the parent's ``posting_*``
copies of both. Offering those as separate picker entries made the user pick
the right synonym before they could search.

**And one option can need two different matchers.** ``MatchText`` is a
full-text matcher: it works on strings and only on strings. Author *ids* are
stored as integers (a real collection holds ``author_id`` as ``int``), so a
TEXT index over that key indexes zero points and every id search returned
nothing at all. Ids therefore match by exact value while names match by
prefix — see :data:`FieldSpec.value_keys`. A posting uuid is value-only: it
is the sole identifier of a single posting artifact, and the same value is
what every image, keyframe and transcript segment derived from that posting
carries as its ``posting_uuid`` link, so one exact match returns the post and
everything hanging off it.

Like ``fulltext.py`` and ``index.py`` this module imports nothing from
``core/rag.py``; the client is injected.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from qdrant_client import models

from docint.core.search.fulltext import uuid_match_forms, value_match_forms
from docint.core.search.index import SEARCH_TEXT_FIELD, search_index_params

#: Payload index a key must carry for ``MatchText`` to work on it.
TEXT_INDEX = "text"

#: Payload index a key must carry for ``MatchValue`` to be accelerated.
KEYWORD_INDEX = "keyword"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """The payload keys one picker option searches, split by matcher.

    Attributes:
        text_keys (tuple[str, ...]): Keys matched with ``MatchText`` —
            case-insensitive, prefix-based, phrase-checked for a multi-word
            query. These hold prose: a name, a handle, a filename.
        value_keys (tuple[str, ...]): Keys matched with ``MatchValue`` —
            exact, once per form ``value_forms`` derives from the query.
            These hold identifiers, which are not prose and cannot be
            full-text indexed.
        value_forms (Callable[[str], list[Any]]): Turns the raw query into
            the exact values to try against ``value_keys``. The default
            covers the int/str duality every numeric id shares; a uuid
            supplies its own dash handling rather than burdening every other
            id search with dead branches.
    """

    text_keys: tuple[str, ...] = ()
    value_keys: tuple[str, ...] = field(default_factory=tuple)
    value_forms: Callable[[str], list[Any]] = value_match_forms

    def indexed_keys(self) -> tuple[tuple[str, str], ...]:
        """Pair every key with the payload index its matcher requires.

        Returns:
            tuple[tuple[str, str], ...]: ``(payload key, expected index kind)``
                in declaration order, text keys first.
        """
        return tuple([(key, TEXT_INDEX) for key in self.text_keys] + [(key, KEYWORD_INDEX) for key in self.value_keys])


#: Picker name → the keys it searches. ``"text"`` is first and is the default.
SEARCH_FIELDS: dict[str, FieldSpec] = {
    "text": FieldSpec(text_keys=(SEARCH_TEXT_FIELD,)),
    "author": FieldSpec(
        text_keys=(
            "reference_metadata.author",
            "reference_metadata.vanity",
            "reference_metadata.posting_author",
            "reference_metadata.posting_vanity",
        ),
        value_keys=(
            "reference_metadata.author_id",
            "reference_metadata.posting_author_id",
        ),
    ),
    "network": FieldSpec(text_keys=("reference_metadata.network",)),
    # A posting's own node carries its uuid only at reference_metadata.uuid;
    # every artifact derived from it carries the same value as posting_uuid —
    # the pair _fetch_posting_entity_nodes already ORs for the same reason.
    "uuid": FieldSpec(
        value_keys=("reference_metadata.uuid", "posting_uuid"),
        value_forms=uuid_match_forms,
    ),
}

DEFAULT_SEARCH_FIELD = "text"

#: Fields an ``_images`` companion point can answer. Its payload holds a
#: caption in ``search_text`` and, for social media, the parent posting's
#: ``posting_*`` fields and ``posting_uuid`` link — which is why ``author``
#: and ``uuid`` reach it. It carries no ``network``, so that one searches the
#: text lane only.
IMAGE_LANE_FIELDS: frozenset[str] = frozenset({"text", "author", "uuid"})


class UnknownSearchFieldError(ValueError):
    """Raised when a field name is not in :data:`SEARCH_FIELDS`."""


def search_field_spec(name: str) -> FieldSpec:
    """Resolve a picker name to the keys it searches.

    Args:
        name (str): A key of :data:`SEARCH_FIELDS`.

    Returns:
        FieldSpec: The payload keys and their matchers.

    Raises:
        UnknownSearchFieldError: When ``name`` is not whitelisted.
    """
    try:
        return SEARCH_FIELDS[name]
    except KeyError as exc:
        raise UnknownSearchFieldError(name) from exc


def _index_kind_from_schema(schema: Any, key: str) -> str | None:
    """Extract one payload key's indexed data type from a schema mapping.

    Factored out of :func:`field_index_kind` so callers can read a
    collection's schema once and check every key against it, rather than
    paying a ``get_collection`` round-trip per key.

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


def _read_schema(client: Any, collection: str) -> dict[str, Any]:
    """Read a collection's payload schema, treating an outage as empty.

    Args:
        client (Any): Qdrant client exposing ``get_collection``.
        collection (str): Physical collection name.

    Returns:
        dict[str, Any]: The schema mapping, or ``{}`` when it cannot be read —
            never a partial view that could read as indexed.
    """
    try:
        info = client.get_collection(collection_name=collection)
    except Exception as exc:
        logger.debug("Payload index status unavailable for {}: {}", collection, exc)
        return {}
    return getattr(info, "payload_schema", None) or {}


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
    return _index_kind_from_schema(_read_schema(client, collection), key)


def field_indexes_ready(client: Any, collection: str, name: str) -> bool:
    """Report whether one picker option's keys all carry the right index.

    "The right index" is matcher-specific: a name key needs TEXT or its
    ``MatchText`` under-matches silently (ASCII-only case folding, no prefix),
    and an id key needs KEYWORD because a TEXT index over an integer field
    indexes nothing at all. A field with any key mis-indexed must report
    ``not_indexed`` rather than return a quietly incomplete result set.

    Args:
        client (Any): Qdrant client exposing ``get_collection``.
        collection (str): Physical collection name.
        name (str): A key of :data:`SEARCH_FIELDS`.

    Returns:
        bool: ``True`` when every key carries its expected index. Trivially
            ``True`` for the default text field, whose ``search_text`` index
            is owned and reported by ``index.py`` instead.

    Raises:
        UnknownSearchFieldError: When ``name`` is not whitelisted.
    """
    spec = search_field_spec(name)
    if name == DEFAULT_SEARCH_FIELD:
        return True
    schema = _read_schema(client, collection)
    return all(_index_kind_from_schema(schema, key) == expected for key, expected in spec.indexed_keys())


def _index_schema_for(kind: str) -> Any:
    """Return the Qdrant index parameters for an expected index kind.

    Args:
        kind (str): :data:`TEXT_INDEX` or :data:`KEYWORD_INDEX`.

    Returns:
        Any: The ``field_schema`` to pass to ``create_payload_index``.
    """
    return search_index_params() if kind == TEXT_INDEX else models.PayloadSchemaType.KEYWORD


def ensure_field_indexes(client: Any, collection: str) -> bool:
    """Ensure every searchable metadata key carries the index its matcher needs.

    Idempotent — a key already indexed the right way is left alone — and
    fail-soft. A wrong-kind index is deleted first, since Qdrant holds one
    index per field; that covers both a KEYWORD index left by the old facet
    lane on a name key, and the TEXT index this feature originally put on the
    numeric id keys, which indexed zero points and made every id search come
    back empty. ``wait=True`` so a search issued right after does not race the
    rebuild. ``search_text`` itself is owned by
    :func:`docint.core.search.index.ensure_search_index` and is not touched.

    The collection's schema is read once, up front, and reused for every
    key — checking it per key would cost one ``get_collection`` round-trip
    each. A schema that cannot be read is treated exactly like an empty one,
    so every key is still attempted rather than skipped as if already indexed.

    Args:
        client (Any): Qdrant client exposing ``get_collection``,
            ``create_payload_index`` and ``delete_payload_index``.
        collection (str): Physical collection name.

    Returns:
        bool: ``True`` when every key ends up with its expected index.
    """
    schema = _read_schema(client, collection)

    ok = True
    for name, spec in SEARCH_FIELDS.items():
        if name == DEFAULT_SEARCH_FIELD:
            continue
        for key, expected in spec.indexed_keys():
            try:
                kind = _index_kind_from_schema(schema, key)
                if kind == expected:
                    continue
                if kind is not None:
                    client.delete_payload_index(collection_name=collection, field_name=key, wait=True)
                client.create_payload_index(
                    collection_name=collection,
                    field_name=key,
                    field_schema=_index_schema_for(expected),
                    wait=True,
                )
            except Exception as exc:
                ok = False
                logger.debug("Field index on {} for collection '{}' skipped: {}", key, collection, exc)
    return ok
