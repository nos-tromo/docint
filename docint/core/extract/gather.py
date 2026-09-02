"""Read a collection's points for an extract.

The only Qdrant-touching module in the package: everything else takes
``(point_id, payload)`` pairs and stays pure. Scrolling is unbounded — an
extract is the whole of what was ingested, so a page cap would silently
truncate a transcript.
"""

from __future__ import annotations

from typing import Any

from qdrant_client.http import models

from docint.core.search.fulltext import not_coarse_condition
from docint.core.storage.scroll import iter_scroll
from docint.core.storage.utils import qdrant_collection_exists

__all__ = ["scroll_collection", "source_filters"]

#: Payload keys under which a main-collection point can carry its source id.
_MAIN_SOURCE_KEYS = ("file_hash", "posting_uuid", "reference_metadata.uuid")
#: The same for an ``_images`` companion point.
_IMAGE_SOURCE_KEYS = ("source_doc_id", "posting_uuid", "image_id", "file_hash")

#: Points fetched per scroll page.
PAGE_SIZE = 256


def _should(keys: tuple[str, ...], value: str) -> models.Filter:
    """Match ``value`` under any of ``keys``."""
    return models.Filter(should=[models.FieldCondition(key=key, match=models.MatchValue(value=value)) for key in keys])


def source_filters(source_id: str) -> tuple[models.Filter, models.Filter]:
    """Build the per-source filters for the main and companion collections.

    A source id is whichever identity the Inspector had to hand: a document's
    file hash, a media file's content hash, a posting uuid or an image id. The
    payloads spell those differently, so each lane matches on every key it
    could appear under.

    Args:
        source_id (str): The id to gather.

    Returns:
        tuple[models.Filter, models.Filter]: ``(main, images)`` filters.
    """
    return _should(_MAIN_SOURCE_KEYS, source_id), _should(_IMAGE_SOURCE_KEYS, source_id)


def _scan(client: Any, collection: str, scroll_filter: models.Filter | None) -> list[tuple[str, dict[str, Any]]]:
    """Return every point of ``collection`` matching ``scroll_filter``."""
    if not collection or not qdrant_collection_exists(client, collection):
        return []
    points: list[tuple[str, dict[str, Any]]] = []
    for page in iter_scroll(
        client,
        collection_name=collection,
        scroll_filter=scroll_filter,
        page_size=PAGE_SIZE,
        error_context="extract",
    ):
        for record in page:
            payload = getattr(record, "payload", None)
            if isinstance(payload, dict):
                points.append((str(getattr(record, "id", "")), payload))
    return points


def scroll_collection(
    client: Any,
    collection: str,
    image_collection: str,
    *,
    source_id: str | None = None,
) -> tuple[list[tuple[str, dict[str, Any]]], list[tuple[str, dict[str, Any]]]]:
    """Read a collection and its images companion for an extract.

    Coarse parent chunks are excluded on the main lane, so a hierarchically
    chunked document contributes its text once rather than twice.

    Args:
        client (Any): Qdrant client.
        collection (str): Physical collection name.
        image_collection (str): Its ``_images`` companion.
        source_id (str | None): Restrict both lanes to one source when set.

    Returns:
        tuple: ``(main_points, image_points)`` as ``(point_id, payload)`` pairs.
            A missing companion yields an empty image lane, not an error.
    """
    coarse = not_coarse_condition()
    main_filter: models.Filter = coarse
    image_filter: models.Filter | None = None
    if source_id:
        main_source, image_filter = source_filters(source_id)
        main_filter = models.Filter(must=[main_source], must_not=coarse.must_not)
    return _scan(client, collection, main_filter), _scan(client, image_collection, image_filter)
