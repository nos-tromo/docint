"""The ``search_text`` payload field, its index, and how it gets written."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loguru import logger
from qdrant_client import models

#: Payload key holding a copy of the chunk's text for full-text search.
#: Written payload-only — never through node metadata, which would land the
#: text in the embedding input and in ``_node_content`` as well.
SEARCH_TEXT_FIELD = "search_text"

#: Shortest indexable token. Keywords below this cannot match and are refused
#: at the API rather than silently contributing an unmatchable condition.
SEARCH_MIN_TOKEN_LEN = 2

#: Longest indexable token.
SEARCH_MAX_TOKEN_LEN = 30


def search_index_params() -> models.TextIndexParams:
    """Return the payload-index parameters full-text search depends on.

    ``lowercase=True`` is load-bearing rather than cosmetic: un-indexed
    ``MatchText`` only case-folds ASCII, so a German title-case token would
    not match its lowercase form. ``PREFIX`` lets the head of a compound find
    the compound (``Partei`` finds ``Parteitag``), which is how German
    investigative material is actually searched.

    Returns:
        models.TextIndexParams: Parameters for the ``search_text`` index.
    """
    return models.TextIndexParams(
        type="text",
        tokenizer=models.TokenizerType.PREFIX,
        lowercase=True,
        min_token_len=SEARCH_MIN_TOKEN_LEN,
        max_token_len=SEARCH_MAX_TOKEN_LEN,
    )


def ensure_search_index(client: Any, collection: str) -> bool:
    """Create the ``search_text`` payload index if it is missing.

    Idempotent — Qdrant accepts a repeated creation of the same index — and
    fail-soft, mirroring the existing ``posting_uuid`` index creation in
    ``RAG.create_index``. A Qdrant outage must degrade search, not break
    ingestion.

    Args:
        client (Any): Qdrant client exposing ``create_payload_index``.
        collection (str): Physical collection name.

    Returns:
        bool: ``True`` when the index exists after the call, ``False`` when the
            attempt failed.
    """
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=SEARCH_TEXT_FIELD,
            field_schema=search_index_params(),
        )
    except Exception as exc:
        logger.debug("search_text index on {} skipped: {}", collection, exc)
        return False
    return True


def write_search_text(
    client: Any,
    collection: str,
    texts: Mapping[str | int, str],
    *,
    batch_size: int = 256,
    wait: bool = False,
) -> int:
    """Write ``search_text`` onto points, several per request.

    Qdrant's ``set_payload`` applies one payload to many points, so distinct
    per-point texts would otherwise cost one request each.
    ``batch_update_points`` carries many independent set-payload operations in
    a single request, which is what makes both the ingest hook and the backfill
    affordable. Setting a payload key leaves the point's other keys untouched.

    Args:
        client (Any): Qdrant client exposing ``batch_update_points``.
        collection (str): Physical collection name.
        texts (Mapping[str | int, str]): ``{point_id: chunk text}``. Point ids
            keep their own type — Qdrant ids are unsigned ints or UUIDs, and
            coercing an int id to a string would target a point that does not
            exist, so the write would silently land nowhere.
        batch_size (int): Operations per request.
        wait (bool): Whether Qdrant should confirm the write before replying.
            ``False`` on the ingest path (throughput), ``True`` for the backfill
            so its progress reporting is truthful.

    Returns:
        int: Number of points written.
    """
    items = [(point_id, text) for point_id, text in texts.items() if text]
    if not items:
        return 0

    size = max(1, int(batch_size))
    written = 0
    for start in range(0, len(items), size):
        chunk = items[start : start + size]
        client.batch_update_points(
            collection_name=collection,
            update_operations=[
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(
                        payload={SEARCH_TEXT_FIELD: text},
                        points=[point_id],
                    )
                )
                for point_id, text in chunk
            ],
            wait=wait,
        )
        written += len(chunk)
    return written
