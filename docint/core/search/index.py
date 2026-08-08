"""The ``search_text`` payload field, its index, and how it gets written."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger
from qdrant_client import models

from docint.core.storage.scroll import iter_scroll

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
    texts: Mapping[Any, str],
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
        texts (Mapping[Any, str]): ``{point_id: chunk text}``. Point ids
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


@dataclass(frozen=True, slots=True)
class BackfillSummary:
    """Outcome of one ``search_text`` backfill run.

    Attributes:
        scanned (int): Points examined.
        written (int): Points given a ``search_text`` value.
        skipped (int): Points left alone — already populated, or carrying no
            extractable text.
    """

    scanned: int = 0
    written: int = 0
    skipped: int = 0


def backfill_search_text(
    client: Any,
    collection: str,
    *,
    extract_text: Callable[[Any], str],
    batch_size: int = 256,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> BackfillSummary:
    """Populate ``search_text`` across an already-ingested collection.

    Payload-only: no re-embedding, no inference, no model download, so this is
    safe to run on an airgapped host. Points that already carry the field are
    skipped unless ``force`` is set, which makes re-running cheap.

    Args:
        client (Any): Qdrant client exposing ``scroll`` and ``batch_update_points``.
        collection (str): Physical collection name.
        extract_text (Callable[[Any], str]): Pulls the chunk text
            out of a point payload. Injected so this module never imports
            ``core.rag``.
        batch_size (int): Points per scroll page and per write request.
        force (bool): Rewrite points that already have a value.
        progress (Callable[[str], None] | None): Optional progress sink.

    Returns:
        BackfillSummary: Counts for the run.
    """
    scanned = written = skipped = 0
    pending: dict[Any, str] = {}

    for page in iter_scroll(
        client,
        collection_name=collection,
        page_size=max(1, int(batch_size)),
        error_context="search_text backfill",
    ):
        for point in page:
            scanned += 1
            payload = getattr(point, "payload", None)
            if not isinstance(payload, Mapping):
                skipped += 1
                continue
            if not force:
                existing = payload.get(SEARCH_TEXT_FIELD)
                if isinstance(existing, str) and existing.strip():
                    skipped += 1
                    continue
            text = extract_text(payload)
            if not text:
                skipped += 1
                continue
            # Keep the id's own type. Qdrant point ids are unsigned ints or
            # UUIDs; coercing an int id to a string would target a point that
            # does not exist, and the backfill would silently write nothing.
            point_id = getattr(point, "id", None)
            if point_id is None:
                skipped += 1
                continue
            pending[point_id] = text

        if len(pending) >= batch_size:
            written += write_search_text(client, collection, pending, batch_size=batch_size, wait=True)
            pending.clear()
            if progress is not None:
                progress(f"{written} point(s) written, {scanned} scanned")

    if pending:
        written += write_search_text(client, collection, pending, batch_size=batch_size, wait=True)

    if progress is not None:
        progress(f"done: {scanned} scanned, {written} written, {skipped} skipped")
    return BackfillSummary(scanned=scanned, written=written, skipped=skipped)


def search_index_status(
    client: Any,
    collection: str,
    *,
    sample_pages: int = 4,
) -> dict[str, Any]:
    """Report whether a collection is ready for full-text search.

    Samples the head of the collection rather than scanning it: the question
    the caller needs answered is "has the migration run here", and an
    unmigrated collection shows up immediately. A search that finds nothing
    must be able to say *why*, so this never conflates "not indexed yet" with
    "no matches".

    Args:
        client (Any): Qdrant client exposing ``get_collection`` and ``scroll``.
        collection (str): Physical collection name.
        sample_pages (int): Scroll pages to sample when looking for the field.

    Returns:
        dict[str, Any]: ``{"indexed": bool, "has_search_text": bool}`` —
            ``indexed`` reflects the payload index, ``has_search_text`` whether
            any sampled point carries the field.
    """
    indexed = False
    try:
        info = client.get_collection(collection_name=collection)
        schema = getattr(info, "payload_schema", None) or {}
        indexed = SEARCH_TEXT_FIELD in set(schema)
    except Exception as exc:
        logger.debug("search_text index status unavailable for {}: {}", collection, exc)

    has_text = False
    for page in iter_scroll(
        client,
        collection_name=collection,
        page_size=64,
        max_pages=max(1, int(sample_pages)),
        on_error="debug",
        error_context="search_text status",
    ):
        for point in page:
            payload = getattr(point, "payload", None)
            if isinstance(payload, Mapping):
                value = payload.get(SEARCH_TEXT_FIELD)
                if isinstance(value, str) and value.strip():
                    has_text = True
                    break
        if has_text:
            break

    return {"indexed": indexed, "has_search_text": has_text}
