"""The ``search_text`` payload field, its index, and how it gets written."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger
from qdrant_client import models

from docint.core.storage.scroll import iter_scroll
from docint.utils.retry import is_transient_qdrant_error, retry_with_backoff

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
    # ``is not None`` rather than truthiness: an explicit empty string is the
    # backfill's marker for "looked, nothing to index", and it has to reach
    # Qdrant or the point counts as unprocessed forever.
    items = [(point_id, text) for point_id, text in texts.items() if text is not None]
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
        skipped (int): Points left alone because they were already populated.
        empty (int): Points with no extractable text, marked with an empty
            value so they stop counting as unprocessed.
    """

    scanned: int = 0
    written: int = 0
    skipped: int = 0
    empty: int = 0


def backfill_search_text(
    client: Any,
    collection: str,
    *,
    extract_text: Callable[[Any], str],
    batch_size: int = 256,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
    retry_sleep: Callable[[float], None] | None = None,
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
        retry_sleep (Callable[[float], None] | None): Sleep used between write
            retries; injected by tests so they do not actually wait.

    Returns:
        BackfillSummary: Counts for the run.
    """
    scanned = skipped = 0
    written = empty = 0
    pending: dict[Any, str] = {}

    def _flush() -> tuple[int, int]:
        """Write the pending batch and report ``(searchable, marker)`` counts.

        Returns:
            tuple[int, int]: Points given searchable text, and points marked
                with an empty value because they had none.
        """
        if not pending:
            return 0, 0
        markers = sum(1 for value in pending.values() if not value)
        flushed = _write_with_retry(client, collection, pending, batch_size, retry_sleep)
        pending.clear()
        return max(0, flushed - markers), markers

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
            # Keep the id's own type. Qdrant point ids are unsigned ints or
            # UUIDs; coercing an int id to a string would target a point that
            # does not exist, and the backfill would silently write nothing.
            point_id = getattr(point, "id", None)
            if point_id is None:
                skipped += 1
                continue
            # A point with no extractable text is marked with an empty value
            # rather than left alone: otherwise it counts as missing forever,
            # every search reports `partial`, and the warning stops meaning
            # anything. An empty string is a value, so Qdrant's is-empty check
            # treats the point as covered.
            pending[point_id] = extract_text(payload)

        if len(pending) >= batch_size:
            batch_written, batch_empty = _flush()
            written += batch_written
            empty += batch_empty
            if progress is not None:
                progress(f"{written} point(s) written, {scanned} scanned")

    batch_written, batch_empty = _flush()
    written += batch_written
    empty += batch_empty

    if progress is not None:
        progress(f"done: {scanned} scanned, {written} written, {skipped} skipped, {empty} without text")
    return BackfillSummary(scanned=scanned, written=written, skipped=skipped, empty=empty)


def _write_with_retry(
    client: Any,
    collection: str,
    texts: Mapping[Any, str],
    batch_size: int,
    retry_sleep: Callable[[float], None] | None,
) -> int:
    """Write one pending batch, retrying transient Qdrant failures.

    Every other Qdrant write in the codebase is wrapped this way. Without it a
    single connection blip partway through a large migration kills the run and
    leaves the collection permanently half-indexed — and a search would then
    serve results from that partial state.

    Args:
        client (Any): Qdrant client.
        collection (str): Physical collection name.
        texts (Mapping[Any, str]): ``{point_id: chunk text}`` for this batch.
        batch_size (int): Operations per request.
        retry_sleep (Callable[[float], None] | None): Injected sleep for tests.

    Returns:
        int: Number of points written.
    """
    snapshot = dict(texts)
    return retry_with_backoff(
        "search_text_backfill",
        lambda: write_search_text(client, collection, snapshot, batch_size=batch_size, wait=True),
        max_retries=3,
        initial_backoff=0.5,
        max_backoff=4.0,
        is_retryable=is_transient_qdrant_error,
        sleep=retry_sleep,
    )


def search_index_status(
    client: Any,
    collection: str,
) -> dict[str, Any]:
    """Report how much of a collection is ready for full-text search.

    Counts coverage exactly rather than sampling. A head sample cannot tell a
    finished backfill from one that has only written its first page: the
    backfill walks the collection from the same offset a sample would, so the
    moment its first page lands, a first-hit sample would call the whole
    collection ready. A search issued during that window would then silently
    omit every chunk not yet written while still reporting success — the worst
    failure mode for an investigative tool. Two counts are also cheaper than
    the several payload-bearing scroll pages a sample needed.

    Args:
        client (Any): Qdrant client exposing ``get_collection`` and ``count``.
        collection (str): Physical collection name.

    Returns:
        dict[str, Any]: ``indexed`` (the payload index exists), ``total``,
            ``with_search_text``, ``missing``, and ``complete`` (every point
            carries the field). ``complete`` is ``False`` when the counts are
            unavailable, so an unknown state never reads as a finished one.
    """
    indexed = False
    try:
        info = client.get_collection(collection_name=collection)
        schema = getattr(info, "payload_schema", None) or {}
        indexed = SEARCH_TEXT_FIELD in set(schema)
    except Exception as exc:
        logger.debug("search_text index status unavailable for {}: {}", collection, exc)

    total = with_search_text = 0
    try:
        total = int(client.count(collection_name=collection, exact=True).count)
        with_search_text = int(
            client.count(
                collection_name=collection,
                count_filter=models.Filter(
                    must_not=[models.IsEmptyCondition(is_empty=models.PayloadField(key=SEARCH_TEXT_FIELD))]
                ),
                exact=True,
            ).count
        )
    except Exception as exc:
        logger.debug("search_text coverage unavailable for {}: {}", collection, exc)

    missing = max(0, total - with_search_text)
    return {
        "indexed": indexed,
        "total": total,
        "with_search_text": with_search_text,
        "missing": missing,
        "complete": with_search_text > 0 and missing == 0,
    }
