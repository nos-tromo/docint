"""Which documents a derived item draws on, and the deadline it inherits from them.

A chat session, a report, an extract or a topic run never outlives the
earliest-expiring document it draws on (``docs/retention.md``). Where docint
knows the sources (a chat answer's citations, a report item's sources), their
deadlines count; where it does not (summaries, topic runs, the collection
overview, a turn that used collection-wide context), the collection's oldest
live document counts. A source that is no longer indexed counts as the oldest
too, never as "never".
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from docint.core.retention import inherit
from docint.core.summary.units import source_file_hash
from docint.utils.env_cfg import load_document_retention_env


def _oldest(manager: Any, physical: str) -> datetime | None:
    """Return the deadline of the collection's earliest-expiring indexed document.

    Args:
        manager (Any): The document retention manager.
        physical (str): The Qdrant collection.

    Returns:
        datetime | None: ``None`` when nothing is indexed (or retention is off).
    """
    row = manager.oldest_live(physical)
    value = getattr(row, "expires_at", None)
    return value if isinstance(value, datetime) else None


def source_expiry(
    rag: Any,
    physical: str,
    file_hashes: Iterable[str],
    *,
    now: datetime,
    period_days: int | None = None,
) -> datetime | None:
    """Return the deadline a derived item inherits from the documents it draws on.

    Args:
        rag (Any): The RAG facade (its document retention manager is used).
        physical (str): The Qdrant collection the item draws on.
        file_hashes (Iterable[str]): The documents it names; empty for
            collection-wide items.
        now (datetime): The item's creation time.
        period_days (int | None): The retention period; read from the
            environment when ``None``.

    Returns:
        datetime | None: The earliest source deadline. With nothing indexed
            to inherit from, the item's own period after ``now``; ``None``
            only while retention is off.
    """
    manager = rag.ensure_document_retention_manager()
    wanted = {value for value in file_hashes if value}
    values: list[datetime] = []
    if wanted:
        found = manager.expiry_of(physical, wanted)
        known = found if isinstance(found, Mapping) else {}
        values = [value for value in known.values() if isinstance(value, datetime)]
        if wanted - set(known):
            oldest = _oldest(manager, physical)
            if oldest is not None:
                values.append(oldest)
    else:
        oldest = _oldest(manager, physical)
        if oldest is not None:
            values.append(oldest)
    result = inherit(None, *values)
    if result is not None:
        return result
    days = load_document_retention_env().days if period_days is None else period_days
    return None if days is None else now + timedelta(days=days)


def report_item_sources(artifact_type: str, snapshot: Mapping[str, Any]) -> tuple[set[str], set[str], bool]:
    """Read which documents a report item draws on.

    Args:
        artifact_type (str): The item's kind.
        snapshot (Mapping[str, Any]): Its frozen content.

    Returns:
        tuple[set[str], set[str], bool]: The file hashes it names, the chunk
            (point) ids it names, and whether it (also) draws on the whole
            collection.
    """
    if artifact_type == "chat_answer":
        hashes: set[str] = set()
        collection_wide = False
        for source in snapshot.get("sources") or []:
            value = str((source or {}).get("file_hash") or "").strip() if isinstance(source, Mapping) else ""
            if value:
                hashes.add(value)
            else:
                collection_wide = True
        return hashes, set(), collection_wide or not hashes
    if artifact_type in {"entity_finding", "hate_speech_finding"}:
        chunk = str(snapshot.get("chunk_id") or "").strip()
        return set(), ({chunk} if chunk else set()), not chunk
    return set(), set(), True


def report_item_expiry(
    rag: Any, physical: str, artifact_type: str, snapshot: Mapping[str, Any], *, now: datetime
) -> datetime | None:
    """Return the deadline a report item inherits.

    A finding names a chunk (a Qdrant point); its document is read from the
    point's payload. A chunk that is gone counts as collection-wide.

    Args:
        rag (Any): The RAG facade.
        physical (str): The report's Qdrant collection.
        artifact_type (str): The item's kind.
        snapshot (Mapping[str, Any]): Its frozen content.
        now (datetime): The item's creation time.

    Returns:
        datetime | None: The inherited deadline.
    """
    hashes, chunks, collection_wide = report_item_sources(artifact_type, snapshot)
    if chunks:
        points = rag.qdrant_client.retrieve(collection_name=physical, ids=sorted(chunks), with_payload=True)
        resolved = {source_file_hash(dict(getattr(point, "payload", None) or {})) for point in points or []}
        resolved.discard("")
        hashes |= resolved
        collection_wide = collection_wide or len(resolved) < len(chunks)
    return source_expiry(rag, physical, set() if collection_wide else hashes, now=now)


def extract_expiry(rag: Any, physical: str, *, now: datetime, retention_days: int) -> datetime:
    """Return when a stored extract goes: after its own retention, or with the oldest document if earlier.

    Args:
        rag (Any): The RAG facade.
        physical (str): The extract's Qdrant collection.
        now (datetime): The build time.
        retention_days (int): ``EXTRACT_RETENTION_DAYS``.

    Returns:
        datetime: The earlier deadline.
    """
    own = now + timedelta(days=retention_days)
    try:
        inherited = source_expiry(rag, physical, [], now=now)
    except Exception as exc:
        logger.warning("Extract deadline taken from its own retention only: {}", exc)
        return own
    return inherit(own, inherited) or own


def collection_expiry_iso(rag: Any, physical: str, *, now: datetime) -> str | None:
    """Return the deadline a collection-wide item made now inherits, as ISO text (for ``run.json``).

    Args:
        rag (Any): The RAG facade.
        physical (str): The Qdrant collection.
        now (datetime): The creation time.

    Returns:
        str | None: The deadline, or ``None`` while retention is off.
    """
    value = source_expiry(rag, physical, [], now=now)
    return None if value is None else value.isoformat()
