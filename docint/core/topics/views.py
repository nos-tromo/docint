"""The topics read side: a run's overview and one topic's evidence, documents and members.

Read-only. Each function takes the active run, already checked for staleness,
and shows its topics as its reduction presents them (``TopicRun.view``): a group
is one topic, read as the union of its leaves. A function returns ``None`` for
a topic the view lacks, a leaf merged into a group and the unassigned bucket
``-1`` included: unassigned chunks are counted, never listed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from docint.core.topics.gather import (
    DenseVector,
    central_points,
    count,
    document_counts,
    eligible_filter,
    member_page,
    pending_filter,
    topic_counts,
)
from docint.core.topics.model import TopicInfo
from docint.core.topics.store import TopicRun
from docint.utils.cursor import InvalidCursorError, decode_cursor, encode_cursor

SourceOf = Callable[[Any], dict[str, Any]]


def overview(
    client: Any,
    run: TopicRun | None,
    physical: str,
    *,
    embed_model: str,
    ensure_indexes: Callable[[str], None],
) -> dict[str, Any] | None:
    """Describe a run: its settings, its coverage and its topics by size.

    Args:
        client: Qdrant client.
        run: The active run, ``None`` when there is none.
        physical: Physical collection name.
        embed_model: The embedding model configured now.
        ensure_indexes: Gives the topic keys their indexes; called before the facet.

    Returns:
        The overview, or ``None`` without a run.
    """
    if run is None:
        return None
    ensure_indexes(physical)
    view = run.view
    # The facet counts leaves, so its limit is the leaf count, whatever the view shows.
    counts = topic_counts(client, physical, run.run_id, limit=len(run.table.topics) + 2)
    total = count(client, physical, eligible_filter())
    sizes = view.sizes(counts)
    topics = view.table.topics
    record = run.record
    return {
        "run_id": run.run_id,
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "embed_model": record.get("embed_model"),
        "embed_model_changed": record.get("embed_model") != embed_model,
        "params": record.get("params"),
        "leaf_count": len(run.table.topics),
        "coverage": {
            "total": total,
            "assigned": sum(sizes.values()),
            "unassigned": counts.get(-1, 0),
            "pending": count(client, physical, pending_filter(run.run_id)),
        },
        "topics": [
            _topic_row(t, sizes[t.id], total, view.members[t.id])
            for t in sorted(topics, key=lambda t: (-sizes[t.id], t.id))
        ],
    }


def _topic_row(topic: TopicInfo, size: int, total: int, members: tuple[int, ...]) -> dict[str, Any]:
    """Shape one topic for the overview.

    Args:
        topic: The topic's record.
        size: Its member count.
        total: Eligible chunks in the collection.
        members: Its leaf ids; just its own id unless it is a group.

    Returns:
        The topic's public fields.
    """
    return {
        "id": topic.id,
        "members": list(members),
        "label": topic.label,
        "description": topic.description,
        "keywords": [[word, score] for word, score in topic.keywords],
        "size": size,
        "share": round(size / total, 4) if total else 0,
        "origin": topic.origin,
        "added_at": topic.added_at,
        "first_seen": topic.first_seen,
        "last_seen": topic.last_seen,
    }


def evidence(
    client: Any,
    run: TopicRun,
    physical: str,
    topic_id: int,
    *,
    dense: DenseVector,
    limit: int,
    source_of: SourceOf,
) -> list[dict[str, Any]] | None:
    """Return a topic's members nearest its centre.

    Args:
        client: Qdrant client.
        run: The active run.
        physical: Physical collection name.
        topic_id: Topic id.
        dense: The collection's dense vector.
        limit: Most passages.
        source_of: Shapes a point as a source.

    Returns:
        Sources, closest first; empty when the vectors no longer match the
        run's size. ``None`` for a topic the view lacks.
    """
    view = run.view
    centre, leaves = view.table.centre(topic_id), view.members.get(topic_id)
    if centre is None or leaves is None:
        return None
    if dense.dim != run.table.dim:
        return []
    points = central_points(
        client, physical, centre=centre, dense=dense, run_id=run.run_id, topic_ids=leaves, limit=limit
    )
    return [source_of(p) for p in points]


def documents(
    client: Any,
    run: TopicRun,
    physical: str,
    topic_id: int,
    *,
    limit: int,
    names: Callable[[], Mapping[str, str]],
) -> dict[str, Any] | None:
    """Count a topic's members per source file.

    Args:
        client: Qdrant client.
        run: The active run.
        physical: Physical collection name.
        topic_id: Topic id.
        limit: Most files.
        names: Reads the file-hash-to-filename map; called only when there are files.

    Returns:
        ``{"documents": [{file_hash, filename, count}], "more"}``, largest
        first; a hash with no known name is named by itself. ``None`` for a
        topic the view lacks.
    """
    leaves = run.view.members.get(topic_id)
    if leaves is None:
        return None
    rows = document_counts(client, physical, run.run_id, leaves, limit=limit + 1)
    shown = rows[:limit]
    known = names() if shown else {}
    return {
        "documents": [
            {"file_hash": file_hash, "filename": known.get(file_hash, file_hash), "count": n} for file_hash, n in shown
        ],
        "more": len(rows) > limit,
    }


def chunks(
    client: Any,
    run: TopicRun,
    physical: str,
    topic_id: int,
    *,
    cursor: str | None,
    limit: int,
    source_of: SourceOf,
) -> dict[str, Any] | None:
    """Read one page of a topic's members.

    Args:
        client: Qdrant client.
        run: The active run.
        physical: Physical collection name.
        topic_id: Topic id.
        cursor: The previous page's cursor, ``None`` for the first.
        limit: Page size.
        source_of: Shapes a point as a source.

    Returns:
        ``{"items", "next_cursor"}``, or ``None`` for a topic the view lacks.

    Raises:
        InvalidCursorError: When the cursor is malformed.
    """
    leaves = run.view.members.get(topic_id)
    if leaves is None:
        return None
    offset = decode_cursor(cursor)["o"]
    if offset is not None and (isinstance(offset, bool) or not isinstance(offset, int | str)):
        raise InvalidCursorError("cursor offset is not a point id")
    points, next_offset = member_page(client, physical, run_id=run.run_id, topic_ids=leaves, offset=offset, limit=limit)
    return {"items": [source_of(p) for p in points], "next_cursor": encode_cursor(next_offset)}
