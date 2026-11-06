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
    UNASSIGNED,
    DenseVector,
    and_filters,
    central_points,
    count,
    document_counts,
    eligible_filter,
    member_page,
    pending_filter,
    relaxed_filter,
    topic_counts,
    topic_filter,
)
from docint.core.topics.model import TopicInfo
from docint.core.topics.reduce import LABELS_KEY, RELAX_KEY, merge_tree
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


def tuning(
    client: Any,
    run: TopicRun | None,
    physical: str,
    *,
    ensure_indexes: Callable[[str], None],
) -> dict[str, Any] | None:
    """Describe what fine-tuning a run starts from: its own topics and the tree merges follow.

    A client cuts the tree itself, so previewing a reduction asks nothing more of the server.

    Args:
        client: Qdrant client.
        run: The active run, ``None`` when there is none.
        physical: Physical collection name.
        ensure_indexes: Gives the topic keys their indexes; called before the facet.

    Returns:
        ``run_id``; the build's ``min_topic_size``; every topic of the run
        unreduced (``leaves``: id, label, fit size, size); the ``merges``
        (``[a, b, similarity]`` in merge order, see :func:`merge_tree`); the
        stored ``groups``; the labels written for groups (``group_labels``,
        by member key); and the last relax (``relax``, see
        :func:`_relax_state`). ``None`` without a run.
    """
    if run is None:
        return None
    ensure_indexes(physical)
    counts = topic_counts(client, physical, run.run_id, limit=len(run.table.topics) + 2)
    params = run.record.get("params")
    written = run.record.get(LABELS_KEY)
    labels = {
        str(key): entry["label"]
        for key, entry in (written.items() if isinstance(written, Mapping) else ())
        if isinstance(entry, Mapping) and isinstance(entry.get("label"), str) and entry["label"]
    }
    return {
        "run_id": run.run_id,
        "min_topic_size": params.get("min_topic_size") if isinstance(params, Mapping) else None,
        "leaves": [
            {"id": t.id, "label": t.label, "fit_size": t.fit_size, "size": counts.get(t.id, 0)}
            for t in run.table.topics
        ],
        "merges": [[a, b, similarity] for a, b, similarity in merge_tree(run.table)],
        "groups": [list(leaves) for leaves in run.view.members.values() if len(leaves) > 1],
        "group_labels": labels,
        "relax": _relax_state(client, run, physical),
    }


def _relax_state(client: Any, run: TopicRun, physical: str) -> dict[str, Any]:
    """Describe a run's last relax and whether its histogram still describes the pool.

    Args:
        client: Qdrant client.
        run: The run.
        physical: Physical collection name.

    Returns:
        ``threshold`` (``None`` when none is applied), ``histogram`` (``None``
        before the first relax), ``applied`` (chunks placed under the bar now)
        and ``stale``: true without a histogram, and once the chunks the rule
        leaves out are no longer the ones it counted (an ingest or a discover
        since).
    """
    stored = run.record.get(RELAX_KEY)
    last: Mapping[str, Any] = stored if isinstance(stored, Mapping) else {}
    applied = count(client, physical, relaxed_filter(run.run_id))
    histogram = last.get("histogram")
    pool = count(client, physical, and_filters(topic_filter(run.run_id, UNASSIGNED), eligible_filter()))
    return {
        "threshold": last.get("threshold") if applied else None,
        "histogram": histogram if isinstance(histogram, list) else None,
        "applied": applied,
        "stale": not isinstance(histogram, list) or last.get("basis") != pool + applied,
    }
