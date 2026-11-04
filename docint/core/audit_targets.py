"""Audit targets for docint's payloads: what a response returned, by reference.

Every builder takes the rows a route is about to return and names them by
identifier only — point ids, file hashes, entity keys, session and report
ids — never by content. ``count`` is the number of rows, so it stays right
when some row carries no identifier. What each identifier means is listed in
``docs/audit.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from docint.utils.audit_event import Target

#: Where a chunk or image row keeps its identity: the point id first (what
#: ``GET /search/chunk`` and a chat scope address), then the reader-minted id.
_CHUNK_KEYS = ("id", "chunk_id", "node_id")


def _first(row: Any, keys: Sequence[str]) -> str | None:
    """Return a row's first non-empty identifier among ``keys``.

    Args:
        row (Any): A payload row; anything but a mapping has none.
        keys (Sequence[str]): Candidate keys, in order of preference.

    Returns:
        str | None: The identifier.
    """
    if not isinstance(row, Mapping):
        return None
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _unique(values: Iterable[str | None]) -> list[str]:
    """Drop empty and repeated identifiers, keeping first-seen order.

    Args:
        values (Iterable[str | None]): Identifiers.

    Returns:
        list[str]: The distinct identifiers.
    """
    return list(dict.fromkeys(value for value in values if value))


def _target(kind: str, rows: Sequence[Any], ids: Iterable[str | None], container: str | None) -> Target:
    """Build a target over ``rows`` with the given identifiers.

    Args:
        kind (str): The target kind.
        rows (Sequence[Any]): The rows returned; their number is the count.
        ids (Iterable[str | None]): Their identifiers.
        container (str | None): The logical collection.

    Returns:
        Target: The target.
    """
    return Target(kind=kind, ids=_unique(ids), count=len(rows), container=container)


def chunk_target(rows: Sequence[Any], *, container: str | None = None) -> Target:
    """Name the chunks and image points rows refer to (sources, hits, findings).

    Args:
        rows (Sequence[Any]): Source, hit or finding rows.
        container (str | None): The logical collection.

    Returns:
        Target: ``kind="chunk"``, by point id.
    """
    return _target("chunk", rows, (_first(row, _CHUNK_KEYS) for row in rows), container)


def document_target(rows: Sequence[Any], *, container: str | None = None) -> Target:
    """Name the documents rows refer to.

    Args:
        rows (Sequence[Any]): Document rows.
        container (str | None): The logical collection.

    Returns:
        Target: ``kind="document"``, by file hash.
    """
    return _target("document", rows, (_first(row, ("file_hash",)) for row in rows), container)


def entity_key(row: Any) -> str | None:
    """Return an entity row's key, ``<text>::<type>`` as the SPA builds it.

    Args:
        row (Any): An entity row.

    Returns:
        str | None: The key, ``None`` for a row with no text.
    """
    if not isinstance(row, Mapping):
        return None
    explicit = _first(row, ("key",))
    if explicit:
        return explicit
    text = str(row.get("text") or "").strip()
    if not text:
        return None
    return f"{text}::{str(row.get('type') or '').strip() or 'Unlabeled'}"


def entity_target(rows: Sequence[Any], *, container: str | None = None) -> Target:
    """Name the entities rows refer to (ranking, search results).

    Args:
        rows (Sequence[Any]): Entity rows.
        container (str | None): The logical collection.

    Returns:
        Target: ``kind="entity"``, by entity key.
    """
    return _target("entity", rows, (entity_key(row) for row in rows), container)


def graph_target(nodes: Sequence[Any], *, container: str | None = None) -> Target:
    """Name the entities an entity graph shows.

    Args:
        nodes (Sequence[Any]): Graph nodes; their ids are cluster keys.
        container (str | None): The logical collection.

    Returns:
        Target: ``kind="entity"``, by node id.
    """
    return _target("entity", nodes, (_first(node, ("id",)) or entity_key(node) for node in nodes), container)


def session_target(rows: Sequence[Any], *, container: str | None = None) -> Target:
    """Name chat sessions.

    Args:
        rows (Sequence[Any]): Session rows.
        container (str | None): The logical collection.

    Returns:
        Target: ``kind="session"``, by session id.
    """
    return _target("session", rows, (_first(row, ("id", "session_id")) for row in rows), container)


def report_target(rows: Sequence[Any], *, container: str | None = None) -> Target:
    """Name reports.

    Args:
        rows (Sequence[Any]): Report rows.
        container (str | None): The logical collection.

    Returns:
        Target: ``kind="report"``, by report id.
    """
    return _target("report", rows, (_first(row, ("id",)) for row in rows), container)


def report_items_target(report: Mapping[str, Any], *, container: str | None = None) -> Target:
    """Name the evidence a report holds.

    The dedupe key names what an item is (``entity:<chunk id>``,
    ``chat:<session>:<turn>``, ``summary:<collection>`` …), so it is the
    reference a later reader can trace.

    Args:
        report (Mapping[str, Any]): The report with its items.
        container (str | None): The logical collection.

    Returns:
        Target: ``kind="report_item"``, by dedupe key.
    """
    items = [item for item in report.get("items") or [] if isinstance(item, Mapping)]
    return _target("report_item", items, (_first(item, ("dedupe_key", "id")) for item in items), container)


def topic_target(topics: Sequence[Any], *, container: str | None = None) -> Target:
    """Name topics.

    Args:
        topics (Sequence[Any]): Topic rows.
        container (str | None): The logical collection.

    Returns:
        Target: ``kind="topic"``, by topic id.
    """
    return _target("topic", topics, (_first(topic, ("id",)) for topic in topics), container)


def unit_target(units: Sequence[Any], *, container: str | None = None) -> Target:
    """Name the units an extract holds: documents, clips, postings, pictures.

    One kind of unit is named by its key alone; several kinds make a ``mixed``
    target whose ids carry their kind (``posting:<uuid>``).

    Args:
        units (Sequence[Any]): Extract units, each with a ``kind`` and a ``key``.
        container (str | None): The logical collection.

    Returns:
        Target: The units, by key.
    """
    kinds = {str(getattr(unit, "kind", "unit")) for unit in units}
    if len(kinds) == 1:
        kind = kinds.pop()
        ids: Iterable[str | None] = (str(getattr(unit, "key", "") or "") for unit in units)
    else:
        kind = "mixed"
        ids = (f"{getattr(unit, 'kind', 'unit')}:{getattr(unit, 'key', '')}" for unit in units)
    return _target(kind, units, ids, container)


def filter_criteria(rules: Any) -> list[Any]:
    """Turn a request's metadata filter rules into plain criteria.

    Args:
        rules (Any): The rules as the request model holds them.

    Returns:
        list[Any]: One mapping per rule; the audit library bounds the size.
    """
    if not rules:
        return []
    out: list[Any] = []
    for rule in rules:
        dump = getattr(rule, "model_dump", None)
        out.append(dump() if callable(dump) else rule)
    return out
