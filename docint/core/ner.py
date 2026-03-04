"""Named entity extraction aggregation and graph helpers.

This module provides pure helper functions for:
- canonicalizing entity/relation payloads
- collection-wide NER aggregation
- stats/search views
- lightweight derived graph construction for GraphRAG-style exploration
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any


def _safe_float(value: Any) -> float | None:
    """Return a float when possible, otherwise ``None``.

    Args:
        value: Candidate numeric value.

    Returns:
        Parsed float or ``None``.
    """
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_type(value: Any) -> str:
    """Normalize entity type labels.

    Args:
        value: Raw type/label value.

    Returns:
        Sanitized type label.
    """
    txt = str(value or "").strip()
    return txt if txt else "Unlabeled"


def _entity_key(text: str, entity_type: str) -> str:
    """Build a canonical entity key.

    Args:
        text: Entity text.
        entity_type: Entity type/label.

    Returns:
        Canonical key for indexing.
    """
    return f"{text.lower()}::{entity_type.lower()}"


def normalize_entities(entities: list[Any] | None) -> list[dict[str, Any]]:
    """Normalize entity payloads.

    Args:
        entities: Raw entity list from source metadata.

    Returns:
        Sanitized entity dictionaries.
    """
    normalized: list[dict[str, Any]] = []
    for ent in entities or []:
        if not isinstance(ent, dict):
            continue
        text = str(ent.get("text") or ent.get("name") or "").strip()
        if not text:
            continue
        entity_type = _normalize_type(ent.get("type") or ent.get("label"))
        normalized.append(
            {
                "text": text,
                "type": entity_type,
                "score": _safe_float(ent.get("score")),
            }
        )
    return normalized


def normalize_relations(relations: list[Any] | None) -> list[dict[str, Any]]:
    """Normalize relation payloads.

    Args:
        relations: Raw relation list from source metadata.

    Returns:
        Sanitized relation dictionaries.
    """
    normalized: list[dict[str, Any]] = []
    for rel in relations or []:
        if not isinstance(rel, dict):
            continue
        head = str(rel.get("head") or rel.get("subject") or "").strip()
        tail = str(rel.get("tail") or rel.get("object") or "").strip()
        if not head or not tail:
            continue
        label = str(rel.get("label") or rel.get("type") or "rel").strip() or "rel"
        normalized.append(
            {
                "head": head,
                "tail": tail,
                "label": label,
                "score": _safe_float(rel.get("score")),
            }
        )
    return normalized


def aggregate_ner_sources(sources: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Aggregate IE payloads across source rows.

    Args:
        sources: Source metadata rows containing optional ``entities`` and ``relations``.

    Returns:
        Aggregation dictionary used by stats/search/graph helpers.
    """
    entity_index: dict[str, dict[str, Any]] = {}
    relation_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    doc_index: dict[str, dict[str, Any]] = {}
    source_entity_sets: list[set[str]] = []

    for src in sources or []:
        if not isinstance(src, dict):
            continue
        filename = str(src.get("filename") or src.get("file_path") or "Unknown")
        doc_entry = doc_index.setdefault(
            filename,
            {
                "filename": filename,
                "ie_source_count": 0,
                "entity_mentions": 0,
                "entity_counter": defaultdict(int),
            },
        )

        normalized_entities = normalize_entities(src.get("entities"))
        normalized_relations = normalize_relations(src.get("relations"))
        if normalized_entities or normalized_relations:
            doc_entry["ie_source_count"] += 1

        source_keys: set[str] = set()
        for ent in normalized_entities:
            text = str(ent["text"])
            entity_type = str(ent["type"])
            key = _entity_key(text, entity_type)
            source_keys.add(key)

            if key not in entity_index:
                entity_index[key] = {
                    "key": key,
                    "text": text,
                    "type": entity_type,
                    "mentions": 0,
                    "best_score": ent.get("score"),
                    "sources": set(),
                }
            row = entity_index[key]
            row["mentions"] += 1
            row["sources"].add(filename)
            if ent.get("score") is not None:
                current = row.get("best_score")
                row["best_score"] = (
                    max(float(current), float(ent["score"]))
                    if current is not None
                    else ent["score"]
                )

            doc_entry["entity_mentions"] += 1
            doc_entry["entity_counter"][key] += 1

        if source_keys:
            source_entity_sets.append(source_keys)

        for rel in normalized_relations:
            rel_key = (
                str(rel["head"]).lower(),
                str(rel["label"]).lower(),
                str(rel["tail"]).lower(),
            )
            if rel_key not in relation_index:
                relation_index[rel_key] = {
                    "head": rel["head"],
                    "tail": rel["tail"],
                    "label": rel["label"],
                    "mentions": 0,
                    "best_score": rel.get("score"),
                    "sources": set(),
                }
            row = relation_index[rel_key]
            row["mentions"] += 1
            row["sources"].add(filename)
            if rel.get("score") is not None:
                current = row.get("best_score")
                row["best_score"] = (
                    max(float(current), float(rel["score"]))
                    if current is not None
                    else rel["score"]
                )

    entity_rows: list[dict[str, Any]] = []
    for row in entity_index.values():
        entity_rows.append(
            {
                "key": row["key"],
                "text": row["text"],
                "type": row["type"],
                "mentions": int(row["mentions"]),
                "best_score": row.get("best_score"),
                "sources": sorted(row["sources"]),
                "source_count": len(row["sources"]),
            }
        )
    entity_rows.sort(key=lambda x: (-int(x["mentions"]), str(x["text"]).lower()))

    relation_rows: list[dict[str, Any]] = []
    for row in relation_index.values():
        relation_rows.append(
            {
                "head": row["head"],
                "tail": row["tail"],
                "label": row["label"],
                "mentions": int(row["mentions"]),
                "best_score": row.get("best_score"),
                "sources": sorted(row["sources"]),
                "source_count": len(row["sources"]),
            }
        )
    relation_rows.sort(
        key=lambda x: (
            -int(x["mentions"]),
            str(x["head"]).lower(),
            str(x["label"]).lower(),
            str(x["tail"]).lower(),
        )
    )

    documents: list[dict[str, Any]] = []
    for doc in doc_index.values():
        ie_source_count = int(doc["ie_source_count"])
        entity_mentions = int(doc["entity_mentions"])
        entity_counter: dict[str, int] = dict(doc["entity_counter"])
        unique_entities = len(entity_counter)
        documents.append(
            {
                "filename": doc["filename"],
                "ie_source_count": ie_source_count,
                "entity_mentions": entity_mentions,
                "unique_entities": unique_entities,
                "entity_density": (
                    float(entity_mentions) / float(ie_source_count)
                    if ie_source_count > 0
                    else 0.0
                ),
                "entity_counter": entity_counter,
            }
        )
    documents.sort(key=lambda x: (-int(x["entity_mentions"]), str(x["filename"])))

    text_to_keys: dict[str, list[str]] = defaultdict(list)
    for ent in entity_rows:
        text_to_keys[str(ent["text"]).lower()].append(str(ent["key"]))

    return {
        "entities": entity_rows,
        "relations": relation_rows,
        "documents": documents,
        "source_entity_sets": source_entity_sets,
        "text_to_keys": dict(text_to_keys),
    }


def build_ner_stats(
    aggregate: dict[str, Any],
    *,
    top_k: int = 15,
    min_mentions: int = 2,
    entity_type: str | None = None,
    include_relations: bool = True,
) -> dict[str, Any]:
    """Build dashboard-friendly IE statistics from an aggregate payload.

    Args:
        aggregate: Output of ``aggregate_ner_sources`` helper.
        top_k: Number of top entities and relations to return.
        min_mentions: Minimum mention count for entities and relations to be included.
        entity_type: Optional filter to include only entities of a given type/label.
        include_relations: Whether to include relations in the statistics.

    Returns:
        dict[str, Any]: A dictionary containing the dashboard-friendly IE statistics.
    """
    entities_all = list(aggregate.get("entities") or [])
    relations_all = list(aggregate.get("relations") or [])
    documents_all = list(aggregate.get("documents") or [])

    type_filter = str(entity_type or "").strip().lower()
    entities = (
        [e for e in entities_all if str(e.get("type") or "").lower() == type_filter]
        if type_filter
        else entities_all
    )

    allowed_texts = {str(e.get("text") or "").lower() for e in entities}
    if type_filter:
        relations = [
            r
            for r in relations_all
            if str(r.get("head") or "").lower() in allowed_texts
            or str(r.get("tail") or "").lower() in allowed_texts
        ]
    else:
        relations = relations_all

    totals = {
        "unique_entities": len(entities),
        "entity_mentions": int(sum(int(e.get("mentions", 0) or 0) for e in entities)),
        "unique_relations": len(relations) if include_relations else 0,
    }

    ranked_entities = [
        e for e in entities if int(e.get("mentions", 0) or 0) >= min_mentions
    ]
    ranked_entities.sort(
        key=lambda x: (
            -int(x.get("mentions", 0) or 0),
            str(x.get("text") or "").lower(),
        )
    )
    top_entities = [
        {
            "text": row["text"],
            "type": row["type"],
            "mentions": int(row["mentions"]),
            "best_score": row.get("best_score"),
            "source_count": int(row.get("source_count", 0) or 0),
        }
        for row in ranked_entities[:top_k]
    ]

    type_rollup: dict[str, dict[str, Any]] = {}
    for ent in entities:
        label = str(ent.get("type") or "Unlabeled")
        entry = type_rollup.setdefault(
            label, {"type": label, "mentions": 0, "unique_entities": 0}
        )
        entry["mentions"] += int(ent.get("mentions", 0) or 0)
        entry["unique_entities"] += 1
    entity_types = sorted(
        type_rollup.values(),
        key=lambda x: (-int(x["mentions"]), str(x["type"]).lower()),
    )

    top_relations: list[dict[str, Any]] = []
    if include_relations:
        ranked_relations = [
            r for r in relations if int(r.get("mentions", 0) or 0) >= min_mentions
        ]
        ranked_relations.sort(
            key=lambda x: (
                -int(x.get("mentions", 0) or 0),
                str(x.get("head") or "").lower(),
                str(x.get("label") or "").lower(),
                str(x.get("tail") or "").lower(),
            )
        )
        top_relations = [
            {
                "head": row["head"],
                "label": row["label"],
                "tail": row["tail"],
                "mentions": int(row["mentions"]),
            }
            for row in ranked_relations[:top_k]
        ]

    documents: list[dict[str, Any]] = []
    for doc in documents_all:
        entity_counter = dict(doc.get("entity_counter") or {})
        if type_filter:
            filtered_keys = [
                k
                for k in entity_counter
                if k.rsplit("::", maxsplit=1)[-1] == type_filter
            ]
            mentions = int(sum(entity_counter[k] for k in filtered_keys))
            unique_entities = len(filtered_keys)
        else:
            mentions = int(doc.get("entity_mentions", 0) or 0)
            unique_entities = int(doc.get("unique_entities", 0) or 0)
        if mentions < min_mentions:
            continue
        ie_source_count = int(doc.get("ie_source_count", 0) or 0)
        documents.append(
            {
                "filename": doc.get("filename"),
                "entity_mentions": mentions,
                "unique_entities": unique_entities,
                "ie_source_count": ie_source_count,
                "entity_density": (
                    float(mentions) / float(ie_source_count) if ie_source_count else 0.0
                ),
            }
        )
    documents.sort(
        key=lambda x: (-int(x["entity_mentions"]), str(x.get("filename") or "").lower())
    )

    return {
        "totals": totals,
        "top_entities": top_entities,
        "entity_types": entity_types,
        "top_relations": top_relations,
        "documents": documents,
    }


def search_entities(
    aggregate: dict[str, Any],
    *,
    q: str = "",
    entity_type: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Search canonicalized entities in aggregated IE payloads.

    Args:
        aggregate: Output of ``aggregate_ner_sources`` helper.
        q: Optional text query to match against entity text.
        entity_type: Optional filter to include only entities of a given type/label.
        limit: Maximum number of results to return.

    Returns:
        List of matching entities with their metadata.
    """
    entities = list(aggregate.get("entities") or [])
    query = str(q or "").strip().lower()
    type_filter = str(entity_type or "").strip().lower()

    rows = []
    for ent in entities:
        if type_filter and str(ent.get("type") or "").lower() != type_filter:
            continue
        if query and query not in str(ent.get("text") or "").lower():
            continue
        rows.append(
            {
                "text": ent.get("text"),
                "type": ent.get("type"),
                "mentions": int(ent.get("mentions", 0) or 0),
                "best_score": ent.get("best_score"),
                "source_count": int(ent.get("source_count", 0) or 0),
            }
        )

    rows.sort(key=lambda x: (-int(x["mentions"]), str(x.get("text") or "").lower()))
    return rows[: max(1, int(limit))]


def build_entity_graph(
    aggregate: dict[str, Any],
    *,
    top_k_nodes: int = 100,
    min_edge_weight: int = 1,
) -> dict[str, Any]:
    """Build a lightweight graph from aggregated entities and relations.

    Args:
        aggregate: Output of ``aggregate_ner_sources`` helper.
        top_k_nodes: Maximum number of top entities to include as nodes.
        min_edge_weight: Minimum weight threshold for edges to be included.

    Returns:
        A dictionary containing nodes, edges, and metadata for graph exploration.
    """
    entities = list(aggregate.get("entities") or [])
    relations = list(aggregate.get("relations") or [])
    source_entity_sets = list(aggregate.get("source_entity_sets") or [])
    text_to_keys = dict(aggregate.get("text_to_keys") or {})

    selected_entities = entities[: max(1, int(top_k_nodes))]
    node_map: dict[str, dict[str, Any]] = {}
    for ent in selected_entities:
        key = str(ent["key"])
        node_map[key] = {
            "id": key,
            "text": ent.get("text"),
            "type": ent.get("type"),
            "mentions": int(ent.get("mentions", 0) or 0),
        }

    edge_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def _add_edge(source: str, target: str, label: str, kind: str, weight: int) -> None:
        """
        Add an edge to the edge index.

        Args:
            source (str): The source node ID.
            target (str): The target node ID.
            label (str): The label for the edge.
            kind (str): The kind of the edge (e.g., "relation" or "cooccurrence").
            weight (int): The weight of the edge.
        """
        if source == target:
            return
        if kind == "cooccurrence":
            source, target = sorted([source, target])
        edge_key = (source, target, label.lower(), kind)
        if edge_key not in edge_index:
            edge_index[edge_key] = {
                "source": source,
                "target": target,
                "label": label,
                "kind": kind,
                "weight": 0,
            }
        edge_index[edge_key]["weight"] += int(weight)

    for rel in relations:
        head_text = str(rel.get("head") or "").lower()
        tail_text = str(rel.get("tail") or "").lower()
        label = str(rel.get("label") or "rel")
        head_key = next(
            (k for k in text_to_keys.get(head_text, []) if k in node_map), None
        )
        tail_key = next(
            (k for k in text_to_keys.get(tail_text, []) if k in node_map), None
        )
        if not head_key or not tail_key:
            continue
        _add_edge(
            head_key,
            tail_key,
            label=label,
            kind="relation",
            weight=int(rel.get("mentions", 1) or 1),
        )

    for keys in source_entity_sets:
        scoped = sorted([k for k in keys if k in node_map])
        if len(scoped) < 2:
            continue
        for left, right in combinations(scoped, 2):
            _add_edge(left, right, label="co_occurs", kind="cooccurrence", weight=1)

    edges = [
        edge
        for edge in edge_index.values()
        if int(edge.get("weight", 0) or 0) >= int(min_edge_weight)
    ]
    edges.sort(
        key=lambda x: (
            -int(x.get("weight", 0) or 0),
            str(x.get("source") or ""),
            str(x.get("target") or ""),
        )
    )

    return {
        "nodes": list(node_map.values()),
        "edges": edges,
        "meta": {"node_count": len(node_map), "edge_count": len(edges)},
    }


def graph_neighbors(
    graph: dict[str, Any],
    *,
    entity: str,
    hops: int = 1,
) -> dict[str, Any]:
    """Return graph neighborhood around an entity (by id or text).

    Args:
        graph: Graph dictionary containing "nodes" and "edges".
        entity: Entity text or ID to find the neighborhood for.
        hops: Number of hops to include in the neighborhood.

    Returns:
        A dictionary containing the center node, its neighbors, and the subgraph
        of nodes and edges within the specified hops.
    """
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    if not nodes:
        return {
            "center": None,
            "neighbors": [],
            "nodes": [],
            "edges": [],
            "meta": {"hops": hops},
        }

    entity_q = str(entity or "").strip().lower()
    if not entity_q:
        return {
            "center": None,
            "neighbors": [],
            "nodes": [],
            "edges": [],
            "meta": {"hops": hops},
        }

    node_by_id = {str(n.get("id")): n for n in nodes}
    center_id = None
    for node in nodes:
        node_id = str(node.get("id") or "")
        node_text = str(node.get("text") or "").lower()
        if entity_q == node_id.lower() or entity_q == node_text:
            center_id = node_id
            break
    if center_id is None:
        for node in nodes:
            node_text = str(node.get("text") or "").lower()
            if entity_q in node_text:
                center_id = str(node.get("id") or "")
                break
    if center_id is None:
        return {
            "center": None,
            "neighbors": [],
            "nodes": [],
            "edges": [],
            "meta": {"hops": hops},
        }

    adjacency: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        weight = int(edge.get("weight", 0) or 0)
        if source and target:
            adjacency[source].append((target, weight))
            adjacency[target].append((source, weight))

    max_hops = max(1, int(hops))
    frontier = {center_id}
    visited: dict[str, int] = {center_id: 0}
    scores: dict[str, float] = defaultdict(float)

    for depth in range(1, max_hops + 1):
        next_frontier: set[str] = set()
        for node_id in frontier:
            for nxt, weight in adjacency.get(node_id, []):
                if nxt not in visited:
                    visited[nxt] = depth
                    next_frontier.add(nxt)
                if visited.get(nxt, depth) <= max_hops and nxt != center_id:
                    scores[nxt] += float(weight) / float(depth)
        frontier = next_frontier
        if not frontier:
            break

    neighborhood_ids = {k for k, v in visited.items() if v <= max_hops}
    neighborhood_edges = [
        edge
        for edge in edges
        if str(edge.get("source") or "") in neighborhood_ids
        and str(edge.get("target") or "") in neighborhood_ids
    ]
    neighbors = []
    for node_id, depth in visited.items():
        if node_id == center_id:
            continue
        node = node_by_id.get(node_id)
        if not node:
            continue
        neighbors.append(
            {
                "id": node_id,
                "text": node.get("text"),
                "type": node.get("type"),
                "mentions": int(node.get("mentions", 0) or 0),
                "depth": depth,
                "score": round(float(scores.get(node_id, 0.0)), 6),
            }
        )
    neighbors.sort(
        key=lambda x: (
            -float(x["score"]),
            int(x["depth"]),
            str(x["text"] or "").lower(),
        )
    )

    neighborhood_nodes = [
        node_by_id[nid] for nid in neighborhood_ids if nid in node_by_id
    ]
    neighborhood_nodes.sort(
        key=lambda x: (
            -int(x.get("mentions", 0) or 0),
            str(x.get("text") or "").lower(),
        )
    )

    return {
        "center": node_by_id.get(center_id),
        "neighbors": neighbors,
        "nodes": neighborhood_nodes,
        "edges": neighborhood_edges,
        "meta": {
            "hops": max_hops,
            "node_count": len(neighborhood_nodes),
            "edge_count": len(neighborhood_edges),
        },
    }
