"""Heuristic graph query planner for analysis and graph-backed retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from docint.core.graph.models import GraphSeed


@dataclass(slots=True)
class GraphQueryPlan:
    """Structured plan for graph lookup, traversal, and synthesis."""

    mode: str
    seeds: list[GraphSeed] = field(default_factory=list)
    path_terms: tuple[str, str] | None = None


def plan_graph_query(
    query: str, *, requested_mode: str | None = None
) -> GraphQueryPlan:
    """Infer a graph retrieval or analysis plan from a user query.
    
    Args:
        query: The raw user query string to analyze for graph retrieval planning.
        requested_mode: An optional explicitly requested retrieval mode to override inference.

    Returns:
        GraphQueryPlan: A structured plan containing the inferred retrieval mode, query seeds, 
            and any path terms for graph traversal.
    """

    text = str(query or "").strip()
    lowered = text.casefold()
    mode = requested_mode or "graph_lookup"
    if requested_mode in {None, "", "answer"}:
        if any(
            term in lowered for term in ("relationship", "connected", "path between")
        ):
            mode = "graph_path"
        elif any(
            term in lowered for term in ("neighbors", "neighborhood", "network around")
        ):
            mode = "graph_neighborhood"
        elif any(
            term in lowered
            for term in (
                "themes",
                "trend",
                "trends",
                "compare",
                "contradiction",
                "outlier",
                "across",
            )
        ):
            mode = "graph_synthesis"
        else:
            mode = "graph_lookup"

    seeds: list[GraphSeed] = []
    for url in re.findall(r"https?://\S+", text):
        seeds.append(GraphSeed(kind="url", value=url.rstrip(".,;")))
    for tag in re.findall(r"(?:^|\s)#([A-Za-z0-9_./-]+)", text):
        seeds.append(GraphSeed(kind="tag", value=tag))
    for token in re.findall(
        r"\b[a-z]{1,8}-\d+\b|\b[a-z0-9_]+(?:-[a-z0-9_]+){1,}\b", lowered
    ):
        seeds.append(GraphSeed(kind="text_id", value=token))
    for date in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text):
        seeds.append(GraphSeed(kind="date", value=date))
    for phrase in re.findall(r'"([^"]+)"', text):
        seeds.append(GraphSeed(kind="phrase", value=phrase))

    capitalized_chunks = re.findall(
        r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)\b",
        text,
    )
    for chunk in capitalized_chunks[:4]:
        seeds.append(GraphSeed(kind="entity", value=chunk))

    path_terms: tuple[str, str] | None = None
    if mode == "graph_path":
        match = re.search(
            r"between\s+(.+?)\s+and\s+(.+)$|(.+?)\s*(?:->|to)\s*(.+)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            left = next((group for group in match.groups()[:2] if group), "") or (
                match.group(3) or ""
            )
            right = match.group(2) or match.group(4) or ""
            if left.strip() and right.strip():
                path_terms = (left.strip(), right.strip())

    deduped: list[GraphSeed] = []
    seen: set[tuple[str, str]] = set()
    for seed in seeds:
        key = (seed.kind, seed.value.casefold())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(seed)

    return GraphQueryPlan(mode=mode, seeds=deduped, path_terms=path_terms)
