"""Structured graph models shared by ingestion and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class GraphSeed:
    """A structured query seed extracted from user input."""

    kind: Literal["entity", "author", "url", "tag", "text_id", "date", "phrase"]
    value: str


@dataclass(slots=True)
class GraphSourceRecord:
    """Graph-ready representation of one evidence-bearing source record."""

    node_id: str
    collection: str
    source_kind: str
    record_kind: str
    text: str
    filename: str | None = None
    file_hash: str | None = None
    text_id: str | None = None
    thread_id: str | None = None
    parent_record_id: str | None = None
    author: str | None = None
    author_id: str | None = None
    platform: str | None = None
    timestamp: str | None = None
    page: int | None = None
    row: int | None = None
    url: str | None = None
    domain: str | None = None
    tags: list[str] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    search_blob: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphCandidate:
    """Rankable graph retrieval candidate tied back to a source node."""

    node_id: str
    exact_score: float
    graph_score: float
    matched_on: str


@dataclass(slots=True)
class GraphTraversalResult:
    """Graph retrieval result with candidates and trace metadata."""

    candidates: list[GraphCandidate]
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphPathResult:
    """Path lookup result for relational analysis."""

    source: str
    target: str
    nodes: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
