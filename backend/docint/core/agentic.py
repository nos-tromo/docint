"""Agentic orchestration utilities used by the RAG pipeline.

This module defines small data containers and helper utilities that power the
agentic, multi-stage workflow described in the product requirements.  The
helpers are intentionally lightweight so they can be reused by both the FastAPI
layer and the core ``RAG`` implementation without introducing additional
runtime dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from textwrap import shorten
from typing import Any, Iterable, Sequence

import json
import re
import uuid


# --- Data containers ----------------------------------------------------------------


@dataclass(slots=True)
class ConversationContext:
    """Compact snapshot of the recent conversation history."""

    summary: str = ""
    topics: list[str] = field(default_factory=list)
    last_user_query: str | None = None


@dataclass(slots=True)
class ClarifiedQuery:
    """Result of the query clarification stage."""

    rewritten_query: str
    sub_queries: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_request: str | None = None
    reasoning: str = ""

    def iter_queries(self) -> Iterable[str]:
        """Yield the primary rewritten query followed by any sub-queries."""

        yield self.rewritten_query
        for candidate in self.sub_queries:
            if candidate.strip():
                yield candidate.strip()


@dataclass(slots=True)
class ParentChunkRecord:
    """Represents a parent chunk stored outside of the vector index."""

    id: str
    title: str
    level: int
    text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class ChildChunkRecord:
    """Represents a child chunk destined for embedding in Qdrant."""

    id: str
    parent_id: str
    text: str
    metadata: dict[str, Any]
    start_char: int
    end_char: int


@dataclass(slots=True)
class AgenticResponse:
    """Lightweight container mimicking the llama-index ``Response`` type."""

    response: str
    source_nodes: Sequence[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    reasoning: str | None = None


# --- Conversation understanding ------------------------------------------------------


_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "can",
    "for",
    "from",
    "have",
    "how",
    "into",
    "more",
    "need",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "will",
    "would",
}


def extract_topics(candidates: Sequence[str], limit: int = 4) -> list[str]:
    """Extracts lightweight keyword-style topics from the provided texts."""

    counts: dict[str, int] = {}
    token_pattern = re.compile(r"[A-Za-z][A-Za-z0-9_/-]+")
    for text in candidates:
        for token in token_pattern.findall(text or ""):
            token_lower = token.lower()
            if token_lower in _STOPWORDS or len(token_lower) < 3:
                continue
            counts[token_lower] = counts.get(token_lower, 0) + 1

    sorted_tokens = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [token for token, _ in sorted_tokens[:limit]]


def summarise_conversation(turns: Sequence[Any]) -> ConversationContext:
    """Create a short, 2–3 sentence synopsis describing recent turns."""

    if not turns:
        return ConversationContext()

    recent_turns = list(turns)[-5:]
    user_prompts = [
        (getattr(t, "rewritten_query", None) or getattr(t, "user_text", ""))
        for t in recent_turns
    ]
    user_prompts = [p for p in user_prompts if p]
    topics = extract_topics(user_prompts)

    sentences: list[str] = []
    if topics:
        focus = ", ".join(topics[:2])
        sentences.append(f"The user has been exploring {focus} recently.")

    last_turn = recent_turns[-1]
    last_user = (
        getattr(last_turn, "rewritten_query", None)
        or getattr(last_turn, "user_text", "")
    )
    if last_user:
        sentences.append(
            f"Their latest question was: {shorten(last_user, width=120, placeholder='…')}"
        )

    last_response = getattr(last_turn, "model_response", "")
    if last_response:
        sentences.append(
            "The assistant responded with "
            + shorten(last_response, width=120, placeholder="…")
            + "."
        )

    summary = " ".join(sentences[:3]).strip()
    return ConversationContext(summary=summary, topics=topics, last_user_query=last_user)


# --- Query clarification -------------------------------------------------------------


_UNCLEAR_PATTERNS = [
    re.compile(r"^[!?]+$"),
    re.compile(r"^[\W_]+$"),
    re.compile(r"\b(?:stupid|idiot|dumb)\b", re.IGNORECASE),
]

_PRONOUN_PATTERN = re.compile(r"\b(it|they|them|this|that|these|those|he|she)\b", re.I)


def clarify_query(question: str, context: ConversationContext) -> ClarifiedQuery:
    """Heuristically rewrite the incoming question for retrieval."""

    cleaned = question.strip()
    reasoning_steps: list[str] = []
    if not cleaned:
        return ClarifiedQuery(
            rewritten_query="",
            needs_clarification=True,
            clarification_request="Could you share more details about what you need?",
            reasoning="Received an empty query and requested clarification.",
        )

    for pattern in _UNCLEAR_PATTERNS:
        if pattern.search(cleaned):
            return ClarifiedQuery(
                rewritten_query=cleaned,
                needs_clarification=True,
                clarification_request="I want to help, but I need a clearer question to proceed.",
                reasoning="Detected a low-information or hostile query and deferred to the user.",
            )

    anchor: str | None = None
    if context.topics:
        anchor = context.topics[0]
    elif context.last_user_query:
        tokens = extract_topics([context.last_user_query], limit=1)
        anchor = tokens[0] if tokens else context.last_user_query

    def _replace(match: re.Match[str]) -> str:
        return anchor or match.group(0)

    rewritten = _PRONOUN_PATTERN.sub(_replace, cleaned)
    if rewritten != cleaned and anchor:
        reasoning_steps.append(
            f"Resolved a reference to '{anchor}' based on the recent conversation."
        )

    # Split multi-part questions into sub-queries.
    sub_queries: list[str] = []
    if rewritten.count("?") > 1:
        pieces = [p.strip() for p in rewritten.split("?") if p.strip()]
        if pieces:
            rewritten = pieces[0] + "?"
            sub_queries = [piece + "?" for piece in pieces[1:]]
            reasoning_steps.append("Split a multi-question prompt into separate sub-queries.")
    else:
        connective_split = re.split(r"\b(?: and | then | as well as )\b", rewritten, flags=re.I)
        if len(connective_split) > 1:
            rewritten = connective_split[0].strip()
            sub_queries = [frag.strip() for frag in connective_split[1:] if frag.strip()]
            reasoning_steps.append("Identified multiple intents joined by conjunctions.")

    # Augment with topical keywords for retrieval.
    keyword_suffix = ""
    if context.topics:
        missing = [topic for topic in context.topics if topic.lower() not in rewritten.lower()]
        if missing:
            keyword_suffix = " " + " ".join(sorted(set(missing)))
            rewritten = rewritten.rstrip("?") + keyword_suffix + ("?" if rewritten.endswith("?") else "")
            reasoning_steps.append("Augmented the query with topical keywords from prior turns.")

    if not sub_queries and keyword_suffix:
        # Provide an alternative view that emphasises keywords for retrieval.
        sub_queries = [f"{rewritten} {keyword_suffix}".strip()]

    reasoning = " ".join(reasoning_steps).strip()
    return ClarifiedQuery(
        rewritten_query=rewritten,
        sub_queries=sub_queries,
        reasoning=reasoning,
    )


# --- Hierarchical chunking -----------------------------------------------------------


class HierarchicalChunker:
    """Split documents into parent/child chunks as per the agentic design."""

    def __init__(self, child_characters: int = 500) -> None:
        self.child_characters = child_characters
        self._heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    def build(
        self, text: str, metadata: dict[str, Any], *, default_title: str
    ) -> tuple[list[ParentChunkRecord], list[ChildChunkRecord]]:
        """Return parent and child chunk records for the provided text."""

        text = text or ""
        if not text.strip():
            return [], []

        parents = self._split_parents(text, metadata, default_title=default_title)
        children = self._split_children(parents)
        return parents, children

    def _split_parents(
        self, text: str, metadata: dict[str, Any], *, default_title: str
    ) -> list[ParentChunkRecord]:
        matches = list(self._heading_pattern.finditer(text))
        parents: list[ParentChunkRecord] = []

        if not matches:
            parent_id = f"{metadata.get('file_hash') or uuid.uuid4().hex}-p0"
            parents.append(
                ParentChunkRecord(
                    id=parent_id,
                    title=default_title,
                    level=1,
                    text=text.strip(),
                    metadata=metadata,
                )
            )
            return parents

        for idx, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if not body:
                continue
            parent_id = f"{metadata.get('file_hash') or uuid.uuid4().hex}-p{idx}"
            parents.append(
                ParentChunkRecord(
                    id=parent_id,
                    title=title or default_title,
                    level=level,
                    text=body,
                    metadata=metadata,
                )
            )
        if not parents:
            # Fall back to treating the entire document as one parent chunk.
            parent_id = f"{metadata.get('file_hash') or uuid.uuid4().hex}-p0"
            parents.append(
                ParentChunkRecord(
                    id=parent_id,
                    title=default_title,
                    level=1,
                    text=text.strip(),
                    metadata=metadata,
                )
            )
        return parents

    def _split_children(self, parents: Sequence[ParentChunkRecord]) -> list[ChildChunkRecord]:
        children: list[ChildChunkRecord] = []
        for parent in parents:
            text = parent.text
            cursor = 0
            chunk_idx = 0
            while cursor < len(text):
                window = text[cursor : cursor + self.child_characters]
                if not window:
                    break
                if cursor + self.child_characters < len(text):
                    # Try to break at a natural boundary.
                    boundary = max(window.rfind(". "), window.rfind("\n"))
                    if boundary <= 0:
                        boundary = window.rfind(" ")
                    if boundary <= 0:
                        boundary = len(window)
                    window = window[:boundary]
                window = window.strip()
                if not window:
                    cursor += self.child_characters
                    continue
                start = cursor
                end = start + len(window)
                child_id = f"{parent.id}-c{chunk_idx}"
                metadata = {
                    "parent_id": parent.id,
                    "section_title": parent.title,
                    "section_level": parent.level,
                    "chunk_index": chunk_idx,
                }
                children.append(
                    ChildChunkRecord(
                        id=child_id,
                        parent_id=parent.id,
                        text=window,
                        metadata=metadata,
                        start_char=start,
                        end_char=end,
                    )
                )
                chunk_idx += 1
                cursor = end
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
        return children


# --- Persistence helpers -------------------------------------------------------------


def serialise_parent_chunks(parents: Sequence[ParentChunkRecord]) -> list[dict[str, Any]]:
    """Convert parent chunks into a JSON-serialisable structure."""

    def _normalise(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {k: _normalise(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_normalise(v) for v in value]
        return str(value)

    payload: list[dict[str, Any]] = []
    for parent in parents:
        payload.append(
            {
                "id": parent.id,
                "title": parent.title,
                "level": parent.level,
                "text": parent.text,
                "metadata": _normalise(parent.metadata),
            }
        )
    return payload


def dump_parent_chunks(path: Path, parents: Sequence[ParentChunkRecord]) -> None:
    """Persist parent chunk definitions to the provided path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialise_parent_chunks(parents)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_parent_chunks(path: Path) -> dict[str, dict[str, Any]]:
    """Read parent chunk definitions into a dictionary keyed by ID."""

    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["id"]: item for item in data}


# --- Utility functions ----------------------------------------------------------------


def combine_reasoning(*pieces: str) -> str:
    """Join non-empty reasoning fragments into a readable paragraph."""

    filtered = [piece.strip() for piece in pieces if piece and piece.strip()]
    if not filtered:
        return ""
    return " ".join(filtered)

