"""Agent orchestration utilities for the Document Intelligence backend.

This module implements the multi-stage agentic pipeline requested for the
application.  The pipeline is split into four stages:

1. Conversation understanding – summarize the ongoing conversation and extract
   the active context for a session.
2. Query clarification – rewrite and/or split user prompts into actionable
   sub-queries while flagging unclear requests.
3. Hierarchical indexing – split documents into parent/child chunks for hybrid
   retrieval.
4. Intelligent retrieval – execute refined sub-queries, evaluate the results,
   and combine the answers with parent context.

The implementation deliberately avoids introducing new entry points.  Instead
it exposes lightweight helpers consumed by :mod:`docint.core.rag` so that the
existing FastAPI routes and CLI commands can take advantage of the agentic
behaviour without further changes.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from loguru import logger

try:  # Optional dependency (fast path when llama-index is available)
    from llama_index.core.schema import TextNode
except ImportError:  # pragma: no cover - fallback for missing dependency
    TextNode = object  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Exceptions


class ClarificationRequiredError(RuntimeError):
    """Raised when the user prompt is too vague or hostile to answer."""


# ---------------------------------------------------------------------------
# Stage 1 – Conversation understanding


@dataclass(slots=True)
class ConversationContext:
    """Lightweight container with the active conversation context."""

    summary: str
    recent_turns: list[dict[str, str]]


class ConversationUnderstandingAgent:
    """Summarise recent turns to maintain conversational continuity."""

    def __init__(self, rag: "RAG", *, max_turns: int = 5) -> None:  # pragma: no cover - imported lazily
        self._rag = rag
        self._max_turns = max_turns

    def analyse(self, session_id: str | None) -> ConversationContext:
        turns = self._rag.get_recent_turns(session_id, limit=self._max_turns)
        summary = self._rag.get_conversation_summary(session_id, max_sentences=3)
        if not summary and turns:
            # Fallback: join the recent turns and clip to 3 sentences
            joined = " ".join(
                f"User: {t['user']} Assistant: {t['assistant']}" for t in turns
            )
            summary = _clip_sentences(joined, 3)
        return ConversationContext(summary=summary, recent_turns=turns)


def _clip_sentences(text: str, max_sentences: int) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return ""
    return " ".join(sentences[:max_sentences]).strip()


# ---------------------------------------------------------------------------
# Stage 2 – Query clarification


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


@dataclass(slots=True)
class ClarifiedQuery:
    """Represents the rewritten structure of a user query."""

    original: str
    context_summary: str
    sub_queries: list[str]
    rewritten_queries: list[str]


class QueryClarifierAgent:
    """Resolve references, split questions, and generate retrieval queries."""

    def __init__(self, *, chunk_size: int = 500) -> None:
        self._chunk_size = chunk_size

    def clarify(self, query: str, context: ConversationContext) -> ClarifiedQuery:
        cleaned = query.strip()
        if not cleaned:
            raise ClarificationRequiredError("Query cannot be empty.")

        if self._is_insult(cleaned):
            raise ClarificationRequiredError(
                "The question appears to be hostile or insulting. Please rephrase."
            )

        if self._is_gibberish(cleaned):
            raise ClarificationRequiredError(
                "I'm not sure I understand the request. Could you clarify?"
            )

        resolved = self._resolve_references(cleaned, context.summary)
        sub_queries = self._split_into_subqueries(resolved)
        rewritten = [self._rewrite_for_retrieval(q, context.summary) for q in sub_queries]
        return ClarifiedQuery(
            original=cleaned,
            context_summary=context.summary,
            sub_queries=sub_queries,
            rewritten_queries=rewritten,
        )

    def refine(
        self,
        query: str,
        context_summary: str,
        sources: Sequence[dict[str, Any]],
        attempt: int,
    ) -> str:
        """Generate a refined retrieval query when earlier attempts fail."""

        keywords: list[str] = []
        for src in sources:
            parent_title = src.get("parent_title") or ""
            parent_text = src.get("parent_text") or ""
            keywords.extend(self._extract_keywords(parent_title))
            keywords.extend(self._extract_keywords(parent_text))
        if not keywords and context_summary:
            keywords.extend(self._extract_keywords(context_summary))
        keywords = list(dict.fromkeys(keywords))  # preserve order, drop duplicates
        if keywords:
            tail = ", ".join(keywords[:6])
            return f"{query}. Focus on: {tail}."
        suffix = "Provide additional factual details." if attempt == 0 else ""
        return f"{context_summary}\n\n{query}. {suffix}".strip()

    @staticmethod
    def _is_insult(text: str) -> bool:
        lowered = text.lower()
        return any(word in lowered for word in {"idiot", "stupid", "dumb"})

    @staticmethod
    def _is_gibberish(text: str) -> bool:
        letters = sum(ch.isalpha() for ch in text)
        vowels = sum(ch.lower() in "aeiou" for ch in text)
        return letters > 0 and vowels == 0

    def _resolve_references(self, query: str, summary: str) -> str:
        pronouns = {"it", "this", "that", "they", "them", "those", "these"}
        if not summary:
            return query

        keywords = self._extract_keywords(summary)
        if not keywords:
            return query

        replacement = keywords[-1]
        pattern = re.compile(r"\b(" + "|".join(pronouns) + r")\b", re.IGNORECASE)
        return pattern.sub(replacement, query)

    @staticmethod
    def _split_into_subqueries(query: str) -> list[str]:
        raw_parts = [part.strip() for part in query.split("?") if part.strip()]
        if len(raw_parts) > 1:
            return [f"{part}?" for part in raw_parts]

        # Split on " and " when the sentence is long enough to likely contain
        # multiple intents.
        if " and " in query.lower() and len(query) > 80:
            segments = [seg.strip().capitalize() for seg in query.split(" and ") if seg.strip()]
            return [seg if seg.endswith("?") else f"{seg}?" for seg in segments]

        return [query if query.endswith("?") else f"{query}?".replace("??", "?")]

    def _rewrite_for_retrieval(self, query: str, summary: str) -> str:
        keywords = self._extract_keywords(query)
        context_keywords = self._extract_keywords(summary)
        important_terms = list(dict.fromkeys(keywords + context_keywords))
        if important_terms:
            joined = ", ".join(important_terms[:12])
            return f"{query} -- key topics: {joined}"
        return query

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9_-]+", text)
        keywords = [tok for tok in tokens if tok.lower() not in STOPWORDS and len(tok) > 2]
        return keywords


# ---------------------------------------------------------------------------
# Stage 3 – Hierarchical indexing


@dataclass(slots=True)
class ParentChunk:
    """Represents a parent chunk stored in JSON."""

    id: str
    title: str
    text: str
    metadata: dict[str, Any]


class HierarchicalIndexer:
    """Create parent/child chunks from raw documents."""

    def __init__(self, *, child_chunk_size: int = 500) -> None:
        self._child_chunk_size = child_chunk_size

    def build(
        self,
        docs: Sequence[Any],
        *,
        collection: str,
        parent_store_path: Path,
    ) -> tuple[list[TextNode], dict[str, ParentChunk]]:
        parent_store: dict[str, ParentChunk] = {}
        child_nodes: list[TextNode] = []

        for doc_idx, doc in enumerate(docs):
            text = self._extract_text(doc)
            if not text:
                continue

            metadata = self._normalise_metadata(getattr(doc, "metadata", {}) or {})
            source_id = getattr(doc, "id_", None) or metadata.get("doc_id")
            parents = self._split_into_parents(text)
            for parent_order, parent in enumerate(parents):
                parent_id = f"parent-{collection}-{uuid.uuid4().hex}"
                parent_meta = {
                    **metadata,
                    "parent_order": parent_order,
                    "source_id": str(source_id) if source_id is not None else None,
                    "heading": parent["title"],
                }
                parent_chunk = ParentChunk(
                    id=parent_id,
                    title=parent["title"],
                    text=parent["text"],
                    metadata=parent_meta,
                )
                parent_store[parent_id] = parent_chunk

                for child_order, child_text in enumerate(
                    self._chunk_text(parent["text"])
                ):
                    node_id = f"child-{collection}-{uuid.uuid4().hex}"
                    node_metadata = {
                        **parent_meta,
                        "parent_id": parent_id,
                        "parent_title": parent["title"],
                        "parent_order": parent_order,
                        "child_order": child_order,
                    }
                    child_nodes.append(
                        TextNode(text=child_text, id_=node_id, metadata=node_metadata)
                    )

            if doc_idx and doc_idx % 100 == 0:
                logger.info("Processed {} documents for collection '{}'.", doc_idx, collection)

        self._persist_parent_store(parent_store, parent_store_path)
        return child_nodes, parent_store

    @staticmethod
    def _extract_text(doc: Any) -> str:
        if hasattr(doc, "get_content") and callable(doc.get_content):
            text = doc.get_content()
            if isinstance(text, str):
                return text
        text = getattr(doc, "text", None)
        if isinstance(text, str):
            return text
        return str(doc)

    @staticmethod
    def _normalise_metadata(meta: dict[str, Any]) -> dict[str, Any]:
        normalised: dict[str, Any] = {}
        for key, value in meta.items():
            if isinstance(value, Path):
                normalised[key] = str(value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                normalised[key] = value
            else:
                normalised[key] = str(value)
        return normalised

    @staticmethod
    def _split_into_parents(text: str) -> list[dict[str, str]]:
        heading_pattern = re.compile(r"^(#{1,3}\s+.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(text))
        if not matches:
            cleaned = text.strip()
            title = cleaned.splitlines()[0][:120] if cleaned else "Document"
            return [{"title": title or "Document", "text": cleaned}]

        parents: list[dict[str, str]] = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            section = text[start:end].strip()
            title = match.group(1).lstrip("# ").strip()
            parents.append({"title": title or f"Section {idx+1}", "text": section})
        return parents

    def _chunk_text(self, text: str) -> Iterable[str]:
        cleaned = text.strip()
        if not cleaned:
            return []
        size = self._child_chunk_size
        return [cleaned[i : i + size] for i in range(0, len(cleaned), size)]

    @staticmethod
    def _persist_parent_store(store: dict[str, ParentChunk], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = [
            {
                "id": parent.id,
                "title": parent.title,
                "text": parent.text,
                "metadata": parent.metadata,
            }
            for parent in store.values()
        ]
        with path.open("w", encoding="utf-8") as f:
            json.dump(serialisable, f, ensure_ascii=False, indent=2)
        logger.info(
            "Stored {} parent chunks for collection '{}' at {}.",
            len(serialisable),
            path.stem,
            path,
        )


# ---------------------------------------------------------------------------
# Stage 4 – Intelligent retrieval


@dataclass(slots=True)
class RetrievalResult:
    response: dict[str, Any]
    raw_responses: list[Any]


class RetrievalAgent:
    """Run refined sub-queries and combine their responses."""

    def __init__(self, rag: "RAG", clarifier: QueryClarifierAgent) -> None:  # pragma: no cover - rag imported lazily
        self._rag = rag
        self._clarifier = clarifier

    def retrieve(
        self,
        clarified: ClarifiedQuery,
        *,
        session_id: str,
    ) -> RetrievalResult:
        aggregated_answer: list[str] = []
        aggregated_sources: list[dict[str, Any]] = []
        aggregated_reasoning: list[str] = []
        raw_responses: list[Any] = []

        for idx, rewritten in enumerate(clarified.rewritten_queries):
            raw_resp, payload = self._run_single_query(
                query=rewritten,
                original=clarified.sub_queries[idx],
                context_summary=clarified.context_summary,
            )
            raw_responses.append(raw_resp)
            aggregated_answer.append(payload.get("response") or "")
            aggregated_sources.extend(payload.get("sources", []))
            if payload.get("reasoning"):
                aggregated_reasoning.append(payload["reasoning"])

        final_answer = self._combine_answers(clarified, aggregated_answer)
        deduped_sources = self._deduplicate_sources(aggregated_sources)
        reasoning = "\n\n".join(aggregated_reasoning).strip()
        response_payload = {
            "query": clarified.original,
            "response": final_answer,
            "sources": deduped_sources,
            "reasoning": reasoning,
            "context_summary": clarified.context_summary,
            "sub_queries": clarified.sub_queries,
            "rewritten_queries": clarified.rewritten_queries,
        }
        return RetrievalResult(response=response_payload, raw_responses=raw_responses)

    def _run_single_query(
        self,
        *,
        query: str,
        original: str,
        context_summary: str,
        max_attempts: int = 3,
    ) -> tuple[Any, dict[str, Any]]:
        attempt = 0
        last_resp: Any | None = None
        last_payload: dict[str, Any] | None = None
        refined_query = query

        while attempt < max_attempts:
            resp = self._rag.query_engine.query(refined_query)
            payload = self._rag._normalize_response_data(original, resp)
            payload["sources"] = self._rag.attach_parent_context(payload.get("sources", []))
            payload["reasoning"] = (
                payload.get("reasoning") or f"Attempt {attempt + 1}: answered with agentic retrieval."
            )
            if self._is_sufficient(payload):
                return resp, payload
            refined_query = self._clarifier.refine(
                original,
                context_summary,
                payload.get("sources", []),
                attempt,
            )
            attempt += 1
            last_resp, last_payload = resp, payload

        assert last_resp is not None and last_payload is not None
        return last_resp, last_payload

    @staticmethod
    def _is_sufficient(payload: dict[str, Any]) -> bool:
        response = payload.get("response", "").strip()
        sources = payload.get("sources", [])
        if len(response) > 40 and sources:
            return True
        if len(sources) >= 2:
            return True
        return False

    @staticmethod
    def _combine_answers(clarified: ClarifiedQuery, answers: Sequence[str]) -> str:
        answers = [ans.strip() for ans in answers if ans.strip()]
        if not answers:
            return "I could not find relevant information."
        if len(answers) == 1:
            return answers[0]
        sections = [
            f"Sub-question {idx + 1}: {clarified.sub_queries[idx]}\n{answer}"
            for idx, answer in enumerate(answers)
        ]
        return "\n\n".join(sections)

    @staticmethod
    def _deduplicate_sources(sources: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        for src in sources:
            key = src.get("parent_id") or src.get("text") or uuid.uuid4().hex
            if key not in seen:
                seen[key] = src
        return list(seen.values())


# ---------------------------------------------------------------------------
# Public façade


class AgenticPipeline:
    """High-level helper that wires the individual agents together."""

    def __init__(self, rag: "RAG") -> None:  # pragma: no cover - rag imported lazily
        self._rag = rag
        self._conversation = ConversationUnderstandingAgent(rag)
        self._clarifier = QueryClarifierAgent()
        self._retriever = RetrievalAgent(rag, self._clarifier)

    def run(self, prompt: str, *, session_id: str) -> RetrievalResult:
        context = self._conversation.analyse(session_id)
        clarified = self._clarifier.clarify(prompt, context)
        return self._retriever.retrieve(clarified, session_id=session_id)


# NOTE: Circular imports – type checking helper
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for mypy/pyright only
    from docint.core.rag import RAG

