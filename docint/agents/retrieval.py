"""Retrieval agents that bridge to RAG."""

import time
from typing import TYPE_CHECKING, Any

from docint.agents.types import (
    PriorTurn,
    RetrievalAgent,
    RetrievalRequest,
    RetrievalResult,
)

if TYPE_CHECKING:
    from docint.core.rag import RAG


class RAGRetrievalAgent(RetrievalAgent):
    """Adapter that uses the existing RAG pipeline for retrieval/response.

    Depending on the intent detected in the turn, it may invoke different tools.
    """

    def __init__(self, rag: "RAG") -> None:
        """Initialize the RAGRetrievalAgent.

        Args:
            rag (RAG): The RAG instance to use for retrieval.
        """
        self.rag = rag

    def _build_prior_turn(self, history: list[dict[str, str]]) -> PriorTurn | None:
        """Extract the immediately preceding user/assistant exchange from history.

        Scans the history tail for the most recent assistant message and the
        user message that triggered it. Returns ``None`` when the conversation
        has no prior exchange yet (first turn) or when the history is
        malformed (e.g., only system messages).

        Args:
            history: Ordered list of ``{"role": ..., "content": ...}`` messages.

        Returns:
            The prior exchange, or ``None`` when not derivable.
        """
        if not history:
            return None
        last_assistant_idx: int | None = None
        for idx in range(len(history) - 1, -1, -1):
            if history[idx].get("role") == "assistant":
                last_assistant_idx = idx
                break
        if last_assistant_idx is None:
            return None
        assistant_text = (history[last_assistant_idx].get("content") or "").strip()
        if not assistant_text:
            return None
        user_text = ""
        for idx in range(last_assistant_idx - 1, -1, -1):
            if history[idx].get("role") == "user":
                user_text = (history[idx].get("content") or "").strip()
                break
        return PriorTurn(user_text=user_text, assistant_text=assistant_text)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Invoke the appropriate tool based on intent; default to RAG chat.

        Args:
            request (RetrievalRequest): The retrieval request containing the turn and analysis.

        Returns:
            RetrievalResult: The result of the retrieval or generation step.
        """
        turn = request.turn
        session_id = self.rag.start_session(turn.session_id)
        analysis = request.analysis
        intent = analysis.intent
        start = time.monotonic()

        if intent in {"ner", "extract"}:
            raw_sources = self.rag.get_collection_ner()
            sources = self._filter_ner_sources(raw_sources, analysis.entities)
            latency = (time.monotonic() - start) * 1000
            return RetrievalResult(
                answer="NER results attached",
                sources=sources,
                session_id=session_id,
                intent=intent,
                confidence=analysis.confidence,
                tool_used="ner_sources",
                latency_ms=latency,
                retrieval_query=turn.user_input,
                rewritten_query=analysis.rewritten_query,
            )

        if intent == "table":
            latency = (time.monotonic() - start) * 1000
            return RetrievalResult(
                answer="Table lookup not yet implemented",
                sources=[],
                session_id=session_id,
                intent=intent,
                confidence=analysis.confidence,
                tool_used="table_lookup",
                latency_ms=latency,
                retrieval_query=turn.user_input,
                rewritten_query=analysis.rewritten_query,
            )

        query_text = analysis.rewritten_query or turn.user_input
        prior_turn = self._build_prior_turn(request.history)
        data = self.rag.chat(query_text, prior_turn=prior_turn)
        latency = (time.monotonic() - start) * 1000

        answer = str(data.get("response") or data.get("answer") or "") if isinstance(data, dict) else ""
        sources = data.get("sources", []) if isinstance(data, dict) else []

        return RetrievalResult(
            answer=answer,
            sources=sources,
            session_id=session_id,
            intent=intent,
            confidence=analysis.confidence,
            tool_used="rag_chat",
            latency_ms=latency,
            retrieval_query=query_text,
            rewritten_query=analysis.rewritten_query,
        )

    def _filter_ner_sources(self, sources: list[dict[str, Any]], entities: dict[str, Any]) -> list[dict[str, Any]]:
        """Filter NER sources using simple entity/page heuristics.

        Args:
            sources (list[dict]): The list of NER sources.
            entities (dict): Extracted entities from the user input.

        Returns:
            list[dict]: Filtered list of sources.
        """
        if not sources or not entities:
            return sources

        query = str(entities.get("query") or "").lower()
        page = str(entities.get("page") or "").strip()

        def match(src: dict[str, Any]) -> bool:
            """Determine if a source matches the given page or query.

            Args:
                src (dict): The source dictionary to check.

            Returns:
                bool: True if the source matches the page or query, False otherwise.
            """
            if page and str(src.get("page") or "") == page:
                return True
            fname = str(src.get("filename") or "").lower()
            if query and query in fname:
                return True
            ents = src.get("entities") or []
            for ent in ents:
                text = str(ent.get("text") or "").lower()
                if query and query in text:
                    return True
            return False

        filtered = [s for s in sources if match(s)]
        return filtered or sources
