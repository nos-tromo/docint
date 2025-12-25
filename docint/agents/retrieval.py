"""Retrieval agents that bridge to RAG."""

import time

from typing import TYPE_CHECKING

from docint.agents.types import RetrievalAgent, RetrievalRequest, RetrievalResult

if TYPE_CHECKING:
    from docint.core.rag import RAG


class RAGRetrievalAgent(RetrievalAgent):
    """Adapter that uses the existing RAG pipeline for retrieval/response."""

    def __init__(self, rag: "RAG"):
        self.rag = rag

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Invoke the appropriate tool based on intent; default to RAG chat."""
        turn = request.turn
        session_id = self.rag.start_session(turn.session_id)
        analysis = request.analysis
        intent = analysis.intent
        start = time.monotonic()

        if intent in {"ie", "extract"}:
            sources = self.rag.get_collection_ie()
            latency = (time.monotonic() - start) * 1000
            return RetrievalResult(
                answer="IE results attached",
                sources=sources,
                session_id=session_id,
                intent=intent,
                confidence=analysis.confidence,
                tool_used="ie_sources",
                latency_ms=latency,
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
            )

        data = self.rag.chat(turn.user_input)
        latency = (time.monotonic() - start) * 1000

        answer = (
            str(data.get("response") or data.get("answer") or "")
            if isinstance(data, dict)
            else ""
        )
        sources = data.get("sources", []) if isinstance(data, dict) else []

        return RetrievalResult(
            answer=answer,
            sources=sources,
            session_id=session_id,
            intent=intent,
            confidence=analysis.confidence,
            tool_used="rag_chat",
            latency_ms=latency,
        )
