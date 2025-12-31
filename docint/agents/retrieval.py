"""Retrieval agents that bridge to RAG."""

import time
from typing import TYPE_CHECKING

from docint.agents.types import RetrievalAgent, RetrievalRequest, RetrievalResult

if TYPE_CHECKING:
    from docint.core.rag import RAG


class RAGRetrievalAgent(RetrievalAgent):
    """
    Adapter that uses the existing RAG pipeline for retrieval/response.
    Depending on the intent detected in the turn, it may invoke different tools.
    """

    def __init__(self, rag: "RAG"):
        """
        Initialize the RAGRetrievalAgent.

        Args:
            rag (RAG): The RAG instance to use for retrieval.
        """
        self.rag = rag

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """
        Invoke the appropriate tool based on intent; default to RAG chat.
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

        if intent in {"ie", "extract"}:
            raw_sources = self.rag.get_collection_ie()
            sources = self._filter_ie_sources(raw_sources, analysis.entities)
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

    def _filter_ie_sources(self, sources: list[dict], entities: dict) -> list[dict]:
        """
        Filter IE sources using simple entity/page heuristics.

        Args:
            sources (list[dict]): The list of IE sources.
            entities (dict): Extracted entities from the user input.

        Returns:
            list[dict]: Filtered list of sources.
        """
        if not sources or not entities:
            return sources

        query = str(entities.get("query") or "").lower()
        page = str(entities.get("page") or "").strip()

        def match(src: dict) -> bool:
            """
            Determine if a source matches the given page or query.

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
