from __future__ import annotations

from pathlib import Path

import pytest

from docint.core.agents import (
    AgenticPipeline,
    ClarificationRequiredError,
    ClarifiedQuery,
    ConversationContext,
    ConversationUnderstandingAgent,
    HierarchicalIndexer,
    QueryClarifierAgent,
    RetrievalAgent,
)


class DummyRAGForConversation:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_recent_turns(
        self, session_id: str | None, limit: int
    ) -> list[dict[str, str]]:
        self.calls.append("turns")
        return [
            {"user": "How do I migrate?", "assistant": "Use SQL."},
            {"user": "What about rollback?", "assistant": "Apply scripts."},
        ]

    def get_conversation_summary(
        self, session_id: str | None, max_sentences: int
    ) -> str:
        self.calls.append("summary")
        return ""


class DummyDocument:
    def __init__(
        self,
        text: str,
        metadata: dict[str, object] | None = None,
        doc_id: str = "doc-1",
    ) -> None:
        self._text = text
        self.metadata = metadata or {}
        self.id_ = doc_id

    def get_content(self) -> str:
        return self._text


class DummyClarifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[dict[str, object]], int]] = []
        self.next_queries: list[str] = ["refined query"]

    def refine(
        self,
        query: str,
        context_summary: str,
        sources: list[dict[str, object]],
        attempt: int,
    ) -> str:
        self.calls.append((query, context_summary, sources, attempt))
        if self.next_queries:
            return self.next_queries.pop(0)
        return query


class DummyRAGForRetrieval:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = payloads
        self.query_engine = self
        self.queries: list[str] = []

    def query(self, query: str) -> object:  # pragma: no cover - simple stub container
        self.queries.append(query)
        return object()

    def _normalize_response_data(
        self, original: str, resp: object
    ) -> dict[str, object]:
        index = len(self.queries) - 1
        return dict(self.payloads[index])

    def attach_parent_context(
        self, sources: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        return sources


class DummyRAGForPipeline(DummyRAGForRetrieval):
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        super().__init__(payloads)
        self.conversation_calls: list[str] = []

    def get_recent_turns(
        self, session_id: str | None, limit: int
    ) -> list[dict[str, str]]:
        self.conversation_calls.append("turns")
        return []

    def get_conversation_summary(
        self, session_id: str | None, max_sentences: int
    ) -> str:
        self.conversation_calls.append("summary")
        return "Previous discussion focused on SQL migrations."


def test_conversation_understanding_falls_back_to_recent_turns() -> None:
    rag = DummyRAGForConversation()
    agent = ConversationUnderstandingAgent(rag)
    context = agent.analyse("session-1")
    assert context.summary.startswith("User: How do I migrate?")
    assert len(context.recent_turns) == 2


def test_query_clarifier_rewrites_and_splits_queries() -> None:
    clarifier = QueryClarifierAgent()
    context = ConversationContext(
        summary="We discussed SQL databases and migrations.",
        recent_turns=[],
    )
    result = clarifier.clarify("How do I update it? What are the risks?", context)
    assert result.sub_queries == ["How do I update migrations?", "What are the risks?"]
    assert all("key topics" in rewritten for rewritten in result.rewritten_queries)


def test_query_clarifier_detects_insults() -> None:
    clarifier = QueryClarifierAgent()
    context = ConversationContext(summary="", recent_turns=[])
    with pytest.raises(ClarificationRequiredError):
        clarifier.clarify("You are stupid", context)


def test_hierarchical_indexer_builds_and_persists_parents(tmp_path: Path) -> None:
    indexer = HierarchicalIndexer(child_chunk_size=50)
    doc = DummyDocument(
        """# Title\nIntro text.\n## Section\nDetails about topic.""",
        metadata={"path": Path("/tmp/example.md")},
    )
    parent_store_path = tmp_path / "collection.json"
    nodes, parent_store = indexer.build(
        [doc], collection="demo", parent_store_path=parent_store_path
    )

    assert parent_store_path.exists()
    assert len(parent_store) == 2
    assert all(isinstance(node.metadata.get("parent_id"), str) for node in nodes)

    persisted = parent_store_path.read_text(encoding="utf-8")
    assert "Title" in persisted and "Section" in persisted


def test_retrieval_agent_retries_until_sufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = [
        {"response": "Too short", "sources": []},
        {
            "response": "This answer provides extensive actionable steps and references for SQL migrations.",
            "sources": [{"text": "Example", "parent_id": "parent-1"}],
        },
    ]
    rag = DummyRAGForRetrieval(payloads)
    clarifier = DummyClarifier()
    agent = RetrievalAgent(rag, clarifier)  # type: ignore[arg-type]

    raw, payload = agent._run_single_query(
        query="Initial query",
        original="How do I migrate?",
        context_summary="Discussing SQL migrations",
    )

    assert rag.queries == ["Initial query", "refined query"]
    assert clarifier.calls[0][3] == 0  # first attempt triggered refine
    assert payload["sources"][0]["parent_id"] == "parent-1"
    assert payload["reasoning"].startswith("Attempt")


def test_agentic_pipeline_uses_conversation_and_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = [
        {
            "response": "Comprehensive instructions with citations and steps for SQL migrations.",
            "sources": [{"text": "Example", "parent_id": "parent-1"}],
        }
    ]
    rag = DummyRAGForPipeline(payloads)
    pipeline = AgenticPipeline(rag)  # type: ignore[arg-type]

    # Override internal clarifier to avoid randomness and complex logic
    clarifier = DummyClarifier()
    clarifier.next_queries = []
    clarified = ClarifiedQuery(
        original="How do I update migrations?",
        context_summary="Previous discussion focused on SQL migrations.",
        sub_queries=["How do I update migrations?"],
        rewritten_queries=["How do I update migrations?"],
    )

    def fake_analyse(session_id: str | None) -> ConversationContext:
        return ConversationContext(
            summary="Previous discussion focused on SQL migrations.",
            recent_turns=[],
        )

    def fake_clarify(prompt: str, context: ConversationContext) -> ClarifiedQuery:
        return clarified

    pipeline._conversation.analyse = fake_analyse  # type: ignore[assignment]
    pipeline._clarifier = clarifier  # type: ignore[assignment]
    pipeline._clarifier.clarify = fake_clarify  # type: ignore[assignment]

    result = pipeline.run("How do I update it?", session_id="s-1")

    assert rag.conversation_calls == []  # replaced by fake analyse
    assert result.response["response"].startswith("Comprehensive instructions")
    assert result.response["sources"][0]["parent_id"] == "parent-1"
    assert result.response["query"] == "How do I update migrations?"
