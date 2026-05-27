"""Tests for :class:`RAGRetrievalAgent` prior-turn extraction."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from docint.agents import (
    IntentAnalysis,
    PriorTurn,
    RAGRetrievalAgent,
    RetrievalRequest,
    Turn,
)


def _make_rag_mock() -> MagicMock:
    """Build a RAG mock that satisfies ``RAGRetrievalAgent`` defaults."""
    rag = MagicMock()
    rag.start_session.return_value = "session-1"
    rag.chat.return_value = {
        "response": "ok",
        "sources": [{"id": 1}],
        "session_id": "session-1",
    }
    return rag


def test_retrieve_passes_prior_turn_when_history_has_assistant() -> None:
    """A trailing assistant message should be packaged into a PriorTurn."""
    rag = _make_rag_mock()
    agent = RAGRetrievalAgent(rag)
    history: list[dict[str, str]] = [
        {"role": "user", "content": "Which interpretation is correct?"},
        {"role": "assistant", "content": "The text mentions the UN Security Council."},
    ]
    request = RetrievalRequest(
        turn=Turn(user_input="Please elaborate.", session_id="session-1"),
        analysis=IntentAnalysis(
            intent="qa",
            confidence=0.9,
            entities={"query": "Please elaborate."},
            rewritten_query="UN Security Council references",
        ),
        history=history,
    )

    agent.retrieve(request)

    assert rag.chat.call_count == 1
    args, kwargs = rag.chat.call_args
    assert args == ("UN Security Council references",)
    prior_turn = kwargs.get("prior_turn")
    assert isinstance(prior_turn, PriorTurn)
    assert prior_turn.assistant_text == "The text mentions the UN Security Council."
    assert prior_turn.user_text == "Which interpretation is correct?"


def test_retrieve_passes_none_when_history_empty() -> None:
    """Empty history must yield ``prior_turn=None`` so SessionManager rewrites normally."""
    rag = _make_rag_mock()
    agent = RAGRetrievalAgent(rag)
    request = RetrievalRequest(
        turn=Turn(user_input="hello", session_id="session-1"),
        analysis=IntentAnalysis(
            intent="qa", confidence=0.9, entities={"query": "hello"}
        ),
    )

    agent.retrieve(request)

    _, kwargs = rag.chat.call_args
    assert kwargs.get("prior_turn") is None


def test_retrieve_uses_rewritten_query_when_present() -> None:
    """Retrieval text should be analysis.rewritten_query when set, else user_input."""
    rag = _make_rag_mock()
    agent = RAGRetrievalAgent(rag)
    request = RetrievalRequest(
        turn=Turn(user_input="raw", session_id="session-1"),
        analysis=IntentAnalysis(
            intent="qa",
            confidence=0.9,
            entities={"query": "raw"},
            rewritten_query="standalone query",
        ),
    )

    agent.retrieve(request)

    args, _ = rag.chat.call_args
    assert args == ("standalone query",)


def test_build_prior_turn_handles_assistant_only_history() -> None:
    """When history has only an assistant message, ``PriorTurn.user_text`` is empty."""
    rag = _make_rag_mock()
    agent = RAGRetrievalAgent(rag)
    history: list[dict[str, str]] = [
        {"role": "assistant", "content": "Standalone assistant note."},
    ]
    prior = agent._build_prior_turn(history)
    assert prior is not None
    assert prior.assistant_text == "Standalone assistant note."
    assert prior.user_text == ""


def test_build_prior_turn_returns_none_without_assistant() -> None:
    """No assistant in history → no prior turn (first user message of the session)."""
    rag = _make_rag_mock()
    agent = RAGRetrievalAgent(rag)
    history: list[dict[str, str]] = [{"role": "user", "content": "first question"}]
    assert agent._build_prior_turn(history) is None


def test_retrieve_skips_chat_for_ner_intent() -> None:
    """The NER branch must not call ``rag.chat`` (no PriorTurn forwarded there)."""
    rag = _make_rag_mock()
    rag.get_collection_ner.return_value = []
    agent = RAGRetrievalAgent(rag)
    request = RetrievalRequest(
        turn=Turn(user_input="extract entities", session_id="session-1"),
        analysis=IntentAnalysis(
            intent="ner",
            confidence=0.9,
            entities={"query": "extract entities"},
        ),
    )

    result = agent.retrieve(request)

    assert rag.chat.call_count == 0
    assert result.tool_used == "ner_sources"


def test_retrieve_pulls_assistant_from_tail_when_more_recent_user_follows() -> None:
    """The trailing user message in history is the *current* turn; pair with previous assistant."""
    rag = _make_rag_mock()
    agent = RAGRetrievalAgent(rag)
    history: list[dict[str, str]] = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1 referencing UN Security Council."},
        {"role": "user", "content": "Please elaborate."},
    ]
    request = RetrievalRequest(
        turn=Turn(user_input="Please elaborate.", session_id="session-1"),
        analysis=IntentAnalysis(
            intent="qa",
            confidence=0.9,
            entities={"query": "Please elaborate."},
            rewritten_query="UN Security Council references",
        ),
        history=history,
    )

    agent.retrieve(request)

    _, kwargs = rag.chat.call_args
    prior_turn = kwargs.get("prior_turn")
    assert isinstance(prior_turn, PriorTurn)
    assert prior_turn.assistant_text == "A1 referencing UN Security Council."
    assert prior_turn.user_text == "Q1"


def test_retrieve_history_field_defaults_empty_list() -> None:
    """A RetrievalRequest without history should not crash; prior_turn=None."""
    rag = _make_rag_mock()
    agent = RAGRetrievalAgent(rag)
    request = RetrievalRequest(
        turn=Turn(user_input="hello"),
        analysis=IntentAnalysis(
            intent="qa", confidence=0.9, entities={"query": "hello"}
        ),
    )
    # history defaults to []; verify here
    assert request.history == []
    agent.retrieve(request)
    _, kwargs = rag.chat.call_args
    assert kwargs.get("prior_turn") is None
