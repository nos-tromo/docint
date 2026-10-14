"""Assessment turns through the session manager. All text is invented."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from assessment_fakes import llm_view, scored_node, streaming_llm
from llama_index.core.schema import NodeWithScore
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from docint.agents.assessment.runner import AssessmentRunner
from docint.agents.assessment.split import SENTINEL
from docint.agents.types import AgentProgress
from docint.core.rag import RAG
from docint.core.state.base import Base
from docint.core.state.session_manager import SessionManager
from docint.core.state.turn import Turn

QUESTION = "What could be improved in this study?"
EVIDENCE = "The study evaluates on one dataset [1]."
ANSWER = [EVIDENCE, f"\n{SENTINEL}\n", "A second dataset would help."]


def _normalize(query: str, result: Any, **kwargs: Any) -> dict[str, Any]:
    """Mirror the normalizer's shape: text, numbered sources and routing fields.

    Args:
        query (str): The user's message.
        result (Any): The response object built from the runner's result.
        **kwargs (Any): The normalizer's keyword arguments.

    Returns:
        dict[str, Any]: A normalized payload; an empty answer becomes ``No answer.``.
    """
    return {
        "query": query,
        "response": result.response or "No answer.",
        "sources": [
            {"id": scored.node.node_id, "citation_index": scored.node.metadata.get("citation_index")}
            for scored in result.source_nodes
        ],
        "reasoning": None,
        "retrieval_query": kwargs["retrieval_query"],
        "coverage_unit": kwargs["coverage_unit"],
        "retrieval_mode": kwargs["retrieval_mode"],
        "rerank": None,
        "visual": None,
    }


@pytest.fixture(autouse=True)
def _english(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the label language.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")


@pytest.fixture
def session_manager() -> Generator[SessionManager, None, None]:
    """SessionManager on an in-memory DB with a RAG double wired for assessments.

    Returns:
        Generator[SessionManager, None, None]: The session manager.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    rag_mock = MagicMock()
    rag_mock.index = object()
    rag_mock.qdrant_collection = "test_collection"
    rag_mock.rewrite_retrieval_query.return_value = "rewritten question"
    rag_mock.expand_query_with_graph_with_debug.return_value = ("expanded question", {"applied": False})
    rag_mock.graph_debug_skipped.return_value = {"applied": False, "reason": "scoped"}
    rag_mock._infer_collection_profile.return_value = {"coverage_unit": "documents"}
    mode = MagicMock()
    mode.value = "compact"
    rag_mock._resolve_chat_response_mode.return_value = mode
    rag_mock._normalize_response_data.side_effect = _normalize
    cast(Any, rag_mock.get_source_by_node_id).return_value = None
    sm = SessionManager(rag=rag_mock)
    sm._SessionMaker = sessionmaker(bind=engine)
    yield sm
    engine.dispose()


def _seed() -> list[NodeWithScore]:
    """Return two passages of an invented study, each with its own file hash.

    Returns:
        list[NodeWithScore]: The seed evidence.
    """
    return [
        scored_node("n1", "We evaluate on one dataset.", filename="study.pdf", page=4, file_hash="h1"),
        scored_node("n2", "Results improve by two points.", filename="study.pdf", page=6, file_hash="h2"),
    ]


def _arm(sm: SessionManager, *, nodes: list[NodeWithScore], tokens: list[str] | None = None) -> MagicMock:
    """Route the turn's engine and runner to doubles.

    Args:
        sm (SessionManager): The session manager under test.
        nodes (list[NodeWithScore]): What the engine's ``retrieve`` returns.
        tokens (list[str] | None): The answer tokens; ``ANSWER`` by default.

    Returns:
        MagicMock: The engine double.
    """
    engine = MagicMock()
    engine.retrieve.return_value = nodes
    rag = cast(Any, sm.rag)
    rag.build_query_engine.return_value = engine
    rag.query_engine = engine
    rag.assessment_runner.side_effect = lambda engine, scoped: AssessmentRunner(
        answer_llm=cast(Any, streaming_llm(tokens or ANSWER)),
        llm_view=llm_view,
        ctx_window=8192,
        char_token_ratio=3.5,
        reasoning=False,
    )
    return engine


def test_an_assessment_turn_streams_progress_text_then_a_final_frame(session_manager: SessionManager) -> None:
    """Progress and labelled text stream first; the final frame names the mode and trail.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    engine = _arm(session_manager, nodes=_seed())

    items = list(session_manager.stream_chat(QUESTION, session_id="s1", answer_mode="assessment"))

    progress = [item for item in items if isinstance(item, AgentProgress)]
    text = "".join(item for item in items if isinstance(item, str))
    final = items[-1]
    assert [step.action for step in progress] == ["seed", "answer"]
    assert "**Model assessment**" in text
    assert isinstance(final, dict)
    assert final["answer_mode"] == "assessment"
    assert final["retrieval_mode"] == "agent"
    assert final["agent"]["answer"] == "split"
    assert final["agent"]["evidence_chars"] == len(EVIDENCE)
    assert final["response"].startswith(EVIDENCE)
    assert final["turn_idx"] == 0
    assert engine.retrieve.call_args.args[0].query_str == "expanded question"


def test_assessment_citations_persist_in_ledger_order(session_manager: SessionManager) -> None:
    """Replay numbers citations by position, so they must be stored in ledger order.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(session_manager, nodes=_seed())

    list(session_manager.stream_chat(QUESTION, session_id="s1", answer_mode="assessment"))

    assistant = session_manager.get_session_history("s1", owner=None)[1]
    assert [(src["file_hash"], src["citation_index"]) for src in assistant["sources"]] == [("h1", 1), ("h2", 2)]
    assert assistant["answer_mode"] == "assessment"


def test_a_scoped_assessment_keeps_its_scope_and_never_researches(session_manager: SessionManager) -> None:
    """A pinned scope is never widened, and the turn still reports itself as scoped.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(session_manager, nodes=_seed())

    final = list(
        session_manager.stream_chat(QUESTION, session_id="s1", scoped_node_ids=["n1", "n2"], answer_mode="assessment")
    )[-1]

    assert isinstance(final, dict)
    assert final["retrieval_mode"] == "scoped"
    assert final["scoped_chunk_count"] == 2
    assert cast(Any, session_manager.rag).assessment_runner.call_args.kwargs["scoped"] is True


def test_a_lookup_turn_never_builds_a_runner(session_manager: SessionManager) -> None:
    """The default mode is today's pipeline, byte for byte.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    streaming_engine = MagicMock()
    response = MagicMock()
    response.response_gen = iter(["Alice [1]."])
    response.source_nodes = cast(list[Any], [])
    streaming_engine.query.return_value = response
    cast(Any, session_manager.rag).build_query_engine.return_value = streaming_engine

    final = list(session_manager.stream_chat("Who wrote it?", session_id="s1"))[-1]

    assert isinstance(final, dict)
    assert "answer_mode" not in final
    cast(Any, session_manager.rag).assessment_runner.assert_not_called()


def test_chat_drains_the_assessment_to_its_final_payload(session_manager: SessionManager) -> None:
    """The non-streaming path returns the same final payload.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(session_manager, nodes=_seed())

    response = session_manager.chat(QUESTION, session_id="s1", answer_mode="assessment")

    assert response["answer_mode"] == "assessment"
    assert response["agent"]["answer"] == "split"


def test_an_unsplit_answer_is_labelled_whole(session_manager: SessionManager) -> None:
    """Without a sentinel nothing may pass as source-backed.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(session_manager, nodes=_seed(), tokens=["No separation here [1]."])

    final = list(session_manager.stream_chat(QUESTION, session_id="s1", answer_mode="assessment"))[-1]

    assert isinstance(final, dict)
    assert final["agent"]["answer"] == "unsplit"
    assert final["agent"]["evidence_chars"] == 0
    assert final["response"].startswith("**Model assessment**")


def test_no_evidence_answers_with_the_unlabelled_fallback(session_manager: SessionManager) -> None:
    """There is nothing to judge, so the normalizer's fallback stands unlabelled.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(session_manager, nodes=[])

    final = list(session_manager.stream_chat(QUESTION, session_id="s1", answer_mode="assessment"))[-1]

    assert isinstance(final, dict)
    assert (final["agent"]["reason"], final["agent"]["answer"]) == ("no_evidence", "empty")
    assert final["response"] == "No answer."


def test_a_sentinel_only_reply_never_reaches_the_user_as_text(session_manager: SessionManager) -> None:
    """A reply holding nothing but the sentinel is empty, so the fallback stands, not the token.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(session_manager, nodes=_seed(), tokens=[SENTINEL])

    final = list(session_manager.stream_chat(QUESTION, session_id="s1", answer_mode="assessment"))[-1]

    assert isinstance(final, dict)
    assert (final["agent"]["answer"], final["agent"]["evidence_chars"]) == ("empty", None)
    assert final["response"] == "No answer."
    assert session_manager.get_session_history("s1", owner=None)[1]["content"] == "No answer."


def test_a_reply_of_only_reasoning_still_reaches_the_normalizer(session_manager: SessionManager) -> None:
    """The normalizer lifts reasoning off the raw reply, so a reply with no answer must not be blanked.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    reply = "<think>weighing the passages</think>"
    _arm(session_manager, nodes=_seed(), tokens=[reply])

    final = list(session_manager.stream_chat(QUESTION, session_id="s1", answer_mode="assessment"))[-1]

    assert isinstance(final, dict)
    assert final["agent"]["answer"] == "empty"
    assert cast(Any, session_manager.rag)._normalize_response_data.call_args.args[1].response == reply


def test_a_stateless_assessment_persists_nothing(session_manager: SessionManager) -> None:
    """``persist=False`` serves stateless requests: no turn row, no turn index.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    engine = _arm(session_manager, nodes=_seed())

    final = list(
        session_manager.stream_assessment(
            QUESTION,
            engine=engine,
            expanded_query=QUESTION,
            retrieval_query=QUESTION,
            prior_turn=None,
            scoped_node_ids=None,
            graph_debug={},
            coverage_unit="documents",
            metadata_filters_active=False,
            metadata_filter_rules=None,
            retrieval_target="all",
            session_id=None,
            owner=None,
            replace_turn_idx=None,
            turn_started_at=0.0,
            persist=False,
        )
    )[-1]

    assert isinstance(final, dict)
    assert final["turn_idx"] is None
    with session_manager._session_scope() as s:
        assert s.query(Turn).count() == 0


def test_a_stateless_assessment_runs_without_a_session() -> None:
    """The stateless entry point builds the turn's engine and never persists."""
    rag = MagicMock()
    rag._infer_collection_profile.return_value = {"coverage_unit": "documents"}
    manager = rag.ensure_session_manager.return_value
    manager.stream_assessment.return_value = iter(["text", {"answer_mode": "assessment"}])

    items = list(RAG.stream_stateless_assessment(rag, QUESTION, expanded_query="expanded", graph_debug={}))

    kwargs = manager.stream_assessment.call_args.kwargs
    assert items[-1] == {"answer_mode": "assessment"}
    assert (kwargs["persist"], kwargs["session_id"], kwargs["prior_turn"]) == (False, None, None)
    assert kwargs["engine"] is rag.build_query_engine.return_value
    assert kwargs["expanded_query"] == "expanded"
    assert rag.build_query_engine.call_args.kwargs["streaming"] is True


def test_a_stateless_assessment_streams_through_the_real_manager(session_manager: SessionManager) -> None:
    """The entry point's keywords fit ``stream_assessment``, filters count as active, nothing is written.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    rag = cast(Any, session_manager.rag)
    rag.ensure_session_manager.return_value = session_manager
    _arm(session_manager, nodes=_seed())

    items = list(
        RAG.stream_stateless_assessment(
            rag, QUESTION, expanded_query="expanded", graph_debug={}, vector_store_kwargs={"qdrant_filters": object()}
        )
    )

    final = items[-1]
    assert isinstance(final, dict)
    assert [item.action for item in items if isinstance(item, AgentProgress)] == ["seed", "answer"]
    assert (final["answer_mode"], final["session_id"], final["turn_idx"]) == ("assessment", None, None)
    assert rag.build_query_engine.call_args.kwargs["metadata_filters_active"] is True
    assert rag._normalize_response_data.call_args.kwargs["metadata_filters_active"] is True
    with session_manager._session_scope() as s:
        assert s.query(Turn).count() == 0
