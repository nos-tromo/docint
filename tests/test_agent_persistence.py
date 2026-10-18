"""A turn stores how it was answered; reads no longer infer it from its trail. All text is invented."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from agent_fakes import answer_round, scripted_loop, streaming_tool_llm, turn_budget
from assessment_fakes import bad_request
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from docint.agents.assessment.ledger import EvidenceLedger
from docint.agents.assessment.split import SENTINEL
from docint.agents.types import AnswerMode
from docint.core.rag import AGENT_LEDGER_CAP, AgentTurn
from docint.core.state.base import Base
from docint.core.state.session_manager import SessionManager
from docint.core.state.turn import Turn

QUESTION = "How many boats were counted at the pier?"
ANSWER = "Eleven boats were counted [1]."

#: An assessment's trail: its source-backed part ends nine characters in.
ASSESSMENT_TRAIL: dict[str, Any] = {
    "steps": 2,
    "actions": [{"action": "search", "detail": "harbour survey method", "found": 3}],
    "reason": None,
    "answer": "split",
    "shortened": 0,
    "evidence_chars": 9,
}

#: A researched lookup's trail: it searched twice, then answered plainly.
LOOKUP_TRAIL: dict[str, Any] = {
    "steps": 2,
    "actions": [
        {"action": "search", "detail": "pier counts by month", "found": 4},
        {"action": "search", "detail": "ferry days", "found": 1},
    ],
    "reason": None,
    "answer": "plain",
    "evidence_chars": None,
}


def _normalize(query: str, result: Any, **kwargs: Any) -> dict[str, Any]:
    """Mirror the normalizer's shape closely enough for a turn to persist.

    Args:
        query (str): The user's message.
        result (Any): The response object built from the turn's result.
        **kwargs (Any): The normalizer's keyword arguments.

    Returns:
        dict[str, Any]: A normalized payload.
    """
    return {
        "query": query,
        "response": result.response or "No answer.",
        "sources": [],
        "reasoning": None,
        "retrieval_query": kwargs["retrieval_query"],
        "coverage_unit": kwargs["coverage_unit"],
        "retrieval_mode": kwargs["retrieval_mode"],
        "rerank": None,
        "visual": None,
    }


@pytest.fixture
def session_manager() -> Generator[SessionManager, None, None]:
    """SessionManager on an in-memory SQLite DB with a mocked RAG.

    Returns:
        Generator[SessionManager, None, None]: The session manager.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    rag_mock = MagicMock()
    rag_mock.index = object()
    rag_mock.qdrant_collection = "test_collection"
    rag_mock.qdrant_host = "http://qdrant:6333"
    rag_mock.embed_model_id = "embed-model"
    rag_mock.sparse_model_id = "sparse-model"
    rag_mock.text_model_id = "text-model"
    rag_mock.retrieve_similarity_top_k = 20
    rag_mock.rerank_top_n = 5
    rag_mock.rewrite_retrieval_query.return_value = "rewritten question"
    rag_mock.expand_query_with_graph_with_debug.return_value = ("expanded question", {"applied": False})
    rag_mock._infer_collection_profile.return_value = {"coverage_unit": "documents"}
    rag_mock._normalize_response_data.side_effect = _normalize
    cast(Any, rag_mock.get_source_by_node_id).return_value = None
    rag_mock.reasoning_requested = False
    rag_mock.agent_char_token_ratio = 3.5
    rag_mock.graphrag_enabled = False
    sm = SessionManager(rag=rag_mock)
    sm._SessionMaker = sessionmaker(bind=engine)
    yield sm
    engine.dispose()


def _resp(*node_ids: str) -> MagicMock:
    """Build a response whose sources carry the given node ids, in order.

    Args:
        *node_ids (str): Node ids in the order the answer cites them.

    Returns:
        MagicMock: A response double with ``source_nodes``.
    """
    resp = MagicMock()
    resp.metadata = cast(dict[str, Any], {})
    nodes = []
    for node_id in node_ids:
        node = MagicMock()
        node.node_id = node_id
        node.metadata = {"filename": "survey.pdf", "file_hash": f"h-{node_id}", "source": "document"}
        scored = MagicMock()
        scored.node = node
        scored.score = 0.5
        nodes.append(scored)
    resp.source_nodes = nodes
    return resp


def _stored_modes(sm: SessionManager) -> list[str | None]:
    """Read the raw answer mode column of every stored turn, oldest first.

    Args:
        sm (SessionManager): The session manager under test.

    Returns:
        list[str | None]: The stored values; ``None`` where a row carries no mode.
    """
    with sm._session_scope() as s:
        return [mode for (mode,) in s.query(Turn.answer_mode).order_by(Turn.idx)]


def _exported(sm: SessionManager, out_dir: Path) -> list[dict[str, Any]]:
    """Export session ``s1`` and read its messages back.

    Args:
        sm (SessionManager): The session manager under test.
        out_dir (Path): Where to export.

    Returns:
        list[dict[str, Any]]: One entry per turn, oldest first.
    """
    out = sm.export_session("s1", out_dir=out_dir)
    return [json.loads(line) for line in (out / "messages.jsonl").read_text(encoding="utf-8").splitlines()]


def test_a_researched_lookup_reloads_as_a_lookup_with_its_trail(session_manager: SessionManager) -> None:
    """A trail no longer makes an assessment: the mode stored beside it says what the turn was.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    session_manager._persist_turn(
        "s1", QUESTION, _resp("n1"), {"response": ANSWER}, agent_trail=LOOKUP_TRAIL, answer_mode="lookup"
    )

    assistant = session_manager.get_session_history("s1", owner=None)[1]

    assert assistant["answer_mode"] == "lookup"
    assert assistant["agent"] == LOOKUP_TRAIL


def test_a_legacy_assessment_row_still_reloads_as_an_assessment(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    """A row written before the mode was stored has none, and its trail still marks it an assessment.

    Args:
        session_manager (SessionManager): The session manager fixture.
        tmp_path (Path): A temporary directory.
    """
    with session_manager._session_scope() as s:
        session_manager._load_or_create_convo(s, "s1")
        # Raw SQL names no answer_mode column, exactly as the release that wrote such a row did.
        s.execute(
            text(
                "INSERT INTO turns (conversation_id, idx, user_text, model_response, agent_trail, created_at) "
                "VALUES ('s1', 0, :question, :answer, :trail, '2026-01-01 00:00:00')"
            ),
            {"question": QUESTION, "answer": ANSWER, "trail": json.dumps(ASSESSMENT_TRAIL)},
        )
        s.commit()
        assert s.execute(text("SELECT answer_mode FROM turns")).scalar() is None

    assistant = session_manager.get_session_history("s1", owner=None)[1]
    exported = _exported(session_manager, tmp_path)[0]

    assert assistant["answer_mode"] == "assessment"
    assert assistant["agent"] == ASSESSMENT_TRAIL
    assert exported["answer_mode"] == "assessment"
    assert exported["agent"] == ASSESSMENT_TRAIL


@pytest.mark.parametrize(
    ("first_mode", "first_trail", "second_mode", "second_trail"),
    [
        ("assessment", ASSESSMENT_TRAIL, "lookup", LOOKUP_TRAIL),
        ("lookup", LOOKUP_TRAIL, "assessment", ASSESSMENT_TRAIL),
        ("assessment", ASSESSMENT_TRAIL, "lookup", None),
        ("assessment", ASSESSMENT_TRAIL, None, None),
    ],
    ids=[
        "an assessment replaced by a researched lookup",
        "a researched lookup replaced by an assessment",
        "an assessment replaced by a plain lookup",
        "an assessment replaced by a caller that names no mode",
    ],
)
def test_answer_mode_is_written_on_replace(
    session_manager: SessionManager,
    tmp_path: Path,
    first_mode: AnswerMode,
    first_trail: dict[str, Any] | None,
    second_mode: AnswerMode | None,
    second_trail: dict[str, Any] | None,
) -> None:
    """Replacing a turn rewrites its mode with its trail, so a retry never keeps the first attempt's label.

    Args:
        session_manager (SessionManager): The session manager fixture.
        tmp_path (Path): A temporary directory.
        first_mode (AnswerMode): The mode of the turn being replaced.
        first_trail (dict[str, Any] | None): Its trail.
        second_mode (AnswerMode | None): The mode the replacement names, if any.
        second_trail (dict[str, Any] | None): The replacement's trail.
    """
    session_manager._persist_turn(
        "s1", QUESTION, _resp("n1"), {"response": "First [1]."}, agent_trail=first_trail, answer_mode=first_mode
    )
    session_manager._persist_turn(
        "s1",
        QUESTION,
        _resp("n2"),
        {"response": "Second [1]."},
        replace_idx=0,
        agent_trail=second_trail,
        answer_mode=second_mode,
    )

    exported = _exported(session_manager, tmp_path)

    assert _stored_modes(session_manager) == [second_mode]
    assert [message["answer_mode"] for message in exported] == [second_mode or "lookup"]
    assert [message["agent"] for message in exported] == [second_trail]


@pytest.mark.parametrize("stored", [None, "lookup"], ids=["no stored mode", "a stored lookup"])
def test_a_plain_lookup_replays_without_agent_keys_whatever_its_stored_mode(
    session_manager: SessionManager, stored: AnswerMode | None
) -> None:
    """History keeps the keys for turns the agent loop answered; a plain lookup is unchanged.

    Args:
        session_manager (SessionManager): The session manager fixture.
        stored (AnswerMode | None): The mode written with the turn.
    """
    session_manager._persist_turn("s1", QUESTION, _resp("n1"), {"response": ANSWER}, answer_mode=stored)

    assistant = session_manager.get_session_history("s1", owner=None)[1]

    assert "answer_mode" not in assistant
    assert "agent" not in assistant


@pytest.mark.parametrize(
    ("stored", "trail", "expected"),
    [
        ("lookup", LOOKUP_TRAIL, "lookup"),
        ("assessment", ASSESSMENT_TRAIL, "assessment"),
        ("lookup", None, "lookup"),
        (None, ASSESSMENT_TRAIL, "assessment"),
        (None, None, "lookup"),
    ],
    ids=["a researched lookup", "an assessment", "a plain lookup", "a legacy assessment", "a legacy lookup"],
)
def test_the_export_names_every_turn_by_the_same_rule_as_history(
    session_manager: SessionManager,
    tmp_path: Path,
    stored: AnswerMode | None,
    trail: dict[str, Any] | None,
    expected: str,
) -> None:
    """The export and the history reader agree: the stored mode wins, and legacy rows fall back to the trail.

    Args:
        session_manager (SessionManager): The session manager fixture.
        tmp_path (Path): A temporary directory.
        stored (AnswerMode | None): The mode written with the turn.
        trail (dict[str, Any] | None): The trail written with the turn.
        expected (str): The mode both readers must report.
    """
    session_manager._persist_turn(
        "s1", QUESTION, _resp("n1"), {"response": ANSWER}, agent_trail=trail, answer_mode=stored
    )

    exported = _exported(session_manager, tmp_path)[0]
    assistant = session_manager.get_session_history("s1", owner=None)[1]

    assert exported["answer_mode"] == expected
    assert exported["agent"] == trail
    assert assistant.get("answer_mode") == (expected if trail is not None else None)


@pytest.mark.parametrize("entry", ["chat", "stream_chat"])
@pytest.mark.parametrize(
    ("agent", "script", "stored"),
    [
        (False, [], "lookup"),
        (True, [answer_round(ANSWER)], "lookup"),
        (True, [answer_round(f"{ANSWER}\n{SENTINEL}\nA second count would help.")], "assessment"),
        (True, [bad_request("The invented endpoint refuses tools.")], "lookup"),
    ],
    ids=["a plain lookup", "an agent lookup", "an agent assessment", "an agent turn that falls back"],
)
def test_each_path_stores_the_mode_it_answered_in(
    session_manager: SessionManager, entry: str, agent: bool, script: list[Any], stored: str
) -> None:
    """Each path stores the mode it answered in: a turn that falls back is a lookup, whatever was asked for.

    Args:
        session_manager (SessionManager): The session manager fixture.
        entry (str): The entry point under test, ``chat`` or ``stream_chat``.
        agent (bool): Whether the caller asked for an agent turn.
        script (list[Any]): The agent's model rounds.
        stored (str): The mode that must end up on the row.
    """
    engine = MagicMock()
    engine.retrieve.return_value = cast(list[Any], [])
    engine.query.side_effect = lambda _query: SimpleNamespace(
        response=ANSWER, response_gen=iter([ANSWER]), source_nodes=[], metadata={}
    )
    rag = cast(Any, session_manager.rag)
    rag.query_engine = engine
    rag.build_query_engine.return_value = engine
    llm = streaming_tool_llm(script)
    rag.agent_turn.side_effect = lambda _engine, **kwargs: AgentTurn(
        loop=scripted_loop(llm, target=kwargs["retrieval_target"]),
        ledger=EvidenceLedger(cap=AGENT_LEDGER_CAP),
        budget=turn_budget(),
        reasoning=False,
    )

    if entry == "chat":
        session_manager.chat(QUESTION, session_id="s1", agent=agent)
    else:
        list(session_manager.stream_chat(QUESTION, session_id="s1", agent=agent))

    assert _stored_modes(session_manager) == [stored]
