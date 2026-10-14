"""Persistence and replay of assessment turns. All text is invented."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from docint.agents.types import PriorTurn
from docint.core.state.base import Base
from docint.core.state.session_manager import SessionManager

TRAIL: dict[str, Any] = {
    "steps": 2,
    "actions": [{"action": "search", "detail": "evaluation datasets", "found": 3}],
    "reason": None,
    "answer": "split",
    "shortened": 0,
    "evidence_chars": 9,
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
    rag_mock.qdrant_collection = "test_collection"
    rag_mock.qdrant_host = "http://qdrant:6333"
    rag_mock.embed_model_id = "embed-model"
    rag_mock.sparse_model_id = "sparse-model"
    rag_mock.text_model_id = "text-model"
    rag_mock.retrieve_similarity_top_k = 20
    rag_mock.rerank_top_n = 5
    cast(Any, rag_mock.get_source_by_node_id).return_value = None
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
        node.metadata = {"filename": "study.pdf", "file_hash": f"h-{node_id}", "source": "document"}
        scored = MagicMock()
        scored.node = node
        scored.score = 0.5
        nodes.append(scored)
    resp.source_nodes = nodes
    return resp


def test_history_replays_the_trail(session_manager: SessionManager) -> None:
    """A reloaded assessment turn still says what it was and what it did.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    session_manager._persist_turn(
        "s1", "What could be improved?", _resp("n1"), {"response": "Fact [1]. View."}, agent_trail=TRAIL
    )

    assistant = session_manager.get_session_history("s1", owner=None)[1]

    assert assistant["answer_mode"] == "assessment"
    assert assistant["agent"] == TRAIL


def test_a_lookup_turn_replays_without_assessment_keys(session_manager: SessionManager) -> None:
    """Lookup history stays exactly as it was.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    session_manager._persist_turn("s1", "Who wrote it?", _resp("n1"), {"response": "Alice [1]."})

    assistant = session_manager.get_session_history("s1", owner=None)[1]

    assert "answer_mode" not in assistant
    assert "agent" not in assistant


def test_a_replaced_turn_drops_a_trail_it_no_longer_has(session_manager: SessionManager) -> None:
    """Replacing a turn rewrites its provenance too.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    session_manager._persist_turn("s1", "Q", _resp("n1"), {"response": "Fact [1]. View."}, agent_trail=TRAIL)
    session_manager._persist_turn("s1", "Q", _resp("n2"), {"response": "Plain [1]."}, replace_idx=0)

    assert "agent" not in session_manager.get_session_history("s1", owner=None)[1]


def test_get_prior_turn_reads_the_last_exchange_with_its_evidence_boundary(session_manager: SessionManager) -> None:
    """The prior-turn reader needs neither citation rehydration nor full history.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    session_manager._persist_turn("s1", "q1", _resp("n1"), {"response": "First [1]."})
    session_manager._persist_turn("s1", "q2", _resp("n2"), {"response": "Fact [1]. View."}, agent_trail=TRAIL)

    prior = session_manager.get_prior_turn("s1", owner=None)

    assert prior == PriorTurn(user_text="q2", assistant_text="Fact [1]. View.", evidence_chars=9)
    cast(Any, session_manager.rag.get_source_by_node_id).assert_not_called()


def test_get_prior_turn_is_owner_scoped(session_manager: SessionManager) -> None:
    """Another principal's session reads as no prior turn at all.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    session_manager._persist_turn("s1", "q", _resp("n1"), {"response": "a"}, owner="alice")

    assert session_manager.get_prior_turn("s1", owner="mallory") is None
    assert session_manager.get_prior_turn("s1", owner="alice") is not None


def test_get_prior_turn_is_none_for_an_unknown_session(session_manager: SessionManager) -> None:
    """A first turn has nothing to build on.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    assert session_manager.get_prior_turn("missing", owner=None) is None


def test_export_writes_the_answer_mode(session_manager: SessionManager, tmp_path: Path) -> None:
    """The session export keeps the trail beside the answer text.

    Args:
        session_manager (SessionManager): The session manager fixture.
        tmp_path (Path): A temporary directory.
    """
    session_manager._persist_turn("s1", "Q", _resp("n1"), {"response": "Fact [1]. View."}, agent_trail=TRAIL)

    out = session_manager.export_session("s1", out_dir=tmp_path)

    message = json.loads((out / "messages.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert message["answer_mode"] == "assessment"
    assert message["agent"] == TRAIL
    assert json.loads((out / "session.json").read_text(encoding="utf-8"))["schema_version"] == "1.1.0"
