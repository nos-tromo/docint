"""Tests for replace-in-place turn persistence used by the corrective retry.

One user message must persist as exactly one turn. When the corrective retry
re-answers a rejected turn, the second attempt overwrites the first row instead
of appending — otherwise a reloaded session shows the user asking twice, with
the discarded weak answer still on screen.
"""

from collections.abc import Generator
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from docint.core.state.base import Base
from docint.core.state.session_manager import SessionManager
from docint.core.state.turn import Turn


@pytest.fixture
def session_manager() -> Generator[SessionManager, None, None]:
    """SessionManager bound to an in-memory SQLite DB.

    Mirrors the fixture in ``test_session_manager_validation.py`` so tests
    share the same RAG mock surface.

    Yields:
        SessionManager: Manager under test.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine)

    rag_mock = MagicMock()
    rag_mock.index = None
    rag_mock.qdrant_collection = "test_collection"
    rag_mock.qdrant_host = "http://qdrant:6333"
    rag_mock.embed_model_id = "embed-model"
    rag_mock.sparse_model_id = "sparse-model"
    rag_mock.text_model_id = "text-model"
    rag_mock.retrieve_similarity_top_k = 20
    rag_mock.rerank_top_n = 5
    rag_mock.conversation_summary_prompt = "Summarize turns:\n"
    rag_mock.rewrite_retrieval_query.return_value = "rewritten question"
    rag_mock._infer_collection_profile.return_value = {"coverage_unit": "documents"}
    mode = MagicMock()
    mode.value = "compact"
    rag_mock._resolve_chat_response_mode.return_value = mode
    cast(Any, rag_mock.get_source_by_node_id).return_value = None

    sm = SessionManager(rag=rag_mock)
    sm._SessionMaker = SessionMaker
    yield sm
    engine.dispose()


def _response(node_ids: list[str]) -> MagicMock:
    """Build a response mock carrying one source node per id.

    Args:
        node_ids: Node ids the persisted citations should reference.

    Returns:
        MagicMock: Response stand-in with ``metadata`` and ``source_nodes``.
    """
    resp = MagicMock()
    resp.metadata = cast(dict[str, Any], {})
    src_nodes = []
    for node_id in node_ids:
        node = MagicMock()
        node.node_id = node_id
        node.metadata = {"filename": f"{node_id}.pdf", "file_hash": f"hash-{node_id}", "source": "document"}
        src = MagicMock()
        src.node = node
        src.score = 0.5
        src_nodes.append(src)
    resp.source_nodes = src_nodes
    return resp


def _turns(sm: SessionManager, session_id: str) -> list[dict[str, Any]]:
    """Read every persisted turn for a conversation, ordered by index.

    Returns plain dicts rather than ORM rows: the rows detach when the session
    scope closes, and reading a relationship off a detached instance raises.

    Args:
        sm: The session manager under test.
        session_id: Conversation id to read.

    Returns:
        list[dict[str, Any]]: One mapping per turn, in index order.
    """
    with sm._session_scope() as s:
        rows = s.query(Turn).filter_by(conversation_id=session_id).order_by(Turn.idx).all()
        return [
            {
                "idx": t.idx,
                "user_text": t.user_text,
                "rewritten_query": t.rewritten_query,
                "model_response": t.model_response,
                "citation_node_ids": [c.node_id for c in t.citations],
            }
            for t in rows
        ]


def test_replace_overwrites_the_turn_in_place(session_manager: SessionManager) -> None:
    """The retry's answer replaces the first one under the same index.

    Args:
        session_manager: The session manager fixture.
    """
    session_id = "sess-replace"
    first_idx = session_manager._persist_turn(
        session_id,
        "What did the UN say?",
        _response(["n1"]),
        {"response": "Evidence insufficient.", "reasoning": None, "retrieval_query": "UN say"},
    )

    second_idx = session_manager._persist_turn(
        session_id,
        "Security Council resolutions",
        _response(["n2", "n3"]),
        {"response": "The Council adopted three resolutions.", "reasoning": None},
        replace_idx=first_idx,
    )

    assert second_idx == first_idx
    turns = _turns(session_manager, session_id)
    assert len(turns) == 1
    assert turns[0]["model_response"] == "The Council adopted three resolutions."


def test_replace_preserves_the_user_text(session_manager: SessionManager) -> None:
    """The reformulated query never overwrites what the user typed.

    Args:
        session_manager: The session manager fixture.
    """
    session_id = "sess-user-text"
    idx = session_manager._persist_turn(
        session_id,
        "What did the UN say?",
        _response([]),
        {"response": "Evidence insufficient.", "reasoning": None},
    )

    session_manager._persist_turn(
        session_id,
        "Security Council resolutions on sanctions",
        _response([]),
        {"response": "Better answer.", "reasoning": None, "retrieval_query": "Security Council resolutions"},
        replace_idx=idx,
    )

    turn = _turns(session_manager, session_id)[0]
    assert turn["user_text"] == "What did the UN say?"
    # The machine-side query is still recorded, just not as the user's message.
    assert turn["rewritten_query"] == "Security Council resolutions"


def test_replace_swaps_the_citations(session_manager: SessionManager) -> None:
    """The rejected answer's sources do not survive the replacement.

    Args:
        session_manager: The session manager fixture.
    """
    session_id = "sess-citations"
    idx = session_manager._persist_turn(
        session_id,
        "What did the UN say?",
        _response(["stale-1", "stale-2"]),
        {"response": "Evidence insufficient.", "reasoning": None},
    )

    session_manager._persist_turn(
        session_id,
        "Security Council resolutions",
        _response(["fresh-1"]),
        {"response": "Better answer.", "reasoning": None},
        replace_idx=idx,
    )

    turn = _turns(session_manager, session_id)[0]
    assert turn["citation_node_ids"] == ["fresh-1"]


def test_replace_falls_back_to_appending_when_the_row_is_missing(
    session_manager: SessionManager,
) -> None:
    """A stale index appends rather than losing the answer.

    Args:
        session_manager: The session manager fixture.
    """
    session_id = "sess-missing"

    idx = session_manager._persist_turn(
        session_id,
        "What did the UN say?",
        _response([]),
        {"response": "An answer.", "reasoning": None},
        replace_idx=41,
    )

    assert idx == 0
    assert len(_turns(session_manager, session_id)) == 1


def test_append_still_advances_the_index(session_manager: SessionManager) -> None:
    """Ordinary turns keep appending after a replaced one.

    Args:
        session_manager: The session manager fixture.
    """
    session_id = "sess-append"
    first = session_manager._persist_turn(
        session_id, "q1", _response([]), {"response": "a1", "reasoning": None}
    )
    session_manager._persist_turn(
        session_id, "retry", _response([]), {"response": "a1-better", "reasoning": None}, replace_idx=first
    )
    second = session_manager._persist_turn(
        session_id, "q2", _response([]), {"response": "a2", "reasoning": None}
    )

    assert (first, second) == (0, 1)
    turns = _turns(session_manager, session_id)
    assert [t["user_text"] for t in turns] == ["q1", "q2"]
    assert [t["model_response"] for t in turns] == ["a1-better", "a2"]


def _wire_chat_engine(sm: SessionManager) -> None:
    """Point the RAG mock's query engine at a canned response.

    Args:
        sm: The session manager whose RAG mock should answer chat turns.
    """
    engine = MagicMock()
    engine.query.return_value = _response([])
    sm.rag.query_engine = engine
    cast(Any, sm.rag).build_query_engine.return_value = engine
    cast(Any, sm.rag).expand_query_with_graph_with_debug.return_value = ("expanded", {})
    # A fresh dict per call: production's ``_normalize_response_data`` builds
    # one each time, and the turn index is stamped onto it in place.
    cast(Any, sm.rag)._normalize_response_data.side_effect = lambda *a, **k: {"response": "Hi", "sources": []}


def test_chat_returns_the_persisted_turn_index(
    session_manager: SessionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``chat`` hands back the join key a retry needs to replace the turn.

    Args:
        session_manager: The session manager fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    _wire_chat_engine(session_manager)
    monkeypatch.setattr(SessionManager, "_maybe_update_summary", lambda *args, **kwargs: None)

    first = session_manager.chat("hello", session_id="sess-turn-idx")
    second = session_manager.chat("again", session_id="sess-turn-idx")

    assert (first["turn_idx"], second["turn_idx"]) == (0, 1)


def test_chat_replacing_a_turn_skips_the_rolling_summary(
    session_manager: SessionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replaced turn must not fold the answer it just discarded into context.

    Args:
        session_manager: The session manager fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    _wire_chat_engine(session_manager)
    calls: list[str] = []
    monkeypatch.setattr(
        SessionManager,
        "_maybe_update_summary",
        lambda self, session_id, *args, **kwargs: calls.append(session_id),
    )

    first = session_manager.chat("hello", session_id="sess-summary")
    session_manager.chat("reformulated", session_id="sess-summary", replace_turn_idx=first["turn_idx"])

    assert calls == ["sess-summary"]
    assert len(_turns(session_manager, "sess-summary")) == 1
