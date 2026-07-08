"""Tests for session manager collection-scoped conversations."""

from collections.abc import Generator
from typing import cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from docint.core.state.base import Base
from docint.core.state.citation import Citation
from docint.core.state.conversation import Conversation
from docint.core.state.session_manager import SessionManager
from docint.core.state.turn import Turn


@pytest.fixture
def session_manager() -> Generator[SessionManager, None, None]:
    """Fixture to create a SessionManager with an in-memory SQLite database.

    Returns:
        Generator[SessionManager, None, None]: The SessionManager instance.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine)

    rag_mock = MagicMock()
    rag_mock.index = None
    rag_mock.qdrant_collection = None  # Initialize

    sm = SessionManager(rag=rag_mock)
    sm._SessionMaker = SessionMaker
    yield sm
    engine.dispose()


def test_conversation_stores_collection_name(session_manager: SessionManager) -> None:
    """Test that the conversation stores the current collection name.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    session_manager.rag.qdrant_collection = "my-collection"
    session_manager.rag.query_engine = MagicMock()  # Mock engine to avoid RuntimeError in start_session

    session_id = session_manager.start_session()

    # Check simple listing
    sessions = session_manager.list_sessions(owner=None)
    assert len(sessions) == 1
    assert sessions[0]["collection"] == "my-collection"

    # Verify persistence
    with session_manager._session_scope() as s:
        from docint.core.state.conversation import Conversation

        c = s.get(Conversation, session_id)
        assert c is not None
        assert cast(str, c.collection_name) == "my-collection"


def test_list_sessions_filters_by_collection(session_manager: SessionManager) -> None:
    """A physical-collection filter returns only sessions pinned to it."""
    session_manager.rag.qdrant_collection = "col-a"
    a = session_manager.start_session()
    session_manager.rag.qdrant_collection = "col-b"
    b = session_manager.start_session()

    only_a = session_manager.list_sessions(owner=None, collection="col-a")
    assert {s["id"] for s in only_a} == {a}

    only_b = session_manager.list_sessions(owner=None, collection="col-b")
    assert {s["id"] for s in only_b} == {b}

    all_sessions = session_manager.list_sessions(owner=None)
    assert {s["id"] for s in all_sessions} == {a, b}


def test_list_sessions_excludes_null_collection_under_filter(session_manager: SessionManager) -> None:
    """An unpinned (NULL-collection) session never matches a physical filter."""
    session_manager.rag.qdrant_collection = None  # pyrefly: ignore[bad-assignment]  # RAG.qdrant_collection is typed str; runtime property allows None
    unpinned = session_manager.start_session()
    session_manager.rag.qdrant_collection = "col-a"
    pinned = session_manager.start_session()

    scoped = session_manager.list_sessions(owner=None, collection="col-a")
    assert {s["id"] for s in scoped} == {pinned}
    assert unpinned in {s["id"] for s in session_manager.list_sessions(owner=None)}


def _seed_convo(sm: SessionManager, session_id: str, collection: str | None) -> None:
    """Insert a conversation with one turn and one citation, pinned to a collection."""
    with sm._session_scope() as s:
        s.add(Conversation(id=session_id, owner=None, collection_name=collection))
        s.flush()
        turn = Turn(conversation_id=session_id, idx=0, user_text="q", model_response="a")
        s.add(turn)
        s.flush()
        s.add(Citation(turn_id=turn.id, filename="f.pdf"))
        s.commit()


def test_delete_sessions_for_collection_cascades(session_manager: SessionManager) -> None:
    """Deleting a collection's sessions removes their turns and citations too."""
    _seed_convo(session_manager, "s-a", "col-a")
    _seed_convo(session_manager, "s-b", "col-b")

    deleted = session_manager.delete_sessions_for_collection("col-a")
    assert deleted == 1

    with session_manager._session_scope() as s:
        assert s.get(Conversation, "s-a") is None
        assert s.get(Conversation, "s-b") is not None
        assert s.query(Turn).filter_by(conversation_id="s-a").count() == 0
        # Only col-b's single citation survives.
        assert s.query(Citation).count() == 1


def test_delete_sessions_for_collection_is_idempotent(session_manager: SessionManager) -> None:
    """A second run finds nothing to delete and returns 0."""
    _seed_convo(session_manager, "s-a", "col-a")
    assert session_manager.delete_sessions_for_collection("col-a") == 1
    assert session_manager.delete_sessions_for_collection("col-a") == 0
