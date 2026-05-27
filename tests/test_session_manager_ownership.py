"""Tests for owner-scoped session listing, history, and deletion."""

from collections.abc import Generator
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from docint.core.state.base import Base
from docint.core.state.session_manager import SessionManager


@pytest.fixture
def session_manager() -> Generator[SessionManager, None, None]:
    """SessionManager bound to an in-memory SQLite DB.

    Mirrors the fixture in ``test_session_manager.py`` so tests share the
    same RAG mock surface.

    Returns:
        Generator[SessionManager, None, None]: The SessionManager instance.
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


def _persist_owned_turn(sm: SessionManager, session_id: str, owner: str) -> None:
    """Create an owned conversation and persist one empty turn into it.

    Args:
        sm (SessionManager): The session manager under test.
        session_id (str): The conversation id.
        owner (str): The owning principal.
    """
    with sm._session_scope() as s:
        sm._load_or_create_convo(s, session_id, owner)

    resp_mock = MagicMock()
    resp_mock.metadata = {}
    resp_mock.source_nodes = []
    sm._owner = owner
    sm._persist_turn(
        session_id,
        "hello",
        resp_mock,
        {"response": "world", "reasoning": None},
    )


def test_load_or_create_convo_stamps_owner(
    session_manager: SessionManager,
) -> None:
    """A newly created conversation records the supplied owner.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    with session_manager._session_scope() as s:
        conv = session_manager._load_or_create_convo(s, "sess-a", "alice")
        assert conv.owner == "alice"


def test_list_sessions_is_owner_scoped(
    session_manager: SessionManager,
) -> None:
    """list_sessions only returns conversations owned by the caller.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _persist_owned_turn(session_manager, "sess-a", "alice")
    _persist_owned_turn(session_manager, "sess-b", "bob")

    alice_sessions = session_manager.list_sessions("alice")
    bob_sessions = session_manager.list_sessions("bob")

    assert {s["id"] for s in alice_sessions} == {"sess-a"}
    assert {s["id"] for s in bob_sessions} == {"sess-b"}


def test_get_session_history_cross_owner_is_not_found(
    session_manager: SessionManager,
) -> None:
    """B reading A's session history gets an empty list (treated as 404).

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _persist_owned_turn(session_manager, "sess-a", "alice")

    own = session_manager.get_session_history("sess-a", "alice")
    cross = session_manager.get_session_history("sess-a", "bob")

    assert len(own) == 2  # user + assistant
    assert cross == []


def test_delete_session_cross_owner_is_not_found(
    session_manager: SessionManager,
) -> None:
    """B deleting A's session returns False and does not delete it.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _persist_owned_turn(session_manager, "sess-a", "alice")

    assert session_manager.delete_session("sess-a", "bob") is False
    # Still present for the real owner.
    assert len(session_manager.get_session_history("sess-a", "alice")) == 2
    # Real owner can delete it.
    assert session_manager.delete_session("sess-a", "alice") is True
    assert session_manager.get_session_history("sess-a", "alice") == []


def test_get_session_history_missing_session_is_empty(
    session_manager: SessionManager,
) -> None:
    """An unknown session id yields an empty history (no existence leak).

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    assert session_manager.get_session_history("nope", "alice") == []


def test_delete_session_missing_session_is_false(
    session_manager: SessionManager,
) -> None:
    """Deleting an unknown session id returns False.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    assert session_manager.delete_session("nope", "alice") is False
