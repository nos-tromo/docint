from typing import Generator, cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from docint.core.session_manager import SessionManager
from docint.core.state.base import Base


@pytest.fixture
def session_manager() -> Generator[SessionManager, None, None]:
    """
    Fixture to create a SessionManager with an in-memory SQLite database.

    Returns:
        Generator[SessionManager, None, None]: The SessionManager instance.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine)

    rag_mock = MagicMock()
    rag_mock.index = None
    rag_mock.qdrant_collection = None # Initialize

    sm = SessionManager(rag=rag_mock)
    sm._SessionMaker = SessionMaker
    yield sm
    engine.dispose()

def test_conversation_stores_collection_name(session_manager: SessionManager) -> None:
    """
    Test that the conversation stores the current collection name.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    session_manager.rag.qdrant_collection = "my-collection"
    session_manager.rag.query_engine = MagicMock()  # Mock engine to avoid RuntimeError in start_session

    session_id = session_manager.start_session()
    
    # Check simple listing
    sessions = session_manager.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["collection"] == "my-collection"
    
    # Verify persistence
    with session_manager._session_scope() as s:
        from docint.core.state.conversation import Conversation
        c = s.get(Conversation, session_id)
        assert c is not None
        assert cast(str, c.collection_name) == "my-collection"
