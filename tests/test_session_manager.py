from typing import Generator
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
    rag_mock.index = None  # Simulate no index initially

    sm = SessionManager(rag=rag_mock)
    sm._SessionMaker = SessionMaker
    yield sm
    engine.dispose()


def test_persist_and_retrieve_citation_with_hash(
    session_manager: SessionManager,
) -> None:
    """
    Test persisting and retrieving a turn with citations that include file_hash.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    session_id = "test_session"
    user_msg = "hello"

    # Mock response object
    resp_mock = MagicMock()
    resp_mock.metadata = {}
    resp_mock.source_nodes = []

    # Create a source node with file_hash
    node_mock = MagicMock()
    node_mock.node_id = "node1"
    node_mock.metadata = {
        "filename": "test.pdf",
        "file_hash": "hash123",
        "page_label": "1",
        "source": "document",
    }

    src_node_mock = MagicMock()
    src_node_mock.node = node_mock
    src_node_mock.score = 0.9

    resp_mock.source_nodes = [src_node_mock]

    data = {"response": "Hi there", "reasoning": "None"}

    # Persist
    session_manager._persist_turn(session_id, user_msg, resp_mock, data)

    # Retrieve
    # We don't mock _get_node_text_by_id, so it will return None/empty string

    history = session_manager.get_session_history(session_id)
    assert len(history) == 2  # User + Assistant

    assistant_msg = history[1]
    assert assistant_msg["role"] == "assistant"
    assert len(assistant_msg["sources"]) == 1

    source = assistant_msg["sources"][0]
    assert source["filename"] == "test.pdf"
    assert source["file_hash"] == "hash123"
    assert source["page"] == 1
    # assert source["preview_text"] == "Preview text" # This will be empty string
