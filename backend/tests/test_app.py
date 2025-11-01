"""Tests for the FastAPI application layer."""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docint import app as app_module


class DummyRAG:
    """Lightweight stand-in for the global RAG instance."""

    def __init__(self, qdrant_collection: str = "demo", session_return: str = "generated-session") -> None:
        self.qdrant_collection = qdrant_collection
        self.index: Any = object()
        self.query_engine: Any = object()
        self._session_return = session_return
        self.received_session_id: str | None = None
        self.received_question: str | None = None
        self.chat_called = False

    def start_session(self, session_id: str | None) -> str:
        self.received_session_id = session_id
        return session_id or self._session_return

    def chat(self, question: str) -> dict[str, Any]:
        self.chat_called = True
        self.received_question = question
        return {
            "response": "Here is an answer.",
            "sources": [{"title": "Doc", "page": 1}],
        }


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, DummyRAG]]:
    """Provide a TestClient with the global RAG dependency replaced."""

    dummy_rag = DummyRAG()
    monkeypatch.setattr(app_module, "rag", dummy_rag)
    with TestClient(app_module.app) as client:
        yield client, dummy_rag


def test_query_endpoint_returns_answer_and_session(app_client: tuple[TestClient, DummyRAG]) -> None:
    """The /query endpoint should proxy question handling to the RAG instance."""

    client, dummy_rag = app_client

    response = client.post("/query", json={"question": "What is new?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Here is an answer."
    assert payload["sources"] == [{"title": "Doc", "page": 1}]
    assert payload["session_id"] == "generated-session"
    assert dummy_rag.received_session_id is None
    assert dummy_rag.received_question == "What is new?"
    assert dummy_rag.chat_called is True


def test_query_endpoint_reuses_existing_session(app_client: tuple[TestClient, DummyRAG]) -> None:
    """If a session_id is provided it should be reused and echoed back."""

    client, dummy_rag = app_client

    response = client.post(
        "/query",
        json={"question": "Follow up?", "session_id": "existing-session"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "existing-session"
    assert dummy_rag.received_session_id == "existing-session"
    assert dummy_rag.chat_called is True
