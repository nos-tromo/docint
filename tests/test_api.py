import types
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

import docint.core.api as api_module


class DummySessionManager:
    """
    Dummy session manager for testing purposes.
    """

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.

        Returns:
            list[dict[str, Any]]: A list of session dictionaries.
        """
        return [{"id": "123", "created_at": "2023-01-01", "title": "Test Chat"}]

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """
        Get the message history for a session.

        Args:
            session_id (str): The ID of the session.

        Returns:
            list[dict[str, Any]]: A list of message dictionaries.
        """
        return [{"role": "user", "content": "hi"}]

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session by ID.

        Args:
            session_id (str): The ID of the session.

        Returns:
            bool: True if the session was successfully deleted, False otherwise.
        """
        return True

    def get_agent_context(self, session_id: str):
        class Ctx:
            clarifications = 0

        return Ctx()


class DummyRAG:
    """
    Dummy Retrieval-Augmented Generation (RAG) class for testing purposes.
    """

    def __init__(self) -> None:
        """
        Initialize the DummyRAG instance.
        """
        self.qdrant_collection = "alpha"
        self.index = object()
        self.query_engine = object()
        self.selected: list[str] = []
        self.sessions = DummySessionManager()
        self.chats: list[str] = []
        self.created_index = 0  # Tracks the number of times an index is created
        self.created_query_engine = 0
        self.ner_sources: list[dict[str, Any]] = []

    def list_collections(self) -> list[str]:
        """
        List all available collections.

        Returns:
            list[str]: A list of collection names.
        """
        return ["alpha", "beta"]

    def select_collection(self, name: str) -> None:
        """
        Select a collection by name.

        Args:
            name (str): The name of the collection to select.
        """
        self.selected.append(name)
        self.qdrant_collection = name
        self.index = None
        self.query_engine = None

    def create_index(self) -> None:
        """
        Create a new index for the selected collection.
        """
        self.created_index += 1
        self.index = object()

    def create_query_engine(self) -> None:
        """
        Create a new query engine for the selected collection.
        """
        self.created_query_engine += 1
        self.query_engine = object()

    def start_session(self, session_id: str | None = None) -> str:
        """
        Start a new session or resume an existing one.

        Args:
            session_id (str | None, optional): The ID of the session to resume. Defaults to None.

        Returns:
            str: The session ID.
        """
        return session_id or "generated-session"

    def chat(self, question: str) -> dict[str, Any]:
        """
        Chat with the RAG system.

        Args:
            question (str): The question to ask the RAG system.

        Returns:
            dict[str, Any]: The response from the RAG system.
        """
        self.chats.append(question)
        return {"response": "answer", "sources": [{"id": 1}]}

    def stream_chat(self, question: str):
        yield "chunk"
        yield {"sources": [{"id": 1}], "session_id": "generated-session"}

    def get_collection_ner(self) -> list[dict[str, Any]]:
        """
        Get information extraction data for the selected collection.

        Returns:
            list[dict[str, Any]]: Information extraction data for the selected collection.
        """
        return self.ner_sources


@pytest.fixture(autouse=True)
def _patch_rag(monkeypatch: pytest.MonkeyPatch) -> Any | None:
    """
    Patch the RAG instance for testing.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.

    Returns:
        Any | None: Yields None after patching.
    """
    dummy = DummyRAG()
    monkeypatch.setattr(api_module, "rag", dummy)
    yield


@pytest.fixture
def client() -> TestClient:
    """
    Create a TestClient for testing the FastAPI application.

    Returns:
        TestClient: The TestClient instance.
    """
    return TestClient(api_module.app)


def test_collections_list_success(client: TestClient) -> None:
    """
    Test the successful retrieval of the collections list.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.get("/collections/list")
    assert response.status_code == 200
    assert response.json() == ["alpha", "beta"]


def test_collections_list_failure(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """
    Test the failed retrieval of the collections list.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raiser() -> list[str]:
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module.rag, "list_collections", raiser)
    response = client.get("/collections/list")
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_collections_select_success(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """
    Test the successful selection of a collection.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    # Force lazy creation paths
    api_module.rag.index = None
    api_module.rag.query_engine = None
    response = client.post("/collections/select", json={"name": " gamma "})
    assert response.status_code == 200
    payload = response.json()
    assert payload == {"ok": True, "name": "gamma"}
    rag = cast(Any, api_module.rag)
    assert rag.qdrant_collection == "gamma"
    assert rag.created_index == 1  # type: ignore[attr-defined]
    assert rag.created_query_engine == 1  # type: ignore[attr-defined]


def test_collections_select_blank_name(client: TestClient) -> None:
    """
    Test the selection of a collection with a blank name.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.post("/collections/select", json={"name": "   "})
    assert response.status_code == 500
    assert "Collection name required" in response.json()["detail"]


def test_collections_ner_success(client: TestClient) -> None:
    """
    Test the successful retrieval of information extraction data.

    Args:
        client (TestClient): The TestClient instance.
    """
    api_module.rag.ner_sources = [
        {"filename": "doc1.pdf", "page": 1, "row": 2, "entities": [], "relations": []}
    ]
    response = client.get("/collections/ie")
    assert response.status_code == 200
    assert response.json() == {"sources": api_module.rag.ner_sources}


def test_agent_chat_answers(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Agent chat should return an answer when confidence is sufficient."""

    def fake_chat(question: str) -> dict[str, Any]:
        return {"response": f"echo:{question}", "sources": [{"id": 1}]}

    monkeypatch.setattr(api_module.rag, "chat", fake_chat)

    payload = {"message": "hello"}
    response = client.post("/agent/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "answer"
    assert data["answer"] == "echo:hello"
    assert data["sources"] == [{"id": 1}]
    assert data["session_id"] == "generated-session"
    assert data["intent"] is not None
    assert data["confidence"] is not None


def test_agent_chat_clarifies(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Agent chat should request clarification when policy requires it."""

    monkeypatch.setattr(
        api_module,
        "_clarification_policy",
        api_module.ClarificationPolicy(
            api_module.ClarificationConfig(
                confidence_threshold=1.0, require_entities=True
            )
        ),
    )  # type: ignore[arg-type]

    payload = {"message": "hello"}
    response = client.post("/agent/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "clarification"
    assert data["message"]
    assert data["intent"] is not None
    assert data["confidence"] is not None


def test_agent_chat_stream_clarifies(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Streaming endpoint should emit clarification event when policy demands it."""

    monkeypatch.setattr(
        api_module,
        "_clarification_policy",
        api_module.ClarificationPolicy(
            api_module.ClarificationConfig(
                confidence_threshold=1.0, require_entities=True
            )
        ),
    )  # type: ignore[arg-type]

    with client.stream("POST", "/agent/chat/stream", json={"message": "hello"}) as resp:
        assert resp.status_code == 200
        text = "".join([chunk.decode() for chunk in resp.iter_raw()])
    assert "clarification" in text
    assert "status" in text


def test_collections_ner_requires_selection(client: TestClient) -> None:
    """
    Test that information extraction requires a collection to be selected.

    Args:
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/ie")
    assert response.status_code == 400
    assert "No collection selected" in response.json()["detail"]


def test_collections_ner_failure(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """
    Test the failed retrieval of information extraction data.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raiser() -> list[dict[str, Any]]:
        """
        Get information extraction data for the selected collection.

        Raises:
            RuntimeError: If there is an error retrieving the information extraction data.

        Returns:
            list[dict[str, Any]]: Information extraction data for the selected collection.
        """
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module.rag, "get_collection_ner", raiser)
    response = client.get("/collections/ie")
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_query_requires_collection(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """
    Test the query endpoint requires a collection to be selected.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = ""
    response = client.post("/query", json={"question": "hi"})
    assert response.status_code == 500
    assert "No collection selected" in response.json()["detail"]


def test_query_success(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """
    Test the successful query execution.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    api_module.rag.index = None
    api_module.rag.query_engine = None
    response = client.post(
        "/query",
        json={"question": "What?", "session_id": "abc"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "answer"
    assert body["sources"] == [{"id": 1}]
    assert body["session_id"] == "abc"


def test_query_handles_missing_sources(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """
    Test the query handles missing sources.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def fake_chat(question: str) -> str:
        """
        Fake chat function for testing.

        Args:
            question (str): The question to ask.

        Returns:
            str: The response from the chat.
        """
        return "plain response"

    monkeypatch.setattr(api_module.rag, "chat", fake_chat)
    response = client.post("/query", json={"question": "What?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == ""
    assert body["sources"] == []
    assert body["session_id"] == "generated-session"


def test_ingest_success(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """
    Test the successful ingestion of documents.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    called = types.SimpleNamespace(args=None)

    def fake_ingest(
        collection: str,
        path,
        hybrid: bool = True,
        progress_callback=None,
    ) -> None:
        called.args = (
            collection,
            path,
            hybrid,
            progress_callback,
        )

    monkeypatch.setattr(api_module, "_resolve_data_dir", lambda: data_dir)
    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)

    response = client.post(
        "/ingest",
        json={"collection": "docs", "hybrid": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "ok": True,
        "collection": "docs",
        "data_dir": str(data_dir),
        "hybrid": False,
    }
    assert called.args[0:3] == ("docs", data_dir, False)


def test_sessions_endpoints(client: TestClient) -> None:
    """Test session management endpoints."""
    # List
    resp = client.get("/sessions/list")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["id"] == "123"

    # History
    resp = client.get("/sessions/123/history")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["messages"]) == 1
    assert data["messages"][0]["content"] == "hi"

    # Delete
    resp = client.delete("/sessions/123")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_ingest_missing_directory(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """
    Test the ingestion of documents when the data directory is missing.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture.
    """
    missing = tmp_path / "missing"
    monkeypatch.setattr(api_module, "_resolve_data_dir", lambda: missing)
    response = client.post("/ingest", json={"collection": "abc"})
    assert response.status_code == 500
    assert "Data directory does not exist" in response.json()["detail"]
