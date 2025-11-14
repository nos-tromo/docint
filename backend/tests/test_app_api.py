import types
from typing import Any

import pytest
from fastapi.testclient import TestClient

import docint.app as app_module


class DummyRAG:
    def __init__(self) -> None:
        self.qdrant_collection = "alpha"
        self.index = object()
        self.query_engine = object()
        self.selected: list[str] = []
        self.sessions: list[Any] = []
        self.chats: list[str] = []
        self.created_index = 0  # Tracks the number of times an index is created
        self.created_query_engine = 0

    def list_collections(self) -> list[str]:
        return ["alpha", "beta"]

    def select_collection(self, name: str) -> None:
        self.selected.append(name)
        self.qdrant_collection = name
        self.index = None
        self.query_engine = None

    def create_index(self) -> None:
        self.created_index += 1
        self.index = object()

    def create_query_engine(self) -> None:
        self.created_query_engine += 1
        self.query_engine = object()

    def start_session(self, session_id: str | None = None) -> str:
        self.sessions.append(session_id)
        return session_id or "generated-session"

    def chat(self, question: str) -> dict[str, Any]:
        self.chats.append(question)
        return {"response": "answer", "sources": [{"id": 1}]}


@pytest.fixture(autouse=True)
def _patch_rag(monkeypatch: pytest.MonkeyPatch) -> Any | None:
    dummy = DummyRAG()
    monkeypatch.setattr(app_module, "rag", dummy)
    yield


@pytest.fixture
def client() -> TestClient:
    return TestClient(app_module.app)


def test_collections_list_success(client: TestClient) -> None:
    response = client.get("/collections/list")
    assert response.status_code == 200
    assert response.json() == ["alpha", "beta"]


def test_collections_list_failure(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    def raiser() -> list[str]:
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module.rag, "list_collections", raiser)
    response = client.get("/collections/list")
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_collections_select_success(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    # Force lazy creation paths
    app_module.rag.index = None
    app_module.rag.query_engine = None
    response = client.post("/collections/select", json={"name": " gamma "})
    assert response.status_code == 200
    payload = response.json()
    assert payload == {"ok": True, "name": "gamma"}
    assert app_module.rag.qdrant_collection == "gamma"
    assert app_module.rag.created_index == 1
    assert app_module.rag.created_query_engine == 1


def test_collections_select_blank_name(client: TestClient) -> None:
    response = client.post("/collections/select", json={"name": "   "})
    assert response.status_code == 500
    assert "Collection name required" in response.json()["detail"]


def test_query_requires_collection(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    app_module.rag.qdrant_collection = ""
    response = client.post("/query", json={"question": "hi"})
    assert response.status_code == 500
    assert "No collection selected" in response.json()["detail"]


def test_query_success(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    app_module.rag.index = object()
    app_module.rag.query_engine = object()
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
    def fake_chat(question: str) -> str:
        return "plain response"

    monkeypatch.setattr(app_module.rag, "chat", fake_chat)
    response = client.post("/query", json={"question": "What?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == ""
    assert body["sources"] == []
    assert body["session_id"] == "generated-session"


def test_ingest_success(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    called = types.SimpleNamespace(args=None)

    def fake_ingest(collection: str, path, hybrid: bool = True) -> None:
        called.args = (collection, path, hybrid)

    monkeypatch.setattr(app_module, "_resolve_data_dir", lambda: data_dir)
    monkeypatch.setattr(app_module.ingest_module, "ingest_docs", fake_ingest)

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
    assert called.args == ("docs", data_dir, False)


def test_ingest_missing_directory(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path
) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setattr(app_module, "_resolve_data_dir", lambda: missing)
    response = client.post("/ingest", json={"collection": "abc"})
    assert response.status_code == 500
    assert "Data directory does not exist" in response.json()["detail"]
