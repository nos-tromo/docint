"""Tests for the FastAPI application layer."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable, ContextManager

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from docint import app as app_module


class TrackingRAG:
    """Stand-in for the global RAG instance with rich instrumentation."""

    def __init__(
        self,
        *,
        qdrant_collection: str = "demo",
        session_return: str = "generated-session",
        with_index: bool = True,
        with_query_engine: bool = True,
    ) -> None:
        self.qdrant_collection = qdrant_collection
        self.index: Any | None = object() if with_index else None
        self.query_engine: Any | None = object() if with_query_engine else None
        self.session_return = session_return
        self.received_session_id: str | None = None
        self.received_question: str | None = None
        self.events: list[str | tuple[str, Any]] = []

    # --- lifecycle helpers -------------------------------------------------
    def create_index(self) -> None:
        self.events.append("create_index")
        self.index = object()

    def create_query_engine(self) -> None:
        self.events.append("create_query_engine")
        self.query_engine = object()

    # --- query related helpers ---------------------------------------------
    def start_session(self, session_id: str | None) -> str:
        self.events.append(("start_session", session_id))
        self.received_session_id = session_id
        return session_id or self.session_return

    def chat(self, question: str) -> dict[str, Any]:
        self.events.append(("chat", question))
        self.received_question = question
        return {
            "response": "Here is an answer.",
            "sources": [{"title": "Doc", "page": 1}],
        }


@pytest.fixture
def client_factory(monkeypatch: pytest.MonkeyPatch) -> Callable[[TrackingRAG], ContextManager[tuple[TestClient, TrackingRAG]]]:
    """Provide a context-managed TestClient wired to the supplied RAG stub."""

    @contextmanager
    def _factory(rag: TrackingRAG) -> Iterator[tuple[TestClient, TrackingRAG]]:
        monkeypatch.setattr(app_module, "rag", rag)
        with TestClient(app_module.app) as client:
            yield client, rag

    return _factory


def _get_route(path: str) -> APIRoute:
    for route in app_module.app.routes:
        if isinstance(route, APIRoute) and route.path == path:
            return route
    raise AssertionError(f"Route {path!r} not found")


def test_query_route_configuration() -> None:
    """The /query route should expose the expected schema contract."""

    route = _get_route("/query")
    assert route.response_model is app_module.QueryOut
    assert "Query" in (route.tags or [])

    body_params = [param for param in route.dependant.body_params if param.name == "payload"]
    assert body_params, "/query endpoint should receive a payload body parameter"
    assert body_params[0].type_ is app_module.QueryIn


def test_query_endpoint_returns_answer_and_session(
    client_factory: Callable[[TrackingRAG], ContextManager[tuple[TestClient, TrackingRAG]]]
) -> None:
    """The /query endpoint should proxy question handling to the RAG instance."""

    with client_factory(TrackingRAG()) as (client, rag):
        response = client.post("/query", json={"question": "What is new?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "answer": "Here is an answer.",
        "sources": [{"title": "Doc", "page": 1}],
        "session_id": "generated-session",
    }
    assert rag.received_session_id is None
    assert rag.received_question == "What is new?"
    assert rag.events == [("start_session", None), ("chat", "What is new?")]


def test_query_endpoint_reuses_existing_session(
    client_factory: Callable[[TrackingRAG], ContextManager[tuple[TestClient, TrackingRAG]]]
) -> None:
    """If a session_id is provided it should be reused and echoed back."""

    with client_factory(TrackingRAG()) as (client, rag):
        response = client.post(
            "/query",
            json={"question": "Follow up?", "session_id": "existing-session"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "existing-session"
    assert rag.received_session_id == "existing-session"
    assert rag.events == [("start_session", "existing-session"), ("chat", "Follow up?")]


def test_query_endpoint_initializes_missing_components(
    client_factory: Callable[[TrackingRAG], ContextManager[tuple[TestClient, TrackingRAG]]]
) -> None:
    """When index/query engine are missing they should be created before handling the chat."""

    rag_instance = TrackingRAG(with_index=False, with_query_engine=False)
    with client_factory(rag_instance) as (client, rag):
        response = client.post("/query", json={"question": "Warm up?"})

    assert response.status_code == 200
    assert response.json()["answer"]
    assert rag.events[0:2] == ["create_index", "create_query_engine"]
    assert ("chat", "Warm up?") in rag.events


def test_query_endpoint_requires_collection_selection(
    client_factory: Callable[[TrackingRAG], ContextManager[tuple[TestClient, TrackingRAG]]]
) -> None:
    """Requests should fail fast if no collection has been selected."""

    rag_instance = TrackingRAG(qdrant_collection="", with_index=True, with_query_engine=True)
    with client_factory(rag_instance) as (client, rag):
        response = client.post("/query", json={"question": "Any docs?"})

    assert response.status_code == 500
    payload = response.json()
    assert "No collection selected" in payload["detail"]
    assert rag.events == []
