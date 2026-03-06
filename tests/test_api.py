import types
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

import docint.core.api as api_module
from docint.agents.types import IntentAnalysis, OrchestratorResult, RetrievalResult


class DummySessionManager:
    """Dummy session manager for testing purposes."""

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions.

        Returns:
            list[dict[str, Any]]: A list of session dictionaries.
        """
        return [{"id": "123", "created_at": "2023-01-01", "title": "Test Chat"}]

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """Get the message history for a session.

        Args:
            session_id (str): The ID of the session.

        Returns:
            list[dict[str, Any]]: A list of message dictionaries.
        """
        return [{"role": "user", "content": "hi"}]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID.

        Args:
            session_id (str): The ID of the session.

        Returns:
            bool: True if the session was successfully deleted, False otherwise.
        """
        return True

    def get_agent_context(self, session_id: str) -> Any:
        """
        Get the agent context for a session.

        Args:
            session_id (str): The ID of the session.

        Returns:
            Any: The agent context for the session.
        """

        class Ctx:
            """Dummy context object for testing purposes."""

            clarifications = 0

        return Ctx()


class DummyRAG:
    """Dummy Retrieval-Augmented Generation (RAG) class for testing purposes."""

    def __init__(self) -> None:
        """Initialize the DummyRAG instance."""
        self.qdrant_collection = "alpha"
        self.summarize_prompt = "Summarize collection"
        self.index = object()
        self.query_engine = object()
        self.selected: list[str] = []
        self.sessions = DummySessionManager()
        self.chats: list[str] = []
        self.created_index = 0  # Tracks the number of times an index is created
        self.created_query_engine = 0
        self.ner_sources: list[dict[str, Any]] = []
        self.hate_speech_rows: list[dict[str, Any]] = []
        self.summary_payload: dict[str, Any] = {
            "response": "summary",
            "sources": [{"id": "s1"}],
            "summary_diagnostics": {
                "total_documents": 2,
                "covered_documents": 2,
                "coverage_ratio": 1.0,
                "uncovered_documents": [],
                "coverage_target": 0.7,
            },
        }

    def list_collections(self) -> list[str]:
        """List all available collections.

        Returns:
            list[str]: A list of collection names.
        """
        return ["alpha", "beta"]

    def select_collection(self, name: str) -> None:
        """Select a collection by name.

        Args:
            name (str): The name of the collection to select.
        """
        self.selected.append(name)
        self.qdrant_collection = name
        self.index = None
        self.query_engine = None

    def create_index(self) -> None:
        """Create a new index for the selected collection."""
        self.created_index += 1
        self.index = object()

    def create_query_engine(self) -> None:
        """Create a new query engine for the selected collection."""
        self.created_query_engine += 1
        self.query_engine = object()

    def start_session(self, session_id: str | None = None) -> str:
        """Start a new session or resume an existing one.

        Args:
            session_id (str | None, optional): The ID of the session to resume. Defaults to None.

        Returns:
            str: The session ID.
        """
        return session_id or "generated-session"

    def chat(self, question: str) -> dict[str, Any]:
        """Chat with the RAG system.

        Args:
            question (str): The question to ask the RAG system.

        Returns:
            dict[str, Any]: The response from the RAG system.
        """
        self.chats.append(question)
        return {"response": "answer", "sources": [{"id": 1}]}

    def stream_chat(self, question: str) -> Generator[str | dict[str, Any], None, None]:
        """
        Stream chat responses from the RAG system.

        Args:
            question (str): The question to ask the RAG system.

        Yields:
            str | dict[str, Any]: Chunks of the chat response as they are generated.
        """
        _ = question
        yield "chunk"
        yield {"sources": [{"id": 1}], "session_id": "generated-session"}

    def summarize_collection(self) -> dict[str, Any]:
        """Return canned summarize payload.

        Returns:
            dict[str, Any]: A dictionary containing the summary response, sources, and diagnostics.
        """
        return self.summary_payload

    def stream_summarize_collection(
        self,
    ) -> Generator[str | dict[str, Any], None, None]:
        """Stream canned summary payload.

        Returns:
            Generator[str | dict[str, Any], None, None]: A generator that yields chunks of the summary response and the final payload.

        Yields:
            str | dict[str, Any]: Chunks of the summary response as they are generated, followed by the final summary payload.
        """
        yield "sum"
        yield self.summary_payload

    def get_collection_ner(self) -> list[dict[str, Any]]:
        """Get information extraction data for the selected collection.

        Returns:
            list[dict[str, Any]]: Information extraction data for the selected collection.
        """
        return self.ner_sources

    def get_collection_hate_speech(self) -> list[dict[str, Any]]:
        """Get hate-speech findings for the selected collection."""
        return self.hate_speech_rows

    def get_collection_ner_stats(
        self,
        *,
        top_k: int = 15,
        min_mentions: int = 2,
        entity_type: str | None = None,
        include_relations: bool = True,
    ) -> dict[str, Any]:
        """Return canned IE stats payload.

        Args:
            top_k (int, optional): The number of top entities to return. Defaults to 15.
            min_mentions (int, optional): The minimum number of mentions for an entity to be included. Defaults to 2.
            entity_type (str | None, optional): Filter entities by type. Defaults to None.
            include_relations (bool, optional): Whether to include relation statistics. Defaults to True.

        Returns:
            dict[str, Any]: A dictionary containing information extraction statistics, including totals, top entities, entity types, top relations, and document-level stats.
        """
        _ = (top_k, min_mentions, entity_type, include_relations)
        return {
            "totals": {
                "unique_entities": 1,
                "entity_mentions": 3,
                "unique_relations": 1,
            },
            "top_entities": [
                {
                    "text": "Acme",
                    "type": "ORG",
                    "mentions": 3,
                    "best_score": 0.9,
                    "source_count": 2,
                }
            ],
            "entity_types": [{"type": "ORG", "mentions": 3, "unique_entities": 1}],
            "top_relations": [
                {"head": "Acme", "label": "owns", "tail": "Widget", "mentions": 2}
            ],
            "documents": [
                {
                    "filename": "doc1.pdf",
                    "entity_mentions": 3,
                    "unique_entities": 1,
                    "ie_source_count": 2,
                    "entity_density": 1.5,
                }
            ],
        }

    def search_collection_ner_entities(
        self,
        *,
        q: str = "",
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return simple entity search results.

        Args:
            q (str, optional): The search query string. Defaults to "".
            entity_type (str | None, optional): Filter entities by type. Defaults to None.
            limit (int, optional): The maximum number of results to return. Defaults to 100.

        Returns:
            list[dict[str, Any]]: A list of entity dictionaries that match the search criteria.
        """
        _ = (entity_type, limit)
        if q and q.lower() not in "acme":
            return []
        return [
            {
                "text": "Acme",
                "type": "ORG",
                "mentions": 3,
                "best_score": 0.9,
                "source_count": 2,
            }
        ]

    def get_collection_ner_graph(
        self,
        *,
        top_k_nodes: int = 100,
        min_edge_weight: int = 1,
    ) -> dict[str, Any]:
        """Return a minimal graph payload.

        Args:
            top_k_nodes (int, optional): The maximum number of nodes to include in the graph. Defaults to 100.
            min_edge_weight (int, optional): The minimum weight for edges to be included in the graph. Defaults to 1.

        Returns:
            dict[str, Any]: A dictionary representing the graph structure, including nodes, edges, and metadata.
        """
        _ = (top_k_nodes, min_edge_weight)
        return {
            "nodes": [
                {"id": "acme::org", "text": "Acme", "type": "ORG", "mentions": 3}
            ],
            "edges": [
                {
                    "source": "acme::org",
                    "target": "widget::unlabeled",
                    "label": "owns",
                    "kind": "relation",
                    "weight": 2,
                }
            ],
            "meta": {"node_count": 1, "edge_count": 1},
        }

    def get_collection_ner_graph_neighbors(
        self,
        *,
        entity: str,
        hops: int = 1,
        top_k_nodes: int = 100,
        min_edge_weight: int = 1,
    ) -> dict[str, Any]:
        """Return canned neighborhood payload.

        Args:
            entity (str): The central entity for which to retrieve neighbors.
            hops (int, optional): The number of hops to include in the neighborhood. Defaults to 1.
            top_k_nodes (int, optional): The maximum number of neighbor nodes to include. Defaults to 100.
            min_edge_weight (int, optional): The minimum weight for edges to be included in the neighborhood. Defaults to 1.

        Returns:
            dict[str, Any]: A dictionary representing the neighborhood of the specified entity, including the center node, neighboring nodes, and metadata.
        """
        _ = (entity, hops, top_k_nodes, min_edge_weight)
        return {
            "center": {"id": "acme::org", "text": "Acme", "type": "ORG", "mentions": 3},
            "neighbors": [
                {
                    "id": "widget::unlabeled",
                    "text": "Widget",
                    "type": "Unlabeled",
                    "mentions": 1,
                    "depth": 1,
                    "score": 2.0,
                }
            ],
            "nodes": [],
            "edges": [],
            "meta": {"hops": 1, "node_count": 2, "edge_count": 1},
        }


@pytest.fixture(autouse=True)
def _patch_rag(monkeypatch: pytest.MonkeyPatch) -> Any | None:
    """Patch the RAG instance for testing.

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
    """Create a TestClient for testing the FastAPI application.

    Returns:
        TestClient: The TestClient instance.
    """
    return TestClient(api_module.app)


def test_collections_list_success(client: TestClient) -> None:
    """Test the successful retrieval of the collections list.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.get("/collections/list")
    assert response.status_code == 200
    assert response.json() == ["alpha", "beta"]


def test_collections_list_failure(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test the failed retrieval of the collections list.

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
    """Test the successful selection of a collection.

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
    assert rag.created_index == 1
    assert rag.created_query_engine == 1


def test_collections_select_blank_name(client: TestClient) -> None:
    """Test the selection of a collection with a blank name.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.post("/collections/select", json={"name": "   "})
    assert response.status_code == 500
    assert "Collection name required" in response.json()["detail"]


def test_collections_ner_success(client: TestClient) -> None:
    """Test the successful retrieval of information extraction data.

    Args:
        client (TestClient): The TestClient instance.
    """
    api_module.rag.ner_sources = [
        {"filename": "doc1.pdf", "page": 1, "row": 2, "entities": [], "relations": []}
    ]
    response = client.get("/collections/ner")
    assert response.status_code == 200
    assert response.json() == {"sources": api_module.rag.ner_sources}


def test_collections_ner_stats_success(client: TestClient) -> None:
    """Stats endpoint should return IE summary payload.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.get("/collections/ner/stats")
    assert response.status_code == 200
    payload = response.json()
    assert payload["totals"]["unique_entities"] == 1
    assert payload["top_entities"][0]["text"] == "Acme"


def test_collections_hate_speech_success(client: TestClient) -> None:
    """Hate-speech endpoint should return flagged rows."""
    api_module.rag.hate_speech_rows = [
        {
            "chunk_id": "c1",
            "chunk_text": "flagged text",
            "category": "ethnicity",
            "confidence": "high",
            "reason": "Contains slur",
            "source_ref": "doc1.pdf",
            "page": 2,
        }
    ]
    response = client.get("/collections/hate-speech")
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["chunk_id"] == "c1"


def test_collections_ner_search_success(client: TestClient) -> None:
    """Search endpoint should return matching entities.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.get("/collections/ner/search", params={"q": "ac"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["text"] == "Acme"


def test_collections_ner_graph_success(client: TestClient) -> None:
    """Graph endpoint should return graph payload.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.get("/collections/ner/graph")
    assert response.status_code == 200
    payload = response.json()
    assert payload["meta"]["edge_count"] == 1
    assert payload["nodes"][0]["text"] == "Acme"


def test_collections_ner_graph_neighbors_success(client: TestClient) -> None:
    """Neighbors endpoint should return center + neighbors.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.get(
        "/collections/ner/graph/neighbors",
        params={"entity": "Acme", "hops": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["center"]["text"] == "Acme"
    assert payload["neighbors"][0]["text"] == "Widget"


def test_agent_chat_answers(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Agent chat should return an answer when confidence is sufficient.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def fake_chat(question: str) -> dict[str, Any]:
        """
        Fake implementation of the RAG chat method for testing purposes.

        Args:
            question (str): The question to ask the RAG system.

        Returns:
            dict[str, Any]: The response from the RAG system, including an answer and sources.
        """
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
    """Agent chat should request clarification when policy requires it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    monkeypatch.setattr(
        api_module,
        "_clarification_policy",
        api_module.ClarificationPolicy(
            api_module.ClarificationConfig(
                confidence_threshold=1.0, require_entities=True
            )
        ),
    )

    payload = {"message": "hello"}
    response = client.post("/agent/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "clarification"
    assert data["message"]
    assert data["intent"] is not None
    assert data["confidence"] is not None


def test_agent_chat_returns_validation_alert(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Agent chat should surface response-validation metadata.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    class _StubOrchestrator:
        """Stub orchestrator that returns a canned retrieval result with validation metadata for testing purposes."""

        def handle_turn(self, turn, context=None) -> OrchestratorResult:
            """Handle a turn by returning a canned retrieval result with validation metadata.

            Args:
                turn (_type_): The user turn to process.
                context (_type_, optional): The context for the turn. Defaults to None.

            Returns:
                OrchestratorResult: The result of processing the turn.
            """
            _ = turn, context
            analysis = IntentAnalysis(
                intent="qa", confidence=0.9, entities={"query": "hello"}
            )
            retrieval = RetrievalResult(
                answer="answer",
                sources=[{"id": 1}],
                session_id="generated-session",
                validation_checked=True,
                validation_mismatch=True,
                validation_reason="mismatch",
            )
            return OrchestratorResult(
                clarification=None, retrieval=retrieval, analysis=analysis
            )

    monkeypatch.setattr(api_module, "_build_orchestrator", lambda: _StubOrchestrator())

    response = client.post("/agent/chat", json={"message": "hello"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "answer"
    assert data["validation_checked"] is True
    assert data["validation_mismatch"] is True
    assert data["validation_reason"] == "mismatch"


def test_agent_chat_stream_clarifies(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Streaming endpoint should emit clarification event when policy demands it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    monkeypatch.setattr(
        api_module,
        "_clarification_policy",
        api_module.ClarificationPolicy(
            api_module.ClarificationConfig(
                confidence_threshold=1.0, require_entities=True
            )
        ),
    )

    with client.stream("POST", "/agent/chat/stream", json={"message": "hello"}) as resp:
        assert resp.status_code == 200
        text = "".join([chunk.decode() for chunk in resp.iter_raw()])
    assert "clarification" in text
    assert "status" in text


def test_stream_query_includes_validation_metadata(client: TestClient) -> None:
    """Streaming query endpoint should emit validation metadata in final payload.

    Args:
        client (TestClient): The TestClient instance.
    """
    with client.stream("POST", "/stream_query", json={"question": "hello"}) as resp:
        assert resp.status_code == 200
        text = "".join([chunk.decode() for chunk in resp.iter_raw()])
    assert '"validation_checked"' in text
    assert '"validation_mismatch"' in text


def test_summarize_includes_summary_diagnostics(client: TestClient) -> None:
    """Summarize endpoint should expose summary diagnostics and validation metadata.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.post("/summarize")
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == "summary"
    assert payload["sources"] == [{"id": "s1"}]
    assert payload["summary_diagnostics"]["total_documents"] == 2
    assert payload["summary_diagnostics"]["covered_documents"] == 2
    assert "validation_checked" in payload
    assert "validation_mismatch" in payload
    assert "validation_reason" in payload


def test_summarize_stream_includes_summary_diagnostics(client: TestClient) -> None:
    """Streaming summarize endpoint should emit diagnostics in final payload.

    Args:
        client (TestClient): The TestClient instance.
    """
    with client.stream("POST", "/summarize/stream") as resp:
        assert resp.status_code == 200
        text = "".join([chunk.decode() for chunk in resp.iter_raw()])
    assert '"summary_diagnostics"' in text
    assert '"coverage_ratio"' in text


def test_collections_ner_requires_selection(client: TestClient) -> None:
    """Test that information extraction requires a collection to be selected.

    Args:
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/ner")
    assert response.status_code == 400
    assert "No collection selected" in response.json()["detail"]


def test_collections_ner_stats_requires_selection(client: TestClient) -> None:
    """Stats endpoint should require active collection selection.

    Args:
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/ner/stats")
    assert response.status_code == 400
    assert "No collection selected" in response.json()["detail"]


def test_collections_ner_search_requires_selection(client: TestClient) -> None:
    """Search endpoint should require active collection selection.

    Args:
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/ner/search", params={"q": "acme"})
    assert response.status_code == 400
    assert "No collection selected" in response.json()["detail"]


def test_collections_hate_speech_requires_selection(client: TestClient) -> None:
    """Hate-speech endpoint should require active collection selection."""
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/hate-speech")
    assert response.status_code == 400
    assert "No collection selected" in response.json()["detail"]


def test_collections_ner_failure(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test the failed retrieval of information extraction data.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raiser() -> list[dict[str, Any]]:
        """Get information extraction data for the selected collection.

        Returns:
            list[dict[str, Any]]: Information extraction data for the selected collection.

        Raises:
            RuntimeError: If there is an error retrieving the information extraction data.
        """
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module.rag, "get_collection_ner", raiser)
    response = client.get("/collections/ner")
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_collections_ner_stats_failure(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Stats endpoint should surface backend failures.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raiser(**kwargs) -> dict[str, Any]:
        """
        Fake implementation of get_collection_ner_stats that raises an error for testing purposes.

        Returns:
            dict[str, Any]: Information extraction statistics for the selected collection.

        Raises:
            RuntimeError: If there is an error retrieving the information extraction stats.
        """
        _ = kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module.rag, "get_collection_ner_stats", raiser)
    response = client.get("/collections/ner/stats")
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_collections_ner_search_failure(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Search endpoint should surface backend failures.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raiser(**kwargs) -> list[dict[str, Any]]:
        """
        Fake implementation of search_collection_ner_entities that raises an error for testing purposes.

        Returns:
            list[dict[str, Any]]: The search results.

        Raises:
            RuntimeError: If there is an error during the search.
        """
        _ = kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module.rag, "search_collection_ner_entities", raiser)
    response = client.get("/collections/ner/search", params={"q": "ac"})
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_collections_hate_speech_failure(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Hate-speech endpoint should surface backend failures."""

    def raiser() -> list[dict[str, Any]]:
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module.rag, "get_collection_hate_speech", raiser)
    response = client.get("/collections/hate-speech")
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_query_requires_collection(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test the query endpoint requires a collection to be selected.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = ""
    response = client.post("/query", json={"question": "hi"})
    assert response.status_code == 500
    assert "No collection selected" in response.json()["detail"]


def test_query_success(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Test the successful query execution.

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
    """Test the query handles missing sources.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def fake_chat(question: str) -> str:
        """Fake chat function for testing.

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
    """Test the successful ingestion of documents.

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
        """
        Fake implementation of the ingest_docs function for testing purposes.

        Args:
            collection (str): The name of the collection to ingest into.
            path (Path): The path to the data to ingest.
            hybrid (bool, optional): Whether to use hybrid ingestion. Defaults to True.
            progress_callback (callable, optional): A callback function for progress updates. Defaults to None.
        """
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
    """Test session management endpoints.

    Args:
        client (TestClient): The TestClient instance.
    """
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
    """Test the ingestion of documents when the data directory is missing.

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
