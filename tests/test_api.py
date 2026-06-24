"""Tests for the FastAPI application endpoints."""

import asyncio
import io
import types
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

import docint.core.api as api_module
from docint.agents.types import IntentAnalysis, OrchestratorResult, PriorTurn, RetrievalResult
from docint.agents.understanding import ContextualUnderstandingAgent
from docint.core.entities.resolution import ResolutionSummary


class DummySessionManager:
    """Dummy session manager for testing purposes."""

    def list_sessions(self, owner: str | None = None) -> list[dict[str, Any]]:
        """List the caller's sessions.

        Args:
            owner (str | None): The owning principal.

        Returns:
            list[dict[str, Any]]: A list of session dictionaries.
        """
        return [{"id": "123", "created_at": "2023-01-01", "title": "Test Chat"}]

    def get_session_history(self, session_id: str, owner: str | None = None) -> list[dict[str, Any]]:
        """Get the message history for a session.

        Args:
            session_id (str): The ID of the session.
            owner (str | None): The owning principal.

        Returns:
            list[dict[str, Any]]: A list of message dictionaries.
        """
        return [{"role": "user", "content": "hi"}]

    def delete_session(self, session_id: str, owner: str | None = None) -> bool:
        """Delete a session by ID.

        Args:
            session_id (str): The ID of the session.
            owner (str | None): The owning principal.

        Returns:
            bool: True if the session was successfully deleted, False otherwise.
        """
        return True

    def get_agent_context(self, session_id: str) -> Any:
        """Get the agent context for a session.

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
        self.stateless_queries: list[str] = []
        self.stateless_query_filters: list[dict[str, Any]] = []
        self.entity_occurrence_queries: list[str] = []
        self.entity_occurrence_filters: list[Any] = []
        self.entity_occurrence_merge_modes: list[str] = []
        self.multi_entity_occurrence_queries: list[str] = []
        self.multi_entity_occurrence_filters: list[Any] = []
        self.multi_entity_occurrence_merge_modes: list[str] = []
        self.chat_filters: list[Any] = []
        self.stream_filters: list[Any] = []
        self.created_index = 0  # Tracks the number of times an index is created
        self.created_query_engine = 0
        self.ner_sources: list[dict[str, Any]] = []
        self.ner_refresh_calls: list[bool] = []
        self.ner_stats_merge_modes: list[str] = []
        self.ner_search_merge_modes: list[str] = []
        self.ner_graph_merge_modes: list[str] = []
        self.hate_speech_rows: list[dict[str, Any]] = []
        self.documents: list[dict[str, Any]] = []
        self.summary_refresh_calls: list[bool] = []
        self.summary_stream_refresh_calls: list[bool] = []
        self.summary_payload: dict[str, Any] = {
            "response": "summary",
            "sources": [{"id": "s1"}],
            "summary_diagnostics": {
                "total_documents": 2,
                "covered_documents": 2,
                "coverage_ratio": 1.0,
                "uncovered_documents": [],
                "coverage_target": 0.7,
                "coverage_unit": "documents",
                "candidate_count": 2,
                "deduped_count": 2,
                "sampled_count": 2,
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

    def ensure_session_manager(self) -> DummySessionManager:
        """Return the SessionManager stub, mirroring RAG.ensure_session_manager.

        Returns:
            DummySessionManager: The pre-initialized session manager stub.
        """
        return self.sessions

    def start_session(self, session_id: str | None = None, owner: str | None = None) -> str:
        """Start a new session or resume an existing one.

        Args:
            session_id (str | None, optional): The ID of the session to resume. Defaults to None.
            owner (str | None, optional): The owning principal. Defaults to None.

        Returns:
            str: The session ID.
        """
        _ = owner
        return session_id or "generated-session"

    def chat(
        self,
        question: str,
        *,
        metadata_filters: Any = None,
        metadata_filters_active: bool = False,
        metadata_filter_rules: Any = None,
        vector_store_kwargs: Any = None,
    ) -> dict[str, Any]:
        """Chat with the RAG system.

        Args:
            question (str): The question to ask the RAG system.
            metadata_filters (Any): Optional compiled metadata filters.
            metadata_filters_active (bool): Whether request filters were active.
            metadata_filter_rules (Any): Optional raw request filter rules.
            vector_store_kwargs (Any): Optional native vector-store query kwargs.

        Returns:
            dict[str, Any]: The response from the RAG system.
        """
        self.chats.append(question)
        self.chat_filters.append(
            {
                "filters": metadata_filters,
                "active": metadata_filters_active,
                "rules": metadata_filter_rules,
                "vector_store_kwargs": vector_store_kwargs,
            }
        )
        return {
            "response": "answer",
            "sources": [{"id": 1}],
            "retrieval_query": f"rewritten::{question}",
            "coverage_unit": "documents",
            "retrieval_mode": "rewrite_compact_graph",
            "graph_debug": {
                "enabled": True,
                "applied": True,
                "original_query": question,
                "expanded_query": f"{question}\n\nRelated entities for retrieval: Acme",
                "anchor_entities": ["Acme"],
                "neighbor_entities": ["Widget"],
            },
        }

    def stream_chat(
        self,
        question: str,
        *,
        metadata_filters: Any = None,
        metadata_filters_active: bool = False,
        metadata_filter_rules: Any = None,
        vector_store_kwargs: Any = None,
        prior_turn: Any = None,
        skip_query_rewrite: Any = None,
    ) -> Generator[str | dict[str, Any], None, None]:
        """Stream chat responses from the RAG system.

        Args:
            question (str): The question to ask the RAG system.
            metadata_filters (Any): Optional compiled metadata filters.
            metadata_filters_active (bool): Whether request filters were active.
            metadata_filter_rules (Any): Optional raw request filter rules.
            vector_store_kwargs (Any): Optional native vector-store query kwargs.
            prior_turn (Any): Optional prior user/assistant exchange for context.
            skip_query_rewrite (Any): Accepted for parity with RAG.stream_chat; ignored by the stub.

        Yields:
            str | dict[str, Any]: Chunks of the chat response as they are generated.
        """
        self.stream_filters.append(
            {
                "filters": metadata_filters,
                "active": metadata_filters_active,
                "rules": metadata_filter_rules,
                "vector_store_kwargs": vector_store_kwargs,
            }
        )
        yield "chunk"
        yield {
            "response": "answer",
            "sources": [{"id": 1}],
            "session_id": "generated-session",
            "retrieval_query": f"rewritten::{question}",
            "coverage_unit": "documents",
            "retrieval_mode": "rewrite_compact_graph",
            "graph_debug": {
                "enabled": True,
                "applied": True,
                "original_query": question,
                "expanded_query": f"{question}\n\nRelated entities for retrieval: Acme",
                "anchor_entities": ["Acme"],
                "neighbor_entities": ["Widget"],
            },
        }

    def run_query(
        self,
        prompt: str,
        *,
        metadata_filters: Any = None,
        metadata_filter_rules: Any = None,
        vector_store_kwargs: Any = None,
    ) -> dict[str, Any]:
        """Run a stateless retrieval query.

        Args:
            prompt: Query prompt.
            metadata_filters: Optional compiled metadata filters.
            metadata_filter_rules: Optional raw request filter rules.
            vector_store_kwargs: Optional native vector-store query kwargs.

        Returns:
            dict[str, Any]: Response payload.
        """
        _ = metadata_filters
        _ = metadata_filter_rules
        _ = vector_store_kwargs
        self.stateless_queries.append(prompt)
        return {
            "response": "answer",
            "sources": [{"id": 1}],
        }

    async def run_query_async(
        self,
        prompt: str,
        *,
        metadata_filters: Any = None,
        metadata_filter_rules: Any = None,
        vector_store_kwargs: Any = None,
        retrieval_options: Any = None,
    ) -> dict[str, Any]:
        """Async stateless retrieval query mirroring :meth:`run_query`.

        Args:
            prompt: Query prompt.
            metadata_filters: Optional compiled metadata filters.
            metadata_filter_rules: Optional raw request filter rules.
            vector_store_kwargs: Optional native vector-store query kwargs.
            retrieval_options: Optional runtime retrieval overrides.

        Returns:
            dict[str, Any]: Response payload.
        """
        _ = retrieval_options
        self.stateless_query_filters.append(
            {
                "filters": metadata_filters,
                "rules": metadata_filter_rules,
                "vector_store_kwargs": vector_store_kwargs,
            }
        )
        return self.run_query(
            prompt,
            metadata_filters=metadata_filters,
            metadata_filter_rules=metadata_filter_rules,
            vector_store_kwargs=vector_store_kwargs,
        )

    def run_entity_occurrence_query(
        self,
        prompt: str,
        *,
        qdrant_filter: Any = None,
        limit: int = 100,
        refresh: bool = False,
        entity_merge_mode: str = "orthographic",
    ) -> dict[str, Any]:
        """Return canned entity-occurrence results for tests.

        Args:
            prompt: Query prompt.
            qdrant_filter: Optional native Qdrant filter.
            limit: Maximum number of occurrence rows to return.
            refresh: Whether cache refresh was requested.
            entity_merge_mode: Entity clustering mode (recorded for assertions).

        Returns:
            dict[str, Any]: Canned occurrence payload.
        """
        _ = (limit, refresh)
        self.entity_occurrence_queries.append(prompt)
        self.entity_occurrence_filters.append(qdrant_filter)
        self.entity_occurrence_merge_modes.append(entity_merge_mode)
        if prompt == "Ambiguous":
            return {
                "response": "Your query matches multiple entities equally well.",
                "sources": [],
                "retrieval_query": prompt,
                "coverage_unit": "entity_mentions",
                "retrieval_mode": "entity_occurrence_ambiguous",
                "entity_match_candidates": [
                    {"text": "Acme", "type": "ORG", "mentions": 3},
                    {"text": "Acme", "type": "PRODUCT", "mentions": 2},
                ],
                "entity_match_groups": [],
            }
        return {
            "response": "Found 3 occurrence(s) of 'Acme' across 2 chunk(s) in 2 document(s).",
            "sources": [{"id": "occ-1"}, {"id": "occ-2"}],
            "retrieval_query": prompt,
            "coverage_unit": "entity_mentions",
            "retrieval_mode": "entity_occurrence",
            "entity_match_candidates": [{"text": "Acme", "type": "ORG", "mentions": 3}],
            "entity_match_groups": [
                {
                    "entity": {"text": "Acme", "type": "ORG", "mentions": 3},
                    "sources": [{"id": "occ-1"}, {"id": "occ-2"}],
                    "chunk_count": 2,
                    "document_count": 2,
                    "truncated": False,
                }
            ],
        }

    def run_multi_entity_occurrence_query(
        self,
        prompt: str,
        *,
        qdrant_filter: Any = None,
        limit: int = 100,
        refresh: bool = False,
        entity_merge_mode: str = "orthographic",
    ) -> dict[str, Any]:
        """Return canned multi-entity occurrence results for tests."""
        _ = (limit, refresh)
        self.multi_entity_occurrence_queries.append(prompt)
        self.multi_entity_occurrence_filters.append(qdrant_filter)
        self.multi_entity_occurrence_merge_modes.append(entity_merge_mode)
        return {
            "response": "Found 2 equally strong entity match(es) for 'Acme', covering 2 chunk(s) across 2 document(s).",
            "sources": [{"id": "occ-org"}, {"id": "occ-product"}],
            "retrieval_query": prompt,
            "coverage_unit": "entity_mentions",
            "retrieval_mode": "entity_occurrence_multi",
            "entity_match_candidates": [
                {"text": "Acme", "type": "ORG", "mentions": 3},
                {"text": "Acme", "type": "PRODUCT", "mentions": 2},
            ],
            "entity_match_groups": [
                {
                    "entity": {"text": "Acme", "type": "ORG", "mentions": 3},
                    "sources": [{"id": "occ-org"}],
                    "chunk_count": 1,
                    "document_count": 1,
                    "truncated": False,
                },
                {
                    "entity": {"text": "Acme", "type": "PRODUCT", "mentions": 2},
                    "sources": [{"id": "occ-product"}],
                    "chunk_count": 1,
                    "document_count": 1,
                    "truncated": False,
                },
            ],
        }

    def expand_query_with_graph_with_debug(self, query: str) -> tuple[str, dict[str, Any]]:
        """Return deterministic GraphRAG expansion metadata for tests.

        Args:
            query: Input query.

        Returns:
            tuple[str, dict[str, Any]]: Expanded query and debug metadata.
        """
        return (
            f"{query}\n\nRelated entities for retrieval: Acme",
            {
                "enabled": True,
                "applied": True,
                "original_query": query,
                "expanded_query": f"{query}\n\nRelated entities for retrieval: Acme",
                "anchor_entities": ["Acme"],
                "neighbor_entities": ["Widget"],
            },
        )

    def summarize_collection(self, refresh: bool = False) -> dict[str, Any]:
        """Return canned summarize payload.

        Args:
            refresh (bool, optional): Whether to bypass summary cache.

        Returns:
            dict[str, Any]: A dictionary containing the summary response, sources, and diagnostics.
        """
        self.summary_refresh_calls.append(bool(refresh))
        return self.summary_payload

    def stream_summarize_collection(
        self,
        refresh: bool = False,
    ) -> Generator[str | dict[str, Any], None, None]:
        """Stream canned summary payload.

        Args:
            refresh (bool, optional): Whether to bypass summary cache.

        Returns:
            Generator[str | dict[str, Any], None, None]: Streams summary chunks plus the final
                payload.

        Yields:
            str | dict[str, Any]: Summary chunks, then the final summary payload.
        """
        self.summary_stream_refresh_calls.append(bool(refresh))
        yield "sum"
        yield self.summary_payload

    def get_collection_ner(self, refresh: bool = False) -> list[dict[str, Any]]:
        """Get information extraction data for the selected collection.

        Args:
            refresh (bool, optional): Whether to bypass cached NER rows.

        Returns:
            list[dict[str, Any]]: Information extraction data for the selected collection.
        """
        self.ner_refresh_calls.append(bool(refresh))
        return self.ner_sources

    def get_collection_hate_speech(self) -> list[dict[str, Any]]:
        """Get hate-speech findings for the selected collection.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing metadata about hate-speech
            findings, such as chunk ID, text, category, confidence, reason, source reference,
            and page number.
        """
        return self.hate_speech_rows

    def list_documents(self) -> list[dict[str, Any]]:
        """Return the canned list of documents for the active collection."""
        return self.documents

    def iter_documents(
        self,
        *,
        cursor: str | None = None,
        limit: int = 50,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Stub paginated document iterator that slices a fixed list."""
        from docint.utils.cursor import decode_cursor, encode_cursor

        offset = int(decode_cursor(cursor).get("o") or 0)
        rows = getattr(self, "documents", [])
        end = offset + max(1, int(limit))
        page = rows[offset:end]
        next_cursor = encode_cursor(end) if end < len(rows) else None
        return page, next_cursor

    def iter_hate_speech(
        self,
        *,
        cursor: str | None = None,
        limit: int = 50,
        category: str | None = None,
        min_confidence: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Stub paginated hate-speech iterator that slices ``hate_speech_rows``."""
        from docint.utils.cursor import decode_cursor, encode_cursor

        offset = int(decode_cursor(cursor).get("o") or 0)
        rows = self.hate_speech_rows
        end = offset + max(1, int(limit))
        page = rows[offset:end]
        next_cursor = encode_cursor(end) if end < len(rows) else None
        return page, next_cursor

    def iter_collection_ner_sources(
        self,
        *,
        cursor: str | None = None,
        limit: int = 50,
        entity_key: str | None = None,
        entity_text: str | None = None,
        entity_type: str | None = None,
        entity_merge_mode: str = "orthographic",
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Stub paginated NER source iterator that slices ``ner_sources``."""
        from docint.utils.cursor import decode_cursor, encode_cursor

        self.last_ner_sources_filter = {
            "entity_key": entity_key,
            "entity_text": entity_text,
            "entity_type": entity_type,
            "entity_merge_mode": entity_merge_mode,
        }
        offset = int(decode_cursor(cursor).get("o") or 0)
        rows = self.ner_sources
        end = offset + max(1, int(limit))
        page = rows[offset:end]
        next_cursor = encode_cursor(end) if end < len(rows) else None
        return page, next_cursor

    def _get_collection_ner_aggregate(self, **_: Any) -> dict[str, Any]:
        """Stub aggregate warm-up that returns an empty payload."""
        self.warm_calls = getattr(self, "warm_calls", 0) + 1
        return {"entities": [], "relations": []}

    def get_collection_ner_stats(
        self,
        *,
        top_k: int = 15,
        min_mentions: int = 2,
        entity_type: str | None = None,
        include_relations: bool = True,
        entity_merge_mode: str = "orthographic",
    ) -> dict[str, Any]:
        """Return canned NER stats payload.

        Args:
            top_k (int, optional): Number of top entities to return. Defaults to 15.
            min_mentions (int, optional): Minimum mention count for inclusion. Defaults to 2.
            entity_type (str | None, optional): Filter entities by type. Defaults to None.
            include_relations (bool, optional): Whether to include relation statistics.
                Defaults to True.
            entity_merge_mode (str): Entity clustering mode (recorded for assertions).

        Returns:
            dict[str, Any]: NER stats payload with totals, top entities, entity types, top
                relations, and document-level stats.
        """
        _ = (top_k, min_mentions, entity_type, include_relations)
        self.ner_stats_merge_modes.append(entity_merge_mode)
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
            "top_relations": [{"head": "Acme", "label": "owns", "tail": "Widget", "mentions": 2}],
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
        entity_merge_mode: str = "orthographic",
    ) -> list[dict[str, Any]]:
        """Return simple entity search results.

        Args:
            q (str, optional): The search query string. Defaults to "".
            entity_type (str | None, optional): Filter entities by type. Defaults to None.
            limit (int, optional): The maximum number of results to return. Defaults to 100.
            entity_merge_mode (str): Entity clustering mode (recorded for assertions).

        Returns:
            list[dict[str, Any]]: A list of entity dictionaries that match the search criteria.
        """
        _ = (entity_type, limit)
        self.ner_search_merge_modes.append(entity_merge_mode)
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
        top_k_nodes: int = 80,
        min_edge_weight: int = 1,
        entity_merge_mode: str = "orthographic",
    ) -> dict[str, Any]:
        """Return a canned entity graph payload.

        Args:
            top_k_nodes (int): Node cap (recorded indirectly via the payload meta).
            min_edge_weight (int): Edge weight threshold (ignored by the stub).
            entity_merge_mode (str): Entity clustering mode (recorded for assertions).

        Returns:
            dict[str, Any]: Graph payload with nodes, edges, and meta counts.
        """
        _ = (top_k_nodes, min_edge_weight)
        self.ner_graph_merge_modes.append(entity_merge_mode)
        return {
            "nodes": [
                {"id": "acme::org", "text": "Acme", "type": "ORG", "mentions": 3},
                {"id": "widget::product", "text": "Widget", "type": "PRODUCT", "mentions": 2},
            ],
            "edges": [
                {
                    "source": "acme::org",
                    "target": "widget::product",
                    "label": "owns",
                    "kind": "relation",
                    "weight": 2,
                }
            ],
            "meta": {"node_count": 2, "edge_count": 1},
        }

    def resolve_entities(self, *, progress_callback: Any = None) -> ResolutionSummary:
        """Record a resolution call and return a fixed summary.

        Args:
            progress_callback (Any): Optional progress sink (ignored).

        Returns:
            ResolutionSummary: Fixed counts for endpoint assertions.
        """
        _ = progress_callback
        self.resolve_called = True
        return ResolutionSummary(processed=4, minted=2, attached=1, skipped=1, entities_touched=2)


@pytest.fixture(autouse=True)
def _patch_rag(monkeypatch: pytest.MonkeyPatch) -> Any | None:
    """Patch the RAG instance for testing.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.

    Returns:
        Any | None: Yields None after patching.
    """
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "test-operator")
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


def test_collections_list_failure(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
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


def test_collections_select_success(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Selecting a collection must succeed without warming the query engine.

    Regression guard for the same OOM pattern that commit 18a47a6 removed
    from ``/ingest`` and ``/ingest/upload``: ``/collections/select``
    previously called ``rag.create_index`` + ``rag.create_query_engine`` +
    ``rag.get_collection_ner(refresh=True)`` immediately after
    ``rag.select_collection``. That chain loads bge-m3 (~2 GB),
    bge-reranker-v2-m3 (~1 GB), and GLiNER on every collection switch,
    causing OOM-kill on CPU Docker. The query engine is now built lazily
    on the first chat query.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    response = client.post("/collections/select", json={"name": " gamma "})
    assert response.status_code == 200
    payload = response.json()
    assert payload == {"ok": True, "name": "gamma"}
    rag = cast(Any, api_module.rag)
    assert rag.qdrant_collection == "gamma"
    # Neither the index nor the query engine may be built eagerly on select.
    assert rag.created_index == 0, (
        "rag.create_index() must NOT be called from /collections/select; "
        "it triggers bge-m3 load and OOM-kills CPU Docker on collection switch."
    )
    assert rag.created_query_engine == 0, (
        "rag.create_query_engine() must NOT be called from /collections/select; "
        "it triggers reranker + embedding loads and OOM-kills CPU Docker."
    )
    # NER pre-warm must also be skipped — it previously triggered GLiNER on select.
    assert rag.ner_refresh_calls == [], (
        "get_collection_ner() must NOT be called from /collections/select; "
        "it triggers GLiNER load and compounds warmup memory pressure."
    )


def test_collections_select_blank_name(client: TestClient) -> None:
    """Blank collection names must surface as a structured HTTP 400.

    Regression guard: a prior handler caught ``HTTPException`` and
    re-raised it as 500, collapsing the original 400 into a generic
    server error. The handler must now propagate the original status.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.post("/collections/select", json={"name": "   "})
    assert response.status_code == 400
    assert "Collection name required" in response.json()["detail"]


def test_collections_select_returns_404_on_nonexistent_collection(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """``ValueError`` from ``select_collection`` must surface as HTTP 404.

    Regression guard for the post-ingest navigation bug: when the
    sidebar (or the ingest page) POSTs ``/collections/select`` for a
    collection that Qdrant hasn't yet exposed, ``rag.select_collection``
    raises ``ValueError``. The handler previously only caught
    ``HTTPException``, so the ``ValueError`` escaped as an unhandled
    server exception, the UI logged "Failed to select collection" with
    no usable detail, and ``st.session_state.selected_collection`` was
    left stale. The endpoint must now translate this into a structured
    404 so the UI can react and recover.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raise_missing(name: str) -> None:
        raise ValueError(f"Collection '{name}' does not exist")

    monkeypatch.setattr(api_module.rag, "select_collection", raise_missing)
    response = client.post("/collections/select", json={"name": "ghost"})
    assert response.status_code == 404
    assert "does not exist" in response.json()["detail"]


def test_collections_select_returns_500_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Unexpected backend exceptions must surface as HTTP 500 with detail.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raise_unexpected(name: str) -> None:
        raise RuntimeError("qdrant exploded")

    monkeypatch.setattr(api_module.rag, "select_collection", raise_unexpected)
    response = client.post("/collections/select", json={"name": "alpha"})
    assert response.status_code == 500
    assert "qdrant exploded" in response.json()["detail"]


def test_collections_ner_success(client: TestClient) -> None:
    """Test the successful retrieval of information extraction data.

    Args:
        client (TestClient): The TestClient instance.
    """
    api_module.rag.ner_sources = [{"filename": "doc1.pdf", "page": 1, "row": 2, "entities": [], "relations": []}]
    response = client.get("/collections/ner")
    assert response.status_code == 200
    assert response.json() == {"sources": api_module.rag.ner_sources}
    assert cast(DummyRAG, api_module.rag).ner_refresh_calls[-1] is False


def test_collections_ner_refresh_success(client: TestClient) -> None:
    """NER endpoint should forward explicit refresh requests.

    Args:
        client (TestClient): The TestClient instance.
    """
    dummy_rag = cast(DummyRAG, api_module.rag)
    dummy_rag.ner_sources = [{"filename": "doc1.pdf", "entities": [], "relations": []}]

    response = client.get("/collections/ner", params={"refresh": "true"})

    assert response.status_code == 200
    assert response.json()["sources"] == dummy_rag.ner_sources
    assert dummy_rag.ner_refresh_calls[-1] is True


def test_collections_ner_stats_success(client: TestClient) -> None:
    """Stats endpoint should return NER summary payload.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.get("/collections/ner/stats")
    assert response.status_code == 200
    payload = response.json()
    assert payload["totals"]["unique_entities"] == 1
    assert payload["top_entities"][0]["text"] == "Acme"
    assert cast(DummyRAG, api_module.rag).ner_stats_merge_modes[-1] == "orthographic"


def test_collections_ner_stats_support_exact_merge_mode(client: TestClient) -> None:
    """Stats endpoint should forward explicit merge-mode overrides."""
    response = client.get("/collections/ner/stats", params={"entity_merge_mode": "exact"})
    assert response.status_code == 200
    assert cast(DummyRAG, api_module.rag).ner_stats_merge_modes[-1] == "exact"


def test_collections_hate_speech_success(client: TestClient) -> None:
    """Hate-speech endpoint should return flagged rows.

    Args:
        client (TestClient): The TestClient instance.
    """
    dummy_rag = cast(DummyRAG, api_module.rag)
    dummy_rag.hate_speech_rows = [
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
    assert cast(DummyRAG, api_module.rag).ner_search_merge_modes[-1] == "orthographic"


def test_collections_ner_search_support_exact_merge_mode(client: TestClient) -> None:
    """Search endpoint should forward explicit merge-mode overrides."""
    response = client.get(
        "/collections/ner/search",
        params={"q": "ac", "entity_merge_mode": "exact"},
    )
    assert response.status_code == 200
    assert cast(DummyRAG, api_module.rag).ner_search_merge_modes[-1] == "exact"


def test_collections_ner_stats_support_resolved_merge_mode(client: TestClient) -> None:
    """Stats endpoint should accept and forward the resolved merge mode."""
    response = client.get("/collections/ner/stats", params={"entity_merge_mode": "resolved"})
    assert response.status_code == 200
    assert cast(DummyRAG, api_module.rag).ner_stats_merge_modes[-1] == "resolved"


def test_collections_ner_graph_success(client: TestClient) -> None:
    """Graph endpoint should return nodes, edges, and meta counts.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.get("/collections/ner/graph")
    assert response.status_code == 200
    payload = response.json()
    assert payload["nodes"][0]["text"] == "Acme"
    assert payload["edges"][0]["kind"] == "relation"
    assert payload["meta"]["node_count"] == 2
    assert cast(DummyRAG, api_module.rag).ner_graph_merge_modes[-1] == "orthographic"


def test_collections_ner_graph_forwards_merge_mode(client: TestClient) -> None:
    """Graph endpoint should forward explicit merge-mode overrides."""
    response = client.get("/collections/ner/graph", params={"entity_merge_mode": "resolved"})
    assert response.status_code == 200
    assert cast(DummyRAG, api_module.rag).ner_graph_merge_modes[-1] == "resolved"


def test_collections_ner_graph_requires_selection(client: TestClient) -> None:
    """Graph endpoint should 400 when no collection is selected."""
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/ner/graph")
    assert response.status_code == 400


def test_collections_ner_graph_failure(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Graph endpoint should 500 when the RAG layer raises.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raiser(**_: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module.rag, "get_collection_ner_graph", raiser)
    response = client.get("/collections/ner/graph")
    assert response.status_code == 500


def test_resolve_entities_success(client: TestClient) -> None:
    """The resolve endpoint returns the resolution summary counts."""
    response = client.post("/collections/entities/resolve")
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "processed": 4,
        "minted": 2,
        "attached": 1,
        "skipped": 1,
        "entities_touched": 2,
    }


def test_resolve_entities_requires_selected_collection(client: TestClient) -> None:
    """The resolve endpoint 400s when no collection is selected."""
    api_module.rag.qdrant_collection = ""
    response = client.post("/collections/entities/resolve")
    assert response.status_code == 400


def test_agent_chat_answers(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Agent chat should return an answer when confidence is sufficient.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def fake_chat(question: str, **_: Any) -> dict[str, Any]:
        """Fake implementation of the RAG chat method for testing purposes.

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


def test_agent_chat_clarifies(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Agent chat should request clarification when policy requires it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setattr(
        api_module,
        "_clarification_policy",
        api_module.ClarificationPolicy(api_module.ClarificationConfig(confidence_threshold=1.0, require_entities=True)),
    )

    payload = {"message": "hello"}
    response = client.post("/agent/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "clarification"
    assert data["message"]
    assert data["intent"] is not None
    assert data["confidence"] is not None


def test_agent_chat_falls_back_to_clarification_on_weak_validation_mismatch(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """A weak (refusal-shaped) answer with validation_mismatch must surface as clarification.

    Exercises the orchestrator's post-responder fallback end-to-end: the
    monkeypatched orchestrator returns a ``RetrievalResult`` shaped exactly
    like the production failure (answer="Evidence insufficient.",
    validation_mismatch=True), and the API must respond with
    ``status="clarification"`` and a helpful nudge instead of echoing the
    refusal back to the user.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    # Use a real orchestrator (post-responder fallback lives there).
    monkeypatch.setattr(
        api_module.rag,
        "chat",
        lambda *_, **__: {
            "response": "Evidence insufficient.",
            "sources": [],
        },
    )

    # Force the response validator to flag mismatch by stubbing it.
    from docint.agents.generation import ResultValidationResponseAgent
    from docint.agents.types import RetrievalResult as _RR

    def _flag_mismatch(self: Any, result: _RR, turn: Any) -> _RR:
        """Mark every result as mismatched for this test."""
        _ = self, turn
        result.validation_checked = True
        result.validation_mismatch = True
        result.validation_reason = "no UN content in sources"
        return result

    monkeypatch.setattr(ResultValidationResponseAgent, "finalize", _flag_mismatch)

    response = client.post("/agent/chat", json={"message": "Please elaborate."})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "clarification"
    assert data["message"]
    assert "previous answer" in data["message"].lower() or "elaborate" in data["message"].lower()


def test_agent_chat_returns_validation_alert(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Agent chat should surface response-validation metadata.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    class _StubOrchestrator:
        """Stub orchestrator that returns a canned retrieval result with validation metadata for testing purposes."""

        def handle_turn(self, turn: Any, context: Any = None) -> OrchestratorResult:
            """Handle a turn by returning a canned retrieval result with validation metadata.

            Args:
                turn (_type_): The user turn to process.
                context (_type_, optional): The context for the turn. Defaults to None.

            Returns:
                OrchestratorResult: The result of processing the turn.
            """
            _ = turn, context
            analysis = IntentAnalysis(intent="qa", confidence=0.9, entities={"query": "hello"})
            retrieval = RetrievalResult(
                answer="answer",
                sources=[{"id": 1}],
                session_id="generated-session",
                validation_checked=True,
                validation_mismatch=True,
                validation_reason="mismatch",
            )
            return OrchestratorResult(clarification=None, retrieval=retrieval, analysis=analysis)

    monkeypatch.setattr(api_module, "_build_orchestrator", lambda: _StubOrchestrator())

    response = client.post("/agent/chat", json={"message": "hello"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "answer"
    assert data["validation_checked"] is True
    assert data["validation_mismatch"] is True
    assert data["validation_reason"] == "mismatch"


def test_agent_chat_stream_clarifies(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Streaming endpoint should emit clarification event when policy demands it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setattr(
        api_module,
        "_clarification_policy",
        api_module.ClarificationPolicy(api_module.ClarificationConfig(confidence_threshold=1.0, require_entities=True)),
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
    assert '"graph_debug"' in text
    assert '"retrieval_query"' in text
    assert '"retrieval_mode"' in text
    assert '"response": "answer"' in text


def test_query_stamps_default_identity_on_session_start(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Session-backed query requests must stamp the resolved principal.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    seen: dict[str, Any] = {}

    def record_start_session(session_id: str | None = None, owner: str | None = None) -> str:
        seen["session_id"] = session_id
        seen["owner"] = owner
        return session_id or "generated-session"

    monkeypatch.setattr(api_module.rag, "start_session", record_start_session)

    response = client.post("/query", json={"question": "hello"})

    assert response.status_code == 200
    assert seen == {"session_id": None, "owner": "operator"}


def test_stream_query_stamps_default_identity_on_session_start(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Session-backed stream queries must stamp the resolved principal.

    The frontend uses ``/stream_query`` for chat. If the write path starts
    sessions without the same owner that ``/sessions/list`` later filters by,
    chats persist but never appear in the sidebar.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    seen: dict[str, Any] = {}

    def record_start_session(session_id: str | None = None, owner: str | None = None) -> str:
        seen["session_id"] = session_id
        seen["owner"] = owner
        return session_id or "generated-session"

    monkeypatch.setattr(api_module.rag, "start_session", record_start_session)

    with client.stream("POST", "/stream_query", json={"question": "hello"}) as resp:
        assert resp.status_code == 200
        list(resp.iter_lines())

    assert seen == {"session_id": None, "owner": "operator"}


def test_agent_chat_stamps_default_identity_on_session_start(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Agent chat must stamp the resolved principal on session start.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    seen: dict[str, Any] = {}

    class _StubOrchestrator:
        def handle_turn(self, turn: Any, context: Any = None) -> OrchestratorResult:
            _ = turn, context
            analysis = IntentAnalysis(intent="qa", confidence=0.9, entities={"query": "hello"})
            retrieval = RetrievalResult(answer="answer", sources=[{"id": 1}], session_id="generated-session")
            return OrchestratorResult(clarification=None, retrieval=retrieval, analysis=analysis)

    def record_start_session(session_id: str | None = None, owner: str | None = None) -> str:
        seen["session_id"] = session_id
        seen["owner"] = owner
        return session_id or "generated-session"

    monkeypatch.setattr(api_module, "_build_orchestrator", lambda: _StubOrchestrator())
    monkeypatch.setattr(api_module.rag, "start_session", record_start_session)

    response = client.post("/agent/chat", json={"message": "hello"})

    assert response.status_code == 200
    assert seen == {"session_id": None, "owner": "operator"}


def test_agent_chat_stream_stamps_default_identity_on_session_start(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Streaming agent chat must stamp the resolved principal on session start.

    ``/agent/chat/stream`` resolves the principal eagerly (before the SSE
    generator) and passes it to ``start_session``. Without it, agent chats would
    persist unowned and never surface in ``/sessions/list``. The clarification
    policy is forced so the generator reaches ``start_session`` and returns
    without depending on the chat stream.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    seen: dict[str, Any] = {}

    def record_start_session(session_id: str | None = None, owner: str | None = None) -> str:
        seen["session_id"] = session_id
        seen["owner"] = owner
        return session_id or "generated-session"

    monkeypatch.setattr(api_module.rag, "start_session", record_start_session)
    monkeypatch.setattr(
        api_module,
        "_clarification_policy",
        api_module.ClarificationPolicy(api_module.ClarificationConfig(confidence_threshold=1.0, require_entities=True)),
    )

    with client.stream("POST", "/agent/chat/stream", json={"message": "hello"}) as resp:
        assert resp.status_code == 200
        list(resp.iter_lines())

    assert seen == {"session_id": None, "owner": "operator"}


def test_agent_chat_stream_uses_history_and_prior_turn(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Streaming agent chat must feed prior history into understanding and stream_chat.

    Verifies parity with /agent/chat: prior_turn + history-rewritten query
    are both forwarded to stream_chat.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    seeded_history = [
        {"role": "user", "content": "Who chairs the council?"},
        {"role": "assistant", "content": "The Security Council has a rotating presidency."},
    ]
    monkeypatch.setattr(
        api_module.rag.sessions,
        "get_session_history",
        lambda session_id, owner=None: seeded_history,
    )

    seen: dict[str, Any] = {}

    class _RecordingUnderstanding:
        def analyze(self, turn: Any, context: Any = None) -> IntentAnalysis:
            seen["context_history"] = list(context.history) if context is not None else None
            return IntentAnalysis(
                intent="qa",
                confidence=0.9,
                entities={"query": turn.user_input},
                rewritten_query="REWRITTEN QUERY",
            )

    def record_stream_chat(
        user_msg: str, *, prior_turn: Any = None, **kwargs: Any
    ) -> Generator[str | dict[str, Any], None, None]:
        seen["stream_query"] = user_msg
        seen["prior_turn"] = prior_turn
        yield "token"
        yield {"sources": [], "session_id": "generated-session"}

    monkeypatch.setattr(api_module, "_understanding_agent", _RecordingUnderstanding())
    monkeypatch.setattr(
        api_module,
        "_clarification_policy",
        api_module.ClarificationPolicy(
            api_module.ClarificationConfig(confidence_threshold=0.0, require_entities=False)
        ),
    )
    monkeypatch.setattr(api_module.rag, "stream_chat", record_stream_chat)

    with client.stream("POST", "/agent/chat/stream", json={"message": "And who is she?"}) as resp:
        assert resp.status_code == 200
        list(resp.iter_lines())

    assert seen["context_history"] == seeded_history
    assert seen["stream_query"] == "REWRITTEN QUERY"
    assert isinstance(seen["prior_turn"], PriorTurn)
    assert seen["prior_turn"].user_text == "Who chairs the council?"
    assert seen["prior_turn"].assistant_text == "The Security Council has a rotating presidency."


def test_stream_query_session_mode_feeds_prior_turn(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """``/stream_query`` (the endpoint the React SPA actually calls) must feed the prior turn.

    In session mode it should build the immediately preceding user/assistant
    exchange from owner-scoped history and pass it to ``stream_chat`` together
    with ``skip_query_rewrite=False`` — so generation becomes history-aware while
    the endpoint keeps its own internal retrieval-query rewrite.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    seeded_history = [
        {"role": "user", "content": "Was ist im Bild sichtbar?"},
        {"role": "assistant", "content": "Ein grüner Baum auf einer Wiese."},
    ]
    monkeypatch.setattr(
        api_module.rag.sessions,
        "get_session_history",
        lambda session_id, owner=None: seeded_history,
    )

    seen: dict[str, Any] = {}

    def record_stream_chat(
        user_msg: str,
        *,
        prior_turn: Any = None,
        skip_query_rewrite: Any = None,
        **kwargs: Any,
    ) -> Generator[str | dict[str, Any], None, None]:
        seen["user_msg"] = user_msg
        seen["prior_turn"] = prior_turn
        seen["skip_query_rewrite"] = skip_query_rewrite
        yield "tok"
        yield {"response": "answer", "sources": [], "session_id": "generated-session"}

    monkeypatch.setattr(api_module.rag, "stream_chat", record_stream_chat)

    with client.stream("POST", "/stream_query", json={"question": "Enthält es Menschen?"}) as resp:
        assert resp.status_code == 200
        list(resp.iter_lines())

    # The raw user message reaches stream_chat (the internal rewrite still runs there).
    assert seen["user_msg"] == "Enthält es Menschen?"
    assert seen["skip_query_rewrite"] is False
    assert isinstance(seen["prior_turn"], PriorTurn)
    assert seen["prior_turn"].user_text == "Was ist im Bild sichtbar?"
    assert seen["prior_turn"].assistant_text == "Ein grüner Baum auf einer Wiese."


def test_agent_chat_history_is_owner_scoped_on_both_endpoints(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Both agent endpoints load session history scoped to the resolved principal.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    owners: list[str | None] = []

    def record_history(session_id: str, owner: str | None = None) -> list[dict[str, str]]:
        owners.append(owner)
        return [{"role": "user", "content": "hi"}]

    monkeypatch.setattr(api_module.rag.sessions, "get_session_history", record_history)

    class _StubOrchestrator:
        def handle_turn(self, turn: Any, context: Any = None) -> OrchestratorResult:
            _ = turn, context
            analysis = IntentAnalysis(intent="qa", confidence=0.9, entities={"query": "hi"})
            retrieval = RetrievalResult(answer="a", sources=[], session_id="generated-session")
            return OrchestratorResult(clarification=None, retrieval=retrieval, analysis=analysis)

    monkeypatch.setattr(api_module, "_build_orchestrator", lambda: _StubOrchestrator())
    resp1 = client.post("/agent/chat", json={"message": "hi"})
    assert resp1.status_code == 200

    monkeypatch.setattr(
        api_module,
        "_clarification_policy",
        api_module.ClarificationPolicy(api_module.ClarificationConfig(confidence_threshold=1.0, require_entities=True)),
    )
    with client.stream("POST", "/agent/chat/stream", json={"message": "hi"}) as resp2:
        assert resp2.status_code == 200
        list(resp2.iter_lines())

    assert owners == ["operator", "operator"]


def test_select_understanding_agent_falls_back_to_simple_without_llm(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Without a configured LLM, the shared selector returns the module-level simple agent."""
    monkeypatch.setattr(api_module.rag, "text_model_id", None, raising=False)
    assert api_module._select_understanding_agent() is api_module._understanding_agent


def test_select_understanding_agent_prefers_contextual_when_llm_configured(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """With an LLM configured, the selector returns the history-aware contextual agent."""
    monkeypatch.setattr(api_module.rag, "text_model_id", "test-model", raising=False)
    monkeypatch.setattr(api_module.rag, "text_model", object(), raising=False)
    assert isinstance(api_module._select_understanding_agent(), ContextualUnderstandingAgent)


def test_query_stateless_mode_skips_session_chat(client: TestClient) -> None:
    """Stateless query mode should use direct retrieval without chat session state.

    Args:
        client: The TestClient instance.
    """
    rag = cast(DummyRAG, api_module.rag)
    before_chats = len(rag.chats)

    response = client.post(
        "/query",
        json={"question": "What?", "retrieval_mode": "stateless"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "answer"
    assert body["session_id"] == "stateless"
    assert len(rag.chats) == before_chats
    assert rag.stateless_queries[-1].startswith("What?")
    assert body["graph_debug"]["applied"] is True


def test_stream_query_stateless_mode_emits_tokens(client: TestClient) -> None:
    """Stateless stream mode should emit tokens via the async query path.

    The stateless branch must call the native-async ``run_query_async`` (so the
    event loop is not blocked) and forward the request-scoped filters intact.

    Args:
        client: The TestClient instance.
    """
    rag = cast(DummyRAG, api_module.rag)
    before = len(rag.stateless_query_filters)
    with client.stream(
        "POST",
        "/stream_query",
        json={"question": "hello", "retrieval_mode": "stateless"},
    ) as resp:
        assert resp.status_code == 200
        text = "".join([chunk.decode() for chunk in resp.iter_raw()])

    assert '"token"' in text
    assert '"session_id": "stateless"' in text
    assert '"graph_debug"' in text
    # The async query path was exercised and carried the filter kwargs.
    assert len(rag.stateless_query_filters) == before + 1
    recorded = rag.stateless_query_filters[-1]
    assert set(recorded) == {"filters", "rules", "vector_store_kwargs"}


@pytest.mark.anyio
async def test_stream_simulated_text_applies_visible_pacing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic token replay should keep a small delay between chunks.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
    """
    delays: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(api_module.asyncio, "sleep", _fake_sleep)

    events: list[str] = []
    async for event in api_module._stream_simulated_text("hello world"):
        events.append(event)

    assert len(events) == 2
    assert all('"token"' in event for event in events)
    assert delays == [
        api_module.SIMULATED_STREAM_TOKEN_DELAY_SECONDS,
        api_module.SIMULATED_STREAM_TOKEN_DELAY_SECONDS,
    ]


def test_query_entity_occurrence_mode_skips_chat_and_uses_ner_lookup(
    client: TestClient,
) -> None:
    """Entity occurrence mode should bypass chat and return mention rows.

    Args:
        client: The TestClient instance.
    """
    rag = cast(DummyRAG, api_module.rag)
    before_chats = len(rag.chats)

    response = client.post(
        "/query",
        json={"question": "Acme", "query_mode": "entity_occurrence"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Found 3 occurrence")
    assert body["sources"] == [{"id": "occ-1"}, {"id": "occ-2"}]
    assert body["session_id"] == "stateless"
    assert len(rag.chats) == before_chats
    assert rag.entity_occurrence_queries[-1] == "Acme"
    assert body["retrieval_mode"] == "entity_occurrence"


def test_query_entity_occurrence_mode_passes_native_filters(
    client: TestClient,
) -> None:
    """Entity occurrence mode should reuse native Qdrant metadata filters.

    Args:
        client: The TestClient instance.
    """
    response = client.post(
        "/query",
        json={
            "question": "Acme",
            "query_mode": "entity_occurrence",
            "metadata_filters": [
                {
                    "field": "hate_speech.hate_speech",
                    "operator": "eq",
                    "value": True,
                }
            ],
        },
    )

    assert response.status_code == 200
    rag = cast(DummyRAG, api_module.rag)
    assert rag.entity_occurrence_filters[-1] is not None


def test_stream_query_entity_occurrence_mode_emits_tokens(client: TestClient) -> None:
    """Streaming occurrence mode should stream the synthesized occurrence summary.

    Args:
        client: The TestClient instance.
    """
    with client.stream(
        "POST",
        "/stream_query",
        json={"question": "Acme", "query_mode": "entity_occurrence"},
    ) as resp:
        assert resp.status_code == 200
        text = "".join([chunk.decode() for chunk in resp.iter_raw()])

    assert '"token"' in text
    assert '"retrieval_mode": "entity_occurrence"' in text
    assert '"coverage_unit": "entity_mentions"' in text


def test_query_entity_occurrence_mode_returns_ambiguity_candidates(
    client: TestClient,
) -> None:
    """Ambiguous single-entity occurrence lookups should expose candidates."""
    response = client.post(
        "/query",
        json={"question": "Ambiguous", "query_mode": "entity_occurrence"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_mode"] == "entity_occurrence_ambiguous"
    assert body["sources"] == []
    assert len(body["entity_match_candidates"]) == 2


def test_query_entity_occurrence_multi_mode_returns_groups(client: TestClient) -> None:
    """Multi-entity occurrence mode should return grouped strong matches."""
    response = client.post(
        "/query",
        json={"question": "Acme", "query_mode": "entity_occurrence_multi"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_mode"] == "entity_occurrence_multi"
    assert len(body["entity_match_groups"]) == 2
    assert body["entity_match_groups"][0]["entity"]["text"] == "Acme"


def test_stream_query_entity_occurrence_multi_mode_emits_groups(
    client: TestClient,
) -> None:
    """Streaming multi-entity occurrence mode should include grouped results."""
    with client.stream(
        "POST",
        "/stream_query",
        json={"question": "Acme", "query_mode": "entity_occurrence_multi"},
    ) as resp:
        assert resp.status_code == 200
        text = "".join([chunk.decode() for chunk in resp.iter_raw()])

    assert '"retrieval_mode": "entity_occurrence_multi"' in text
    assert '"entity_match_groups"' in text


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
    assert payload["summary_diagnostics"]["coverage_unit"] == "documents"
    assert "validation_checked" in payload
    assert "validation_mismatch" in payload
    assert "validation_reason" in payload


def test_summarize_refresh_flag_passthrough(client: TestClient) -> None:
    """Summarize endpoint should pass refresh query parameter to RAG.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.post("/summarize?refresh=true")
    assert response.status_code == 200
    rag = cast(DummyRAG, api_module.rag)
    assert rag.summary_refresh_calls[-1] is True


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
    assert '"coverage_unit"' in text


def test_summarize_stream_refresh_flag_passthrough(client: TestClient) -> None:
    """Streaming summarize endpoint should pass refresh query parameter to RAG.

    Args:
        client (TestClient): The TestClient instance.
    """
    with client.stream("POST", "/summarize/stream?refresh=true") as resp:
        assert resp.status_code == 200
        _ = "".join([chunk.decode() for chunk in resp.iter_raw()])
    rag = cast(DummyRAG, api_module.rag)
    assert rag.summary_stream_refresh_calls[-1] is True


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
    """Hate-speech endpoint should require active collection selection.

    Args:
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/hate-speech")
    assert response.status_code == 400
    assert "No collection selected" in response.json()["detail"]


def test_collections_ner_failure(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Test the failed retrieval of information extraction data.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raiser(*, refresh: bool = False) -> list[dict[str, Any]]:
        """Get information extraction data for the selected collection.

        Args:
            refresh (bool, optional): Whether to bypass cached NER rows.

        Returns:
            list[dict[str, Any]]: Information extraction data for the selected collection.

        Raises:
            RuntimeError: If there is an error retrieving the information extraction data.
        """
        _ = refresh
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module.rag, "get_collection_ner", raiser)
    response = client.get("/collections/ner")
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_collections_ner_stats_failure(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Stats endpoint should surface backend failures.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raiser(**kwargs: Any) -> dict[str, Any]:
        """Fake implementation of get_collection_ner_stats that raises an error for testing purposes.

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


def test_collections_ner_search_failure(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Search endpoint should surface backend failures.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raiser(**kwargs: Any) -> list[dict[str, Any]]:
        """Fake implementation of search_collection_ner_entities that raises an error for testing purposes.

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


def test_collections_hate_speech_failure(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Hate-speech endpoint should surface backend failures.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raiser() -> list[dict[str, Any]]:
        """Fake implementation of get_collection_hate_speech that raises an error for testing purposes.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing metadata about hate-speech
            findings, such as chunk ID, text, category, confidence, reason, source reference,
            and page number.

        Raises:
            RuntimeError: If there is an error retrieving the hate-speech findings.
        """
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module.rag, "get_collection_hate_speech", raiser)
    response = client.get("/collections/hate-speech")
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_query_requires_collection(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Empty active collection must surface as a structured HTTP 400.

    Regression guard for the outer-handler antipattern that collapsed
    a 400 into a 500. The handler must now propagate the 400 raised
    by ``_require_active_collection`` unchanged.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = ""
    response = client.post("/query", json={"question": "hi"})
    assert response.status_code == 400
    assert "No collection selected" in response.json()["detail"]


def test_query_returns_404_when_active_collection_missing(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Stale active collection must surface as HTTP 404 with a clean message.

    Regression guard for the chat-after-delete crash: if a collection is
    deleted out-of-band (or the API singleton holds a stale name),
    ``_require_active_collection`` must trip and return a structured 404
    instead of letting llama-index propagate Qdrant's raw 404.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = "ghost"
    monkeypatch.setattr(api_module.rag, "list_collections", lambda: ["alpha", "beta"])
    response = client.post("/query", json={"question": "hi"})
    assert response.status_code == 404
    assert "ghost" in response.json()["detail"]
    assert "no longer exists" in response.json()["detail"]
    # Singleton must self-heal so the user can recover via re-select.
    assert api_module.rag.qdrant_collection == ""
    assert api_module.rag.index is None
    assert api_module.rag.query_engine is None


def test_stream_query_returns_404_when_active_collection_missing(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Stream query must gate on collection existence before opening the SSE stream.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = "ghost"
    monkeypatch.setattr(api_module.rag, "list_collections", lambda: ["alpha", "beta"])
    response = client.post("/stream_query", json={"question": "hi"})
    assert response.status_code == 404
    assert "ghost" in response.json()["detail"]


def test_stream_query_requires_collection(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Empty active collection on stream_query must surface as HTTP 400.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = ""
    response = client.post("/stream_query", json={"question": "hi"})
    assert response.status_code == 400
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
    assert body["graph_debug"]["applied"] is True
    assert body["graph_debug"]["anchor_entities"] == ["Acme"]


def test_query_builds_and_passes_metadata_filters(client: TestClient) -> None:
    """Query endpoint should compile request filters and pass them to RAG chat.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.post(
        "/query",
        json={
            "question": "What?",
            "metadata_filters": [
                {
                    "field": "mimetype",
                    "operator": "mime_match",
                    "value": "image/*",
                },
                {
                    "field": "reference_metadata.timestamp",
                    "operator": "date_on_or_after",
                    "value": "2026-01-01",
                },
            ],
        },
    )

    assert response.status_code == 200
    rag = cast(DummyRAG, api_module.rag)
    last_filters = rag.chat_filters[-1]
    assert last_filters["active"] is True
    assert [rule.model_dump() for rule in last_filters["rules"]] == [
        {
            "field": "mimetype",
            "operator": "mime_match",
            "value": "image/*",
            "values": [],
        },
        {
            "field": "reference_metadata.timestamp",
            "operator": "date_on_or_after",
            "value": "2026-01-01",
            "values": [],
        },
    ]
    compiled = last_filters["filters"]
    assert compiled is not None
    assert len(compiled.filters) == 2
    assert last_filters["vector_store_kwargs"]["qdrant_filters"] is not None


def test_stream_query_passes_metadata_filters(client: TestClient) -> None:
    """Streaming query endpoint should compile and pass request filters.

    Args:
        client (TestClient): The TestClient instance.
    """
    with client.stream(
        "POST",
        "/stream_query",
        json={
            "question": "hello",
            "metadata_filters": [
                {
                    "field": "hate_speech.hate_speech",
                    "operator": "eq",
                    "value": True,
                }
            ],
        },
    ) as resp:
        assert resp.status_code == 200
        assert any(line for line in resp.iter_lines())

    rag = cast(DummyRAG, api_module.rag)
    last_filters = rag.stream_filters[-1]
    assert last_filters["active"] is True
    assert [rule.model_dump() for rule in last_filters["rules"]] == [
        {
            "field": "hate_speech.hate_speech",
            "operator": "eq",
            "value": True,
            "values": [],
        }
    ]
    assert last_filters["vector_store_kwargs"]["qdrant_filters"] is not None


def test_query_handles_missing_sources(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Test the query handles missing sources.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def fake_chat(question: str, **_: Any) -> str:
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


def test_ingest_success(monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path) -> None:
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
        path: Any,
        hybrid: bool = True,
        progress_callback: Any = None,
    ) -> None:
        """Fake implementation of the ingest_docs function for testing purposes.

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
        "empty": False,
    }
    assert called.args[0:3] == ("docs", data_dir, False)


def test_sessions_endpoints(client: TestClient) -> None:
    """Test session management endpoints.

    Args:
        client (TestClient): The TestClient instance.
    """
    headers = {"X-Auth-User": "tester"}

    # List
    resp = client.get("/sessions/list", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["id"] == "123"

    # History
    resp = client.get("/sessions/123/history", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["messages"]) == 1
    assert data["messages"][0]["content"] == "hi"

    # Delete
    resp = client.delete("/sessions/123", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_ingest_missing_directory(monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path) -> None:
    """Test the ingestion of documents when the data directory is missing.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture.
    """
    missing = tmp_path / "missing"
    monkeypatch.setattr(api_module, "_resolve_data_dir", lambda: missing)
    response = client.post("/ingest", json={"collection": "abc"})
    assert response.status_code == 400
    assert "Data directory does not exist" in response.json()["detail"]


def test_ingest_sync_generic_exception_propagates_as_500(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Non-``EmptyIngestionError`` failures in sync ``/ingest`` surface as 500.

    Commit e1060fd added an explicit ``except EmptyIngestionError`` around
    the ``ingest_docs`` call that returns 200 with ``empty=true``. Generic
    runtime errors (Qdrant unreachable, disk full, OOM) must still
    propagate to FastAPI's default handler as an HTTP 500. This test
    guards against an over-broad exception catch accidentally swallowing
    real failures into the soft-empty response.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture, used as the data dir.
    """
    monkeypatch.setattr(api_module, "_resolve_data_dir", lambda: tmp_path)

    def exploding_ingest(
        collection: str,
        path: Path,
        hybrid: bool = True,
        progress_callback: Any = None,
    ) -> None:
        """Simulate a hard runtime failure during ingestion.

        Args:
            collection (str): Collection name (ignored).
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).

        Raises:
            RuntimeError: Always, to mimic an infrastructure failure.
        """
        _ = (collection, path, hybrid, progress_callback)
        raise RuntimeError("Qdrant unreachable")

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", exploding_ingest)

    # The endpoint must convert unexpected runtime failures into a
    # structured HTTP 500 rather than propagating an unhandled exception
    # or coercing them into the soft-empty 200 reserved for
    # ``EmptyIngestionError``.
    response = client.post("/ingest", json={"collection": "col", "hybrid": True})
    assert response.status_code == 500
    assert "Qdrant unreachable" in response.json()["detail"]


def test_ingest_upload_empty_emits_warning_and_completes(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Empty ingestion via /ingest/upload yields a warning + empty completion event.

    Verifies the API translates :class:`EmptyIngestionError` into an SSE
    ``warning`` event followed by an ``ingestion_complete`` event with
    ``"empty": true`` and skips ``rag.select_collection`` (which would
    otherwise raise ``ValueError`` because the collection was never
    created), instead of surfacing a generic ``"Ingestion failed"``.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture.
    """
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)

    def fake_ingest(
        collection: str,
        path: Path,
        hybrid: bool = True,
        progress_callback: Any = None,
    ) -> None:
        """Simulate an ingestion run that produced no documents.

        Args:
            collection (str): Collection name.
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
        """
        _ = (path, hybrid, progress_callback)
        raise api_module.EmptyIngestionError(collection)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)

    response = client.post(
        "/ingest/upload",
        data={"collection": "silence-test", "hybrid": "false"},
        files={"files": ("silence.m4a", b"\x00" * 32, "audio/mp4")},
    )

    assert response.status_code == 200
    body = response.text

    # The SSE payload should contain a warning event referencing the collection
    # and an ingestion_complete event flagged as empty=true. It should NOT
    # contain a generic "Ingestion failed" error event.
    assert "event: warning" in body
    assert "silence-test" in body
    assert "event: ingestion_complete" in body
    assert '"empty": true' in body
    assert "Ingestion failed" not in body

    # select_collection must NOT have been called — DummyRAG.selected stays empty.
    assert cast(Any, api_module.rag).selected == []


def test_ingest_upload_success_does_not_warm_query_engine(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Successful ``/ingest/upload`` must not eagerly warm the query engine.

    Regression guard for an OOM-kill observed on CPU Docker containers with
    the default 8 GB limit: the ingest SSE handler previously called
    ``rag.create_index()`` and ``rag.create_query_engine()`` on the
    module-level ``api.rag`` singleton immediately after a successful
    ingestion. That warmup triggered the reranker (bge-reranker-v2-m3,
    roughly 1 GB) and the embedding model (bge-m3, roughly 2 GB) to load
    on top of the PyTorch allocator memory still held by the just-finished
    ingest pipeline, blowing past the container memory cap and producing
    exit 137.

    The query engine is still built lazily on the next chat query, so the
    only user-visible effect of removing the warmup is a slower first-query
    TTFB. There is no correctness regression. This test pins down the
    behavioral defect (eager warmup) that reproduces the OOM, so that
    deleting the warmup block keeps the test green while any future
    reintroduction would fail it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture, used as the uploads dir.
    """
    # Route uploads into a temp dir so the endpoint does not touch real state.
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)

    def fake_ingest(
        collection: str,
        path: Path,
        hybrid: bool = True,
        progress_callback: Any = None,
    ) -> None:
        """Simulate a successful, no-op ingestion run.

        Returning cleanly causes the SSE handler to enqueue a ``None``
        sentinel and break out of the progress loop, which is exactly the
        code path that previously reached the eager warmup block.

        Args:
            collection (str): Collection name (ignored).
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
        """
        _ = (collection, path, hybrid, progress_callback)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)

    # Sanity: the autouse _patch_rag fixture installs a fresh DummyRAG whose
    # create_index / create_query_engine counters start at zero.
    dummy_rag = cast(DummyRAG, api_module.rag)
    assert dummy_rag.created_index == 0
    assert dummy_rag.created_query_engine == 0

    response = client.post(
        "/ingest/upload",
        data={"collection": "warmup-guard", "hybrid": "true"},
        files={"files": ("hello.txt", b"hello world", "text/plain")},
    )

    # The ingest itself must still succeed and emit the completion event —
    # we are not regressing success signalling, only removing the warmup.
    assert response.status_code == 200
    body = response.text
    assert "event: ingestion_complete" in body
    assert "Ingestion failed" not in body

    # Core assertion: neither the index nor the query engine may be built
    # eagerly during a successful ingest. Both counters must remain zero.
    # Under the buggy code path (now removed), the ingest SSE success path
    # called rag.create_index() and rag.create_query_engine() immediately
    # after the progress loop, loading reranker + embedding and OOM-killing
    # the backend on CPU Docker with default 8 GB limit.
    assert dummy_rag.created_query_engine == 0, (
        "rag.create_query_engine() must NOT be called from the ingest "
        "success path; it triggers reranker + embedding model loads that "
        "OOM-kill the backend on CPU Docker with default 8 GB limit."
    )
    assert dummy_rag.created_index == 0, (
        "rag.create_index() must NOT be called from the ingest success "
        "path either; the next chat query will build the index lazily."
    )


def test_ingest_upload_cancels_awaiter_on_client_disconnect(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Client disconnect mid-ingest cancels the awaiter and exits cleanly.

    The SSE ``/ingest/upload`` endpoint now polls
    ``request.is_disconnected()`` while waiting on the worker queue so
    that a disconnected client doesn't leave an orphan coroutine
    blocked on a queue no one will read. On disconnect, the awaiter is
    cancelled, a warning is logged, and the generator returns without
    emitting ``ingestion_complete``. The worker thread itself runs to
    completion (Python has no safe thread kill), but its output is
    safely discarded.

    Implementation detail: we monkeypatch
    ``INGEST_DISCONNECT_POLL_INTERVAL_S`` to a tiny value so the test
    finishes in well under a second, and override ``is_disconnected``
    on the Starlette ``Request`` class to simulate an immediate
    disconnect. A brief ``time.sleep`` in ``fake_ingest`` guarantees
    the poll path is reached before the worker enqueues the success
    sentinel.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture, used as the uploads dir.
    """
    import time

    from starlette.requests import Request as StarletteRequest

    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    monkeypatch.setattr(api_module, "INGEST_DISCONNECT_POLL_INTERVAL_S", 0.05)

    def blocking_fake_ingest(
        collection: str,
        path: Path,
        hybrid: bool = True,
        progress_callback: Any = None,
    ) -> None:
        """Block long enough for the disconnect poll to fire first.

        Args:
            collection (str): Collection name (ignored).
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
        """
        _ = (collection, path, hybrid, progress_callback)
        time.sleep(0.3)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", blocking_fake_ingest)

    async def always_disconnected(_self: StarletteRequest) -> bool:
        """Simulate an immediate client disconnect.

        Args:
            _self (StarletteRequest): The Request instance (ignored).

        Returns:
            bool: Always ``True``.
        """
        return True

    monkeypatch.setattr(StarletteRequest, "is_disconnected", always_disconnected)

    response = client.post(
        "/ingest/upload",
        data={"collection": "disconnect-guard", "hybrid": "true"},
        files={"files": ("hello.txt", b"hello world", "text/plain")},
    )

    # Connection opened successfully (SSE headers sent)...
    assert response.status_code == 200
    body = response.text
    # ... but the stream was cut short — no completion event, no error event.
    assert "event: ingestion_complete" not in body, (
        "ingestion_complete must NOT be emitted when the awaiter is cancelled; "
        "the worker thread still runs but its output is discarded."
    )
    assert "Ingestion failed" not in body, "Cancellation must not be surfaced as a generic ingestion failure."


def test_ingest_upload_poll_continues_when_still_connected(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Poll timeouts without disconnect must not prematurely exit the stream.

    The ``asyncio.wait_for(queue.get())`` timeout branch has two outcomes:
    ``is_disconnected() -> True`` (cancel & return) and
    ``is_disconnected() -> False`` (continue polling).
    ``test_ingest_upload_cancels_awaiter_on_client_disconnect`` exercises
    the first branch. This test exercises the "still connected → continue"
    branch: the fake worker blocks longer than a single poll interval so
    the timeout fires at least once, but ``is_disconnected`` always
    returns ``False`` — the stream must keep draining until the worker's
    completion sentinel arrives, and ``ingestion_complete`` must still be
    emitted.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture, used as the uploads dir.
    """
    import time

    from starlette.requests import Request as StarletteRequest

    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    monkeypatch.setattr(api_module, "INGEST_DISCONNECT_POLL_INTERVAL_S", 0.05)

    def slow_fake_ingest(
        collection: str,
        path: Path,
        hybrid: bool = True,
        progress_callback: Any = None,
    ) -> None:
        """Block long enough for the poll to fire at least once.

        Args:
            collection (str): Collection name (ignored).
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
        """
        _ = (collection, path, hybrid, progress_callback)
        time.sleep(0.15)  # ~3x poll interval

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", slow_fake_ingest)

    poll_count: dict[str, int] = {"n": 0}

    async def always_connected(_self: StarletteRequest) -> bool:
        """Simulate a stable connection — client stays the full duration.

        Args:
            _self (StarletteRequest): The Request instance (ignored).

        Returns:
            bool: Always ``False``.
        """
        poll_count["n"] += 1
        return False

    monkeypatch.setattr(StarletteRequest, "is_disconnected", always_connected)

    response = client.post(
        "/ingest/upload",
        data={"collection": "connected-guard", "hybrid": "true"},
        files={"files": ("hello.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 200
    body = response.text
    assert "event: ingestion_complete" in body, (
        "With is_disconnected always False, the poll must not cancel the "
        "awaiter — the worker's completion sentinel must reach the client."
    )
    assert "Ingestion failed" not in body
    assert poll_count["n"] >= 1, (
        "Poll interval (0.05 s) is shorter than fake_ingest's sleep (0.15 s), "
        "so is_disconnected must have been consulted at least once."
    )


def test_ingest_sync_empty_returns_empty_flag(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Empty ingestion via sync ``POST /ingest`` returns 200 with ``empty=true``.

    Matches the SSE ``/ingest/upload`` behaviour: ``EmptyIngestionError``
    is a soft-empty outcome (no content parsed), not a server error. The
    sync endpoint previously let the exception propagate to FastAPI's
    default handler, yielding an HTTP 500 which forced SDK/REST/CLI
    consumers to parse tracebacks to distinguish an empty upload from a
    real failure. The endpoint now catches the exception and returns
    ``{"ok": true, "empty": true, ...}``.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture, used as the data dir.
    """
    monkeypatch.setattr(api_module, "_resolve_data_dir", lambda: tmp_path)

    def fake_ingest(
        collection: str,
        path: Path,
        hybrid: bool = True,
        progress_callback: Any = None,
    ) -> None:
        """Simulate an ingestion run that produced no documents.

        Args:
            collection (str): Collection name.
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
        """
        _ = (path, hybrid, progress_callback)
        raise api_module.EmptyIngestionError(collection)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)

    response = client.post(
        "/ingest",
        json={"collection": "silence-sync-test", "hybrid": True},
    )

    # Empty ingestion must NOT surface as a 500; the SSE path already
    # handles this gracefully, and the sync path now mirrors it.
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["empty"] is True
    assert body["collection"] == "silence-sync-test"
    assert body["data_dir"] == str(tmp_path)

    # Collection was never created, so no select_collection should have fired.
    assert cast(Any, api_module.rag).selected == []


def test_ingest_sync_success_does_not_warm_query_engine(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Successful synchronous ``POST /ingest`` must not eagerly warm the query engine.

    Companion guard to :func:`test_ingest_upload_success_does_not_warm_query_engine`:
    the synchronous ``/ingest`` endpoint contained the same warmup pattern
    (``rag.select_collection`` + ``create_index`` + ``create_query_engine``
    + NER pre-warm) and had the same OOM-kill potential on CPU Docker
    containers. This test pins down the behavioral defect so that any
    future reintroduction of a post-ingest warmup on the sync route fails
    the suite.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture, used as the data dir.
    """
    # Point the endpoint at a real, empty temp directory so _resolve_data_dir
    # does not blow up the request before the warmup would have fired.
    monkeypatch.setattr(api_module, "_resolve_data_dir", lambda: tmp_path)

    def fake_ingest(
        collection: str,
        path: Path,
        hybrid: bool = True,
        progress_callback: Any = None,
    ) -> None:
        """Simulate a successful, no-op synchronous ingestion run.

        Returning cleanly is exactly the code path that previously fell
        through to the warmup block on the ``/ingest`` endpoint.

        Args:
            collection (str): Collection name (ignored).
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
        """
        _ = (collection, path, hybrid, progress_callback)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)

    dummy_rag = cast(DummyRAG, api_module.rag)
    assert dummy_rag.created_index == 0
    assert dummy_rag.created_query_engine == 0

    response = client.post(
        "/ingest",
        json={"collection": "warmup-guard-sync", "hybrid": True},
    )

    # The sync endpoint must still return a successful payload.
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["collection"] == "warmup-guard-sync"

    # Core assertion: neither the index nor the query engine may be built
    # eagerly during a successful sync ingest. Both counters must remain zero.
    assert dummy_rag.created_query_engine == 0, (
        "rag.create_query_engine() must NOT be called from the sync "
        "/ingest success path; it triggers reranker + embedding model "
        "loads that OOM-kill the backend on CPU Docker."
    )
    assert dummy_rag.created_index == 0, (
        "rag.create_index() must NOT be called from the sync /ingest "
        "success path either; the next chat query will build the index "
        "lazily."
    )
    # select_collection must also not run eagerly — DummyRAG.selected stays empty.
    assert cast(Any, api_module.rag).selected == []


def test_query_forwards_retrieval_query_to_validation_payload(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """The /query endpoint must forward retrieval_query to _validation_payload.

    The DummyRAG.chat stub returns ``retrieval_query`` in its response dict.
    This test monkeypatches ``_validation_payload`` in the api module and
    captures the kwargs it receives, then asserts the retrieval query is
    forwarded unchanged. ``retrieval_mode`` is the session-routing mode and
    is NOT forwarded as ``tool_used`` (those are semantically distinct).

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    captured: dict[str, Any] = {}

    def _stub_validation_payload(**kwargs: Any) -> dict[str, Any]:
        """Capture kwargs forwarded from the /query handler.

        Args:
            **kwargs: All keyword arguments forwarded to the real helper.

        Returns:
            dict[str, Any]: A minimal valid validation payload.
        """
        captured.update(kwargs)
        return {
            "validation_checked": None,
            "validation_mismatch": None,
            "validation_reason": None,
        }

    monkeypatch.setattr(api_module, "_validation_payload", _stub_validation_payload)

    response = client.post("/query", json={"question": "What?"})

    assert response.status_code == 200
    # DummyRAG.chat returns retrieval_query="rewritten::What?" and retrieval_mode="rewrite_compact_graph".
    assert captured.get("retrieval_query") == "rewritten::What?"
    # retrieval_mode is the session-routing mode and must not be passed as tool_used.
    assert captured.get("tool_used") is None
    assert captured.get("question") == "What?"


def test_stream_query_context_window_overflow_surfaces_descriptive_error(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Context-window overflow should surface a descriptive error instead of 'Internal server error'.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        client: The TestClient instance.
    """
    original_run_query = type(api_module.rag).run_query

    def _exploding_run_query(self: Any, *a: Any, **kw: Any) -> Any:
        raise ValueError(
            "The query and retrieved context exceed the configured "
            "context window (4096 tokens). Increase OPENAI_CTX_WINDOW "
            "to match your model's actual context length or reduce the "
            "retrieval top-k."
        )

    monkeypatch.setattr(type(api_module.rag), "run_query", _exploding_run_query)

    try:
        with client.stream(
            "POST",
            "/stream_query",
            json={"question": "hello", "retrieval_mode": "stateless"},
        ) as resp:
            text = "".join(chunk.decode() for chunk in resp.iter_raw())
    finally:
        monkeypatch.setattr(type(api_module.rag), "run_query", original_run_query)

    assert "OPENAI_CTX_WINDOW" in text
    assert "Internal server error" not in text


# ---------------------------------------------------------------------------
# Paginated reads: /collections/documents, /collections/hate-speech,
# /collections/ner/sources, /collections/ner/warm
# ---------------------------------------------------------------------------


def _select_alpha(client: TestClient) -> None:
    """Drive the API into a state with the canned 'alpha' collection active."""
    response = client.post("/collections/select", json={"name": "alpha"})
    assert response.status_code == 200, response.text


def test_documents_legacy_mode_returns_full_list(client: TestClient) -> None:
    """Without cursor or limit, /collections/documents returns the legacy envelope."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    rag.documents = [{"filename": f"doc{i}.pdf"} for i in range(3)]

    response = client.get("/collections/documents")
    assert response.status_code == 200
    payload = response.json()
    assert payload == {"documents": rag.documents}


def test_documents_paginated_mode_round_trips_cursor(client: TestClient) -> None:
    """Cursor + limit drives the paginated envelope and round-trips correctly."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    rag.documents = [{"filename": f"doc{i:03d}.pdf"} for i in range(25)]

    page1 = client.get("/collections/documents", params={"limit": 10}).json()
    assert len(page1["items"]) == 10
    assert page1["next_cursor"] is not None
    assert page1["items"][0]["filename"] == "doc000.pdf"

    page2 = client.get(
        "/collections/documents",
        params={"limit": 10, "cursor": page1["next_cursor"]},
    ).json()
    assert page2["items"][0]["filename"] == "doc010.pdf"
    assert page2["next_cursor"] is not None

    page3 = client.get(
        "/collections/documents",
        params={"limit": 10, "cursor": page2["next_cursor"]},
    ).json()
    assert len(page3["items"]) == 5
    assert page3["next_cursor"] is None


def test_documents_invalid_cursor_returns_400(client: TestClient) -> None:
    """Malformed cursor tokens must surface as HTTP 400, not 500."""
    _select_alpha(client)
    response = client.get(
        "/collections/documents",
        params={"cursor": "not-a-valid-token", "limit": 10},
    )
    assert response.status_code == 400


def test_documents_no_collection_selected_returns_400(client: TestClient) -> None:
    """The paginated endpoint must require an active collection like the legacy one."""
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/documents", params={"limit": 10})
    assert response.status_code == 400


def test_hate_speech_legacy_mode_returns_results_envelope(client: TestClient) -> None:
    """Without cursor/limit/filter args the response keeps the legacy ``results`` shape."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    rag.hate_speech_rows = [{"chunk_id": "c1", "category": "X"}]

    response = client.get("/collections/hate-speech")
    assert response.status_code == 200
    assert response.json() == {"results": [{"chunk_id": "c1", "category": "X"}]}


def test_hate_speech_paginated_mode(client: TestClient) -> None:
    """Passing ``limit`` switches the response to the paginated envelope."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    rag.hate_speech_rows = [{"chunk_id": f"c{i}"} for i in range(7)]

    payload = client.get("/collections/hate-speech", params={"limit": 3}).json()
    assert payload["items"] == rag.hate_speech_rows[:3]
    assert payload["next_cursor"] is not None


def test_ner_sources_paginates_and_forwards_filters(client: TestClient) -> None:
    """Paginated NER sources slice the cached list and forward filter args."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    rag.ner_sources = [{"chunk_id": f"c{i}", "entities": []} for i in range(8)]

    response = client.get(
        "/collections/ner/sources",
        params={"limit": 5, "entity_key": "Acme::ORG"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["items"] == rag.ner_sources[:5]
    assert payload["next_cursor"] is not None

    forwarded = rag.last_ner_sources_filter
    assert forwarded["entity_key"] == "Acme::ORG"


def test_ner_sources_forwards_resolved_merge_mode(client: TestClient) -> None:
    """The paginated NER-sources endpoint forwards entity_merge_mode=resolved."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    response = client.get(
        "/collections/ner/sources",
        params={"entity_key": "US::loc", "entity_merge_mode": "resolved"},
    )
    assert response.status_code == 200
    assert rag.last_ner_sources_filter["entity_merge_mode"] == "resolved"


def test_export_ner_sources_csv_forwards_resolved_merge_mode(client: TestClient) -> None:
    """The NER-sources CSV export forwards entity_merge_mode=resolved to the iterator."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    response = client.get(
        "/collections/alpha/export/ner-sources.csv",
        params={"entity_text": "US", "entity_type": "loc", "entity_merge_mode": "resolved"},
    )
    assert response.status_code == 200
    assert rag.last_ner_sources_filter["entity_merge_mode"] == "resolved"


def test_ner_sources_invalid_cursor_returns_400(client: TestClient) -> None:
    """Malformed cursors on the paginated NER endpoint must return HTTP 400."""
    _select_alpha(client)
    response = client.get("/collections/ner/sources", params={"cursor": "$$$"})
    assert response.status_code == 400


def test_ner_warm_kicks_aggregate(client: TestClient) -> None:
    """POST /collections/ner/warm triggers exactly one aggregate-build."""
    _select_alpha(client)
    response = client.post("/collections/ner/warm")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    rag = cast(DummyRAG, api_module.rag)
    assert getattr(rag, "warm_calls", 0) == 1


def test_ner_warm_requires_collection(client: TestClient) -> None:
    """Warming with no active collection must surface HTTP 400."""
    api_module.rag.qdrant_collection = ""
    response = client.post("/collections/ner/warm")
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Streaming CSV exports: /collections/{name}/export/*.csv
# ---------------------------------------------------------------------------


def _parse_csv_body(body: bytes) -> list[list[str]]:
    """Decode a streamed CSV body (BOM-tolerant) into rows."""
    import csv
    import io

    text = body.decode("utf-8-sig")
    return list(csv.reader(io.StringIO(text)))


def test_export_documents_csv_streams(client: TestClient) -> None:
    """Document export streams a UTF-8 BOM-prefixed CSV with RFC 6266 headers."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    rag.documents = [
        {
            "filename": "doc1.pdf",
            "mimetype": "application/pdf",
            "file_hash": "abc",
            "node_count": 3,
            "page_count": 2,
            "entity_types": ["PERSON", "ORG"],
        }
    ]

    response = client.get("/collections/alpha/export/documents.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    disp = response.headers["content-disposition"]
    assert "alpha-documents.csv" in disp
    assert "filename*=UTF-8''alpha-documents.csv" in disp

    rows = _parse_csv_body(response.content)
    assert rows[0] == [
        "filename",
        "mimetype",
        "file_hash",
        "node_count",
        "page_count",
        "max_rows",
        "max_duration",
        "entity_types",
    ]
    assert rows[1][0] == "doc1.pdf"
    assert rows[1][-1] == "PERSON;ORG"


def test_export_rejects_collection_mismatch(client: TestClient) -> None:
    """The URL path's collection name must match the active collection (409 otherwise)."""
    _select_alpha(client)
    response = client.get("/collections/beta/export/documents.csv")
    assert response.status_code == 409


def test_export_entities_csv_uses_ner_stats(client: TestClient) -> None:
    """Entity export streams the rank/entity/type/mentions schema from get_collection_ner_stats."""
    _select_alpha(client)
    response = client.get("/collections/alpha/export/entities.csv")
    assert response.status_code == 200
    rows = _parse_csv_body(response.content)
    assert rows[0] == ["rank", "entity", "type", "mentions"]
    assert rows[1] == ["1", "Acme", "ORG", "3"]


def test_export_entities_csv_supports_resolved_merge_mode(client: TestClient) -> None:
    """Entity export accepts entity_merge_mode=resolved and forwards it to stats."""
    _select_alpha(client)
    response = client.get(
        "/collections/alpha/export/entities.csv",
        params={"entity_merge_mode": "resolved"},
    )
    assert response.status_code == 200
    assert cast(DummyRAG, api_module.rag).ner_stats_merge_modes[-1] == "resolved"


def test_export_ner_sources_csv_filters_by_entity(client: TestClient) -> None:
    """NER-source export honors entity_text + entity_type filters and embeds the entity label."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    rag.ner_sources = [
        {
            "chunk_id": "c1",
            "filename": "doc1.pdf",
            "chunk_text": "Acme makes widgets",
            "entities": [{"text": "Acme", "type": "ORG"}],
        }
    ]
    response = client.get(
        "/collections/alpha/export/ner-sources.csv",
        params={"entity_text": "Acme", "entity_type": "ORG"},
    )
    assert response.status_code == 200
    rows = _parse_csv_body(response.content)
    assert rows[0][0] == "entity"
    assert rows[1][0] == "Acme [ORG]"
    assert rows[1][1] == "doc1.pdf"


def test_export_hate_speech_csv_passes_through_rows(client: TestClient) -> None:
    """Hate-speech export streams the per-finding schema with reference_metadata fields."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    rag.hate_speech_rows = [
        {
            "chunk_id": "c1",
            "filename": "doc.pdf",
            "category": "Hateful",
            "confidence": "high",
            "reason": "tagged",
            "chunk_text": "bad text",
            "page": 2,
        }
    ]
    response = client.get("/collections/alpha/export/hate-speech.csv")
    assert response.status_code == 200
    rows = _parse_csv_body(response.content)
    assert rows[0][0] == "source"
    assert rows[1][0] == "doc.pdf"
    assert rows[1][4] == "Hateful"


def test_export_requires_active_collection(client: TestClient) -> None:
    """Export endpoints reject calls when no collection is active (HTTP 400)."""
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/alpha/export/documents.csv")
    assert response.status_code == 400


def test_select_does_not_warm_documents_or_hate_speech_caches(client: TestClient) -> None:
    """Select must remain light — paginated caches populate lazily.

    Companion to test_collections_select_success's NER guard. Documents and
    hate-speech caches must not be eagerly populated by /collections/select;
    they populate on first GET to the paginated endpoint.
    """
    rag = cast(DummyRAG, api_module.rag)
    rag.documents = [{"filename": "x.pdf"}]
    rag.hate_speech_rows = [{"chunk_id": "c1"}]

    _select_alpha(client)

    assert getattr(rag, "warm_calls", 0) == 0
    assert rag.ner_refresh_calls == []


# ---------------------------------------------------------------------------
# /collections/documents/count
# ---------------------------------------------------------------------------


def test_documents_count_returns_size_of_document_list(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The count endpoint returns ``len(list_documents())`` for the active collection."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    rag.documents = [{"filename": f"d{i}.pdf"} for i in range(7)]

    # The endpoint calls rag.get_document_count(); the dummy doesn't have one,
    # so wire a thin lambda for this test.
    monkeypatch.setattr(rag, "get_document_count", lambda: len(rag.documents), raising=False)

    response = client.get("/collections/documents/count")
    assert response.status_code == 200
    assert response.json() == {"count": 7}


def test_documents_count_requires_active_collection(client: TestClient) -> None:
    """Count endpoint must reject calls with no active collection."""
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/documents/count")
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# /sessions/{session_id}/sources.zip
# ---------------------------------------------------------------------------


def test_session_sources_zip_streams_unique_files(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Session-ZIP bundles every unique cited file exactly once and streams as application/zip."""
    import zipfile as _zipfile

    src_a = tmp_path / "a.pdf"
    src_b = tmp_path / "b.pdf"
    src_a.write_bytes(b"hello from a")
    src_b.write_bytes(b"hello from b")

    def fake_history(self: Any, sid: str, owner: str) -> list[dict[str, Any]]:
        assert sid == "sess-1"
        assert owner == "tester"
        return [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "a",
                "sources": [
                    {"file_hash": "h-a", "filename": "a.pdf", "collection": "alpha"},
                    {"file_hash": "h-b", "filename": "b.pdf", "collection": "alpha"},
                    # Duplicate hash must not produce a second entry.
                    {"file_hash": "h-a", "filename": "a.pdf", "collection": "alpha"},
                ],
            },
        ]

    def fake_resolve(
        collection: str,
        file_hash: str,
        *,
        filename_hint: str | None = None,
    ) -> Path | None:
        return {"h-a": src_a, "h-b": src_b}.get(file_hash)

    monkeypatch.setattr(DummySessionManager, "get_session_history", fake_history, raising=False)
    monkeypatch.setattr(api_module, "_resolve_source_file_path", fake_resolve)

    response = client.get("/sessions/sess-1/sources.zip", headers={"X-Auth-User": "tester"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    assert "session-sess-1-sources.zip" in response.headers["content-disposition"]

    archive = _zipfile.ZipFile(io.BytesIO(response.content))
    assert sorted(archive.namelist()) == ["a.pdf", "b.pdf"]
    assert archive.read("a.pdf") == b"hello from a"
    assert archive.read("b.pdf") == b"hello from b"


def test_session_sources_zip_returns_404_when_no_files(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty or unresolvable sessions surface as HTTP 404, not an empty ZIP."""
    monkeypatch.setattr(
        DummySessionManager,
        "get_session_history",
        lambda self, sid, owner: [{"role": "user", "content": "q"}],
        raising=False,
    )
    response = client.get("/sessions/empty/sources.zip", headers={"X-Auth-User": "tester"})
    assert response.status_code == 404


def test_session_sources_zip_skips_unresolved_files(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Files the backend can't resolve are dropped silently, not failed loudly."""
    import zipfile as _zipfile

    src = tmp_path / "kept.pdf"
    src.write_bytes(b"kept")

    monkeypatch.setattr(
        DummySessionManager,
        "get_session_history",
        lambda self, sid, owner: [
            {
                "role": "assistant",
                "content": "a",
                "sources": [
                    {"file_hash": "kept", "filename": "kept.pdf", "collection": "alpha"},
                    {"file_hash": "missing", "filename": "missing.pdf", "collection": "alpha"},
                ],
            }
        ],
        raising=False,
    )
    monkeypatch.setattr(
        api_module,
        "_resolve_source_file_path",
        lambda collection, file_hash, **_: src if file_hash == "kept" else None,
    )

    response = client.get("/sessions/partial/sources.zip", headers={"X-Auth-User": "tester"})
    assert response.status_code == 200
    archive = _zipfile.ZipFile(io.BytesIO(response.content))
    assert archive.namelist() == ["kept.pdf"]


def test_sessions_endpoints_pass_principal_and_404_on_cross_owner(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Session endpoints forward the resolved principal and 404 cross-owner.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    seen: dict[str, Any] = {}

    class OwnerAwareSessions:
        """Session manager stub recording the owner it was called with."""

        def list_sessions(self, owner: str) -> list[dict[str, Any]]:
            seen["list"] = owner
            return [{"id": "s1", "created_at": "2026-01-01", "title": "t"}]

        def get_session_history(self, session_id: str, owner: str) -> list[dict[str, Any]]:
            seen["history"] = (session_id, owner)
            # Simulate a cross-owner / missing session: empty history.
            return [] if owner == "bob" else [{"role": "user", "content": "hi"}]

        def delete_session(self, session_id: str, owner: str) -> bool:
            seen["delete"] = (session_id, owner)
            return owner == "alice"

    monkeypatch.setattr(api_module.rag, "ensure_session_manager", lambda: OwnerAwareSessions())

    # List forwards the header principal.
    resp = client.get("/sessions/list", headers={"X-Auth-User": "alice"})
    assert resp.status_code == 200
    assert seen["list"] == "alice"
    assert resp.json()["sessions"][0]["id"] == "s1"

    # History for the owner succeeds.
    resp = client.get("/sessions/s1/history", headers={"X-Auth-User": "alice"})
    assert resp.status_code == 200
    assert seen["history"] == ("s1", "alice")
    assert resp.json()["messages"][0]["content"] == "hi"

    # Cross-owner history is 404 (empty -> not found, no existence leak).
    resp = client.get("/sessions/s1/history", headers={"X-Auth-User": "bob"})
    assert resp.status_code == 404

    # Cross-owner delete is 404.
    resp = client.delete("/sessions/s1", headers={"X-Auth-User": "bob"})
    assert resp.status_code == 404
    assert seen["delete"] == ("s1", "bob")

    # Owner delete succeeds.
    resp = client.delete("/sessions/s1", headers={"X-Auth-User": "alice"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_sessions_list_401_without_header_or_default(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """With no trusted header and no configured default, endpoints 401.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)

    resp = client.get("/sessions/list")
    assert resp.status_code == 401


def test_sessions_list_uses_default_identity_when_no_header(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """A configured default identity is used as the owner when no header.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    seen: dict[str, Any] = {}

    class OwnerAwareSessions:
        """Session manager stub recording the owner it was called with."""

        def list_sessions(self, owner: str) -> list[dict[str, Any]]:
            seen["list"] = owner
            return []

    monkeypatch.setattr(api_module.rag, "ensure_session_manager", lambda: OwnerAwareSessions())

    resp = client.get("/sessions/list")
    assert resp.status_code == 200
    assert seen["list"] == "operator"


@pytest.mark.anyio
async def test_stream_query_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow streaming query must not stall concurrent requests.

    Regression test for the PR-195 freeze: the chat stream used to iterate a
    blocking sync generator directly on the event loop, so a single in-flight
    ``/stream_query`` starved every other request (nginx 504s, frozen UI). With
    the generator pumped on a worker thread, a cheap ``/collections/list`` must
    still return promptly while the slow stream is mid-flight.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    import time

    import httpx
    from httpx import ASGITransport

    rag = cast(DummyRAG, api_module.rag)

    def slow_stream(question: str, **_kwargs: Any) -> Generator[str | dict[str, Any], None, None]:
        """Block before the first yield, modelling retrieval + first-token latency.

        The heavy synchronous work (query rewrite, embedding, Qdrant search,
        rerank, first LLM token) all happens inside the first ``next()`` before
        any chunk is produced — so a single ``time.sleep`` before the first
        ``yield`` is the faithful reproduction of the freeze.
        """
        time.sleep(1.0)
        yield "tok "
        yield {"response": "answer", "sources": [], "session_id": "generated-session"}

    monkeypatch.setattr(rag, "stream_chat", slow_stream)

    # Measure each request's completion time relative to a shared start. The
    # blocking sync sleep cannot be "raced" on a single event loop — what
    # distinguishes the bug from the fix is *when the loop is free*. With the
    # bug, the stream blocks the loop for its full duration, so the cheap GET
    # cannot complete until the stream releases it (both finish ~together).
    # With the fix, the stream parks on a worker thread immediately, so the GET
    # completes near-instantly while the stream is still sleeping.
    transport = ASGITransport(app=api_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        start = time.perf_counter()
        timings: dict[str, float] = {}

        async def timed_stream() -> int:
            """Drive the slow streaming query to completion, recording its finish."""
            resp = await ac.post("/stream_query", json={"question": "hi", "session_id": "s"})
            timings["stream"] = time.perf_counter() - start
            return resp.status_code

        async def timed_cheap() -> int:
            """Hit a cheap endpoint concurrently, recording its finish."""
            resp = await ac.get("/collections/list")
            timings["cheap"] = time.perf_counter() - start
            return resp.status_code

        stream_status, cheap_status = await asyncio.gather(timed_stream(), timed_cheap())

        assert stream_status == 200
        assert cheap_status == 200
        # The cheap call must clear well before the ~1s stream finishes; if the
        # loop were blocked it could only complete once the stream released it.
        assert timings["cheap"] < timings["stream"] - 0.5, (
            f"concurrent request was blocked behind the stream: "
            f"cheap={timings['cheap']:.2f}s stream={timings['stream']:.2f}s"
        )


def test_stream_query_disconnect_cancels_awaiter(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """A client disconnect mid-stream must stop draining the chat generator.

    Mirrors ``test_ingest_upload_cancels_awaiter_on_client_disconnect``: the
    poll interval is shrunk and ``is_disconnected`` forced ``True`` so the
    disconnect fires before the slow generator yields its first token — no
    token events should reach the client.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    import time

    from starlette.requests import Request as StarletteRequest

    monkeypatch.setattr(api_module, "STREAM_DISCONNECT_POLL_INTERVAL_S", 0.05)

    rag = cast(DummyRAG, api_module.rag)

    def slow_stream(question: str, **_kwargs: Any) -> Generator[str | dict[str, Any], None, None]:
        """Block before the first yield so the disconnect poll wins the race."""
        time.sleep(0.3)
        yield "tok "
        yield {"response": "answer", "sources": [], "session_id": "generated-session"}

    monkeypatch.setattr(rag, "stream_chat", slow_stream)

    async def always_disconnected(_self: StarletteRequest) -> bool:
        """Simulate an immediate client disconnect."""
        return True

    monkeypatch.setattr(StarletteRequest, "is_disconnected", always_disconnected)

    with client.stream("POST", "/stream_query", json={"question": "hi", "session_id": "s"}) as resp:
        assert resp.status_code == 200
        body = "".join(chunk.decode() for chunk in resp.iter_raw())

    # The awaiter was cancelled before the generator produced any token.
    assert '"token"' not in body


def test_stream_query_surfaces_generator_error(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """An exception raised inside the chat generator becomes an SSE error event.

    The thread-bridge re-raises the worker exception on the loop, where the
    endpoint's existing ``except`` clause converts it to an ``error`` payload.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    rag = cast(DummyRAG, api_module.rag)

    def boom_stream(question: str, **_kwargs: Any) -> Generator[str | dict[str, Any], None, None]:
        """Yield one token, then raise to exercise error propagation."""
        yield "tok "
        raise RuntimeError("kaboom")

    monkeypatch.setattr(rag, "stream_chat", boom_stream)

    with client.stream("POST", "/stream_query", json={"question": "hi", "session_id": "s"}) as resp:
        assert resp.status_code == 200
        body = "".join(chunk.decode() for chunk in resp.iter_raw())

    assert '"token": "tok ' in body
    assert '"error"' in body
