"""Unit tests for the Neo4j-backed graph subsystem."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from llama_index.core.schema import TextNode
from llama_index.core.indices.vector_store import VectorStoreIndex

from docint.core.graph.entity_resolution import resolve_entity_identity
from docint.core.graph.models import (
    GraphCandidate,
    GraphSourceRecord,
    GraphTraversalResult,
)
from docint.core.graph.service import Neo4jGraphService
from docint.core.rag import RAG
from docint.utils.env_cfg import load_graph_store_env


def test_resolve_entity_identity_defers_low_confidence_merge() -> None:
    """Low-confidence entities should stay collection-scoped with a candidate global key."""

    resolved = resolve_entity_identity(
        text="Ac",
        entity_type="ORG",
        collection="alpha",
        score=0.3,
        min_confidence=0.85,
    )

    assert resolved.auto_merged is False
    assert resolved.scope == "alpha"
    assert resolved.candidate_key == "org:global:ac"


def test_graph_service_ingest_source_records_writes_provenance_and_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graph ingestion should emit provenance and entity upsert Cypher writes.
    
    Args:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching.
    """

    service = Neo4jGraphService(load_graph_store_env())
    captured: list[tuple[str, dict[str, object]]] = []

    def _capture(query: str, parameters: dict[str, object]) -> list[dict[str, object]]:
        """A capture function to intercept Cypher queries and parameters during testing.

        Args:
            query (str): The Cypher query string being executed.
            parameters (dict[str, object]): The parameters being passed to the Cypher query.

        Returns:
            list[dict[str, object]]: An empty list simulating no results from the query, while capturing 
                the query and parameters for assertions.
        """        
        captured.append((query, parameters))
        return []

    monkeypatch.setattr(
        Neo4jGraphService,
        "_run_write",
        lambda self, query, parameters: _capture(query, parameters),
    )

    service.ingest_source_records(
        collection="alpha",
        records=[
            GraphSourceRecord(
                node_id="n1",
                collection="alpha",
                source_kind="table",
                record_kind="comment",
                text="Alice mentioned Acme",
                filename="comments.csv",
                file_hash="fh1",
                text_id="c1",
                author="Alice",
                platform="Telegram",
                timestamp="2026-01-01T10:00:00Z",
                url="https://example.com/post/1",
                tags=["tag"],
                entities=[{"text": "Acme", "type": "ORG", "score": 0.99}],
                search_blob="Alice mentioned Acme",
            )
        ],
    )

    assert any("MERGE (p:Provenance" in query for query, _ in captured)
    assert any("MERGE (a:EntityAlias" in query for query, _ in captured)
    assert any("MERGE (s)-[:AUTHORED_BY]->(a)" in query for query, _ in captured)


def test_run_graph_query_returns_graph_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Graph-backed RAG queries should synthesize an answer from graph-selected sources.
    
    Args:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching.
    """

    rag = RAG(qdrant_collection="alpha")
    rag.index = cast(VectorStoreIndex, object())
    rag._graph_service = cast(
        Neo4jGraphService,
        SimpleNamespace(
            retrieve_candidates=lambda **_: GraphTraversalResult(
                candidates=[
                    GraphCandidate(
                        "n1", exact_score=1.0, graph_score=0.7, matched_on="entity"
                    )
                ],
                trace={"candidate_count": 1},
            )
        ),
    )
    rag._post_retrieval_text_model = cast(
        Any,
        SimpleNamespace(
            complete=lambda prompt: SimpleNamespace(text=f"Grounded::{prompt[:18]}")
        ),
    )
    rag._reranker = None

    monkeypatch.setattr(RAG, "_collection_supports_graph", lambda self: True)
    monkeypatch.setattr(
        RAG,
        "_infer_collection_profile",
        lambda self: {"is_social_table": True, "coverage_unit": "posts"},
    )
    monkeypatch.setattr(
        RAG,
        "_get_node_by_id",
        lambda self, node_id: TextNode(
            text="Alice mentioned Acme",
            id_=node_id,
            metadata={
                "source": "table",
                "file_hash": "fh1",
                "filename": "comments.csv",
                "table": {"row_index": 2},
                "reference_metadata": {
                    "author": "Alice",
                    "network": "Telegram",
                    "text_id": "c1",
                    "timestamp": "2026-01-01T10:00:00Z",
                },
            },
        ),
    )

    class _EmptyRetriever:
        def retrieve(self, query: str) -> list[object]:
            """A dummy retriever that returns an empty list, used to bypass actual retrieval logic during testing.

            Args:
                query (str): The input query string for which retrieval would normally be performed.

            Returns:
                list[object]: An empty list simulating no retrieval results, allowing the test to focus on the graph 
                    query logic and response synthesis.
            """            
            _ = query
            return []

    monkeypatch.setattr(RAG, "_build_retriever", lambda self, **_: _EmptyRetriever())

    result = rag.run_graph_query(
        "What did Alice say about Acme?", query_mode="graph_lookup"
    )

    assert result["retrieval_mode"] == "graph_lookup"
    assert result["vector_query_mode"] == "graph_hybrid"
    assert result["retrieval_trace"]["candidate_count"] == 1
    assert result["sources"][0]["reference_metadata"]["text_id"] == "c1"
    assert result["response"].startswith("Grounded::")
