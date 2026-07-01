"""Unit tests for RAG._fetch_posting_entity_nodes (link-following sibling fetch)."""

from types import SimpleNamespace
from typing import Any

import pytest

from docint.core.rag import RAG


class _FakeQdrant:
    def __init__(self, by_collection: dict[str, list[dict[str, Any]]]) -> None:
        self.by_collection = by_collection

    def collection_exists(self, collection_name: str) -> bool:
        """Return True when the fake collection has data."""
        return collection_name in self.by_collection

    def scroll(
        self,
        collection_name: str,
        scroll_filter: Any = None,
        limit: int = 64,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[SimpleNamespace], None]:
        """Return fake points for the requested collection."""
        points = [
            SimpleNamespace(id=p["id"], payload=p["payload"]) for p in self.by_collection.get(collection_name, [])
        ]
        return points, None


def test_fetch_posting_entity_collects_siblings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sibling text and image nodes are returned for a given posting_uuid."""
    rag = RAG.__new__(RAG)  # bypass heavy __init__
    rag.qdrant_collection = "c"
    rag._qdrant_client = _FakeQdrant(  # type: ignore[assignment]  # QdrantClient duck-type
        {
            "c": [{"id": "t1", "payload": {"posting_uuid": "u1", "text": "spoken words"}}],
            "c_images": [{"id": "i1", "payload": {"posting_uuid": "u1", "llm_description": "a red banner"}}],
        }
    )
    # RAG uses slots=True, so patch the class method (monkeypatch restores it after test)
    monkeypatch.setattr(RAG, "_image_collection_name", lambda self: "c_images")

    nodes = rag._fetch_posting_entity_nodes("u1", exclude_node_ids={"already"})
    texts = {n.node.text for n in nodes}  # type: ignore[attr-defined]  # node is TextNode at runtime
    assert "spoken words" in texts
    assert any("red banner" in t for t in texts)
