"""Tests that chat sources carry a traceable chunk identity.

A citation the reader cannot pin to a chunk is not verifiable evidence: the
transcript export renders ``Chunk-ID: n/a`` for every source when the
normalized payload has no id, so two chunks of the same file on the same page
are indistinguishable.
"""

from typing import Any, cast

from docint.core.rag import RAG


class _Node:
    """Minimal node stand-in exposing the attributes the normalizer reads.

    Attributes:
        node_id: The node's stable id.
        text: The chunk body.
        metadata: The Qdrant payload carried by the node.
    """

    def __init__(self, node_id: str, text: str, metadata: dict[str, Any]) -> None:
        """Initialize the node stand-in.

        Args:
            node_id (str): The node's stable id.
            text (str): The chunk body.
            metadata (dict[str, Any]): The node payload.
        """
        self.node_id = node_id
        self.text = text
        self.metadata = metadata


class _NodeWithScore:
    """Minimal ``NodeWithScore`` stand-in.

    Attributes:
        node: The wrapped node.
        score: The retrieval score.
    """

    def __init__(self, node: _Node, score: float) -> None:
        """Initialize the scored-node stand-in.

        Args:
            node (_Node): The wrapped node.
            score (float): The retrieval score.
        """
        self.node = node
        self.score = score


def test_retrieved_source_carries_the_node_id_as_its_identity() -> None:
    """A source normalized from a retrieved node exposes ``id``."""
    node = _Node(
        "8f2c9f33-5870-5edc-83b7-2cc2f53ffbdb",
        "Station 3: the sealed cave gate.",
        {"filename": "handbook.pdf", "page": 26},
    )

    source = RAG._source_from_node_with_score(cast(Any, RAG), cast(Any, _NodeWithScore(node, 0.42)))

    assert source is not None
    assert source["id"] == "8f2c9f33-5870-5edc-83b7-2cc2f53ffbdb"


def test_payload_chunk_id_is_surfaced_alongside_the_node_id() -> None:
    """A payload-level ``chunk_id`` survives normalization."""
    payload: dict[str, Any] = {
        "filename": "handbook.pdf",
        "page": 26,
        "chunk_id": "handbook-p26-b3-0",
    }

    src = RAG._source_from_payload(collection="c", payload=payload, node_id="node-1")

    assert src["chunk_id"] == "handbook-p26-b3-0"
    assert src["id"] == "node-1"


def test_payload_node_id_is_used_when_no_node_id_is_passed() -> None:
    """Restored citations carry their id via the payload alone."""
    payload: dict[str, Any] = {"filename": "handbook.pdf", "node_id": "node-42"}

    src = RAG._source_from_payload(collection="c", payload=payload)

    assert src["id"] == "node-42"


def test_source_without_any_identity_omits_the_id_key() -> None:
    """Nothing is invented when the payload carries no identity."""
    src = RAG._source_from_payload(collection="c", payload={"filename": "handbook.pdf"})

    assert "id" not in src
    assert "chunk_id" not in src
