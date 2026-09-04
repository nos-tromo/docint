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


def test_a_transcript_segment_names_the_clip_not_the_parsed_transcript() -> None:
    """The ``.nextext.jsonl`` a segment was parsed from is deleted during ingest.

    Naming it in a citation sends an analyst looking for a file that never
    outlived the run. Segments ingested before the clip's own name was stamped
    carry it, so the name is recovered by stripping the suffix.
    """
    payload = {
        "file_name": "clip.mp4.nextext.jsonl",
        "filename": "clip.mp4.nextext.jsonl",
        "origin": {"filename": "clip.mp4.nextext.jsonl"},
        "docint_doc_kind": "transcript_segment",
    }

    src = RAG._source_from_payload(collection="uabc__docs", payload=payload)

    assert src["filename"] == "clip.mp4"


def test_a_stamped_transcript_segment_names_its_clip_directly() -> None:
    """Once the linker stamps the clip, nothing has to be recovered."""
    payload = {"source_file": "clip.mp4", "file_name": "clip.mp4.nextext.jsonl"}

    src = RAG._source_from_payload(collection="uabc__docs", payload=payload)

    assert src["filename"] == "clip.mp4"


def _retrieve_stub(companion_payload: dict[str, Any]) -> Any:
    """Return a ``qdrant_client`` stand-in that holds one point, in ``_images`` only.

    Args:
        companion_payload (dict[str, Any]): Payload of the companion point.

    Returns:
        Any: Client stand-in whose ``retrieve`` answers by collection name.
    """
    point = _Node("pt-img-1", "", companion_payload)
    point.payload = companion_payload  # type: ignore[attr-defined]

    def retrieve(collection_name: str, ids: list[str]) -> list[Any]:
        """Answer only for the companion collection."""
        return [point] if collection_name.endswith("_images") else []

    client = type("Client", (), {})()
    client.retrieve = retrieve  # type: ignore[attr-defined]
    return client


def test_a_cited_image_is_rehydrated_from_the_companion() -> None:
    """A session revisit resolves an image citation the main collection never held.

    The cited node id is an ``_images`` point; looking it up in the main
    collection alone rehydrated every picture as an empty source, so the
    citation card had nothing to expand.
    """
    rag = RAG(qdrant_collection="uabc__docs")
    rag._qdrant_client = _retrieve_stub(
        {
            "image_id": "img-hash",
            "source_type": "social_media",
            "source_doc_id": "posting-1",
            "posting_uuid": "posting-1",
            "file_name": "pic.png",
            "llm_description": "A harbour at dusk.",
            "reference_metadata": {"type": "image", "posting_author": "someone"},
        }
    )

    src = rag.get_source_by_node_id("pt-img-1", score=0.4)

    assert src is not None
    assert src["id"] == "pt-img-1"
    assert "harbour" in src["text"]
    assert src["reference_metadata"]["posting_author"] == "someone"
    assert src["file_hash"] == "img-hash"
    assert src["filename"] == "pic.png"


def test_a_rehydrated_chunk_keeps_the_node_id_it_was_cited_by() -> None:
    """A Qdrant payload carries no ``node_id``; the citation's own id fills it."""
    rag = RAG(qdrant_collection="uabc__docs")
    client = _retrieve_stub({})
    chunk = _Node("pt-1", "", {"file_name": "a.pdf", "file_hash": "h1", "text": "body"})
    chunk.payload = chunk.metadata  # type: ignore[attr-defined]
    client.retrieve = lambda collection_name, ids: [chunk] if collection_name == "uabc__docs" else []  # type: ignore[attr-defined]
    rag._qdrant_client = client

    src = rag.get_source_by_node_id("pt-1")

    assert src is not None
    assert src["id"] == "pt-1"
