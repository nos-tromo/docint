"""Tests that a document's images count as evidence in the collection summary.

The summary's per-document retrieval only ever queried the main collection, so
a PDF's embedded figures and a clip's keyframes -- which live in the `_images`
companion -- could never be quoted, no matter how much of the document's
substance they carried. A multimodal collection was summarized as if it were
text-only.
"""

import types
from typing import Any, cast

from docint.core.rag import RAG

KEYFRAME_PAYLOAD: dict[str, Any] = {
    "image_id": "img-key-1",
    "source_type": "video_keyframe",
    "source_doc_id": "hash-clip",
    "source_path": "/ingest/batch/briefing.mp4",
    "llm_description": "A slide reading 'Q3 field results'.",
}


class _StubQdrant:
    """Qdrant stub serving one images-companion scroll."""

    def __init__(self, points: list[Any]) -> None:
        """Initialize the stub.

        Args:
            points (list[Any]): Points the images scroll returns.
        """
        self.points = points
        self.scrolled: list[str] = []

    def collection_exists(self, collection_name: str) -> bool:
        """Report every collection as present.

        Args:
            collection_name (str): The collection being probed.

        Returns:
            bool: Always ``True``.
        """
        return True

    def scroll(self, *, collection_name: str, **kwargs: Any) -> tuple[list[Any], Any]:
        """Return the canned points for the images companion.

        Args:
            collection_name (str): The collection being scrolled.
            **kwargs (Any): Scroll arguments (ignored).

        Returns:
            tuple[list[Any], Any]: The points and a null continuation offset.
        """
        self.scrolled.append(collection_name)
        return (self.points, None)


def _rag_with_images(points: list[Any]) -> tuple[RAG, _StubQdrant]:
    """Build a RAG whose images companion holds ``points``.

    Args:
        points (list[Any]): Points the companion scroll returns.

    Returns:
        tuple[RAG, _StubQdrant]: The RAG and its Qdrant stub.
    """
    rag = RAG(qdrant_collection="testbatch")
    client = _StubQdrant(points)
    rag._qdrant_client = cast(Any, client)
    rag._image_ingestion_service = cast(
        Any,
        types.SimpleNamespace(
            _resolve_collection_name=lambda source_collection=None: f"{source_collection}_images",
            img_ingestion_config=types.SimpleNamespace(rerank_min_score=0.05, retrieve_top_k=5),
        ),
    )
    return rag, client


def test_summary_evidence_includes_the_documents_images() -> None:
    """A document's stored images are available as summary evidence."""
    point = types.SimpleNamespace(id="point-77", payload=KEYFRAME_PAYLOAD)
    rag, _ = _rag_with_images([point])

    nodes = rag._summary_image_nodes_for_document(file_hash="hash-clip", top_k=3)

    assert len(nodes) == 1
    assert "Q3 field results" in nodes[0].node.get_content()


def test_summary_image_evidence_is_scoped_to_the_images_companion() -> None:
    """The scroll targets the companion, never the main collection."""
    point = types.SimpleNamespace(id="point-77", payload=KEYFRAME_PAYLOAD)
    rag, client = _rag_with_images([point])

    rag._summary_image_nodes_for_document(file_hash="hash-clip", top_k=3)

    assert client.scrolled == ["testbatch_images"]


def test_summary_image_evidence_is_capped() -> None:
    """A figure-heavy document cannot crowd out its own text evidence."""
    points = [
        types.SimpleNamespace(id=f"point-{i}", payload={**KEYFRAME_PAYLOAD, "image_id": f"img-{i}"}) for i in range(10)
    ]
    rag, _ = _rag_with_images(points)

    nodes = rag._summary_image_nodes_for_document(file_hash="hash-clip", top_k=3)

    assert len(nodes) == 3


def test_a_document_with_no_hash_draws_no_image_evidence() -> None:
    """Without a hash there is nothing to tie an image to the document."""
    point = types.SimpleNamespace(id="point-77", payload=KEYFRAME_PAYLOAD)
    rag, client = _rag_with_images([point])

    nodes = rag._summary_image_nodes_for_document(file_hash=None, top_k=3)

    assert nodes == []
    assert client.scrolled == []


def test_uncaptioned_images_are_not_summary_evidence() -> None:
    """An image with no caption says nothing a brief could quote."""
    point = types.SimpleNamespace(id="point-77", payload={**KEYFRAME_PAYLOAD, "llm_description": ""})
    rag, _ = _rag_with_images([point])

    nodes = rag._summary_image_nodes_for_document(file_hash="hash-clip", top_k=3)

    assert nodes == []
