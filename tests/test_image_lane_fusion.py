"""Tests that images retrieve as sources rather than being appended after.

The image lane used to run *after* generation: matches were normalized and
stapled onto the source list the answer had already been written from. So an
image could never be cited, never be numbered, and never compete with a text
chunk for a slot. Fusing the lane into the retriever puts image captions in
front of the reranker and the generator like any other evidence.
"""

from typing import Any

import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from docint.core.rag import (
    IMAGE_LANE_METADATA_KEY,
    RAG,
    ImageRelevanceFloorPostprocessor,
    MultimodalRetriever,
)

IMAGE_PAYLOAD: dict[str, Any] = {
    "image_id": "img-9f2c",
    "node_id": "point-77",
    "source_doc_id": "hash-abc123",
    "source_path": "/ingest/batch/site-plan.png",
    "llm_description": "A hand-drawn site plan with a numbered legend.",
    "llm_tags": ["plan", "legend"],
    "score": 0.28,
}


class _TextRetriever:
    """Text retriever stand-in returning one fixed hit."""

    def __init__(self, nodes: list[NodeWithScore] | None = None) -> None:
        """Initialize the stand-in.

        Args:
            nodes (list[NodeWithScore] | None): Nodes to return; one hit by default.
        """
        self.nodes = nodes if nodes is not None else [_text_node("t1", "The gate is sealed.", 0.9)]
        self.seen: list[str] = []

    def retrieve(self, query_bundle: Any) -> list[NodeWithScore]:
        """Return the configured nodes.

        Args:
            query_bundle (Any): The incoming query bundle.

        Returns:
            list[NodeWithScore]: The configured hits.
        """
        self.seen.append(getattr(query_bundle, "query_str", str(query_bundle)))
        return self.nodes


def _text_node(node_id: str, text: str, score: float | None) -> NodeWithScore:
    """Build a scored text node.

    Args:
        node_id (str): Node id.
        text (str): Node body.
        score (float | None): Retrieval score.

    Returns:
        NodeWithScore: The wrapped node.
    """
    return NodeWithScore(node=TextNode(id_=node_id, text=text, metadata={"filename": "handbook.pdf"}), score=score)


def _image_node(node_id: str, score: float | None) -> NodeWithScore:
    """Build a scored image-lane node.

    Args:
        node_id (str): Node id.
        score (float | None): Retrieval score.

    Returns:
        NodeWithScore: The wrapped node, marked as image-lane.
    """
    metadata = {**IMAGE_PAYLOAD, IMAGE_LANE_METADATA_KEY: True}
    return NodeWithScore(node=TextNode(id_=node_id, text="A site plan.", metadata=metadata), score=score)


def test_fused_retriever_returns_both_lanes() -> None:
    """Image captions reach the query engine alongside text chunks."""
    text = _TextRetriever()
    fused = MultimodalRetriever(text_retriever=text, image_lane=lambda query: [_image_node("point-77", 0.28)])

    out = fused.retrieve(QueryBundle(query_str="Where is the gate?"))

    assert [n.node.node_id for n in out] == ["t1", "point-77"]


def test_image_lane_failure_degrades_to_text_only() -> None:
    """A CLIP or companion-collection outage must not fail the whole query."""

    def _boom(query: str) -> list[NodeWithScore]:
        raise RuntimeError("clip down")

    fused = MultimodalRetriever(text_retriever=_TextRetriever(), image_lane=_boom)

    out = fused.retrieve(QueryBundle(query_str="Where is the gate?"))

    assert [n.node.node_id for n in out] == ["t1"]


def test_image_lane_receives_the_original_query() -> None:
    """The lane translates for CLIP itself; it must see the user's words."""
    seen: list[str] = []
    fused = MultimodalRetriever(
        text_retriever=_TextRetriever(),
        image_lane=lambda query: (seen.append(query), [])[1],
    )

    fused.retrieve(QueryBundle(query_str="Wo ist das Tor?"))

    assert seen == ["Wo ist das Tor?"]


def test_floor_drops_images_the_reranker_scored_below_it() -> None:
    """A merely-nearest image does not become a citable source."""
    nodes = [_text_node("t1", "The gate is sealed.", 0.5), _image_node("point-77", 0.001)]

    out = ImageRelevanceFloorPostprocessor(min_score=0.05)._postprocess_nodes(nodes, None)

    assert [n.node.node_id for n in out] == ["t1"]


def test_floor_keeps_images_the_reranker_scored_above_it() -> None:
    """A relevant image survives and keeps its place among the text nodes."""
    nodes = [_text_node("t1", "The gate is sealed.", 0.5), _image_node("point-77", 0.42)]

    out = ImageRelevanceFloorPostprocessor(min_score=0.05)._postprocess_nodes(nodes, None)

    assert [n.node.node_id for n in out] == ["t1", "point-77"]


def test_floor_never_touches_text_nodes() -> None:
    """Text sources are the reranker's business, not the image floor's."""
    nodes = [_text_node("t1", "Barely relevant.", 0.001), _text_node("t2", "Unscored.", None)]

    out = ImageRelevanceFloorPostprocessor(min_score=0.05)._postprocess_nodes(nodes, None)

    assert [n.node.node_id for n in out] == ["t1", "t2"]


def test_floor_stands_down_when_the_rerank_degraded() -> None:
    """An unscored set means the reranker failed, not that nothing is relevant.

    ``VLLMRerankPostprocessor`` swallows its transport errors and hands the
    nodes back untouched, so a wholly unscored set is how a degraded rerank
    announces itself. Applying the floor there would blank the image lane on
    every query for as long as the rerank endpoint is down.
    """
    nodes = [_text_node("t1", "The gate is sealed.", None), _image_node("point-77", None)]

    out = ImageRelevanceFloorPostprocessor(min_score=0.05)._postprocess_nodes(nodes, None)

    assert [n.node.node_id for n in out] == ["t1", "point-77"]


@pytest.mark.parametrize("score", [None, 0.0])
def test_floor_drops_unscored_images_when_other_nodes_did_score(score: float | None) -> None:
    """A scored set means the rerank worked; an unscored image lost."""
    nodes = [_text_node("t1", "The gate is sealed.", 0.5), _image_node("point-77", score)]

    out = ImageRelevanceFloorPostprocessor(min_score=0.05)._postprocess_nodes(nodes, None)

    assert [n.node.node_id for n in out] == ["t1"]


def test_the_lane_forwards_the_request_filter_rules() -> None:
    """Image candidates are filtered in memory, so the rules must reach them."""
    rag = RAG(qdrant_collection="testbatch")
    captured: dict[str, Any] = {}

    def _capture(query: str, *, top_k: int, metadata_filter_rules: Any = None) -> list[NodeWithScore]:
        captured["rules"] = metadata_filter_rules
        return []

    rag._retrieve_image_nodes = _capture
    rules = [{"field": "mimetype", "operator": "mime_match", "value": "image/*"}]

    lane = rag._build_image_lane(metadata_filter_rules=rules, metadata_filters_active=True)

    assert lane is not None
    lane("banner")
    assert captured["rules"] == rules


def test_the_lane_stands_down_when_filters_never_reached_the_runtime() -> None:
    """Filtered requests must not answer from unfilterable images.

    The lane post-filters in memory, so without the raw rules the only options
    are unfiltered images or none. None is the safe one.
    """
    rag = RAG(qdrant_collection="testbatch")

    lane = rag._build_image_lane(metadata_filter_rules=None, metadata_filters_active=True)

    assert lane is None


def test_the_lane_runs_unfiltered_when_no_filters_are_in_play() -> None:
    """An ordinary request retrieves images."""
    rag = RAG(qdrant_collection="testbatch")

    lane = rag._build_image_lane(metadata_filter_rules=None, metadata_filters_active=False)

    assert lane is not None
