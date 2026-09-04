"""Tests for the visual retrieval target's two-lane retriever.

The visual target answers from stored imagery alone, so it cannot borrow the
text lane's recall: CLIP finds pictures whose captions never name the thing
asked about, and a keyword pass finds the literal terms CLIP's text tower is
weak at. Both must be optional, because either can be down while the other
still answers.
"""

from typing import Any

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from docint.core.retrieval.visual import (
    BLOB_PAYLOAD_KEYS,
    RETRIEVAL_TARGETS,
    VisualCandidate,
    VisualRetriever,
    fuse_candidates,
    rank_keyword_candidates,
    reciprocal_rank_fusion,
    visual_min_match,
)


class _Point:
    """Qdrant scroll point stand-in."""

    def __init__(self, point_id: str, payload: dict[str, Any]) -> None:
        """Initialize the stand-in.

        Args:
            point_id (str): The point id.
            payload (dict[str, Any]): The point payload.
        """
        self.id = point_id
        self.payload = payload


def _candidate(point_id: str, **kwargs: Any) -> VisualCandidate:
    """Build a candidate with a caption-bearing payload.

    Args:
        point_id (str): The point id.
        **kwargs (Any): Overrides for the candidate fields.

    Returns:
        VisualCandidate: The candidate.
    """
    payload = {"image_id": f"img-{point_id}", "llm_description": f"caption {point_id}"}
    payload.update(kwargs.pop("payload", {}))
    return VisualCandidate(point_id=point_id, payload=payload, **kwargs)


def _node(candidate: VisualCandidate) -> NodeWithScore:
    """Build a citation node for a candidate.

    Args:
        candidate (VisualCandidate): The candidate.

    Returns:
        NodeWithScore: A node carrying the candidate's payload as metadata.
    """
    return NodeWithScore(
        node=TextNode(text=str(candidate.payload.get("llm_description") or ""), metadata=dict(candidate.payload)),
        score=candidate.clip_score,
    )


def _retriever(clip: list[VisualCandidate], keyword: list[VisualCandidate], *, limit: int = 10) -> VisualRetriever:
    """Build a retriever over two fixed lanes.

    Args:
        clip (list[VisualCandidate]): CLIP lane output.
        keyword (list[VisualCandidate]): Keyword lane output.
        limit (int): Fused candidate cap.

    Returns:
        VisualRetriever: The retriever.
    """
    return VisualRetriever(
        clip_lane=lambda _query: list(clip),
        keyword_lane=lambda _query: list(keyword),
        make_node=_node,
        limit=limit,
    )


def test_the_three_targets_are_the_whole_vocabulary() -> None:
    """The target names are fixed; the SPA and the API validate against them."""
    assert RETRIEVAL_TARGETS == ("all", "documents", "visual")


def test_visual_retriever_returns_both_lanes() -> None:
    """A point either lane found is evidence."""
    retriever = _retriever([_candidate("a")], [_candidate("b", keyword_hits=2)])

    nodes = retriever.retrieve(QueryBundle(query_str="what is at the gate"))

    assert {node.node.metadata["image_id"] for node in nodes} == {"img-a", "img-b"}


def test_visual_retriever_survives_a_clip_lane_outage() -> None:
    """A dead CLIP endpoint degrades the turn to keyword-only, never fails it."""

    def _boom(_query: str) -> list[VisualCandidate]:
        """Fail the way an unreachable CLIP endpoint does.

        Args:
            _query (str): Unused.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("clip endpoint unreachable")

    retriever = VisualRetriever(
        clip_lane=_boom,
        keyword_lane=lambda _query: [_candidate("b", keyword_hits=1)],
        make_node=_node,
        limit=10,
    )

    nodes = retriever.retrieve(QueryBundle(query_str="gate"))

    assert [node.node.metadata["image_id"] for node in nodes] == ["img-b"]


def test_visual_retriever_survives_a_keyword_lane_outage() -> None:
    """A failed companion scroll degrades the turn to CLIP-only."""

    def _boom(_query: str) -> list[VisualCandidate]:
        """Fail the way a rejected scroll does.

        Args:
            _query (str): Unused.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("scroll rejected")

    retriever = VisualRetriever(
        clip_lane=lambda _query: [_candidate("a", clip_score=0.3)],
        keyword_lane=_boom,
        make_node=_node,
        limit=10,
    )

    nodes = retriever.retrieve(QueryBundle(query_str="gate"))

    assert [node.node.metadata["image_id"] for node in nodes] == ["img-a"]


def test_visual_retriever_dedupes_the_two_lanes_by_point_id() -> None:
    """One picture found twice is one source, not two citation slots."""
    shared = _candidate("a", clip_score=0.4)
    retriever = _retriever([shared], [_candidate("a", keyword_hits=3)])

    nodes = retriever.retrieve(QueryBundle(query_str="gate"))

    assert len(nodes) == 1


def test_visual_retriever_dedupes_by_image_id_across_point_ids() -> None:
    """The same bytes stored under two points still cite once."""
    first = VisualCandidate(point_id="p1", payload={"image_id": "img-x", "llm_description": "a"})
    second = VisualCandidate(point_id="p2", payload={"image_id": "img-x", "llm_description": "a"})
    retriever = _retriever([first], [second])

    nodes = retriever.retrieve(QueryBundle(query_str="gate"))

    assert len(nodes) == 1


def test_visual_retriever_drops_uncaptioned_images() -> None:
    """An image with nothing written about it is no evidence a reader can judge."""

    def _make_node(candidate: VisualCandidate) -> NodeWithScore | None:
        """Refuse a candidate with no caption.

        Args:
            candidate (VisualCandidate): The candidate.

        Returns:
            NodeWithScore | None: The node, or ``None`` when uncaptioned.
        """
        return None if not candidate.payload.get("llm_description") else _node(candidate)

    retriever = VisualRetriever(
        clip_lane=lambda _query: [VisualCandidate(point_id="p1", payload={"image_id": "img-x"})],
        keyword_lane=lambda _query: [],
        make_node=_make_node,
        limit=10,
    )

    assert retriever.retrieve(QueryBundle(query_str="gate")) == []


def test_visual_retriever_is_empty_when_both_lanes_are() -> None:
    """No candidates is an unanswerable turn, not an error."""
    retriever = _retriever([], [])

    assert retriever.retrieve(QueryBundle(query_str="gate")) == []


def test_visual_retriever_caps_the_fused_set() -> None:
    """The candidate depth is a cap on evidence, not a suggestion."""
    retriever = _retriever([_candidate(str(index)) for index in range(10)], [], limit=3)

    assert len(retriever.retrieve(QueryBundle(query_str="gate"))) == 3


def test_rrf_prefers_items_ranked_by_both_lanes() -> None:
    """Agreement between the lanes is the strongest signal either can give."""
    fused = reciprocal_rank_fusion(["a", "b", "c"], ["c", "d"])

    assert fused[0] == "c"


def test_rrf_keeps_a_single_lane_order_intact() -> None:
    """One lane's ranking must round-trip, so a CLIP-only turn is unreordered."""
    assert reciprocal_rank_fusion(["a", "b", "c"], []) == ["a", "b", "c"]


def test_fusion_merges_what_each_lane_knew_about_a_point() -> None:
    """A fused candidate carries the CLIP score and the keyword evidence."""
    fused = fuse_candidates(
        [_candidate("a", clip_score=0.42)],
        [_candidate("a", keyword_hits=2, phrase=True)],
        limit=5,
    )

    assert (fused[0].clip_score, fused[0].keyword_hits, fused[0].phrase) == (0.42, 2, True)


def test_keyword_candidates_rank_by_matched_count() -> None:
    """``min_should`` says whether a point matched, never how well."""
    points = [
        _Point("a", {"search_text": "ein rotes auto"}),
        _Point("b", {"search_text": "ein rotes auto am tor"}),
    ]

    ranked = rank_keyword_candidates(points, ["rotes", "auto", "tor"])

    assert [candidate.point_id for candidate in ranked] == ["b", "a"]


def test_keyword_candidates_drop_points_that_matched_nothing() -> None:
    """A point the filter returned but no keyword touched is not a candidate."""
    ranked = rank_keyword_candidates([_Point("a", {"search_text": "nichts davon"})], ["auto"])

    assert ranked == []


def test_blob_payload_keys_never_reach_a_candidate() -> None:
    """Pixels reach the model through the synthesizer, never through a node."""
    point = _Point("a", {"search_text": "auto", "thumbnail_b64": "QUJD", "image_id": "img-a"})

    ranked = rank_keyword_candidates([point], ["auto"])

    assert not any(key in ranked[0].payload for key in BLOB_PAYLOAD_KEYS)
    assert ranked[0].payload["image_id"] == "img-a"


def test_half_the_keywords_is_the_match_bar() -> None:
    """A question carries more words than a one-line caption can hold."""
    assert (visual_min_match(1), visual_min_match(4), visual_min_match(5)) == (1, 2, 3)
