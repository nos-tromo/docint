"""Tests that images retrieve as sources rather than being appended after.

The image lane used to run *after* generation: matches were normalized and
stapled onto the source list the answer had already been written from. So an
image could never be cited, never be numbered, and never compete with a text
chunk for a slot. Fusing the lane into the retriever puts image captions in
front of the reranker and the generator like any other evidence.
"""

import urllib.error
import urllib.request
from typing import Any

import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from docint.core.rag import (
    IMAGE_LANE_METADATA_KEY,
    RAG,
    RERANK_APPLIED_KEY,
    RERANK_ERROR_KEY,
    ImageRelevanceFloorPostprocessor,
    LazyRerankerPostprocessor,
    MultimodalRetriever,
    VLLMRerankPostprocessor,
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


def test_the_same_image_is_not_retrieved_twice() -> None:
    """A standalone image file lands in both collections; it is one source.

    ``ImageReader`` writes the caption into the main collection as the
    document's text *and* into the `_images` companion as the CLIP point. Both
    lanes can therefore return the same image. Before fusion that produced two
    citation cards (which the SPA papered over); now it would also spend two
    numbered slots in the prompt on one piece of evidence.
    """
    text_hit = NodeWithScore(
        node=TextNode(id_="doc-node", text="A site plan.", metadata={"filename": "site-plan.png", "image_id": "img-1"}),
        score=0.9,
    )
    fused = MultimodalRetriever(
        text_retriever=_TextRetriever([text_hit]),
        image_lane=lambda query: [_image_node("point-77", 0.28)],
    )

    out = fused.retrieve(QueryBundle(query_str="site plan"))

    assert [n.node.node_id for n in out] == ["doc-node", "point-77"]


def test_the_lane_yields_to_the_document_copy_of_the_same_image() -> None:
    """The main-collection node wins: it is the one with a docstore entry."""
    duplicate = NodeWithScore(
        node=TextNode(
            id_="doc-node",
            text="A hand-drawn site plan.",
            metadata={"filename": "site-plan.png", "image_id": "img-9f2c"},
        ),
        score=0.9,
    )
    fused = MultimodalRetriever(
        text_retriever=_TextRetriever([duplicate]),
        # Same ``image_id`` as IMAGE_PAYLOAD.
        image_lane=lambda query: [_image_node("point-77", 0.28)],
    )

    out = fused.retrieve(QueryBundle(query_str="site plan"))

    assert [n.node.node_id for n in out] == ["doc-node"]


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


def test_the_fused_order_alternates_lanes_by_their_own_rank() -> None:
    """Each lane's best hit leads, so no modality is stranded behind the other.

    Text hits used to come first and image hits last. A rerank that failed
    kept that order and cut to its top-n, so an outage deleted the image lane.
    The raw scores disagree on purpose: text similarity and CLIP cosine are
    not on one scale, so only each lane's own rank may decide the order.
    """
    text_hits = [
        _text_node(f"t{n}", f"Handbook section {n}.", score)
        for n, score in enumerate((0.8, 0.7, 0.6, 0.5, 0.4, 0.3), start=1)
    ]
    image_hits = [_image_node("i1", 0.28), _image_node("i2", 0.27)]
    fused = MultimodalRetriever(text_retriever=_TextRetriever(text_hits), image_lane=lambda query: image_hits)

    out = fused.retrieve(QueryBundle(query_str="Where is the gate?"))

    assert [n.node.node_id for n in out] == ["t1", "i1", "t2", "i2", "t3", "t4", "t5", "t6"]


@pytest.mark.parametrize(
    ("text_ids", "image_ids"),
    [(["t1", "t2", "t3"], []), ([], ["i1", "i2", "i3"])],
    ids=["text-only", "image-only"],
)
def test_a_single_lane_round_trips_unchanged(text_ids: list[str], image_ids: list[str]) -> None:
    """With one lane empty there is nothing to interleave: it keeps its own order.

    The scores are out of order on purpose, so a re-sort by score would show.
    """
    scores = (0.4, 0.9, 0.6)
    text_hits = [_text_node(node_id, "Handbook section.", scores[n]) for n, node_id in enumerate(text_ids)]
    image_hits = [_image_node(node_id, scores[n]) for n, node_id in enumerate(image_ids)]
    fused = MultimodalRetriever(text_retriever=_TextRetriever(text_hits), image_lane=lambda query: image_hits)

    out = fused.retrieve(QueryBundle(query_str="Where is the gate?"))

    assert [n.node.node_id for n in out] == [*text_ids, *image_ids]


def test_a_text_and_an_image_node_with_the_same_id_both_stay() -> None:
    """The fusion keys by lane position, so equal node ids cannot merge two hits.

    ``retrieve`` drops a repeated node id afterwards, which is why this reads
    ``_retrieve``: the fusion itself must not be what loses a hit.
    """
    text_hit = _text_node("shared", "The gate is sealed.", 0.9)
    image_hit = _image_node("shared", 0.28)
    fused = MultimodalRetriever(text_retriever=_TextRetriever([text_hit]), image_lane=lambda query: [image_hit])

    out = fused._retrieve(QueryBundle(query_str="Where is the gate?"))

    assert out == [text_hit, image_hit]


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
    """The legacy shape: a wholly unscored set means no rerank scored anything.

    ``VLLMRerankPostprocessor`` now stamps its own failures, so this is the
    guard for a reranker that drops scores instead. Applying the floor there
    would blank the image lane on every query for as long as the rerank
    endpoint is down.
    """
    nodes = [_text_node("t1", "The gate is sealed.", None), _image_node("point-77", None)]

    out = ImageRelevanceFloorPostprocessor(min_score=0.05)._postprocess_nodes(nodes, None)

    assert [n.node.node_id for n in out] == ["t1", "point-77"]


def _unreranked(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    """Stamp nodes the way a failed rerank leaves them.

    Args:
        nodes (list[NodeWithScore]): Nodes to mark; their scores are untouched.

    Returns:
        list[NodeWithScore]: The same nodes, carrying ``RERANK_APPLIED_KEY=False``.
    """
    for node in nodes:
        node.node.metadata[RERANK_APPLIED_KEY] = False
        node.node.metadata[RERANK_ERROR_KEY] = "rerank endpoint down"
    return nodes


@pytest.mark.parametrize("stamped", [(0, 1), (0,), (1,)], ids=["every-node", "text-node-only", "image-node-only"])
def test_floor_stands_down_on_a_stamped_rerank_failure_even_with_retrieval_scores(stamped: tuple[int, ...]) -> None:
    """A failed rerank keeps the retrieval scores; only the stamp says they are not comparable.

    ``VLLMRerankPostprocessor`` falls back to the nodes it was handed, scores
    intact, so the set is never wholly unscored. An image then carries its CLIP
    cosine, and measuring that against a reranker threshold says nothing about
    relevance: at the default floor every cosine would clear it, and a floor
    above the CLIP band drops them all. The floor sits above the band here,
    which is the setting under which that comparison deletes the lane.
    One stamped node is enough: the mark means the rerank did not run, wherever
    it landed.
    """
    nodes = [_text_node("t1", "The gate is sealed.", 0.9), _image_node("point-77", 0.28)]
    _unreranked([nodes[index] for index in stamped])

    out = ImageRelevanceFloorPostprocessor(min_score=0.5)._postprocess_nodes(nodes, None)

    assert [n.node.node_id for n in out] == ["t1", "point-77"]


def test_a_healthy_stamped_rerank_still_gates_sub_floor_images() -> None:
    """A rerank that ran stamps ``True`` and its scores are on the floor's scale, so the floor applies.

    Every healthy rerank carries the mark, which is why the stand-down has to
    read it as ``False`` and not as merely present: a presence check would turn
    ``IMAGE_RERANK_MIN_SCORE`` into a no-op on every turn.
    """
    nodes = [_text_node("t1", "The gate is sealed.", 0.9), _image_node("point-77", 0.003)]
    for node in nodes:
        node.node.metadata[RERANK_APPLIED_KEY] = True

    out = ImageRelevanceFloorPostprocessor(min_score=0.05)._postprocess_nodes(nodes, None)

    assert [n.node.node_id for n in out] == ["t1"]


def test_a_rerank_outage_keeps_the_best_image_inside_the_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole chain, with the reranker really failing, still shows the best image.

    Retrieval fuses the lanes by rank, the reranker fails the way a stopped
    router does (scores kept, outcome stamped), and the floor must neither
    gate the images on a score from another scale nor let its top-n cut take
    text alone. The stamp must survive to the end, because it is what the
    response lifts into ``rerank.applied``.
    """

    def _refuse(request: object, timeout: float) -> object:
        raise urllib.error.URLError("rerank endpoint down")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)
    text_hits = [
        _text_node(f"t{n}", f"Handbook section {n}.", score)
        for n, score in enumerate((0.8, 0.7, 0.6, 0.5, 0.4, 0.3), start=1)
    ]
    image_hits = [_image_node("i1", 0.28), _image_node("i2", 0.27)]
    bundle = QueryBundle(query_str="Where is the gate?")
    retrieved = MultimodalRetriever(
        text_retriever=_TextRetriever(text_hits), image_lane=lambda query: image_hits
    ).retrieve(bundle)
    rag = RAG(qdrant_collection="testbatch")
    rag.rerank_model_id = "stub-reranker"
    rag._reranker = VLLMRerankPostprocessor(api_base="http://router:8000/v1", model="stub-reranker", top_n=5)

    reranked = LazyRerankerPostprocessor(rag=rag, keep_all=True).postprocess_nodes(retrieved, bundle)
    kept = ImageRelevanceFloorPostprocessor(min_score=0.5, top_n=5).postprocess_nodes(reranked, bundle)

    assert len(kept) == 5
    assert "i1" in [n.node.node_id for n in kept]
    assert all(n.node.metadata[RERANK_APPLIED_KEY] is False for n in kept)


@pytest.mark.parametrize(
    "query_bundle",
    [None, QueryBundle(query_str="   ")],
    ids=["no-bundle", "blank-query"],
)
def test_an_empty_rerank_query_is_stamped_unapplied(query_bundle: QueryBundle | None) -> None:
    """Nothing to rank against is a rerank that did not happen, and the turn must say so.

    The nodes come back in retrieval order either way. Unstamped, a reranker
    that sat in the loop and did nothing reads as one that was never there,
    which is how the response reports a scoped turn.
    """
    reranker = VLLMRerankPostprocessor(api_base="http://router:8000/v1", model="stub-reranker", top_n=2)
    nodes = [_text_node(f"t{n}", f"Handbook section {n}.", 0.9 - n / 10) for n in range(1, 4)]

    out = reranker._postprocess_nodes(nodes, query_bundle)

    assert [n.node.node_id for n in out] == ["t1", "t2"]
    for node in out:
        assert node.node.metadata[RERANK_APPLIED_KEY] is False
        assert node.node.metadata[RERANK_ERROR_KEY] == "empty_query"


@pytest.mark.parametrize("score", [None, 0.0])
def test_floor_drops_unscored_images_when_other_nodes_did_score(score: float | None) -> None:
    """A scored set means the rerank worked; an unscored image lost."""
    nodes = [_text_node("t1", "The gate is sealed.", 0.5), _image_node("point-77", score)]

    out = ImageRelevanceFloorPostprocessor(min_score=0.05)._postprocess_nodes(nodes, None)

    assert [n.node.node_id for n in out] == ["t1"]


def _capturing_lane(rag: RAG, captured: dict[str, Any]) -> None:
    """Record the arguments the image lane hands its retrieval call.

    Args:
        rag (RAG): The instance whose retrieval call to replace.
        captured (dict[str, Any]): Dict the arguments are recorded into.
    """

    def _capture(
        query: str,
        *,
        top_k: int,
        metadata_filter_rules: Any = None,
        qdrant_filter: Any = None,
    ) -> list[NodeWithScore]:
        """Record and return nothing.

        Args:
            query (str): The query.
            top_k (int): Candidate depth.
            metadata_filter_rules (Any): Raw request filters.
            qdrant_filter (Any): Compiled native filter.

        Returns:
            list[NodeWithScore]: Always empty.
        """
        captured["rules"] = metadata_filter_rules
        captured["qdrant_filter"] = qdrant_filter
        return []

    rag._retrieve_image_nodes = _capture


def test_the_lane_forwards_the_request_filter_rules() -> None:
    """The rules are the lane's second pass, so they must reach it."""
    rag = RAG(qdrant_collection="testbatch")
    captured: dict[str, Any] = {}
    _capturing_lane(rag, captured)
    rules = [{"field": "mimetype", "operator": "mime_match", "value": "image/*"}]

    lane = rag._build_image_lane(metadata_filter_rules=rules, metadata_filters_active=True)

    assert lane is not None
    lane("banner")
    assert captured["rules"] == rules


def test_the_lane_forwards_the_compiled_filter_to_the_companion() -> None:
    """Filtering after the top-k cut is what emptied a filtered image lane.

    Five unfiltered candidates rarely survive a clip or time-range rule, so
    the filter has to reach the companion collection, which applies it before
    ranking rather than after.
    """
    rag = RAG(qdrant_collection="testbatch")
    captured: dict[str, Any] = {}
    _capturing_lane(rag, captured)
    native = object()

    lane = rag._build_image_lane(
        metadata_filter_rules=[{"field": "source_type", "operator": "eq", "value": "video_keyframe"}],
        metadata_filters_active=True,
        qdrant_filter=native,
    )

    assert lane is not None
    lane("banner")
    assert captured["qdrant_filter"] is native


def test_the_lane_runs_when_only_the_compiled_filter_reached_it() -> None:
    """A native filter alone is a complete description of what may match."""
    rag = RAG(qdrant_collection="testbatch")
    captured: dict[str, Any] = {}
    _capturing_lane(rag, captured)

    lane = rag._build_image_lane(
        metadata_filter_rules=None,
        metadata_filters_active=True,
        qdrant_filter=object(),
    )

    assert lane is not None


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


def test_the_floor_cuts_after_it_gates_not_before() -> None:
    """A sub-floor image must not hold a slot a text hit would have taken.

    Measured on a live collection: for a question nothing matched, three
    image captions scored ~0.003 against the text chunks' ~0.001, took every
    slot of a top-5 cut, and were then dropped by the floor — the turn
    answered from one source while a dozen text chunks waited behind the cut.
    """
    nodes = [
        _image_node("img-1", 0.003),
        _image_node("img-2", 0.003),
        _image_node("img-3", 0.002),
        _text_node("txt-1", "first chunk", 0.001),
        _text_node("txt-2", "second chunk", 0.0009),
    ]

    kept = ImageRelevanceFloorPostprocessor(min_score=0.05, top_n=2)._postprocess_nodes(nodes)

    assert [node.node.node_id for node in kept] == ["txt-1", "txt-2"]


def test_the_floor_leaves_the_set_uncut_without_a_top_n() -> None:
    """The cut is opt-in: the visual chain does its own."""
    nodes = [_image_node("img-1", 0.9), _text_node("txt-1", "chunk", 0.5)]

    kept = ImageRelevanceFloorPostprocessor(min_score=0.05)._postprocess_nodes(nodes)

    assert [node.node.node_id for node in kept] == ["img-1", "txt-1"]


WEATHER = QueryBundle(query_str="Wie ist das Wetter am Hafen?")


class _Rescore:
    """Description re-score stand-in: fixed scores by node id, every call recorded."""

    def __init__(self, scores: dict[str, float] | None = None, *, error: Exception | None = None) -> None:
        """Store the canned answer.

        Args:
            scores (dict[str, float] | None): Description score per node id.
            error (Exception | None): Raised instead of answering, when set.
        """
        self.scores = scores or {}
        self.error = error
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, query: str, candidates: list[NodeWithScore]) -> dict[str, float]:
        """Record the query and the candidates' ids, then answer.

        Args:
            query (str): The query the floor saw.
            candidates (list[NodeWithScore]): Sub-floor images inside the cut.

        Returns:
            dict[str, float]: The canned scores.

        Raises:
            Exception: The canned error, when one is set.
        """
        self.calls.append((query, [node.node.node_id for node in candidates]))
        if self.error is not None:
            raise self.error
        return dict(self.scores)


def _ids(nodes: list[NodeWithScore]) -> list[str]:
    """Return the node ids in order.

    Args:
        nodes (list[NodeWithScore]): Nodes the floor kept.

    Returns:
        list[str]: Their ids.
    """
    return [node.node.node_id for node in nodes]


@pytest.mark.parametrize("description_score", [0.86, 0.02], ids=["clear-match", "at-the-floor"])
def test_a_sub_floor_image_inside_the_cut_is_kept_when_its_description_score_clears_the_rescore_floor(
    description_score: float,
) -> None:
    """A question about the imagery scores the right image under the floor; its description rescues it.

    Asked as a question, the right picture lands in the band of an unrelated
    one, while the text beside it passes because text is never floored. The
    re-score only decides whether the image stays: it keeps its place and the
    score the turn's rerank gave it.

    Args:
        description_score (float): The image's score against the description.
    """
    nodes = [
        _text_node("t1", "Handbook section 1.", 0.3),
        _image_node("i1", 0.0044),
        _text_node("t2", "Handbook section 2.", 0.0001),
    ]
    rescore = _Rescore({"i1": description_score})

    out = ImageRelevanceFloorPostprocessor(
        min_score=0.05, top_n=5, rescore=rescore, rescore_min_score=0.02
    )._postprocess_nodes(nodes, WEATHER)

    assert _ids(out) == ["t1", "i1", "t2"]
    assert out[1].score == 0.0044
    assert rescore.calls == [("Wie ist das Wetter am Hafen?", ["i1"])]


def test_a_sub_floor_image_inside_the_cut_is_dropped_when_its_description_score_does_not() -> None:
    """An image the description does not match is dropped, as it was before the re-score."""
    nodes = [
        _text_node("t1", "Handbook section 1.", 0.3),
        _image_node("i1", 0.0044),
        _text_node("t2", "Handbook section 2.", 0.0001),
    ]

    out = ImageRelevanceFloorPostprocessor(
        min_score=0.05, top_n=5, rescore=_Rescore({"i1": 0.0072}), rescore_min_score=0.02
    )._postprocess_nodes(nodes, WEATHER)

    assert _ids(out) == ["t1", "t2"]


@pytest.mark.parametrize(
    ("nodes", "rescored"),
    [
        (
            [
                _text_node("t1", "Handbook section 1.", 0.3),
                _image_node("i1", 0.0044),
                _image_node("i2", 0.004),
                _text_node("t2", "Handbook section 2.", 0.0001),
            ],
            [["i1"]],
        ),
        (
            [
                _text_node("t1", "Handbook section 1.", 0.3),
                _text_node("t2", "Handbook section 2.", 0.2),
                _image_node("i1", 0.0044),
            ],
            [],
        ),
    ],
    ids=["one-inside-one-outside", "all-outside"],
)
def test_the_rescore_never_runs_for_an_image_outside_the_cut(
    nodes: list[NodeWithScore], rescored: list[list[str]]
) -> None:
    """Only an image that would land inside the top-n cut is worth a chat call.

    Each sub-floor image is weighed as it is reached, counting every node kept
    or already a candidate ahead of it: with two slots, ``i1`` is reached with
    one taken and ``i2`` with both (``t1`` and the candidate ``i1``). With no
    candidate there is no call at all.
    """
    rescore = _Rescore({"i1": 0.5, "i2": 0.5})

    ImageRelevanceFloorPostprocessor(min_score=0.05, top_n=2, rescore=rescore)._postprocess_nodes(nodes, WEATHER)

    assert [ids for _query, ids in rescore.calls] == rescored


def test_without_a_cut_every_sub_floor_image_is_rescored_in_one_call() -> None:
    """With no ``top_n`` every sub-floor image is inside the cut; an image over the floor needs no second look."""
    nodes = [
        _text_node("t1", "Handbook section 1.", 0.3),
        _image_node("i0", 0.42),
        _image_node("i1", 0.0044),
        _text_node("t2", "Handbook section 2.", 0.2),
        _image_node("i2", 0.004),
    ]
    rescore = _Rescore({"i1": 0.5, "i2": 0.001})

    out = ImageRelevanceFloorPostprocessor(min_score=0.05, rescore=rescore)._postprocess_nodes(nodes, WEATHER)

    assert rescore.calls == [("Wie ist das Wetter am Hafen?", ["i1", "i2"])]
    assert _ids(out) == ["t1", "i0", "i1", "t2"]


@pytest.mark.parametrize("raises", [True, False], ids=["raises", "returns-nothing"])
def test_a_failed_rescore_drops_the_image_as_before(raises: bool) -> None:
    """A re-score that fails costs the image its second look and nothing else.

    Args:
        raises (bool): Whether the re-score raises, rather than scoring nothing.
    """
    nodes = [_text_node("t1", "Handbook section 1.", 0.3), _image_node("i1", 0.0044)]
    rescore = _Rescore(error=RuntimeError("chat endpoint down")) if raises else _Rescore({})

    out = ImageRelevanceFloorPostprocessor(min_score=0.05, top_n=5, rescore=rescore)._postprocess_nodes(nodes, WEATHER)

    assert _ids(out) == ["t1"]
    assert len(rescore.calls) == 1


@pytest.mark.parametrize("shape", ["stamped", "unscored"])
def test_the_rescore_never_runs_on_a_degraded_rerank(shape: str) -> None:
    """A degraded rerank gates nothing, so there is nothing to rescue and no call to pay for.

    Args:
        shape (str): How the rerank degraded: stamped as not applied, or left wholly unscored.
    """
    if shape == "stamped":
        nodes = _unreranked([_text_node("t1", "The gate is sealed.", 0.9), _image_node("i1", 0.28)])
    else:
        nodes = [_text_node("t1", "The gate is sealed.", None), _image_node("i1", None)]
    rescore = _Rescore({"i1": 0.9})

    out = ImageRelevanceFloorPostprocessor(min_score=0.5, top_n=5, rescore=rescore)._postprocess_nodes(nodes, WEATHER)

    assert rescore.calls == []
    assert _ids(out) == ["t1", "i1"]


@pytest.mark.parametrize("query_bundle", [None, QueryBundle(query_str="   ")], ids=["no-bundle", "blank-query"])
def test_the_rescore_needs_a_query_to_describe(query_bundle: QueryBundle | None) -> None:
    """Without a query there is no picture to describe, so the image drops as before.

    Args:
        query_bundle (QueryBundle | None): A missing or blank query.
    """
    nodes = [_text_node("t1", "Handbook section 1.", 0.3), _image_node("i1", 0.0044)]
    rescore = _Rescore({"i1": 0.9})

    out = ImageRelevanceFloorPostprocessor(min_score=0.05, top_n=5, rescore=rescore)._postprocess_nodes(
        nodes, query_bundle
    )

    assert rescore.calls == []
    assert _ids(out) == ["t1"]


def test_the_floor_cuts_after_it_gates_not_before_when_the_rescore_rescues_nothing() -> None:
    """The re-score leaves the gate-then-cut order alone.

    The same nodes as :func:`test_the_floor_cuts_after_it_gates_not_before`:
    the two images inside the cut get their second look and fail it, so the
    text chunks behind them still take both slots.
    """
    nodes = [
        _image_node("img-1", 0.003),
        _image_node("img-2", 0.003),
        _image_node("img-3", 0.002),
        _text_node("txt-1", "first chunk", 0.001),
        _text_node("txt-2", "second chunk", 0.0009),
    ]

    def _nothing_matches(query: str, candidates: list[NodeWithScore]) -> dict[str, float]:
        """Score every candidate 0.0.

        Args:
            query (str): The query.
            candidates (list[NodeWithScore]): Sub-floor images inside the cut.

        Returns:
            dict[str, float]: A zero for each candidate.
        """
        return {node.node.node_id: 0.0 for node in candidates}

    kept = ImageRelevanceFloorPostprocessor(min_score=0.05, top_n=2, rescore=_nothing_matches)._postprocess_nodes(
        nodes, WEATHER
    )

    assert [node.node.node_id for node in kept] == ["txt-1", "txt-2"]
