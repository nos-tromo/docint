"""The visual retrieval target: answering from stored imagery alone.

A chat turn carries a *retrieval target* saying which evidence may answer it:

``all``
    Text chunks and image captions fused into one evidence set, the
    historical behaviour.
``documents``
    Text chunks only; the image lane never runs.
``visual``
    The ``{collection}_images`` companion only, retrieved by
    :class:`VisualRetriever`.

The target is orthogonal to the request's ``retrieval_mode``
(``session``/``stateless``, which is session routing) and to the response's
own ``retrieval_mode`` vocabulary (``scoped``/``rewrite_*``).

This module holds the target vocabulary and the visual retriever. It knows
nothing about :class:`~docint.core.rag.RAG`: the CLIP lane, the keyword lane
and the node builder all arrive as callables.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Literal, get_args

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from loguru import logger
from qdrant_client import models
from typing_extensions import override

from docint.core.search.fulltext import matches_phrase

RetrievalTarget = Literal["all", "documents", "visual"]
"""Which evidence a chat turn may answer from."""

RETRIEVAL_TARGETS: tuple[str, ...] = get_args(RetrievalTarget)
"""Every accepted retrieval target, for validation and tests."""

DEFAULT_RETRIEVAL_TARGET: RetrievalTarget = "all"

BLOB_PAYLOAD_KEYS: tuple[str, ...] = (
    "thumbnail_b64",
    "thumbnail_mime",
    "thumbnail_max_dim",
    "search_text",
    "_node_content",
    "_node_type",
)
"""Companion payload keys a candidate never carries.

The thumbnail fields are pixels — they reach the model through the
synthesizer, by point id, and must never travel on a node where they could
land in ``sources`` or in the LLM-visible metadata. ``search_text`` is an
index artifact and the ``_node_*`` keys are llama-index bookkeeping; both are
noise in a payload that becomes citation metadata.
"""


@dataclass(frozen=True, slots=True)
class VisualCandidate:
    """One companion point that survived a lane.

    Attributes:
        point_id: The Qdrant point id, which is also the citation node id.
        payload: The point's payload minus :data:`BLOB_PAYLOAD_KEYS`.
        clip_score: Raw CLIP cosine when the CLIP lane produced it, else
            ``None``. Not comparable across queries — the reranker decides
            relevance downstream.
        keyword_hits: How many of the query's keywords the point's indexed
            text matched; ``0`` for a CLIP-only candidate.
        phrase: Whether the point matched the query as a phrase.
    """

    point_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    clip_score: float | None = None
    keyword_hits: int = 0
    phrase: bool = False


ClipLane = Callable[[str], list[VisualCandidate]]
KeywordLane = Callable[[str], list[VisualCandidate]]
NodeBuilder = Callable[[VisualCandidate], NodeWithScore | None]


class VisualRetriever(BaseRetriever):
    """Retrieve stored imagery as the whole evidence set for a turn.

    Two lanes run over the same companion collection and are fused by
    reciprocal rank: CLIP similarity finds imagery a caption never names,
    and a keyword pass over the indexed caption/tag/OCR text finds the
    literal terms CLIP's text tower is weak at. Both are optional — an
    outage in either degrades the turn rather than failing it, and a query
    with no usable keywords is simply CLIP-only.

    Attributes:
        clip_lane: Callable taking the query and returning CLIP candidates.
        keyword_lane: Callable taking the query and returning keyword
            candidates.
        make_node: Turns one candidate into a retrieval node, or ``None``
            when it carries no caption a reader could judge.
        limit: How many candidates survive the fusion.
    """

    def __init__(
        self,
        *,
        clip_lane: ClipLane,
        keyword_lane: KeywordLane,
        make_node: NodeBuilder,
        limit: int,
    ) -> None:
        """Initialize the visual retriever.

        Args:
            clip_lane (ClipLane): CLIP candidate producer.
            keyword_lane (KeywordLane): Keyword candidate producer.
            make_node (NodeBuilder): Candidate-to-node builder.
            limit (int): Fused candidate cap.
        """
        self.clip_lane = clip_lane
        self.keyword_lane = keyword_lane
        self.make_node = make_node
        self.limit = max(1, int(limit))
        super().__init__()

    def _run_lane(self, lane: ClipLane | KeywordLane, query: str, name: str) -> list[VisualCandidate]:
        """Run one lane, absorbing its failures.

        Args:
            lane (ClipLane | KeywordLane): The lane to run.
            query (str): The query being answered.
            name (str): Lane name, for the log line.

        Returns:
            list[VisualCandidate]: The lane's candidates; empty on any fault.
        """
        try:
            return list(lane(query))
        except Exception as exc:
            logger.warning("Visual {} lane failed: {}. Continuing without it.", name, exc)
            return []

    @override
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve fused visual candidates as citation nodes.

        Args:
            query_bundle (QueryBundle): The query being answered.

        Returns:
            list[NodeWithScore]: Caption nodes in fused order, capped at
                :attr:`limit`. Empty when both lanes came back empty, which
                the synthesizer reports as an unanswerable turn.
        """
        query = query_bundle.query_str
        clip = self._run_lane(self.clip_lane, query, "CLIP")
        keyword = self._run_lane(self.keyword_lane, query, "keyword")
        if not clip and not keyword:
            return []

        nodes: list[NodeWithScore] = []
        seen_images: set[str] = set()
        for candidate in fuse_candidates(clip, keyword, limit=self.limit):
            image_id = str(candidate.payload.get("image_id") or "").strip()
            if image_id:
                if image_id in seen_images:
                    continue
                seen_images.add(image_id)
            node = self.make_node(candidate)
            if node is not None:
                nodes.append(node)
        return nodes


def reciprocal_rank_fusion(*ranked_ids: Sequence[str], k: int = 60) -> list[str]:
    """Fuse ranked id lists by reciprocal rank.

    Each list contributes ``1 / (k + rank)`` to an id's score, so an item
    both lanes rank moderately well beats one lane's top hit less often the
    larger ``k`` is. ``k=60`` is the conventional value and the same order of
    magnitude as the candidate depth here.

    Args:
        *ranked_ids (Sequence[str]): One ranked id list per lane, best first.
        k (int): Rank damping constant.

    Returns:
        list[str]: Ids in fused order, each appearing once. Ties keep the
            order in which the id was first seen, so a single non-empty lane
            round-trips unchanged.
    """
    scores: dict[str, float] = {}
    order: list[str] = []
    for ranking in ranked_ids:
        for rank, point_id in enumerate(ranking):
            if point_id not in scores:
                scores[point_id] = 0.0
                order.append(point_id)
            scores[point_id] += 1.0 / (k + rank + 1)
    return sorted(order, key=lambda point_id: (-scores[point_id], order.index(point_id)))


def fuse_candidates(
    clip: Sequence[VisualCandidate],
    keyword: Sequence[VisualCandidate],
    *,
    limit: int,
    k: int = 60,
) -> list[VisualCandidate]:
    """Merge the two lanes into one ranked candidate list.

    A point found by both lanes is emitted once, carrying the CLIP score from
    the CLIP entry and the keyword evidence from the keyword entry, so the
    fused candidate says everything known about it.

    Args:
        clip (Sequence[VisualCandidate]): CLIP candidates, best first.
        keyword (Sequence[VisualCandidate]): Keyword candidates, best first.
        limit (int): How many candidates to keep.
        k (int): Reciprocal-rank damping constant.

    Returns:
        list[VisualCandidate]: Fused candidates, best first, at most
            ``limit`` long.
    """
    by_id: dict[str, VisualCandidate] = {}
    for candidate in keyword:
        by_id[candidate.point_id] = candidate
    for candidate in clip:
        existing = by_id.get(candidate.point_id)
        if existing is None:
            by_id[candidate.point_id] = candidate
            continue
        by_id[candidate.point_id] = VisualCandidate(
            point_id=candidate.point_id,
            payload={**existing.payload, **candidate.payload},
            clip_score=candidate.clip_score,
            keyword_hits=existing.keyword_hits,
            phrase=existing.phrase,
        )

    fused = reciprocal_rank_fusion(
        [candidate.point_id for candidate in clip],
        [candidate.point_id for candidate in keyword],
        k=k,
    )
    return [by_id[point_id] for point_id in fused[: max(1, int(limit))] if point_id in by_id]


VISUAL_FILTER_INDEXES: tuple[tuple[str, Any], ...] = (
    ("source_type", models.PayloadSchemaType.KEYWORD),
    ("source_doc_id", models.PayloadSchemaType.KEYWORD),
    ("source_file", models.PayloadSchemaType.KEYWORD),
    ("keyframe_time_sec", models.PayloadSchemaType.FLOAT),
)
"""Companion payload keys the visual filter presets narrow by.

``source_type`` separates video keyframes from social imagery and loose
files, ``source_file``/``source_doc_id`` pin one clip or document, and
``keyframe_time_sec`` carries the time range. Without these indexes Qdrant
still answers, by scanning — correct but slow, and the answer is identical,
so creating them is an optimisation that must never fail a query.
"""


def ensure_visual_filter_indexes(client: Any, collection: str) -> bool:
    """Ensure the companion carries the indexes the visual filters use.

    Idempotent and fail-soft, mirroring
    :func:`docint.core.search.fields.ensure_field_indexes`. Creating an index
    that already exists is a no-op on Qdrant's side, so this only saves
    round-trips; an existing index of a different kind is left alone, because
    these keys are not shared with the search-field matchers.

    Args:
        client (Any): Qdrant client exposing ``create_payload_index``.
        collection (str): Physical companion collection name.

    Returns:
        bool: ``True`` when every index was created or already present.
    """
    ok = True
    for key, schema in VISUAL_FILTER_INDEXES:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=key,
                field_schema=schema,
                wait=True,
            )
        except Exception as exc:
            logger.debug("Visual filter index '{}' on '{}' not created: {}", key, collection, exc)
            ok = False
    return ok


def visual_min_match(keyword_count: int) -> int:
    """Return how many of a query's keywords a caption must match.

    Half, rounded up: a chat question carries more words than a one-line
    caption can hold, so demanding all of them matches nothing, and demanding
    one matches the whole collection.

    Args:
        keyword_count (int): How many keywords the query yielded.

    Returns:
        int: The minimum match count, at least one.
    """
    return max(1, math.ceil(keyword_count / 2))


def rank_keyword_candidates(
    points: Sequence[Any],
    keywords: Sequence[str],
    *,
    text_key: str = "search_text",
) -> list[VisualCandidate]:
    """Rank companion points by how much of the query they matched.

    Qdrant's ``min_should`` says *whether* a point matched, never how well, so
    the lane would otherwise hand the fusion an arbitrary order. Counting the
    matched keywords here is what makes its ranking mean something.

    Args:
        points (Sequence[Any]): Scrolled points carrying ``id`` and ``payload``.
        keywords (Sequence[str]): Keywords in query order.
        text_key (str): Payload key holding the indexed text.

    Returns:
        list[VisualCandidate]: Candidates, best first — most keywords matched,
            phrase matches ahead of scattered ones, then point id so the order
            is stable across calls.
    """
    folded = [keyword.casefold() for keyword in keywords if keyword]
    ranked: list[VisualCandidate] = []
    for point in points:
        payload = dict(getattr(point, "payload", None) or {})
        haystack = str(payload.get(text_key) or "").casefold()
        hits = sum(1 for keyword in folded if keyword in haystack)
        ranked.append(
            VisualCandidate(
                point_id=str(getattr(point, "id", "")),
                payload={key: value for key, value in payload.items() if key not in BLOB_PAYLOAD_KEYS},
                clip_score=None,
                keyword_hits=hits,
                phrase=matches_phrase(haystack, list(folded)) if folded else False,
            )
        )
    ranked.sort(key=lambda candidate: (-candidate.keyword_hits, not candidate.phrase, candidate.point_id))
    return [candidate for candidate in ranked if candidate.keyword_hits]


VISUAL_IMAGE_TOKEN_ESTIMATE = 600
"""Prompt tokens one attached thumbnail is budgeted at.

A 768px image on a 32px patch grid costs about 576 tokens on the Qwen3-VL
family. Rounded up, because under-reserving overflows the context window
while over-reserving only shortens the caption context slightly.
"""

VISUAL_LEGEND_TOKEN_ALLOWANCE = 128
"""Prompt tokens reserved for the legend mapping images to citation numbers."""

VISUAL_IMAGES_ATTACHED_KEY = "docint_visual_images_attached"
"""Node-metadata stamp marking a source whose pixels the model actually saw."""

ThumbnailFetcher = Callable[[Sequence[str]], dict[str, tuple[str, str]]]
"""Fetch ``point_id -> (mime, base64)`` for stored thumbnails."""


def image_token_reserve(max_images: int) -> int:
    """Return the prompt budget to hold back for attached imagery.

    Args:
        max_images (int): How many images may be attached.

    Returns:
        int: Tokens to subtract from the synthesizer's context window.
    """
    return max(0, int(max_images)) * VISUAL_IMAGE_TOKEN_ESTIMATE + VISUAL_LEGEND_TOKEN_ALLOWANCE


def select_answer_images(nodes: Sequence[NodeWithScore], *, max_images: int) -> list[NodeWithScore]:
    """Pick the nodes whose pixels the model should see.

    Citation order, which by this point is relevance order: the numbering
    postprocessor has already run, so the first nodes are the ones the answer
    is most likely to cite.

    Args:
        nodes (Sequence[NodeWithScore]): The postprocessed evidence set.
        max_images (int): How many images may be attached.

    Returns:
        list[NodeWithScore]: The nodes to attach, at most ``max_images``.
    """
    if max_images <= 0:
        return []
    return [node for node in nodes if _node_metadata(node).get("image_id")][:max_images]


def _node_metadata(node: Any) -> dict[str, Any]:
    """Read a scored node's metadata defensively.

    Args:
        node (Any): A ``NodeWithScore``-like object.

    Returns:
        dict[str, Any]: Its metadata, or an empty dict.
    """
    inner = getattr(node, "node", None)
    return dict(getattr(inner, "metadata", None) or {})


def citation_index_of(node: Any, fallback: int) -> int:
    """Return the citation number the answer will use for a node.

    Args:
        node (Any): A ``NodeWithScore``-like object.
        fallback (int): Number to use when the node carries none.

    Returns:
        int: The citation index.
    """
    for key in ("citation_index", "docint_citation_index", "source_number"):
        value = _node_metadata(node).get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return fallback


def render_legend(template: str, images: Sequence[tuple[int, str, str]]) -> str:
    """Render the legend tying attached images to citation numbers.

    Without it the model sees N pictures and a numbered caption list with no
    way to say which is which, so it cannot cite the picture it is describing.

    Args:
        template (str): Locale template carrying an ``{image_legend}`` slot.
        images (Sequence[tuple[int, str, str]]): ``(citation_index, mime,
            base64)`` in attachment order.

    Returns:
        str: The rendered legend, or an empty string when nothing is attached.
    """
    if not images:
        return ""
    lines = "\n".join(f"Image {position}: source [{citation}]" for position, (citation, _, _) in enumerate(images, 1))
    if "{image_legend}" in template:
        return template.replace("{image_legend}", lines)
    return lines


class VisualImagesMixin:
    """Put the retrieved imagery itself in front of the model.

    Every other retrieval path answers from text: a caption is what the model
    reads of a picture. That is enough to *find* imagery and too little to
    answer questions about it, because a caption written at ingest time
    answered a different question than the one being asked now.

    So the visual target attaches the stored thumbnails of the top sources to
    the same synthesis call, alongside the numbered caption context. The
    mechanism is deliberately narrow: the LLM, the prompts, the streaming, the
    citation numbering and the persistence are all unchanged, and the only
    difference is that the user message carries image blocks.

    Thumbnails are fetched by point id at synthesis time and held here, never
    on a node — a node's metadata reaches the citation panel, the persisted
    turn and the LLM-visible context, and pixels belong in none of them.

    Attributes:
        _fetch_thumbnails: Fetches ``point_id -> (mime, base64)``.
        _max_images: How many images may be attached.
        _legend_template: Locale template mapping images to citations.
        _attached: ``(citation_index, mime, base64)`` for this synthesis.
        _stamped: The node metadata dicts marked as shown to the model, so
            the mark can be taken back if the attachment is refused.
    """

    _fetch_thumbnails: ThumbnailFetcher
    _max_images: int
    _legend_template: str
    _attached: list[tuple[int, str, str]]
    _stamped: list[dict[str, Any]]

    def _collect_images(self, nodes: Sequence[NodeWithScore]) -> None:
        """Resolve and stamp the images this synthesis will attach.

        Args:
            nodes (Sequence[NodeWithScore]): The postprocessed evidence set.
        """
        self._attached = []
        self._stamped = []
        selected = select_answer_images(nodes, max_images=self._max_images)
        if not selected:
            return
        point_ids = [str(getattr(getattr(node, "node", None), "node_id", "") or "") for node in selected]
        try:
            thumbnails = self._fetch_thumbnails([point_id for point_id in point_ids if point_id])
        except Exception as exc:
            logger.warning("Visual thumbnails unavailable: {}. Answering from captions alone.", exc)
            return
        for position, node in enumerate(selected):
            point_id = point_ids[position]
            found = thumbnails.get(point_id)
            if not found:
                continue
            mime, data = found
            self._attached.append((citation_index_of(node, position + 1), mime, data))
            inner = getattr(node, "node", None)
            metadata = getattr(inner, "metadata", None)
            if isinstance(metadata, dict):
                # Lifted off the nodes and out of the response by
                # ``RAG._normalize_response_data``, the way the rerank stamp is.
                metadata[VISUAL_IMAGES_ATTACHED_KEY] = True
                self._stamped.append(metadata)

    def _drop_attached_images(self, reason: Exception) -> None:
        """Give up on the imagery and answer from the captions instead.

        A model can refuse a prompt carrying pictures — most often because
        the endpoint caps how many it accepts (vLLM's
        ``--limit-mm-per-prompt``, which the deployment sets independently of
        ``VISUAL_ANSWER_MAX_IMAGES``), sometimes because it has no vision
        tower at all. Failing the turn over that would lose an answer the
        captions alone could have carried, so this is the same degradation a
        thumbnail-fetch outage already takes.

        The stamps go with them: a turn reports how many images the model
        actually saw, and leaving them on would report pictures it never got.

        Args:
            reason (Exception): What the model said, for the operator's log.
        """
        logger.warning(
            "Visual answer refused with {} image(s) attached: {}. Answering from captions alone. "
            "Lower VISUAL_ANSWER_MAX_IMAGES to what the endpoint accepts, or raise the endpoint's own limit.",
            len(self._attached),
            reason,
        )
        self._attached = []
        for metadata in self._stamped:
            metadata.pop(VISUAL_IMAGES_ATTACHED_KEY, None)
        self._stamped = []

    def synthesize(self, query: Any, nodes: Sequence[NodeWithScore], *args: Any, **kwargs: Any) -> Any:
        """Attach imagery, then synthesize as usual.

        Args:
            query (Any): The query bundle.
            nodes (Sequence[NodeWithScore]): The postprocessed evidence set.
            *args (Any): Forwarded upstream.
            **kwargs (Any): Forwarded upstream.

        Returns:
            Any: The upstream response.
        """
        self._collect_images(nodes)
        return super().synthesize(query, nodes, *args, **kwargs)  # type: ignore[misc]

    async def asynthesize(self, query: Any, nodes: Sequence[NodeWithScore], *args: Any, **kwargs: Any) -> Any:
        """Attach imagery, then synthesize as usual, asynchronously.

        Args:
            query (Any): The query bundle.
            nodes (Sequence[NodeWithScore]): The postprocessed evidence set.
            *args (Any): Forwarded upstream.
            **kwargs (Any): Forwarded upstream.

        Returns:
            Any: The upstream response.
        """
        self._collect_images(nodes)
        return await super().asynthesize(query, nodes, *args, **kwargs)  # type: ignore[misc]

    def _messages_with_images(self, prompt: Any, prompt_kwargs: dict[str, Any]) -> list[Any]:
        """Render the prompt as chat messages carrying the attached imagery.

        Args:
            prompt (Any): The refine program's prompt template.
            prompt_kwargs (dict[str, Any]): Prompt variables for this chunk.

        Returns:
            list[Any]: Chat messages, the last user one carrying the legend
                and one image block per attachment.
        """
        from llama_index.core.base.llms.types import ImageBlock, MessageRole, TextBlock

        llm = self._llm  # type: ignore[attr-defined]
        messages = list(llm._get_messages(prompt, **prompt_kwargs))
        if not self._attached:
            return messages

        target = next(
            (message for message in reversed(messages) if getattr(message, "role", None) == MessageRole.USER),
            messages[-1] if messages else None,
        )
        if target is None:
            return messages

        legend = render_legend(self._legend_template, self._attached)
        blocks = list(getattr(target, "blocks", []) or [])
        if legend:
            blocks.append(TextBlock(text=legend))
        blocks.extend(ImageBlock(url=f"data:{mime};base64,{data}") for _, mime, data in self._attached)
        target.blocks = blocks
        return messages

    def _update_response(self, program: Any, program_kwargs: dict[str, Any], response_kwargs: dict[str, Any]) -> Any:
        """Answer this chunk with the imagery attached.

        Args:
            program (Any): The refine program built for the current prompt.
            program_kwargs (dict[str, Any]): Prompt variables for this chunk.
            response_kwargs (dict[str, Any]): Extra LLM kwargs from the caller.

        Returns:
            Any: A token generator when streaming, the answer text otherwise,
                or whatever upstream produces for a structured answer.
        """
        from llama_index.core.llms.llm import stream_chat_response_to_tokens

        prompt = getattr(program, "_prompt", None)
        plain_text = (
            not self._structured_answer_filtering  # type: ignore[attr-defined]
            and self._output_cls is None  # type: ignore[attr-defined]
            and prompt is not None
        )
        if not plain_text or not self._attached:
            return super()._update_response(program, program_kwargs, response_kwargs)  # type: ignore[misc]

        llm = self._llm  # type: ignore[attr-defined]
        messages = self._messages_with_images(prompt, {**program_kwargs, **response_kwargs})
        # Every call below is retried without the imagery on any fault, which
        # is why the catch can be broad without hiding an outage: a dead
        # endpoint fails the retry too and the error still reaches the caller.
        if self._streaming:  # type: ignore[attr-defined]
            tokens = stream_chat_response_to_tokens(llm.stream_chat(messages))
            try:
                # A streaming client sends the request on the first pull, not
                # on the call, so the refusal arrives here rather than above.
                first = next(tokens)
            except StopIteration:
                return iter(())
            except Exception as exc:
                self._drop_attached_images(exc)
                return super()._update_response(program, program_kwargs, response_kwargs)  # type: ignore[misc]
            return chain([first], tokens)
        try:
            return llm.chat(messages).message.content or ""
        except Exception as exc:
            self._drop_attached_images(exc)
            return super()._update_response(program, program_kwargs, response_kwargs)  # type: ignore[misc]

    async def _aupdate_response(
        self,
        program: Any,
        program_kwargs: dict[str, Any],
        response_kwargs: dict[str, Any],
    ) -> Any:
        """Answer this chunk with the imagery attached, asynchronously.

        Args:
            program (Any): The refine program built for the current prompt.
            program_kwargs (dict[str, Any]): Prompt variables for this chunk.
            response_kwargs (dict[str, Any]): Extra LLM kwargs from the caller.

        Returns:
            Any: The answer text, or whatever upstream produces for a
                structured answer.
        """
        prompt = getattr(program, "_prompt", None)
        plain_text = (
            not self._structured_answer_filtering  # type: ignore[attr-defined]
            and self._output_cls is None  # type: ignore[attr-defined]
            and prompt is not None
        )
        if not plain_text or not self._attached:
            return await super()._aupdate_response(program, program_kwargs, response_kwargs)  # type: ignore[misc]

        llm = self._llm  # type: ignore[attr-defined]
        messages = self._messages_with_images(prompt, {**program_kwargs, **response_kwargs})
        try:
            response = await llm.achat(messages)
        except Exception as exc:
            self._drop_attached_images(exc)
            return await super()._aupdate_response(program, program_kwargs, response_kwargs)  # type: ignore[misc]
        return response.message.content or ""
