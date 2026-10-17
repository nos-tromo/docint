"""The agent's image tools, search_images and keyframes, and the RAG methods behind them.

Every clip, caption, person and hash here is invented.
"""

from __future__ import annotations

import functools
import io
import json
import re
import types
import urllib.error
from collections.abc import Sequence
from typing import Any, cast

import pytest
from agent_fakes import ScriptedSearch, programme_passage, shown_numbers, tool_call, tool_context, two_phase_round
from assessment_fakes import scored_node
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from loguru import logger
from qdrant_client import models
from typing_extensions import override

import docint.core.rag as rag_module
from docint.agents.turn.image_tools import KeyframesTool, SearchImagesTool
from docint.agents.turn.stream import ToolCall
from docint.agents.turn.tools import DETAIL_CHARS, PassageReply, ReadTool, ToolContext, ToolRegistry
from docint.core.rag import RAG, ParentContextPostprocessor
from docint.core.retrieval.visual import BLOB_PAYLOAD_KEYS
from docint.core.retrieval_filters import build_qdrant_filter

CLIP = "harbour_walk.webm"
CLIP_HASH = "c" * 64
OTHER_CLIP = "kai_rundgang.mp4"
OTHER_HASH = "d" * 64
DESCRIPTION = "Zwei Personen in Regenjacken am Hafen"
CAPTIONS = (
    "Zwei Personen in Regenjacken am Hafen, bedeckter Himmel",
    "Person A hält einen Schirm vor dem Hafenamt",
    "Ein Fischkutter legt am Kai an, Möwen über dem Wasser",
    "Person A und Person B gehen an den Ständen vorbei",
    "Regen auf dem Kopfsteinpflaster vor der Hafenkneipe",
)
#: The request filter of a turn that asks only about the clip from its thirtieth second on.
FROM_THIRTY_SECONDS = [{"field": "keyframe_time_sec", "operator": "gte", "value": 30}]
#: An audio clip: transcribed, so its segments carry its hash, and without frames.
AUDIO = "hafen_interview.m4a"
AUDIO_HASH = "e" * 64
#: One file name two different clips carry, from two folders of one batch.
SHARED = "hafen_rundgang.mov"
FIRST_HASH = "a" * 64
SECOND_HASH = "b" * 64


def _context(*, reasoning: bool = False) -> ToolContext:
    """Build the context of one turn whose evidence is still empty, seen through the production LLM view.

    Args:
        reasoning (bool): Whether the model thinks before it answers, which shortens replies.

    Returns:
        ToolContext: The context.
    """
    return tool_context(reasoning=reasoning, view=ParentContextPostprocessor._emit_with_llm_exclusion)


def _labels(text: str) -> list[str]:
    """Return the keyframe labels a reply shows, in reply order.

    Args:
        text (str): The reply.

    Returns:
        list[str]: Each shown frame's label.
    """
    return re.findall(r"^keyframe: (.+)$", text, flags=re.MULTILINE)


def _frame_payload(
    index: int,
    *,
    clip: str = CLIP,
    media_hash: str = CLIP_HASH,
    caption: str | None = None,
) -> dict[str, Any]:
    """Build the payload a clip's frame is stored with, as the companion returns it without its blobs.

    Args:
        index (int): The frame's ``keyframe_index``; it appears ``6 * index`` seconds into the clip.
        clip (str): The clip's file name.
        media_hash (str): The clip's content hash.
        caption (str | None): What the frame shows; one of ``CAPTIONS`` when not given.

    Returns:
        dict[str, Any]: The payload.
    """
    return {
        "image_id": f"{clip}-image-{index}",
        "source_type": "video_keyframe",
        "source_doc_id": media_hash,
        "source_file": clip,
        "source_path": f"uploads/{clip}",
        "media_file_hash": media_hash,
        "keyframe_index": index,
        "keyframe_time_sec": float(6 * index),
        "llm_description": caption if caption is not None else CAPTIONS[index % len(CAPTIONS)],
        "llm_tags": ["hafen"],
    }


def _stored(payload: dict[str, Any]) -> dict[str, Any]:
    """Add the blobs a stored companion point carries beside its payload.

    Args:
        payload (dict[str, Any]): The payload.

    Returns:
        dict[str, Any]: The payload with a thumbnail and its indexed search text.
    """
    return {
        **payload,
        "thumbnail_b64": "QUJD",
        "thumbnail_mime": "image/jpeg",
        "search_text": payload["llm_description"],
    }


@functools.cache
def _builder() -> RAG:
    """Return a RAG instance that builds caption nodes as the image lane does; it touches no store.

    Returns:
        RAG: The instance.
    """
    return RAG(qdrant_collection="testbatch")


def _frame(index: int, *, clip: str = CLIP, media_hash: str = CLIP_HASH, caption: str | None = None) -> NodeWithScore:
    """Build one clip frame as the keyframe listing returns it.

    Args:
        index (int): The frame's ``keyframe_index``.
        clip (str): The clip's file name.
        media_hash (str): The clip's content hash.
        caption (str | None): What the frame shows.

    Returns:
        NodeWithScore: The frame's caption node.
    """
    payload = {**_frame_payload(index, clip=clip, media_hash=media_hash, caption=caption), "node_id": f"{clip}-{index}"}
    return _builder()._image_caption_node(payload, RAG._image_caption_text(payload))


def _segment(number: int, *, clip: str = CLIP, media_hash: str | None = CLIP_HASH) -> NodeWithScore:
    """Build one transcript segment of a clip, as a search finds it.

    Args:
        number (int): Which segment; it varies the id.
        clip (str): The clip's file name.
        media_hash (str | None): The clip's content hash; ``None`` for a transcript uploaded on its own.

    Returns:
        NodeWithScore: The segment.
    """
    identity: dict[str, Any] = {"media_file_hash": media_hash} if media_hash else {}
    return scored_node(
        f"{clip}-segment-{number}",
        "Person A: Wir gehen jetzt am Hafen entlang, es fängt gleich an zu regnen.",
        filename=clip,
        source_file=clip,
        docint_doc_kind="transcript_segment",
        **identity,
    )


class _Lister:
    """A keyframe listing double: answers each clip key with its frames and records the keys asked for."""

    def __init__(self, clips: dict[str, list[NodeWithScore]]) -> None:
        """Hold the frames of each clip.

        Args:
            clips (dict[str, list[NodeWithScore]]): Frames in time order, by file name or content hash.
        """
        self._clips = clips
        self.keys: list[str] = []

    def __call__(self, key: str) -> list[NodeWithScore]:
        """List one clip's frames.

        Args:
            key (str): The clip's file name or content hash.

        Returns:
            list[NodeWithScore]: Its frames; empty for a clip it does not hold.
        """
        self.keys.append(key)
        return list(self._clips.get(key, []))


def _registry(*, search_images: ScriptedSearch | None = None, list_keyframes: Any = None) -> ToolRegistry:
    """Build the registry of a visual turn: search_images, keyframes and read.

    Args:
        search_images (ScriptedSearch | None): The image search; one that never runs when not given.
        list_keyframes (Any): The keyframe listing; one that holds no clip when not given.

    Returns:
        ToolRegistry: The registry.
    """
    return ToolRegistry(
        [
            SearchImagesTool(search_images or ScriptedSearch()),
            KeyframesTool(list_keyframes or _Lister({})),
            ReadTool(),
        ]
    )


def _value(payload: dict[str, Any], key: str) -> Any:
    """Read a dotted payload key as Qdrant does.

    Args:
        payload (dict[str, Any]): The payload.
        key (str): The key, such as ``reference_metadata.type``.

    Returns:
        Any: The value; ``None`` when the key is absent.
    """
    value: Any = payload
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _holds(condition: Any, payload: dict[str, Any]) -> bool:
    """Evaluate a Qdrant filter or condition against one payload, as the store would.

    Args:
        condition (Any): A ``models.Filter`` or one of the conditions these tests use.
        payload (dict[str, Any]): The point's payload.

    Returns:
        bool: Whether the point matches.

    Raises:
        AssertionError: For a condition the invented store cannot evaluate.
    """
    if isinstance(condition, models.Filter):
        assert condition.min_should is None, "The invented store does not evaluate min_should."
        return (
            all(_holds(item, payload) for item in condition.must or [])
            and not any(_holds(item, payload) for item in condition.must_not or [])
            and (not condition.should or any(_holds(item, payload) for item in condition.should))
        )
    if isinstance(condition, models.IsEmptyCondition):
        return _value(payload, condition.is_empty.key) in (None, [])
    if isinstance(condition, models.FieldCondition):
        value = _value(payload, condition.key)
        if isinstance(condition.match, models.MatchValue):
            return value == condition.match.value
        if condition.range is not None and isinstance(condition.range, models.Range):
            bounds = condition.range
            return isinstance(value, (int, float)) and all(
                (
                    bounds.gte is None or value >= bounds.gte,
                    bounds.gt is None or value > bounds.gt,
                    bounds.lte is None or value <= bounds.lte,
                    bounds.lt is None or value < bounds.lt,
                )
            )
    raise AssertionError(f"The invented store cannot evaluate {condition!r}.")


class _Companion:
    """An invented ``testbatch_images`` companion: a scroll answers with its matching points, a page at a time.

    It applies the scroll filter itself, so a test sees which frames a filter lets through rather
    than how the filter is spelled, and it honours the payload selector as Qdrant does.
    """

    def __init__(self, payloads: Sequence[dict[str, Any]], *, page: int = 2, exists: bool = True) -> None:
        """Store the points, in the order given.

        Args:
            payloads (Sequence[dict[str, Any]]): Each point's stored payload.
            page (int): The most points one scroll returns.
            exists (bool): Whether the companion collection exists.
        """
        self._points = [types.SimpleNamespace(id=f"point-{position}", payload=p) for position, p in enumerate(payloads)]
        self._page = page
        self._exists = exists
        self.scrolls: list[dict[str, Any]] = []

    def collection_exists(self, collection_name: str) -> bool:
        """Report whether the companion exists.

        Args:
            collection_name (str): The collection asked about.

        Returns:
            bool: Whether it exists.
        """
        return self._exists and collection_name == "testbatch_images"

    def scroll(self, **kwargs: Any) -> tuple[list[types.SimpleNamespace], int | None]:
        """Return the next page of matching points.

        Args:
            **kwargs (Any): The scroll's arguments, as the Qdrant client takes them.

        Returns:
            tuple[list[types.SimpleNamespace], int | None]: The page and the next page's offset.
        """
        self.scrolls.append(kwargs)
        assert kwargs["collection_name"] == "testbatch_images"
        excluded = set(getattr(kwargs["with_payload"], "exclude", None) or ())
        matching = [point for point in self._points if _holds(kwargs["scroll_filter"], point.payload)]
        start = kwargs["offset"] or 0
        end = start + min(kwargs["limit"], self._page)
        page = [
            types.SimpleNamespace(id=point.id, payload={k: v for k, v in point.payload.items() if k not in excluded})
            for point in matching[start:end]
        ]
        return page, end if end < len(matching) else None


def _rag(companion: Any) -> RAG:
    """Build a RAG instance whose store is the given double.

    Args:
        companion (Any): The Qdrant client double.

    Returns:
        RAG: The instance.
    """
    rag = RAG(qdrant_collection="testbatch")
    rag._qdrant_client = cast(Any, companion)
    return rag


class _Retriever(BaseRetriever):
    """Return the same candidates for any query and record each query, as the visual retriever hands them on."""

    def __init__(self, candidates: Sequence[NodeWithScore]) -> None:
        """Hold the candidates.

        Args:
            candidates (Sequence[NodeWithScore]): What every retrieval returns, in fused order.
        """
        super().__init__()
        self._candidates = list(candidates)
        self.queries: list[str] = []

    @override
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Record the query and return the candidates.

        Args:
            query_bundle (QueryBundle): The query.

        Returns:
            list[NodeWithScore]: The candidates.
        """
        self.queries.append(query_bundle.query_str)
        return list(self._candidates)


class _RerankEndpoint:
    """A rerank endpoint double: records the request and answers with fixed scores, best first, cut to ``top_n``."""

    def __init__(self, scores: dict[int, float]) -> None:
        """Hold the score of each document, by its position in the request.

        Args:
            scores (dict[int, float]): Scores by document position.
        """
        self._scores = scores
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: Any, timeout: float) -> Any:
        """Answer one rerank request.

        Args:
            request (Any): The ``urllib`` request.
            timeout (float): The request timeout, unused.

        Returns:
            Any: A response whose body holds the results.
        """
        sent = json.loads(request.data.decode("utf-8"))
        self.requests.append(sent)
        ranked = sorted(self._scores.items(), key=lambda item: item[1], reverse=True)[: sent["top_n"]]
        body = json.dumps({"results": [{"index": index, "relevance_score": score} for index, score in ranked]})

        return io.BytesIO(body.encode("utf-8"))


def _image_search_rag(monkeypatch: pytest.MonkeyPatch, retriever: _Retriever, built: dict[str, Any]) -> RAG:
    """Build a RAG instance whose visual retriever is ``retriever`` and whose rerank endpoint is remote and stubbed.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patching fixture.
        retriever (_Retriever): What ``_build_visual_retriever`` returns.
        built (dict[str, Any]): Filled with the arguments ``_build_visual_retriever`` was called with.

    Returns:
        RAG: The instance, with ``rerank_top_n`` 2 and ``VISUAL_ANSWER_MAX_IMAGES`` 3.
    """

    def build(self: RAG, **kwargs: Any) -> _Retriever:
        """Record the arguments and return the retriever.

        Args:
            self (RAG): The instance.
            **kwargs (Any): The arguments.

        Returns:
            _Retriever: The retriever.
        """
        built.update(kwargs)
        return retriever

    monkeypatch.setenv("VISUAL_ANSWER_MAX_IMAGES", "3")
    monkeypatch.setattr(RAG, "_build_visual_retriever", build)
    rag = RAG(qdrant_collection="testbatch")
    rag.rerank_top_n = 2
    rag.rerank_model_id = "BAAI/bge-reranker-v2-m3"
    rag._reranker = rag_module.VLLMRerankPostprocessor(
        api_base="http://rerank.invalid/v1", model="BAAI/bge-reranker-v2-m3", top_n=2
    )
    return rag


def _candidates() -> list[NodeWithScore]:
    """Build the fused candidates of an image search, one per invented caption.

    Returns:
        list[NodeWithScore]: The candidates, in fused order.
    """
    return [
        NodeWithScore(node=TextNode(id_=f"point-{n}", text=caption), score=None) for n, caption in enumerate(CAPTIONS)
    ]


@pytest.mark.parametrize("filtered", [False, True], ids=["unfiltered", "filtered"])
def test_search_images_reranks_against_the_description_without_a_floor(
    monkeypatch: pytest.MonkeyPatch, filtered: bool
) -> None:
    """Candidates are reranked against the description as deep as a visual answer may look; a low score cuts none.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patching fixture.
        filtered (bool): Whether the turn carries a request filter, which must reach the retriever.
    """
    retriever = _Retriever(_candidates())
    built: dict[str, Any] = {}
    rag = _image_search_rag(monkeypatch, retriever, built)
    endpoint = _RerankEndpoint({0: 0.001, 1: 0.003, 2: 0.002, 3: 0.0004, 4: 0.004})
    monkeypatch.setattr(rag_module.urllib.request, "urlopen", endpoint)
    request = build_qdrant_filter(FROM_THIRTY_SECONDS) if filtered else None

    found = rag._agent_search_images(DESCRIPTION, qdrant_filter=request)

    assert built == {
        "metadata_filter_rules": None,
        "metadata_filters_active": filtered,
        "vector_store_kwargs": {"qdrant_filters": request} if filtered else None,
    }
    assert retriever.queries == [DESCRIPTION]
    assert [(sent["query"], sent["documents"], sent["top_n"]) for sent in endpoint.requests] == [
        (DESCRIPTION, list(CAPTIONS), 3)
    ]
    assert [(scored.node.node_id, scored.score) for scored in found] == [
        ("point-4", 0.004),
        ("point-1", 0.003),
        ("point-2", 0.002),
    ]
    assert all(scored.score is not None and scored.score < rag._image_relevance_floor() for scored in found)


def test_a_rerank_outage_leaves_the_images_in_fused_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreachable rerank endpoint degrades the search to the fused order, as deep as the rerank would have cut.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patching fixture.
    """
    rag = _image_search_rag(monkeypatch, _Retriever(_candidates()), {})

    def unreachable(request: Any, timeout: float) -> Any:
        """Fail as an endpoint that is down does.

        Args:
            request (Any): The request, unused.
            timeout (float): The timeout, unused.

        Raises:
            urllib.error.URLError: Always.
        """
        raise urllib.error.URLError("invented outage")

    monkeypatch.setattr(rag_module.urllib.request, "urlopen", unreachable)

    found = rag._agent_search_images(DESCRIPTION)

    assert [scored.node.node_id for scored in found] == ["point-0", "point-1", "point-2"]


def test_search_images_shows_new_pictures_and_names_ones_shown_before() -> None:
    """The description is the search; pictures new to the turn are shown under the next numbers, as passages are."""
    ctx = _context()
    held, found = _frame(0), _frame(3, clip=OTHER_CLIP, media_hash=OTHER_HASH)
    ctx.ledger.add([held])
    search = ScriptedSearch([found, held])

    reply = _registry(search_images=search).call(
        tool_call("search_images", description="  Zwei Personen   in Regenjacken am Hafen "), ctx
    )

    assert search.queries == [DESCRIPTION]
    assert shown_numbers(reply.text) == [2]
    assert CAPTIONS[3] in reply.text
    assert _labels(reply.text) == ["#4 @ 00:18"]
    assert reply.text.endswith("\n\nAlso matched, shown before: [1].")
    assert isinstance(reply, PassageReply)
    assert (reply.found, reply.action, reply.detail, reply.numbers) == (1, "search_images", DESCRIPTION, (1, 2))


@pytest.mark.parametrize(
    "arguments", ["{}", '{"description": "   "}', '{"description": 7}'], ids=["missing", "blank", "number"]
)
def test_search_images_needs_a_description(arguments: str) -> None:
    """A call without words to look for says what it needs; nothing runs.

    Args:
        arguments (str): The raw arguments the model sent.
    """
    search = ScriptedSearch()

    reply = _registry(search_images=search).call(
        ToolCall(id="call_0", name="search_images", arguments=arguments), _context()
    )

    assert reply.text == "Error: search_images needs a description: what the picture should show."
    assert (reply.found, reply.action) == (0, "search_images")
    assert search.queries == []


def test_a_repeated_image_search_is_answered_not_rerun() -> None:
    """The same description again is answered from memory by the numbers it showed."""
    ctx = _context()
    search = ScriptedSearch([_frame(0), _frame(1)])
    registry = _registry(search_images=search)
    registry.call(tool_call("search_images", description=DESCRIPTION), ctx)

    again = registry.call(tool_call("search_images", description=DESCRIPTION.upper()), ctx)

    assert again.text == "Already called search_images with these arguments; see [1], [2]."
    assert search.queries == [DESCRIPTION]


def test_keyframes_lists_frames_in_time_order() -> None:
    """A clip's frames arrive across scroll pages in store order; they are shown by time, each labelled #n @ mm:ss."""
    companion = _Companion(
        [_stored(_frame_payload(index)) for index in (3, 0, 4, 1, 2)]
        + [_stored(_frame_payload(0, clip=OTHER_CLIP, media_hash=OTHER_HASH))],
        page=2,
    )
    ctx = _context()

    reply = _registry(list_keyframes=_rag(companion)._agent_list_keyframes).call(tool_call("keyframes", file=CLIP), ctx)

    assert shown_numbers(reply.text) == [1, 2, 3, 4, 5]
    assert _labels(reply.text) == ["#1 @ 00:00", "#2 @ 00:06", "#3 @ 00:12", "#4 @ 00:18", "#5 @ 00:24"]
    shown_captions = re.findall("|".join(map(re.escape, CAPTIONS)), reply.text)
    assert [CAPTIONS.index(caption) for caption in shown_captions] == [0, 1, 2, 3, 4]
    assert len(companion.scrolls) == 3
    assert "frames in all" not in reply.text
    assert isinstance(reply, PassageReply)
    assert (reply.found, reply.action, reply.detail, reply.numbers) == (5, "keyframes", CLIP, (1, 2, 3, 4, 5))


@pytest.mark.parametrize("file", [2, "2", "[2]"], ids=["number", "text", "bracketed"])
@pytest.mark.parametrize(
    ("media_hash", "key"),
    [(CLIP_HASH, CLIP_HASH), (None, CLIP)],
    ids=["by-media-hash", "by-file-name"],
)
def test_keyframes_accepts_a_passage_number(file: object, media_hash: str | None, key: str) -> None:
    """A passage from a clip names the clip: its media hash, else its file name, is what the frames are listed by.

    Args:
        file (object): The ``file`` the model sent.
        media_hash (str | None): The passage's media hash; ``None`` for a transcript uploaded on its own.
        key (str): What the clip's frames must be listed by.
    """
    ctx = _context()
    ctx.ledger.add([programme_passage(), _segment(1, media_hash=media_hash)])
    lister = _Lister({key: [_frame(0), _frame(1)]})

    reply = _registry(list_keyframes=lister).call(tool_call("keyframes", file=file), ctx)

    assert lister.keys == [key]
    assert shown_numbers(reply.text) == [3, 4]
    assert _labels(reply.text) == ["#1 @ 00:00", "#2 @ 00:06"]
    assert (reply.found, reply.detail) == (2, CLIP)


def test_a_passage_from_no_clip_names_no_clip() -> None:
    """A document passage belongs to no clip; the reply says so and names the clips the turn has seen."""
    ctx = _context()
    ctx.ledger.add([programme_passage(), _segment(1)])
    lister = _Lister({})

    reply = _registry(list_keyframes=lister).call(tool_call("keyframes", file=1), ctx)

    assert reply.text == "Error: passage [1] is not from a clip; the clips seen so far are 'harbour_walk.webm'."
    assert (reply.found, reply.action, reply.detail) == (0, "keyframes", "1")
    assert lister.keys == []


def test_an_unknown_passage_number_names_the_range() -> None:
    """A number no passage has is answered with the numbers there are, as read answers it."""
    ctx = _context()
    ctx.ledger.add([programme_passage(), _segment(1)])
    lister = _Lister({})

    reply = _registry(list_keyframes=lister).call(tool_call("keyframes", file="[7]"), ctx)

    assert reply.text == "Error: there is no passage [7]; the passages run from [1] to [2]."
    assert (reply.found, reply.detail) == (0, "[7]")
    assert lister.keys == []


def test_a_passage_number_resolves_against_the_evidence_as_the_round_was_prepared() -> None:
    """Calls prepared together see the evidence as it stood before their round, not a number that round admits."""
    ctx = _context()
    registry = _registry(
        search_images=ScriptedSearch([_frame(0)]),
        list_keyframes=_Lister({CLIP_HASH: [_frame(0), _frame(1)]}),
    )
    calls = [tool_call("search_images", description=DESCRIPTION), tool_call("keyframes", file=1)]

    found, early = two_phase_round(registry, calls, ctx)
    later = registry.call(tool_call("keyframes", file=1), ctx)

    assert shown_numbers(found.text) == [1]
    assert early.text == "Error: there is no passage [1]; no passage has been shown yet."
    assert shown_numbers(later.text) == [2]
    assert later.text.endswith("\n\nAlso matched, shown before: [1].")


@pytest.mark.parametrize("index", [4, "4", "#4"], ids=["number", "text", "label"])
def test_keyframes_opens_one_frame_by_its_label_number(index: object) -> None:
    """Index names a frame by its #n label, never its place in the list: #3 was pruned, so #4 is the third frame stored.

    Args:
        index (object): The ``index`` the model sent.
    """
    ctx = _context()
    lister = _Lister({CLIP: [_frame(position) for position in (0, 1, 3, 4)]})

    reply = _registry(list_keyframes=lister).call(tool_call("keyframes", file=CLIP, index=index), ctx)

    assert shown_numbers(reply.text) == [1]
    assert _labels(reply.text) == ["#4 @ 00:18"]
    assert CAPTIONS[3] in reply.text
    assert (reply.found, reply.detail, len(ctx.ledger)) == (1, f"{CLIP} #4", 1)


@pytest.mark.parametrize(
    ("indexes", "index", "known"),
    [
        ((0, 1, 3, 4), 3, "its frames are #1, #2, #4, #5"),
        ((0,), 2, "its only frame is #1"),
        (tuple(range(10)), 12, "its frames run from #1 to #10"),
    ],
    ids=["pruned", "single-frame", "long-clip"],
)
def test_a_frame_the_clip_lacks_is_answered_with_the_frames_it_has(
    indexes: tuple[int, ...], index: int, known: str
) -> None:
    """An index the clip has no frame for is answered with the frames it has; nothing is admitted.

    Nextext numbers frames before it drops near-duplicates, so a clip's numbers have gaps:
    a short clip's are listed, a long clip's range is given.

    Args:
        indexes (tuple[int, ...]): The ``keyframe_index`` of each frame the clip kept.
        index (int): The frame number the model asked for.
        known (str): How the reply names the clip's frames.
    """
    ctx = _context()

    reply = _registry(list_keyframes=_Lister({CLIP: [_frame(position) for position in indexes]})).call(
        tool_call("keyframes", file=CLIP, index=index), ctx
    )

    assert reply.text == f"Error: 'harbour_walk.webm' has no frame #{index}; {known}."
    assert (reply.found, reply.detail, len(ctx.ledger)) == (0, f"{CLIP} #{index}", 0)


@pytest.mark.parametrize("index", [0, -2, "drei", True, 2.5], ids=["zero", "negative", "word", "bool", "fraction"])
def test_an_index_that_is_no_frame_number_gets_an_error_reply(index: object) -> None:
    """An index that names no frame is answered with the form it takes; nothing is listed.

    Args:
        index (object): The ``index`` the model sent.
    """
    lister = _Lister({CLIP: [_frame(0)]})

    reply = _registry(list_keyframes=lister).call(tool_call("keyframes", file=CLIP, index=index), _context())

    assert reply.text == "Error: index is the number of one frame, the n of its #n label, such as 3."
    assert lister.keys == []


@pytest.mark.parametrize("file", [None, "   ", True, 0], ids=["missing", "blank", "bool", "zero"])
def test_keyframes_needs_a_clip(file: object) -> None:
    """A call naming no clip says what file takes; nothing is listed.

    Args:
        file (object): The ``file`` the model sent; ``None`` leaves it out.
    """
    lister = _Lister({})
    arguments: dict[str, object] = {} if file is None else {"file": file}

    reply = _registry(list_keyframes=lister).call(tool_call("keyframes", **arguments), _context())

    assert (
        reply.text == "Error: keyframes needs file: a clip's filename, or the number of a passage from it, such as 3."
    )
    assert lister.keys == []


@pytest.mark.parametrize(
    ("count", "labels"),
    [
        (8, [1, 2, 3, 4, 5, 6, 7, 8]),
        (9, [1, 2, 3, 4, 6, 7, 8, 9]),
        (20, [1, 4, 6, 9, 12, 15, 17, 20]),
    ],
    ids=["eight-frames", "nine-frames", "twenty-frames"],
)
def test_keyframes_caps_long_clips_evenly(count: int, labels: list[int]) -> None:
    """At most eight frames are shown: a longer clip's evenly spaced, its first and last among them, the total noted.

    Args:
        count (int): How many frames the clip kept.
        labels (list[int]): The frame numbers the reply shows, in order.
    """
    ctx = _context()
    frames = [_frame(index, caption=f"Bild {index + 1} am Hafen") for index in range(count)]

    reply = _registry(list_keyframes=_Lister({CLIP: frames})).call(tool_call("keyframes", file=CLIP), ctx)

    assert [int(label.split(" @ ")[0].removeprefix("#")) for label in _labels(reply.text)] == labels
    assert shown_numbers(reply.text) == list(range(1, 9))
    assert reply.text.endswith(
        f"\n\n{count} frames in all; call keyframes(file, index) for one."
        if count > 8
        else f"Bild {count} am Hafen\n\nTags: hafen"
    )
    assert len(reply.text) <= ctx.budget.result_chars


def test_an_uncaptioned_frame_is_listed_with_its_label_and_no_text() -> None:
    """A frame the vision model said nothing about is still a frame of the clip: listed, labelled, its text empty."""
    blank: dict[str, Any] = {**_frame_payload(0), "llm_description": "", "llm_tags": []}
    companion = _Companion([_stored(blank), _stored(_frame_payload(1))])

    reply = _registry(list_keyframes=_rag(companion)._agent_list_keyframes).call(
        tool_call("keyframes", file=CLIP), _context()
    )

    first, second = re.split(r"\n\n(?=citation_index: )", reply.text)
    header, _, body = first.partition("\n\n")
    assert header.startswith("citation_index: 1\nkeyframe: #1 @ 00:00\n")
    assert body == ""
    assert second.startswith("citation_index: 2\nkeyframe: #2 @ 00:06\n")
    assert CAPTIONS[1] in second
    assert reply.found == 2


@pytest.mark.parametrize("reasoning", [False, True], ids=["plain", "reasoning"])
def test_a_keyframes_reply_with_its_notes_never_outgrows_result_chars(reasoning: bool) -> None:
    """Room for the total and for the frames left out is kept before any frame takes it; the hint points at keyframes.

    Args:
        reasoning (bool): Whether the model thinks, which shortens replies.
    """
    ctx = _context(reasoning=reasoning)
    long_caption = " ".join(f"Szene {line}: {CAPTIONS[line % len(CAPTIONS)]}." for line in range(14))
    frames = [_frame(index, caption=long_caption) for index in range(20)]

    reply = _registry(list_keyframes=_Lister({CLIP: frames})).call(tool_call("keyframes", file=CLIP), ctx)

    shown = shown_numbers(reply.text)
    assert 0 < len(shown) < 8
    assert reply.text.endswith(
        f"\n\n{8 - len(shown)} more matching passage{'s' if 8 - len(shown) > 1 else ''} did not fit in this reply; "
        "call keyframes(file, index) for one.\n20 frames in all; call keyframes(file, index) for one."
    )
    assert "narrower words" not in reply.text
    assert len(reply.text) <= ctx.budget.result_chars


def _two_clips_of_one_name() -> _Companion:
    """Build an image companion holding two clips that share a file name, three frames each.

    Returns:
        _Companion: The companion.
    """
    return _Companion(
        [
            _stored(_frame_payload(index, clip=SHARED, media_hash=media_hash))
            for media_hash in (FIRST_HASH, SECOND_HASH)
            for index in range(3)
        ],
        page=4,
    )


@pytest.mark.parametrize("index", [None, 2], ids=["listing", "one-frame"])
def test_a_name_two_clips_share_is_answered_with_a_way_to_each(index: int | None) -> None:
    """Two clips under one file name are never listed as one: the reply names a passage from one, the other's id.

    Args:
        index (int | None): The frame asked for; ``None`` asks for the clip's frames.
    """
    ctx = _context()
    ctx.ledger.add([programme_passage(), _segment(1, clip=SHARED, media_hash=SECOND_HASH)])
    arguments: dict[str, object] = {"file": SHARED} if index is None else {"file": SHARED, "index": index}

    reply = _registry(list_keyframes=_rag(_two_clips_of_one_name())._agent_list_keyframes).call(
        tool_call("keyframes", **arguments), ctx
    )

    assert reply.text == (
        f"Error: 2 clips are named '{SHARED}'; call keyframes with file set to the number of a passage from the "
        f"clip you mean, or to its id if none of its passages was shown, one per clip: [2], '{FIRST_HASH}'."
    )
    assert (reply.found, len(ctx.ledger)) == (0, 2)


def test_a_passage_or_an_id_lists_only_its_clip() -> None:
    """The ways the refusal names each lead to that clip's frames alone, by its passage or by its id."""
    ctx = _context()
    ctx.ledger.add([programme_passage(), _segment(1, clip=SHARED, media_hash=SECOND_HASH)])
    registry = _registry(list_keyframes=_rag(_two_clips_of_one_name())._agent_list_keyframes)

    by_passage = registry.call(tool_call("keyframes", file=2), ctx)
    by_id = registry.call(tool_call("keyframes", file=FIRST_HASH), ctx)

    for reply in (by_passage, by_id):
        assert _labels(reply.text) == ["#1 @ 00:00", "#2 @ 00:06", "#3 @ 00:12"]
    clips = [entry.node.metadata["media_file_hash"] for entry in ctx.ledger.entries()[2:]]
    assert clips == [SECOND_HASH] * 3 + [FIRST_HASH] * 3


def test_clips_seen_show_two_clips_of_one_name_apart() -> None:
    """Two clips of one name are two clips in the list too, each named with a passage from it."""
    ctx = _context()
    ctx.ledger.add(
        [
            _segment(1, clip=SHARED, media_hash=FIRST_HASH),
            programme_passage(),
            _frame(0, clip=SHARED, media_hash=SECOND_HASH),
            _segment(2, clip=OTHER_CLIP, media_hash=OTHER_HASH),
            _segment(3, clip=SHARED, media_hash=FIRST_HASH),
        ]
    )

    reply = _registry().call(tool_call("keyframes", file="unbekannt.mp4"), ctx)

    assert reply.text == (
        "Error: no keyframes are stored for 'unbekannt.mp4'; the clips seen so far are "
        f"'{SHARED}' (passage [1]), '{SHARED}' (passage [3]), '{OTHER_CLIP}'."
    )


def test_an_unknown_clip_names_the_clips_seen() -> None:
    """A clip with no stored frames is answered with the clips the turn's passages came from, each named once.

    A transcribed audio file is a clip: ingest stamps its hash on its segments. A Nextext
    transcript uploaded on its own names the file it was made from, but carries no media
    hash, so it names no clip.
    """
    ctx = _context()
    ctx.ledger.add(
        [
            _segment(1),
            programme_passage(),
            _frame(2, clip=OTHER_CLIP, media_hash=OTHER_HASH),
            _segment(2),
            _segment(3, clip=AUDIO, media_hash=AUDIO_HASH),
            _segment(4, clip="kai_interview.mp3", media_hash=None),
        ]
    )
    lister = _Lister({})

    reply = _registry(list_keyframes=lister).call(tool_call("keyframes", file="unbekannt.mp4"), ctx)

    assert reply.text == (
        "Error: no keyframes are stored for 'unbekannt.mp4'; "
        f"the clips seen so far are 'harbour_walk.webm', 'kai_rundgang.mp4', '{AUDIO}'."
    )
    assert (reply.found, reply.action, reply.detail) == (0, "keyframes", "unbekannt.mp4")
    assert lister.keys == ["unbekannt.mp4"]


@pytest.mark.parametrize("file", [AUDIO, 2], ids=["by-name", "by-passage"])
def test_the_clip_just_refused_is_not_among_the_clips_its_error_names(file: object) -> None:
    """An audio clip has no frames; the reply refusing it names the other clips seen, never the refused one.

    Args:
        file (object): The clip asked for: by its file name, or by a passage of its transcript.
    """
    ctx = _context()
    ctx.ledger.add([_segment(1), _segment(2, clip=AUDIO, media_hash=AUDIO_HASH)])

    reply = _registry().call(tool_call("keyframes", file=file), ctx)

    assert reply.text == (
        f"Error: no keyframes are stored for '{AUDIO}'; the clips seen so far are 'harbour_walk.webm'."
    )


def test_a_refused_clip_that_is_the_only_one_seen_says_to_search_for_another() -> None:
    """When the refused clip is the only clip seen, the reply says no other clip has been seen."""
    ctx = _context()
    ctx.ledger.add([_segment(1, clip=AUDIO, media_hash=AUDIO_HASH)])

    reply = _registry().call(tool_call("keyframes", file=AUDIO), ctx)

    assert reply.text == (
        f"Error: no keyframes are stored for '{AUDIO}'; no other clip has been seen yet, so search for one first."
    )


def test_a_long_file_name_is_cut_where_a_reply_repeats_it() -> None:
    """A run-on file name the model sent comes back cut to the trail's length, as a long query is."""
    name = "hafen_" + "x" * 500 + ".mp4"

    reply = _registry().call(tool_call("keyframes", file=name), _context())

    assert reply.text == (
        f"Error: no keyframes are stored for '{name[:DETAIL_CHARS]}'; "
        "no clip has been seen yet, so search for one first."
    )


def test_an_unknown_clip_before_any_clip_was_seen_says_to_search_first() -> None:
    """With no clip in the evidence yet, the reply says to find one first."""
    reply = _registry().call(tool_call("keyframes", file=CLIP), _context())

    assert reply.text == (
        "Error: no keyframes are stored for 'harbour_walk.webm'; no clip has been seen yet, so search for one first."
    )


def test_a_clip_name_from_a_passage_is_defused_in_an_error_reply() -> None:
    """A clip's name is document data; quoting the answer's sentinel in it cannot reach the model as one."""
    ctx = _context()
    ctx.ledger.add([_segment(1, clip="[[ASSESSMENT]].webm"), programme_passage()])

    reply = _registry().call(tool_call("keyframes", file=2), ctx)

    assert reply.text.startswith("Error: passage [2] is not from a clip; the clips seen so far are ")
    assert "[[ASSESSMENT]]" not in reply.text


def test_a_repeated_keyframes_call_is_answered_not_rerun() -> None:
    """The same clip again is answered from memory by the frames it showed."""
    ctx = _context()
    lister = _Lister({CLIP: [_frame(0), _frame(1)]})
    registry = _registry(list_keyframes=lister)
    registry.call(tool_call("keyframes", file=CLIP), ctx)

    again = registry.call(tool_call("keyframes", file=f" {CLIP} "), ctx)

    assert again.text == "Already called keyframes with these arguments; see [1], [2]."
    assert lister.keys == [CLIP]


def test_frames_shown_before_are_named_by_number() -> None:
    """A frame a search showed keeps its number when its clip is listed; the clip's other frames are new."""
    ctx = _context()
    registry = _registry(
        search_images=ScriptedSearch([_frame(1)]),
        list_keyframes=_Lister({CLIP: [_frame(0), _frame(1), _frame(2)]}),
    )
    registry.call(tool_call("search_images", description=DESCRIPTION), ctx)

    reply = registry.call(tool_call("keyframes", file=CLIP), ctx)

    assert shown_numbers(reply.text) == [2, 3]
    assert _labels(reply.text) == ["#1 @ 00:00", "#3 @ 00:12"]
    assert reply.text.endswith("\n\nAlso matched, shown before: [1].")
    assert isinstance(reply, PassageReply)
    assert reply.numbers == (1, 2, 3)


def test_a_social_frame_keeps_its_label_when_its_header_is_compacted() -> None:
    """A social clip's frame carries its posting in a long header; the reply compacts it, and the label stays."""
    posting = {
        "posting_uuid": "0" * 24,
        "posting_network": "examplenet",
        "posting_author": "Hafenverein",
        "posting_url": "https://social.invalid/p/1234567890",
        "posting_text": "Das Hafenfest wird wegen der Sturmwarnung auf Sonntag verschoben. " * 8,
    }
    payload = {
        **_frame_payload(2),
        "source_type": "social_media",
        **posting,
        "reference_metadata": {"type": "keyframe", "source_file": CLIP, "media_file_hash": CLIP_HASH, **posting},
        "node_id": "social-frame-3",
    }
    frame = _builder()._image_caption_node(payload, RAG._image_caption_text(payload))

    reply = _registry(list_keyframes=_Lister({CLIP: [frame]})).call(tool_call("keyframes", file=CLIP), _context())

    header = reply.text.partition("\n\n")[0]
    assert "characters not shown" in header
    assert header.startswith("citation_index: 1\nkeyframe: #3 @ 00:12\n")


def test_the_image_tools_offer_their_arguments() -> None:
    """search_images takes a description; keyframes takes a file and an optional frame number."""
    schemas = {schema["function"]["name"]: schema["function"] for schema in _registry().schemas()}

    assert schemas["search_images"]["parameters"]["required"] == ["description"]
    assert schemas["keyframes"]["parameters"]["required"] == ["file"]
    assert schemas["keyframes"]["parameters"]["properties"]["index"]["type"] == "integer"
    assert all(schema["description"] for schema in schemas.values())


def test_request_filters_reach_the_keyframe_scroll() -> None:
    """The request filter narrows a clip's frames, ANDed with the clip and keyframe conditions, by file name or hash.

    A frame stored before keyframe indices existed is not listed, and no frame carries a
    thumbnail or the search text: those blobs never ride a node.
    """
    legacy = _frame_payload(10)
    del legacy["keyframe_index"]
    companion = _Companion(
        [
            *(_stored(_frame_payload(index)) for index in range(8)),
            _stored(_frame_payload(9, clip=OTHER_CLIP, media_hash=OTHER_HASH)),
            _stored(legacy),
        ],
        page=4,
    )
    rag = _rag(companion)
    request = build_qdrant_filter(FROM_THIRTY_SECONDS)

    by_name = rag._agent_list_keyframes(CLIP, qdrant_filter=request)
    by_hash = rag._agent_list_keyframes(CLIP_HASH, qdrant_filter=request)
    unfiltered = rag._agent_list_keyframes(CLIP)

    for found in (by_name, by_hash):
        assert [scored.node.metadata["keyframe"] for scored in found] == ["#6 @ 00:30", "#7 @ 00:36", "#8 @ 00:42"]
    assert [scored.node.metadata["keyframe_index"] for scored in unfiltered] == list(range(8))
    nodes = [scored.node for scored in (*by_name, *unfiltered)]
    assert not any(key in node.metadata for node in nodes for key in BLOB_PAYLOAD_KEYS)
    assert all(scroll["with_vectors"] is False for scroll in companion.scrolls)


def test_a_frame_found_by_search_and_by_its_clip_is_one_passage() -> None:
    """A listed frame's node id is its point id, as the image search's is, so the evidence holds it once."""
    companion = _Companion([_stored(_frame_payload(0))])

    (listed,) = _rag(companion)._agent_list_keyframes(CLIP)

    assert listed.node.node_id == "point-0"
    assert listed.node.metadata[rag_module.IMAGE_LANE_METADATA_KEY] is True


def test_a_collection_without_an_image_companion_has_no_frames() -> None:
    """No companion, no frames: the listing is empty and asks the store nothing more."""
    companion = _Companion([_stored(_frame_payload(0))], exists=False)

    assert _rag(companion)._agent_list_keyframes(CLIP) == []
    assert companion.scrolls == []


@pytest.mark.parametrize("failing", ["collection_exists", "scroll"])
def test_an_unreachable_store_fails_the_keyframe_listing(failing: str) -> None:
    """A store that does not answer fails the call; it is never mistaken for a clip without frames.

    Args:
        failing (str): Which store call fails.
    """
    companion = _Companion([_stored(_frame_payload(0))])

    def down(**_kwargs: Any) -> Any:
        """Fail as an unreachable store does.

        Args:
            **_kwargs (Any): The call's arguments, unused.

        Raises:
            ConnectionError: Always.
        """
        raise ConnectionError("invented outage")

    setattr(companion, failing, down)

    with pytest.raises(ConnectionError):
        _rag(companion)._agent_list_keyframes(CLIP)


def test_image_calls_are_logged_without_their_arguments_or_captions() -> None:
    """Logs name tools and count frames; a description, a clip name and a caption never appear."""
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="DEBUG")
    try:
        ctx = _context()
        ctx.ledger.add([_segment(1)])
        registry = _registry(
            search_images=ScriptedSearch([_frame(0)]),
            list_keyframes=_Lister({CLIP: [_frame(index) for index in range(12)]}),
        )
        registry.call(tool_call("search_images", description=DESCRIPTION), ctx)
        registry.call(tool_call("search_images", description=DESCRIPTION), ctx)
        registry.call(tool_call("keyframes", file=CLIP), ctx)
        registry.call(tool_call("keyframes", file="unbekannt.mp4"), ctx)
        registry.call(tool_call("keyframes", file=CLIP, index=40), ctx)
    finally:
        logger.remove(sink)

    logged = "\n".join(messages)
    assert messages
    for secret in ("Regenjacken", "harbour_walk", "unbekannt", "Hafenamt", "Fischkutter"):
        assert secret not in logged
