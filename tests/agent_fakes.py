"""Test doubles for agent turns: model rounds, the tools' searches and context, and the store a keyword match reads.

Every message, query, passage and tool call here is invented.
"""

from __future__ import annotations

import base64
import json
import re
import types
from collections.abc import Callable, Iterator, Sequence
from typing import Any
from unittest.mock import MagicMock

import httpx
import openai
import pytest
from assessment_fakes import llm_view, scored_node
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.schema import BaseNode, NodeWithScore

from docint.agents.assessment.ledger import EvidenceLedger
from docint.agents.turn.budget import TurnBudget
from docint.agents.turn.keyword import KeywordMatcher, KeywordResult
from docint.agents.turn.loop import AgentLoop
from docint.agents.turn.stream import ToolCall
from docint.agents.turn.tools import SearchFn, ToolContext, ToolRegistry, ToolReply
from docint.agents.turn.toolsets import toolset_for
from docint.core.search.fields import SEARCH_FIELDS

ScriptedRound = Sequence[dict[str, Any] | Exception] | Exception
"""One scripted model call: the raw chunks it streams, or the error its first pull raises.

An error among the chunks is raised where it stands, as a stream that breaks off mid-reply.
"""

PROGRAMME = (
    "Das Hafenfest wurde wegen der Sturmwarnung auf Sonntag verschoben. Die Stände öffnen um zehn Uhr, Musik ab Mittag."
)

#: Where the invented chat endpoint lives; nothing is ever sent there.
INVENTED_ENDPOINT = "http://inference.invalid/v1/chat/completions"
#: The ledger cap of a test turn that sets none; a production turn sizes its own (``AGENT_LEDGER_CAP``).
LEDGER_CAP = 24


def status_error(cls: type[openai.APIStatusError], status: int, message: str) -> openai.APIStatusError:
    """Build the error an OpenAI-compatible endpoint raises on an HTTP error status.

    Args:
        cls (type[openai.APIStatusError]): The SDK's error class for the status.
        status (int): The HTTP status.
        message (str): The error message.

    Returns:
        openai.APIStatusError: The error.
    """
    request = httpx.Request("POST", INVENTED_ENDPOINT)
    return cls(message, response=httpx.Response(status, request=request), body=None)


def timed_out() -> openai.APITimeoutError:
    """Build the error a call raises when the endpoint does not answer in time.

    Returns:
        openai.APITimeoutError: The error.
    """
    return openai.APITimeoutError(request=httpx.Request("POST", INVENTED_ENDPOINT))


def unreachable() -> openai.APIConnectionError:
    """Build the error a call raises when the endpoint cannot be reached.

    Returns:
        openai.APIConnectionError: The error.
    """
    return openai.APIConnectionError(request=httpx.Request("POST", INVENTED_ENDPOINT))


def tool_call_delta(
    *,
    index: int | None = None,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> dict[str, Any]:
    """Build one entry of a chunk's ``delta.tool_calls`` as an endpoint sends it.

    Args:
        index (int | None): The call's position in the reply; left out when ``None``.
        call_id (str | None): The call's id; left out when ``None``.
        name (str | None): The tool's name, or a fragment of it; left out when ``None``.
        arguments (str | None): The arguments' JSON text, or a fragment of it; left out when ``None``.

    Returns:
        dict[str, Any]: The entry, carrying only the keys given.
    """
    entry: dict[str, Any] = {}
    if index is not None:
        entry["index"] = index
    if call_id is not None:
        entry["id"] = call_id
        entry["type"] = "function"
    function = {key: value for key, value in (("name", name), ("arguments", arguments)) if value is not None}
    if function:
        entry["function"] = function
    return entry


def chunk(
    content: str | None = None,
    *,
    tool_calls: Sequence[dict[str, Any]] = (),
    finish_reason: str | None = None,
    reasoning_content: str | None = None,
) -> dict[str, Any]:
    """Build one streamed chat-completion chunk as an OpenAI-compatible endpoint sends it.

    Args:
        content (str | None): Answer text in this chunk; left out when ``None``.
        tool_calls (Sequence[dict[str, Any]]): Tool-call entries (see :func:`tool_call_delta`).
        finish_reason (str | None): Why the reply ended, on its last chunk.
        reasoning_content (str | None): Reasoning sent in the separate field a reasoning parser fills.

    Returns:
        dict[str, Any]: The chunk in wire shape.
    """
    delta: dict[str, Any] = {"role": "assistant"}
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    if tool_calls:
        delta["tool_calls"] = list(tool_calls)
    return {
        "id": "chatcmpl-invented",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "invented-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def text_chunks(text: str, size: int) -> list[dict[str, Any]]:
    """Split answer text into chunks of ``size`` characters, as a model streams it.

    Args:
        text (str): The text.
        size (int): Characters per chunk; the last one may be shorter.

    Returns:
        list[dict[str, Any]]: One chunk per piece.
    """
    return [chunk(text[start : start + size]) for start in range(0, len(text), size)]


def streaming_tool_llm(script: Sequence[ScriptedRound]) -> MagicMock:
    """Build a chat model whose ``stream_chat`` replays one scripted round per call.

    Each raw chunk becomes a ``ChatResponse`` the way llama-index's OpenAI client builds
    one: the chunk on ``raw``, its content on ``delta`` and a separate reasoning field on
    ``additional_kwargs["thinking_delta"]``, never on ``delta``.

    Args:
        script (Sequence[ScriptedRound]): Per call, the chunks to stream or the error to raise.

    Returns:
        MagicMock: The model double; ``stream_chat.call_args_list`` records every call.
    """
    rounds = list(script)
    llm = MagicMock()

    def _stream(messages: Sequence[ChatMessage], **kwargs: Any) -> Iterator[ChatResponse]:
        """Raise the next scripted error on the first pull, or stream the next round.

        Args:
            messages (Sequence[ChatMessage]): The prompt, unused.
            **kwargs (Any): Call options, unused.

        Yields:
            ChatResponse: One response per chunk.

        Raises:
            AssertionError: When the model is called more often than the script has rounds.
            Exception: The scripted error, when that is the round or the round reaches it.
        """
        if not rounds:
            raise AssertionError("The model was called more often than its script has rounds.")
        current = rounds.pop(0)
        if isinstance(current, Exception):
            raise current
        content = ""
        for raw in current:
            if isinstance(raw, Exception):
                raise raw
            delta: dict[str, Any] = raw["choices"][0]["delta"] if raw["choices"] else {}
            piece = delta.get("content") or ""
            content += piece
            thinking = delta.get("reasoning_content")
            yield ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
                delta=piece,
                raw=raw,
                additional_kwargs={"thinking_delta": thinking} if thinking else {},
            )

    llm.stream_chat.side_effect = _stream
    return llm


#: What the plan budgets for the tool schemas of an 8k turn; budgets built here take it as given.
#: test_agent_toolsets measures the real schemas against the window.
PLANNED_SCHEMA_TOKENS = 630
#: What the plan budgets for the rendered system prompt of an 8k turn, at 3.5 characters per token.
PLANNED_SYSTEM_TOKENS = 962


def turn_budget(*, reasoning: bool = False) -> TurnBudget:
    """Build one turn's budget at an 8k window; the tools read only its reply and passage sizes.

    Args:
        reasoning (bool): Whether the model thinks before it answers.

    Returns:
        TurnBudget: The budget.
    """
    return TurnBudget(
        ctx_window=8192,
        reasoning=reasoning,
        char_token_ratio=3.5,
        # A sizing fake for tool replies, deliberately not the planned prompt size (PLANNED_SYSTEM_TOKENS).
        system_tokens=900,
        tool_schema_tokens=PLANNED_SCHEMA_TOKENS,
        history_tokens=1300,
        message_tokens=300,
    )


def tool_context(
    *,
    reasoning: bool = False,
    cap: int = LEDGER_CAP,
    view: Callable[[BaseNode], BaseNode] = llm_view,
) -> ToolContext:
    """Build the context of one turn whose evidence is still empty.

    Args:
        reasoning (bool): Whether the model thinks before it answers, which shortens replies.
        cap (int): The most passages the turn's ledger admits.
        view (Callable[[BaseNode], BaseNode]): The LLM view; the whitelist double by default.

    Returns:
        ToolContext: The context.
    """
    return ToolContext(ledger=EvidenceLedger(cap=cap), budget=turn_budget(reasoning=reasoning), llm_view=view)


class ScriptedSearch:
    """A search double: each run answers with its next scripted result and records what it was asked."""

    def __init__(self, *results: list[NodeWithScore]) -> None:
        """Hold the results, one per run.

        Args:
            *results (list[NodeWithScore]): What each run returns, in order.
        """
        self._results = list(results)
        self.queries: list[str] = []

    def __call__(self, query: str) -> list[NodeWithScore]:
        """Run one search.

        Args:
            query (str): What the tool searched for.

        Returns:
            list[NodeWithScore]: The next scripted result.

        Raises:
            AssertionError: When the search runs more often than it has results.
        """
        self.queries.append(query)
        if not self._results:
            raise AssertionError("The search ran more often than its script has results.")
        return self._results.pop(0)


def tool_call(name: str, **arguments: Any) -> ToolCall:
    """Build one tool call as a model round sends it.

    Args:
        name (str): The tool's name.
        **arguments (Any): The arguments, sent as JSON text.

    Returns:
        ToolCall: The call.
    """
    return ToolCall(id="call_0", name=name, arguments=json.dumps(arguments, ensure_ascii=False))


def shown_numbers(text: str) -> list[int]:
    """Return the numbers a reply shows passages under, in reply order.

    Args:
        text (str): The reply.

    Returns:
        list[int]: Each shown passage's number.
    """
    return [int(number) for number in re.findall(r"^citation_index: (\d+)$", text, flags=re.MULTILINE)]


def programme_passage() -> NodeWithScore:
    """Build the invented programme passage, a page of a document and of no clip.

    Returns:
        NodeWithScore: The passage.
    """
    return scored_node("programme", PROGRAMME, filename="hafenfest_programm.pdf", page=1)


#: One round of keyword calls; the first names the author, and the other two name no field, so they match the text.
KEYWORD_CALLS = (
    ("call_a", "keyword", {"words": "Hafenfest", "field": "author"}),
    ("call_b", "keyword", {"words": "Regatta"}),
    ("call_c", "keyword", {"words": "Feuerwerk"}),
)
#: The steps those calls leave in a turn's trail: a count the index only partly covers, a zero, and a failed count.
KEYWORD_STEPS: list[dict[str, Any]] = [
    {"action": "keyword", "detail": "Hafenfest", "found": 1, "field": "author", "matches": 3, "unchecked": 2},
    {"action": "keyword", "detail": "Regatta", "found": 0, "field": "text", "matches": 0},
    {"action": "keyword", "detail": "Feuerwerk", "found": 0, "field": "text", "matches": None},
]


def keyword_match() -> KeywordMatcher:
    """Script the match :data:`KEYWORD_CALLS` reach, so that they leave :data:`KEYWORD_STEPS`.

    Returns:
        KeywordMatcher: A match offering the text and the author, that accepts every call and answers it by its words.
    """
    results = {
        "Hafenfest": KeywordResult(nodes=[programme_passage()], matches=3, unchecked=2, keywords=("hafenfest",)),
        "Regatta": KeywordResult(nodes=[], matches=0, unchecked=0, keywords=("regatta",)),
        "Feuerwerk": KeywordResult(nodes=[], matches=None, unchecked=0, keywords=("feuerwerk",)),
    }

    def _check(words: str, field: str) -> str | None:
        """Accept the call.

        Args:
            words (str): The words.
            field (str): The field.

        Returns:
            str | None: No reason to refuse.
        """
        return None

    def _run(words: str, field: str) -> KeywordResult:
        """Answer one run by its words, whatever order the round's calls run in.

        Args:
            words (str): The words.
            field (str): The field.

        Returns:
            KeywordResult: The scripted result.
        """
        return results[words]

    return KeywordMatcher(fields=("text", "author"), check=_check, run=_run)


def two_phase_round(registry: ToolRegistry, calls: Sequence[ToolCall], ctx: ToolContext) -> list[ToolReply]:
    """Answer calls as a turn with parallel calls does: all prepared in order, all fetched, then all finished in order.

    Args:
        registry (ToolRegistry): The turn's tools.
        calls (Sequence[ToolCall]): One round's calls, in the order the model made them.
        ctx (ToolContext): The turn's context.

    Returns:
        list[ToolReply]: One reply per call, in call order.
    """
    prepared = [registry.prepare(call, ctx) for call in calls]
    fetched = [None if ready.fetch is None else ready.fetch() for ready in prepared]
    return [registry.finish(ready, result, ctx) for ready, result in zip(prepared, fetched, strict=True)]


#: The invented clip the frame passages come from.
CLIP = "harbour_walk.webm"
#: What the invented clip's frames show, one caption per frame in turn.
FRAME_CAPTIONS = (
    "Zwei Personen in Regenjacken am Hafen, bedeckter Himmel",
    "Person A hält einen Schirm vor dem Hafenamt",
    "Ein Fischkutter legt am Kai an, Möwen über dem Wasser",
    "Person A und Person B gehen an den Ständen vorbei",
    "Regen auf dem Kopfsteinpflaster vor der Hafenkneipe",
    "Möwen auf dem Geländer am Anleger, Nebel über dem Wasser",
)


def frame_passage(index: int) -> NodeWithScore:
    """Build one frame of the invented clip as the image tools return it: its caption, carrying its ``image_id``.

    Args:
        index (int): The frame's place in the clip, from zero; it picks the caption.

    Returns:
        NodeWithScore: The frame; its node id is the companion point id its thumbnail is fetched by.
    """
    return scored_node(
        f"frame-{index}",
        FRAME_CAPTIONS[index % len(FRAME_CAPTIONS)],
        filename=CLIP,
        image_id=f"harbour-walk-image-{index}",
        source_file=CLIP,
        keyframe_index=index,
    )


def thumbnail(index: int) -> tuple[str, str]:
    """Return a frame's invented stored thumbnail.

    Args:
        index (int): The frame's place in the clip.

    Returns:
        tuple[str, str]: Its mime type and its bytes in base64.
    """
    return "image/jpeg", base64.b64encode(f"invented thumbnail of frame {index}".encode()).decode()


def thumbnail_url(index: int) -> str:
    """Return the data URL a frame's invented thumbnail is shown to the model by.

    Args:
        index (int): The frame's place in the clip.

    Returns:
        str: The URL, as a picture message carries it.
    """
    mime, data = thumbnail(index)
    return f"data:{mime};base64,{data}"


class ThumbnailStore:
    """A thumbnail fetcher double: answers by point id with the frames it stores, records every fetch, and can fail."""

    def __init__(self, *indexes: int, failure: Exception | None = None) -> None:
        """Store the thumbnails of the frames given.

        Args:
            *indexes (int): The frames whose thumbnails are stored.
            failure (Exception | None): What every fetch raises, as a store outage would; ``None`` to answer.
        """
        self._stored = {f"frame-{index}": thumbnail(index) for index in indexes}
        self._failure = failure
        self.fetches: list[list[str]] = []

    def __call__(self, point_ids: Sequence[str]) -> dict[str, tuple[str, str]]:
        """Fetch the stored thumbnails of the points asked for.

        Args:
            point_ids (Sequence[str]): The points, by id.

        Returns:
            dict[str, tuple[str, str]]: ``point_id -> (mime, base64)`` for each point with a stored thumbnail.

        Raises:
            Exception: The scripted failure, when there is one.
        """
        self.fetches.append(list(point_ids))
        if self._failure is not None:
            raise self._failure
        return {point_id: self._stored[point_id] for point_id in point_ids if point_id in self._stored}


#: What a scripted loop tells its model; nothing reads them back.
LOOP_SYSTEM_PROMPT = "Beantworte Fragen zur Sammlung mit den Werkzeugen und belege jede Aussage mit [n]."
LOOP_FINAL_NOTE = "Antworte jetzt aus den Passagen oben; rufe keine Werkzeuge mehr auf."
LOOP_SCOPE_NOTE = "Der Nutzer hat diese Passagen ausgewählt; antworte nur aus ihnen."
LOOP_LEGEND = "Die Bilder unten sind die Quellen:\n{image_legend}"


def answer_round(text: str, *, reasoning: str | None = None) -> list[dict[str, Any]]:
    """Script a round that answers: its reasoning first, if any, then the text in pieces.

    Args:
        text (str): The answer.
        reasoning (str | None): Reasoning sent in the separate field a reasoning parser fills.

    Returns:
        list[dict[str, Any]]: The round's chunks, the last one ending it.
    """
    head = [chunk(reasoning_content=reasoning)] if reasoning else []
    return [*head, *text_chunks(text, 40), chunk(finish_reason="stop")]


def call_round(*calls: tuple[str, str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Script a round that ends in tool calls, every call whole in one chunk.

    Args:
        *calls (tuple[str, str, dict[str, Any]]): Per call its id, the tool's name and its arguments.

    Returns:
        list[dict[str, Any]]: The round's one chunk.
    """
    deltas = [
        tool_call_delta(index=index, call_id=call_id, name=name, arguments=json.dumps(arguments, ensure_ascii=False))
        for index, (call_id, name, arguments) in enumerate(calls)
    ]
    return [chunk(tool_calls=deltas, finish_reason="tool_calls")]


class GrantedSlots:
    """A model-slot double: the first ``grants`` acquisitions succeed at once, every later one fails at once."""

    def __init__(self, grants: int) -> None:
        """Hold the slots to grant.

        Args:
            grants (int): How many acquisitions succeed.
        """
        self.grants = grants

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        """Grant a slot while any is left, without waiting.

        Args:
            blocking (bool): Unused.
            timeout (float | None): Unused.

        Returns:
            bool: Whether a slot was granted.
        """
        if self.grants <= 0:
            return False
        self.grants -= 1
        return True

    def release(self) -> None:
        """Take a slot back; it is never granted again."""


def scripted_loop(
    llm: Any,
    *,
    target: str = "all",
    search: SearchFn | None = None,
    search_images: SearchFn | None = None,
    keyword: KeywordMatcher | None = None,
    view: Callable[[BaseNode], BaseNode] = llm_view,
    **overrides: Any,
) -> AgentLoop:
    """Build an agent loop at an 8k window over the target's tools, its model a scripted double.

    Args:
        llm (Any): The model double, as :func:`streaming_tool_llm` builds it.
        target (str): The retrieval target whose tools the loop offers.
        search (SearchFn | None): The text search; by default one that fails if it runs.
        search_images (SearchFn | None): The image search; by default one that fails if it runs.
        keyword (KeywordMatcher | None): The keyword match to offer; by default none, so no keyword tool.
        view (Callable[[BaseNode], BaseNode]): The LLM view; the whitelist double by default.
        **overrides (Any): Further loop arguments, such as limits, model slots or a thumbnail fetcher.

    Returns:
        AgentLoop: The loop.
    """
    registry = toolset_for(
        target,
        search=search or ScriptedSearch(),
        search_images=search_images or ScriptedSearch(),
        list_keyframes=ScriptedSearch(),
        keyword=keyword,
    )
    kwargs: dict[str, Any] = {
        "budget": turn_budget(),
        "system_prompt": LOOP_SYSTEM_PROMPT,
        "final_note": LOOP_FINAL_NOTE,
        "max_rounds": 4,
        "time_budget_s": 120.0,
        "max_calls_per_round": 4,
        "char_token_ratio": 3.5,
        "llm_view": view,
        "scope_note": LOOP_SCOPE_NOTE,
        "legend_template": LOOP_LEGEND,
    }
    kwargs.update(overrides)
    return AgentLoop(llm=llm, registry=registry, **kwargs)


def ready_schema(*fields: str) -> dict[str, str]:
    """Return a payload schema in which every key of these keyword fields carries the index its matcher needs.

    Args:
        *fields (str): Keys of :data:`~docint.core.search.fields.SEARCH_FIELDS`.

    Returns:
        dict[str, str]: Index kind per payload key, as :class:`KeywordClient` serves a schema.
    """
    return {key: kind for name in fields for key, kind in SEARCH_FIELDS[name].indexed_keys()}


class KeywordClient:
    """A Qdrant double for keyword matches: payload schemas, counts, and each count, lookup and schema read made."""

    def __init__(self, schemas: dict[str, dict[str, str]], counts: dict[str, int | Exception] | None = None) -> None:
        """Serve these schemas and counts.

        Args:
            schemas (dict[str, dict[str, str]]): Index kind per payload key, per existing collection.
            counts (dict[str, int | Exception] | None): The count, or the error, per collection; 0 otherwise.
        """
        self.schemas = schemas
        self.counts = dict(counts or {})
        self.counted: list[tuple[str, Any]] = []
        self.exists_asked: list[str] = []
        self.schemas_read: list[str] = []
        self.deleted: list[str] = []

    def get_collection(self, collection_name: str) -> Any:
        """Return a collection's payload schema, recording the read.

        Args:
            collection_name (str): The collection.

        Returns:
            Any: An info object carrying ``payload_schema``.
        """
        self.schemas_read.append(collection_name)
        schema = self.schemas.get(collection_name, {})
        return types.SimpleNamespace(
            payload_schema={key: types.SimpleNamespace(data_type=kind, params=None) for key, kind in schema.items()}
        )

    def collection_exists(self, collection_name: str) -> bool:
        """Report whether a collection exists, recording the question.

        Args:
            collection_name (str): The collection.

        Returns:
            bool: Whether a schema was given for it.
        """
        self.exists_asked.append(collection_name)
        return collection_name in self.schemas

    def delete_collection(self, collection_name: str) -> bool:
        """Accept a collection's deletion, recording it; the schemas are left as they are.

        Args:
            collection_name (str): The collection.

        Returns:
            bool: Always ``True``.
        """
        self.deleted.append(collection_name)
        return True

    def count(self, collection_name: str, count_filter: Any = None, exact: bool = True) -> Any:
        """Count, recording the filter, or fail as scripted.

        Args:
            collection_name (str): The collection.
            count_filter (Any): The filter.
            exact (bool): Whether the count is exact.

        Returns:
            Any: An object carrying ``count``.

        Raises:
            Exception: The scripted error.
        """
        self.counted.append((collection_name, count_filter))
        value = self.counts.get(collection_name, 0)
        if isinstance(value, Exception):
            raise value
        return types.SimpleNamespace(count=value)


class CoverageLanes(dict[str, tuple[int, int]]):
    """The lanes a coverage double reports, as ``(total, with_search_text)``, and each read made of them.

    Attributes:
        reads (list[str]): The collection of each coverage read, in order.
    """

    def __init__(self, lanes: dict[str, tuple[int, int]]) -> None:
        """Report these lanes, with no read made yet.

        Args:
            lanes (dict[str, tuple[int, int]]): The lanes counted in full; a lane left out was not counted.
        """
        super().__init__(lanes)
        self.reads: list[str] = []


def keyword_coverage(monkeypatch: pytest.MonkeyPatch, lanes: dict[str, tuple[int, int]]) -> CoverageLanes:
    """Report the keyword index's coverage per lane, as RAG reads it, as ``(total, with_search_text)``.

    Each read reports the lanes as they stand in the dict returned, so a test can change the coverage between reads,
    and notes its collection in that dict's ``reads``.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        lanes (dict[str, tuple[int, int]]): The lanes counted in full; a lane left out was not counted.

    Returns:
        CoverageLanes: The lanes reported, a copy of ``lanes``; change it to change the next read.
    """
    reported = CoverageLanes(lanes)

    def _status(client: Any, collection: str, **kwargs: Any) -> dict[str, Any]:
        """Report the lanes as they stand now, noting the read.

        Args:
            client (Any): The store, unused.
            collection (str): The collection.
            **kwargs (Any): The read's options, unused.

        Returns:
            dict[str, Any]: The status, with every lane reported counted in full.
        """
        reported.reads.append(collection)
        return {
            "indexed": True,
            "lanes": {
                name: {"total": total, "with_search_text": covered} for name, (total, covered) in reported.items()
            },
        }

    monkeypatch.setattr("docint.core.rag.search_index_status", _status)
    return reported
