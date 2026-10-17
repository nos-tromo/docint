"""Test doubles for agent turns: model rounds, and the tools' searches and context.

Every message, query, passage and tool call here is invented.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any
from unittest.mock import MagicMock

from assessment_fakes import llm_view, scored_node
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.schema import BaseNode, NodeWithScore

from docint.agents.assessment.ledger import MAX_LEDGER_ENTRIES, EvidenceLedger
from docint.agents.turn.budget import TurnBudget
from docint.agents.turn.stream import ToolCall
from docint.agents.turn.tools import ToolContext, ToolRegistry, ToolReply

ScriptedRound = Sequence[dict[str, Any]] | Exception
"""One scripted model call: the raw chunks it streams, or the error its first pull raises."""

PROGRAMME = (
    "Das Hafenfest wurde wegen der Sturmwarnung auf Sonntag verschoben. Die Stände öffnen um zehn Uhr, Musik ab Mittag."
)


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
            Exception: The scripted error, when that is the round.
        """
        if not rounds:
            raise AssertionError("The model was called more often than its script has rounds.")
        current = rounds.pop(0)
        if isinstance(current, Exception):
            raise current
        content = ""
        for raw in current:
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
        system_tokens=900,
        tool_schema_tokens=PLANNED_SCHEMA_TOKENS,
        history_tokens=1300,
        message_tokens=300,
    )


def tool_context(
    *,
    reasoning: bool = False,
    cap: int = MAX_LEDGER_ENTRIES,
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
