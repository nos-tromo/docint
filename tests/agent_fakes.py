"""Test doubles for agent turns. Every message, query and tool call here is invented."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any
from unittest.mock import MagicMock

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

ScriptedRound = Sequence[dict[str, Any]] | Exception
"""One scripted model call: the raw chunks it streams, or the error its first pull raises."""


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
