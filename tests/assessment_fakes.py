"""Test doubles for assessment turns. Every passage and query here is invented."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any
from unittest.mock import MagicMock

import httpx
import openai
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole, ToolCallBlock
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode

#: What the production LLM view lets through for these doubles.
_VISIBLE = frozenset({"citation_index", "filename", "page"})


def scored_node(node_id: str, text: str, **metadata: Any) -> NodeWithScore:
    """Build a retrieved passage.

    Args:
        node_id (str): The node id (the ledger's dedupe key).
        text (str): The passage text.
        **metadata (Any): Node metadata such as ``filename`` and ``page``.

    Returns:
        NodeWithScore: The passage with a neutral score.
    """
    return NodeWithScore(node=TextNode(id_=node_id, text=text, metadata=dict(metadata)), score=1.0)


POST_TEXT = "Harbour festival moved to Sunday because of the storm warning. Stalls open at ten, music from noon."


def social_post(number: int, text: str = POST_TEXT) -> NodeWithScore:
    """Build one pinned post of an invented social export, shaped like a dossier row.

    Its ``table`` and ``reference_metadata`` keys stay visible to the production
    LLM view, which renders them as a header of about 335 characters.

    Args:
        number (int): The post's position; it also varies its ids.
        text (str): The post's text.

    Returns:
        NodeWithScore: The post with a neutral score.
    """
    metadata = {
        "file_name": "example_profile/dossier.json",
        "filename": "example_profile/dossier.json",
        "file_hash": "f" * 64,
        "source": "table",
        "table": {"style": "postings", "row_index": number, "n_rows": 120},
        "reference_metadata": {
            "network": "examplenet",
            "type": "posting",
            "uuid": f"{number:024x}",
            "timestamp": "2026-03-14T09:30:00Z",
            "author": "Harbour Events",
            "author_id": "1234567890",
            "text_id": f"{9876543210 + number}",
        },
    }
    return NodeWithScore(node=TextNode(id_=f"post-{number}", text=text, metadata=metadata), score=1.0)


#: About a thousand characters, under the 1 KiB clamp the production LLM view puts on each metadata leaf.
LONG_LEAF = ("Harbour Events community calendar entry for the spring festival. " * 16)[:990]


def heavy_post(number: int) -> NodeWithScore:
    """Build a short post whose visible metadata renders a header of about 3 KB.

    Each ``reference_metadata`` leaf stays under the per-leaf clamp, so nothing
    bounds the header but the number of leaves.

    Args:
        number (int): The post's position; it also varies its id and text.

    Returns:
        NodeWithScore: The post with a neutral score.
    """
    metadata = {
        "filename": "example_profile/dossier.json",
        "reference_metadata": {
            "network": "examplenet",
            "author": LONG_LEAF,
            "text_id": f"{9876543210 + number}",
            "description": LONG_LEAF,
            "summary": LONG_LEAF,
        },
    }
    text = f"Harbour post {number}: the stalls open at ten."
    return NodeWithScore(node=TextNode(id_=f"heavy-{number}", text=text, metadata=metadata), score=1.0)


def llm_view(node: BaseNode) -> BaseNode:
    """Mirror the production LLM view: hide every key but the whitelisted ones.

    Args:
        node (BaseNode): A ledger node.

    Returns:
        BaseNode: A clone whose LLM rendering shows only the whitelisted keys.
    """
    clone = node.model_copy(deep=True)
    clone.excluded_llm_metadata_keys = [key for key in clone.metadata if key not in _VISIBLE]
    return clone


def bad_request(message: str) -> openai.BadRequestError:
    """Build the error an OpenAI-compatible endpoint raises on HTTP 400.

    Args:
        message (str): The error message.

    Returns:
        openai.BadRequestError: The error.
    """
    request = httpx.Request("POST", "http://inference.invalid/v1/chat/completions")
    return openai.BadRequestError(message, response=httpx.Response(400, request=request), body=None)


def streaming_llm(tokens: Sequence[str], *, failures: Sequence[Exception] = ()) -> MagicMock:
    """Build a chat model whose ``stream_chat`` streams ``tokens``, recording each call.

    Args:
        tokens (Sequence[str]): The answer, token by token.
        failures (Sequence[Exception]): Errors the first calls raise, one per call,
            on their first pull — where an endpoint's HTTP error surfaces.

    Returns:
        MagicMock: The model double; ``stream_chat.call_args`` holds the messages and kwargs.
    """
    llm = MagicMock()
    pending = list(failures)

    def _stream(messages: Sequence[ChatMessage], **kwargs: Any) -> Iterator[ChatResponse]:
        """Raise the next scripted failure on the first pull, or stream the answer.

        Args:
            messages (Sequence[ChatMessage]): The prompt, unused.
            **kwargs (Any): Call options, unused.

        Yields:
            ChatResponse: One response per token, the text so far included.

        Raises:
            Exception: The next scripted failure, while any remain.
        """
        if pending:
            raise pending.pop(0)
        text = ""
        for token in tokens:
            text += token
            yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text), delta=token)

    llm.stream_chat.side_effect = _stream
    return llm


def tool_llm(script: Sequence[ToolSelection | str | Exception]) -> MagicMock:
    """Build a function-calling model double that replays one scripted reply per call.

    Args:
        script (Sequence[ToolSelection | str | Exception]): Per call: a tool
            call to return, plain text to answer with, or an error to raise.
            After the script runs out, the model calls ``finish``.

    Returns:
        MagicMock: The model double; ``chat_with_tools.call_args_list`` records every call.
    """
    replies = list(script)
    llm = MagicMock()

    def _chat_with_tools(tools: Sequence[Any], **kwargs: Any) -> ChatResponse:
        """Replay the next scripted reply as a chat response, or raise it.

        Args:
            tools (Sequence[Any]): The tools offered, unused.
            **kwargs (Any): The call's other arguments, unused.

        Returns:
            ChatResponse: A tool call, or plain text.

        Raises:
            Exception: The scripted error, when that is the next reply.
        """
        reply: ToolSelection | str | Exception = (
            replies.pop(0) if replies else ToolSelection(tool_id="", tool_name="finish", tool_kwargs={})
        )
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, ToolSelection):
            block = ToolCallBlock(
                tool_call_id=reply.tool_id or None, tool_name=reply.tool_name, tool_kwargs=reply.tool_kwargs
            )
            return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, blocks=[block]))
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=reply))

    def _tool_calls(response: ChatResponse, error_on_no_tool_call: bool = True, **kwargs: Any) -> list[ToolSelection]:
        """Read the tool calls out of a response, as the real model does.

        Args:
            response (ChatResponse): The response to read.
            error_on_no_tool_call (bool): Accepted for signature parity, unused.
            **kwargs (Any): Other arguments, unused.

        Returns:
            list[ToolSelection]: The calls; empty for a plain-text response.
        """
        return [
            ToolSelection(
                tool_id=block.tool_call_id or "",
                tool_name=block.tool_name,
                tool_kwargs=block.tool_kwargs if isinstance(block.tool_kwargs, dict) else {},
            )
            for block in response.message.blocks
            if isinstance(block, ToolCallBlock)
        ]

    llm.chat_with_tools.side_effect = _chat_with_tools
    llm.get_tool_calls_from_response.side_effect = _tool_calls
    return llm


def search_call(query: str, call_id: str = "call_x") -> ToolSelection:
    """Build a scripted ``search`` tool call.

    Args:
        query (str): The search query.
        call_id (str): The tool-call id the model returns.

    Returns:
        ToolSelection: The scripted call.
    """
    return ToolSelection(tool_id=call_id, tool_name="search", tool_kwargs={"query": query})
