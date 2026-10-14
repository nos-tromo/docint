"""Test doubles for assessment turns. Every passage and query here is invented."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any
from unittest.mock import MagicMock

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
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


def streaming_llm(tokens: Sequence[str]) -> MagicMock:
    """Build a chat model whose ``stream_chat`` streams ``tokens``, recording each call.

    Args:
        tokens (Sequence[str]): The answer, token by token.

    Returns:
        MagicMock: The model double; ``stream_chat.call_args`` holds the messages and kwargs.
    """
    llm = MagicMock()

    def _stream(messages: Sequence[ChatMessage], **kwargs: Any) -> Iterator[ChatResponse]:
        text = ""
        for token in tokens:
            text += token
            yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text), delta=token)

    llm.stream_chat.side_effect = _stream
    return llm
