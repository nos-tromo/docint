"""Test doubles for assessment turns. Every passage and query here is invented."""

from __future__ import annotations

from typing import Any

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
