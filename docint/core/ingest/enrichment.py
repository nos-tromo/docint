"""Attach NER and hate-speech results to nodes without changing what they embed."""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import BaseNode


def set_enrichment(node: BaseNode, updates: dict[str, Any]) -> None:
    """Merge enrichment results into ``node.metadata``, hidden from embed and prompt rendering.

    Enrichment runs after chunking has fixed a node's exclusion lists, so a key
    written plainly is rendered into the dense and sparse embedding input and
    the chat prompt. The results are analysis of the text, not part of it; they
    stay readable on the metadata dict for every consumer that reads it there.

    Args:
        node (BaseNode): The node to annotate.
        updates (dict[str, Any]): Enrichment keys and their values, e.g.
            ``entities``, ``relations`` or ``hate_speech``. Empty values are
            skipped, so an extraction that found nothing writes no key.
    """
    updates = {key: value for key, value in updates.items() if value}
    if not updates:
        return
    node.metadata = {**(node.metadata or {}), **updates}
    for attr in ("excluded_embed_metadata_keys", "excluded_llm_metadata_keys"):
        excluded = list(getattr(node, attr, None) or [])
        excluded.extend(key for key in updates if key not in excluded)
        setattr(node, attr, excluded)
