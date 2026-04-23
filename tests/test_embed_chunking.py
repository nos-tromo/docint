"""RED tests for the pre-embed re-chunking contract.

These tests pin the public contract of ``docint.utils.embed_chunking``
before the implementation is written. They MUST fail on ``HEAD``
(the module does not yet exist) — that red state is the TDD signal
for the implementer to satisfy the contract.
"""

from __future__ import annotations

import pytest
from llama_index.core.schema import BaseNode, TextNode


def test_estimate_tokens_conservative_for_mixed_language() -> None:
    """``estimate_tokens`` should stay conservative for multilingual text.

    The fix-branch contract: the char/token ratio default of ``3.5`` is
    picked so that mixed English + CJK content is never *under*-counted.
    A safe estimator predicts at most ``len(text) / 3.5`` tokens (plus a
    small slack for rounding). An estimator that over-counts on this
    input is still safe — the concern is never *under*-counting.
    """
    from docint.utils.embed_chunking import estimate_tokens

    mixed_text = "Hello 你好 " * 100
    estimated = estimate_tokens(mixed_text)
    conservative_upper_bound = int(len(mixed_text) / 3.5) + 4

    assert estimated <= conservative_upper_bound
    assert estimated > 0


def test_fits_budget_uses_safety_margin() -> None:
    """``fits_budget`` must respect ``safety_margin`` fraction of ``budget_tokens``.

    The safety margin reserves a fraction of the embedding context
    window for BOS/EOS tokens and estimator slop. A text whose
    estimated token count equals ``int(budget * safety_margin)`` fits;
    one whose estimate exceeds that ceiling does not.
    """
    from docint.utils.embed_chunking import effective_budget, fits_budget

    budget_tokens = 8192
    char_token_ratio = 3.5
    safety_margin = 0.95
    effective = effective_budget(budget_tokens, safety_margin)
    at_limit_chars = int(effective * char_token_ratio)
    at_limit_text = "x" * at_limit_chars
    over_limit_text = at_limit_text + "y" * int(char_token_ratio + 1)

    assert (
        fits_budget(
            at_limit_text,
            budget_tokens=budget_tokens,
            char_token_ratio=char_token_ratio,
            safety_margin=safety_margin,
        )
        is True
    )
    assert (
        fits_budget(
            over_limit_text,
            budget_tokens=budget_tokens,
            char_token_ratio=char_token_ratio,
            safety_margin=safety_margin,
        )
        is False
    )


def test_resplit_passes_through_small_nodes() -> None:
    """Within-budget nodes must pass through both output lists unchanged.

    No re-chunking should happen when a node already fits; the same
    ``node_id`` and metadata appear in both the vector and docstore
    output lists, and no ``embedding_split`` marker is added.
    """
    from docint.utils.embed_chunking import resplit_nodes_for_embedding

    small_node = TextNode(
        text="short text",
        id_="small-1",
        metadata={"chunk_id": "cid", "source_ref": "s"},
    )

    vector_nodes, docstore_nodes = resplit_nodes_for_embedding(
        [small_node], budget_tokens=8192
    )

    assert vector_nodes == [small_node]
    assert docstore_nodes == [small_node]
    assert "embedding_split" not in small_node.metadata


def test_resplit_splits_oversize_into_sub_nodes() -> None:
    """Oversize nodes must be split into budget-conforming sub-nodes.

    The parent node is absent from ``vector_nodes`` (it is never
    embedded) but present in ``docstore_nodes`` (retained for citation
    reconstruction). Sub-nodes appear in both lists and each sub-node's
    estimated token count must be at or below ``effective_budget``.
    """
    from docint.utils.embed_chunking import (
        effective_budget,
        estimate_tokens,
        resplit_nodes_for_embedding,
    )

    parent = TextNode(text="x " * 40000, id_="parent-1", metadata={})
    budget_tokens = 8192
    vector_nodes, docstore_nodes = resplit_nodes_for_embedding(
        [parent], budget_tokens=budget_tokens
    )

    parent_ids_in_vector = [n.node_id for n in vector_nodes]
    assert parent.node_id not in parent_ids_in_vector
    assert parent in docstore_nodes

    sub_nodes = [n for n in vector_nodes]
    assert len(sub_nodes) >= 3
    assert all(n in docstore_nodes for n in sub_nodes)

    effective = effective_budget(budget_tokens)
    for sub in sub_nodes:
        assert estimate_tokens(sub.get_content()) <= effective


def test_resplit_sub_nodes_carry_parent_id_and_markers() -> None:
    """Sub-nodes must carry the parent-context contract metadata keys.

    Every sub-node must link back to its parent via ``hier.parent_id``,
    carry a contiguous ``split_part_index`` starting at 0, share the
    same ``split_total_parts``, and declare ``embedding_split=True``,
    ``docint_hier_type='fine'``, ``hier.level`` one deeper than the
    parent's.
    """
    from docint.utils.embed_chunking import resplit_nodes_for_embedding

    parent = TextNode(
        text="x " * 40000,
        id_="parent-2",
        metadata={"hier.level": 2},
    )
    vector_nodes, _ = resplit_nodes_for_embedding([parent], budget_tokens=8192)
    sub_nodes = list(vector_nodes)

    assert len(sub_nodes) >= 2
    total_parts = sub_nodes[0].metadata["split_total_parts"]
    for idx, sub in enumerate(sub_nodes):
        assert sub.metadata["hier.parent_id"] == parent.node_id
        assert sub.metadata["embedding_split"] is True
        assert sub.metadata["split_part_index"] == idx
        assert sub.metadata["split_total_parts"] == total_parts
        assert sub.metadata["hier.level"] == 3
        assert sub.metadata["docint_hier_type"] == "fine"


def test_resplit_sub_nodes_have_unique_uuids() -> None:
    """Every sub-node must receive a fresh ``node_id`` distinct from the parent.

    The vector-store branch at ``rag.py:_select_vector_nodes`` and the
    docstore writes rely on unique IDs to avoid collisions on upsert.
    """
    from docint.utils.embed_chunking import resplit_nodes_for_embedding

    parent = TextNode(text="x " * 40000, id_="parent-uuid", metadata={})
    vector_nodes, _ = resplit_nodes_for_embedding([parent], budget_tokens=8192)
    sub_nodes = list(vector_nodes)

    ids = [n.node_id for n in sub_nodes]
    assert len(ids) == len(set(ids))
    assert parent.node_id not in ids


def test_resplit_preserves_parent_metadata() -> None:
    """Arbitrary parent metadata keys must be copied onto every sub-node.

    Keys the ingestion pipeline relies on for citation and dedup
    (``source_ref``, ``chunk_id``, ``file_hash``) must survive the split
    so that downstream filters and joins still work.
    """
    from docint.utils.embed_chunking import resplit_nodes_for_embedding

    parent = TextNode(
        text="x " * 40000,
        id_="parent-meta",
        metadata={
            "source_ref": "s3://bucket/doc.pdf",
            "chunk_id": "chunk-42",
            "file_hash": "deadbeef",
        },
    )
    vector_nodes, _ = resplit_nodes_for_embedding([parent], budget_tokens=8192)

    for sub in vector_nodes:
        assert sub.metadata["source_ref"] == "s3://bucket/doc.pdf"
        assert sub.metadata["chunk_id"] == "chunk-42"
        assert sub.metadata["file_hash"] == "deadbeef"


def test_resplit_raises_on_irreducible_input() -> None:
    """A node that cannot be split further must raise ``EmbeddingInputTooLongError``.

    A single 60_000-character word (no whitespace) cannot be split by
    sentence-based chunking; the re-splitter must surface the failure
    loudly with diagnostics — ``node_id`` and ``estimated_tokens`` in
    the error message — rather than silently truncate or drop content.
    """
    from docint.utils.embed_chunking import resplit_nodes_for_embedding
    from docint.utils.openai_cfg import EmbeddingInputTooLongError

    parent = TextNode(text="x" * 60_000, id_="irreducible-1", metadata={})

    with pytest.raises(EmbeddingInputTooLongError) as excinfo:
        resplit_nodes_for_embedding([parent], budget_tokens=8192)

    message = str(excinfo.value)
    assert "node_id=" in message
    assert "estimated_tokens=" in message


def test_resplit_raises_on_punctuation_only_stream() -> None:
    """A punctuation-only run must be treated as an irreducible single-token stream.

    ``SentenceSplitter`` has a character-chopping fallback that can produce
    dense-punctuation sub-chunks that still overflow the budget. Pre-check
    must raise instead.
    """
    from docint.utils.embed_chunking import resplit_nodes_for_embedding
    from docint.utils.openai_cfg import EmbeddingInputTooLongError

    parent = TextNode(text="." * 60_000, id_="punct-only-1", metadata={})

    with pytest.raises(EmbeddingInputTooLongError) as excinfo:
        resplit_nodes_for_embedding([parent], budget_tokens=8192)

    message = str(excinfo.value)
    assert "single token stream" in message
    assert "node_id=" in message


def test_resplit_second_pass_shrinks_until_fits() -> None:
    """A second-pass shrink must produce all in-budget sub-nodes.

    The default sentence splitter underchunks pathological inputs with
    only sentence boundaries; the resplitter must transparently retry
    with a smaller chunk size and return sub-nodes whose estimated
    token counts are all within ``effective_budget``.
    """
    from docint.utils.embed_chunking import (
        effective_budget,
        estimate_tokens,
        resplit_nodes_for_embedding,
    )

    sentence = "This is a sentence that has absolutely no paragraph breaks. "
    text = sentence * 800
    parent = TextNode(text=text, id_="parent-fallback", metadata={})

    budget_tokens = 8192
    vector_nodes, _ = resplit_nodes_for_embedding([parent], budget_tokens=budget_tokens)

    effective = effective_budget(budget_tokens)
    assert len(vector_nodes) >= 2
    for sub in vector_nodes:
        sub_node: BaseNode = sub
        assert estimate_tokens(sub_node.get_content()) <= effective
