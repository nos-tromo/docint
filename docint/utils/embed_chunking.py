"""Pre-embed re-chunking helpers that bound inputs to the embedding budget.

Ollama, vLLM, and OpenAI-compatible embedding endpoints reject requests
that exceed the model's configured context window. Rather than silently
truncating oversize chunks at the client layer (which would store lossy
prefix-only vectors alongside the full text and break retrieval), this
module splits oversize nodes into budget-conforming sub-nodes *before*
the embedding call. The sub-nodes link back to their parent via
``hier.parent_id`` so the existing query-time parent-context-attachment
postprocessor can reconstruct the original content for answer synthesis.

Design constraints:

- **Centralize ML workloads at the provider**: we deliberately avoid
  loading a tokenizer in the worker to count tokens exactly. Instead,
  a conservative character/token ratio (``3.5`` chars/token by default)
  under-counts tokens for mixed-language text, and a configurable
  safety margin reserves slop for BOS/EOS and estimator error.
- **Loud-on-failure**: a node that cannot be reduced below the budget
  raises :class:`docint.utils.openai_cfg.EmbeddingInputTooLongError`
  with the offending ``node_id`` and estimated token count so operators
  can diagnose the source rather than quietly drop data.
"""

from __future__ import annotations

import math
import uuid
from typing import cast

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, NodeRelationship, TextNode

from docint.utils.openai_cfg import EmbeddingInputTooLongError


def estimate_tokens(text: str, char_token_ratio: float = 3.5) -> int:
    """Conservatively estimate the number of tokens in *text*.

    The default ratio of ``3.5`` characters/token under-counts tokens
    for English-only text (which typically averages ~4 chars/token)
    but stays close to the realistic upper bound for mixed-language
    content including CJK scripts where a single character can be
    multiple tokens. Under-counting is safer here: every downstream
    budget check treats the estimate as the true token count, so an
    under-count lowers the effective cap and never admits an oversize
    payload by accident.

    Args:
        text: Input string to measure.
        char_token_ratio: Characters per token. Must be positive.

    Returns:
        Upper-bound estimated token count (ceiling of ``len(text) / char_token_ratio``).
    """
    if char_token_ratio <= 0:
        raise ValueError("char_token_ratio must be positive")
    return math.ceil(len(text) / char_token_ratio)


def effective_budget(ctx_tokens: int, safety_margin: float = 0.95) -> int:
    """Return the embedding budget after reserving a safety margin.

    Reserves ``(1 - safety_margin)`` of the raw context window for
    BOS/EOS tokens that the provider inserts and for residual estimator
    error. A ``0.95`` margin on an ``8192``-token window yields an
    ``7782``-token effective budget.

    Args:
        ctx_tokens: Raw context window size in tokens.
        safety_margin: Fraction of ``ctx_tokens`` that remains available
            to the user payload. Must fall in ``(0, 1]``.

    Returns:
        The effective budget in tokens.
    """
    return int(ctx_tokens * safety_margin)


def fits_budget(
    text: str,
    *,
    budget_tokens: int,
    char_token_ratio: float = 3.5,
    safety_margin: float = 0.95,
) -> bool:
    """Return whether *text* fits within the effective embedding budget.

    Args:
        text: Candidate text.
        budget_tokens: Raw context window size in tokens.
        char_token_ratio: Characters per token estimator (see
            :func:`estimate_tokens`).
        safety_margin: Budget reservation fraction (see
            :func:`effective_budget`).

    Returns:
        ``True`` when the estimated token count is at or below the
        effective budget, ``False`` otherwise.
    """
    return estimate_tokens(text, char_token_ratio) <= effective_budget(
        budget_tokens, safety_margin
    )


def _build_sub_node(
    *,
    parent: BaseNode,
    text: str,
    part_index: int,
    total_parts: int,
) -> TextNode:
    """Build a sub-node that links back to *parent* and carries split markers.

    Args:
        parent: The oversize parent node being split.
        text: Sub-chunk text payload.
        part_index: 0-based index of this sub-node within the split.
        total_parts: Total number of sub-nodes produced from *parent*.

    Returns:
        A :class:`TextNode` with fresh ``node_id``, copied parent
        metadata, and the split-tracking markers (``hier.parent_id``,
        ``hier.level``, ``docint_hier_type='fine'``, ``embedding_split``,
        ``split_part_index``, ``split_total_parts``). A
        ``NodeRelationship.PARENT`` relationship links the sub-node back
        to the parent for llama_index-aware consumers.
    """
    parent_level = int(parent.metadata.get("hier.level", 2))
    sub_metadata: dict[str, object] = dict(parent.metadata)
    sub_metadata.update(
        {
            "hier.parent_id": parent.node_id,
            "hier.level": parent_level + 1,
            "docint_hier_type": "fine",
            "embedding_split": True,
            "split_part_index": part_index,
            "split_total_parts": total_parts,
        }
    )
    sub_node = TextNode(
        id_=str(uuid.uuid4()),
        text=text,
        metadata=sub_metadata,
    )
    sub_node.relationships[NodeRelationship.PARENT] = parent.as_related_node_info()
    return sub_node


def _has_word_boundaries(text: str) -> bool:
    """Return whether *text* contains any splittable whitespace boundary.

    The sentence splitter ultimately relies on whitespace (or newlines)
    to carve out tokens; dense punctuation streams like ``"." * 60000``
    would otherwise slip past the pre-check and be fed to
    llama_index's character-chopping fallback, which produces
    dense-punctuation sub-chunks that still overflow the embedding
    budget. Requiring at least one whitespace character closes that
    loophole so irreducible single-token streams are raised up front
    with a clear diagnostic.

    Empty text is treated as trivially splittable (``True``) — it fits
    any budget and the existing "empty is not irreducible" convention
    is preserved so the caller never receives a spurious
    :class:`EmbeddingInputTooLongError` for an empty payload.

    Args:
        text: Text to inspect.

    Returns:
        ``True`` when *text* is empty or contains at least one
        whitespace character (space, tab, newline, etc.) that a
        sentence splitter can use as a break point; ``False`` for
        single-token streams such as a long contiguous word, a
        punctuation-only run, or a base64 blob.
    """
    if not text:
        return True
    for ch in text:
        if ch.isspace():
            return True
    return False


def _split_parent_text(
    parent: BaseNode,
    *,
    budget_tokens: int,
    char_token_ratio: float,
    safety_margin: float,
    sentence_splitter: SentenceSplitter | None,
) -> list[str]:
    """Split *parent*'s text into chunks that each fit the embedding budget.

    Makes up to two passes over the sentence splitter. The first pass
    uses ``chunk_size = effective_budget`` (matching the embedding
    budget). If that still yields an oversize chunk — which can happen
    on pathological inputs without sentence boundaries where
    llama_index's sentence splitter leans toward its default target —
    it retries once with ``chunk_size = effective_budget // 2`` to
    force tighter splits. No overlap is used because parent context is
    reconstructed at query time from ``hier.parent_id``.

    A pre-check guards against single-token streams (no whitespace):
    those are raised as irreducible up front so we do not rely on
    llama_index's character-chopping fallback, which would produce
    sub-chunks that *appear* to fit the estimator but actually carry
    as many tokens as the original contiguous stream.

    Each pass constructs a fresh :class:`SentenceSplitter` with the
    pass-specific ``chunk_size``. A caller-supplied splitter is only
    consulted as a fallback if construction fails; it is never mutated
    so callers can safely reuse their splitter instance.

    Args:
        parent: The oversize parent node.
        budget_tokens: Raw context window in tokens.
        char_token_ratio: Characters per token estimator.
        safety_margin: Budget reservation fraction.
        sentence_splitter: Optional fallback splitter used only when a
            fresh :class:`SentenceSplitter` cannot be constructed for
            the pass. Never mutated.

    Returns:
        A list of text chunks that each fit the effective budget.

    Raises:
        EmbeddingInputTooLongError: When no pass can produce all
            in-budget chunks (e.g. a single whitespace-free token
            stream longer than the budget).
    """
    effective = effective_budget(budget_tokens, safety_margin)
    parent_text = parent.get_content()
    parent_token_estimate = estimate_tokens(parent_text, char_token_ratio)

    if not _has_word_boundaries(parent_text):
        raise EmbeddingInputTooLongError(
            f"node_id={parent.node_id} estimated_tokens={parent_token_estimate} "
            f"budget={effective} — content is a single token stream larger than "
            f"the embedding budget; chunk your input further or raise EMBED_CTX_TOKENS."
        )

    candidate_sizes = [effective, max(1, effective // 2)]
    last_chunks: list[str] = [parent_text]
    for chunk_size in candidate_sizes:
        try:
            splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=0,
            )
        except Exception:
            # Fall back to the caller-supplied splitter only if fresh
            # construction fails; do not mutate the caller's instance.
            if sentence_splitter is None:
                raise
            splitter = sentence_splitter
        chunks = splitter.split_text(parent_text)
        last_chunks = chunks
        if not chunks:
            continue
        if all(
            fits_budget(
                chunk,
                budget_tokens=budget_tokens,
                char_token_ratio=char_token_ratio,
                safety_margin=safety_margin,
            )
            for chunk in chunks
        ):
            return chunks

    raise EmbeddingInputTooLongError(
        f"node_id={parent.node_id} estimated_tokens={parent_token_estimate} "
        f"budget={effective} — content is a single token stream larger than "
        f"the embedding budget; chunk your input further or raise EMBED_CTX_TOKENS "
        f"(last_pass_chunks={len(last_chunks)})."
    )


def resplit_nodes_for_embedding(
    nodes: list[BaseNode],
    *,
    budget_tokens: int,
    char_token_ratio: float = 3.5,
    safety_margin: float = 0.95,
    sentence_splitter: SentenceSplitter | None = None,
) -> tuple[list[BaseNode], list[BaseNode]]:
    """Split oversize nodes into sub-nodes that fit the embedding budget.

    Nodes whose estimated token count is already within the effective
    budget pass through unchanged and appear in both output lists.
    Nodes that exceed the budget are split via
    :class:`~llama_index.core.node_parser.SentenceSplitter`; the
    original parent is kept in ``docstore_nodes`` only (never embedded)
    so the query-time parent-context-attachment postprocessor can
    reconstruct the full text, while the sub-nodes appear in both
    output lists so the vector store holds one embedding per sub-node.

    Args:
        nodes: Ingestion nodes to evaluate.
        budget_tokens: Raw context window in tokens (``EMBED_CTX_TOKENS``).
        char_token_ratio: Characters per token estimator.
        safety_margin: Budget reservation fraction.
        sentence_splitter: Optional pre-built splitter, mainly useful
            for tests. When omitted, each split creates a fresh
            :class:`SentenceSplitter`.

    Returns:
        A ``(vector_nodes, docstore_nodes)`` pair.

        - ``vector_nodes`` — nodes to embed and insert into the vector
          store. Oversize parents are absent; their sub-nodes replace
          them.
        - ``docstore_nodes`` — nodes to persist to the docstore.
          Oversize parents remain here so retrieval can expand sub-node
          hits back to the full parent content.

    Raises:
        EmbeddingInputTooLongError: When a node cannot be reduced to
            within the budget even after the second-pass shrink.
    """
    vector_nodes: list[BaseNode] = []
    docstore_nodes: list[BaseNode] = []
    for node in nodes:
        text = node.get_content()
        if fits_budget(
            text,
            budget_tokens=budget_tokens,
            char_token_ratio=char_token_ratio,
            safety_margin=safety_margin,
        ):
            vector_nodes.append(node)
            docstore_nodes.append(node)
            continue

        chunks = _split_parent_text(
            node,
            budget_tokens=budget_tokens,
            char_token_ratio=char_token_ratio,
            safety_margin=safety_margin,
            sentence_splitter=sentence_splitter,
        )
        total_parts = len(chunks)
        sub_nodes: list[BaseNode] = []
        for idx, chunk in enumerate(chunks):
            sub_nodes.append(
                cast(
                    BaseNode,
                    _build_sub_node(
                        parent=node,
                        text=chunk,
                        part_index=idx,
                        total_parts=total_parts,
                    ),
                )
            )
        # Parent stays in the docstore only; sub-nodes go to both.
        docstore_nodes.append(node)
        vector_nodes.extend(sub_nodes)
        docstore_nodes.extend(sub_nodes)

    return vector_nodes, docstore_nodes
