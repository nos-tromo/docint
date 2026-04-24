"""Pre-embed re-chunking helpers that bound inputs to the embedding budget.

Ollama, vLLM, and OpenAI-compatible embedding endpoints reject requests
that exceed the model's configured context window. Rather than silently
truncating oversize chunks at the client layer (which would store lossy
prefix-only vectors alongside the full text and break retrieval), this
module splits oversize nodes into budget-conforming sub-nodes *before*
the embedding call. The sub-nodes link back to their parent via
``hier.parent_id`` so the existing query-time parent-context-attachment
postprocessor can reconstruct the original content for answer synthesis.

Token counting:

- **Primary (authoritative)**: a ``token_counter: Callable[[str], list[int]]``
  built from the embedding model's real tokenizer (see
  :func:`docint.utils.embedding_tokenizer.build_embedding_token_counter`).
  When supplied, this counter is used end-to-end: for parent fit checks,
  for sub-node fit checks, and as the ``tokenizer`` kwarg of the inner
  :class:`~llama_index.core.node_parser.SentenceSplitter` so chunk
  sizing and fit checking share one definition of a token. The counter
  is loaded from the local HF cache populated by ``uv run load-models``
  and never reaches the network.
- **Fallback**: when no counter is supplied (e.g. the cache is empty or
  the operator runs without ``MODEL_EMBED``), a char/token ratio
  (default ``3.5``) with a configurable safety margin acts as a
  last-resort estimator. It is inherently language-biased and should
  not be relied on for multilingual corpora.

Loud-on-failure: a node that cannot be reduced below the budget raises
:class:`docint.utils.openai_cfg.EmbeddingInputTooLongError` with the
offending ``node_id`` and estimated token count so operators can
diagnose the source rather than quietly drop data.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from typing import cast

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, MetadataMode, NodeRelationship, TextNode

from docint.utils.openai_cfg import EmbeddingInputTooLongError

TokenCounter = Callable[[str], list[int]]


def estimate_tokens(
    text: str,
    char_token_ratio: float = 3.5,
    *,
    token_counter: TokenCounter | None = None,
) -> int:
    """Return the estimated number of tokens in *text*.

    When *token_counter* is supplied it is authoritative: the function
    returns ``len(token_counter(text))``, matching what the embedding
    provider will see on the wire (BOS/EOS included when the counter
    encodes with special tokens). This is the correct path for
    multilingual content where no char/token ratio is accurate.

    When *token_counter* is ``None`` the function falls back to a
    conservative char/token ratio (default ``3.5``) with a safer
    intent: every downstream budget check treats the estimate as the
    true token count, so the ratio is only reliable for English-like
    text. Operators should treat the fallback as a degraded state and
    populate the HF tokenizer cache via ``uv run load-models``.

    Args:
        text: Input string to measure.
        char_token_ratio: Characters per token for the fallback
            estimator. Must be positive.
        token_counter: Optional callable that returns the list of token
            ids for *text* (e.g. the bge-m3 ``AutoTokenizer`` adapter
            from :mod:`docint.utils.embedding_tokenizer`). When
            supplied, overrides *char_token_ratio*.

    Returns:
        Estimated token count — authoritative when *token_counter* is
        supplied, char-ratio upper-bound otherwise.

    Raises:
        ValueError: When *token_counter* is ``None`` and
            *char_token_ratio* is non-positive.
    """
    if token_counter is not None:
        return len(token_counter(text))
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
    token_counter: TokenCounter | None = None,
) -> bool:
    """Return whether *text* fits within the effective embedding budget.

    Args:
        text: Candidate text.
        budget_tokens: Raw context window size in tokens.
        char_token_ratio: Characters per token estimator (see
            :func:`estimate_tokens`).
        safety_margin: Budget reservation fraction (see
            :func:`effective_budget`).
        token_counter: Optional authoritative token counter (see
            :func:`estimate_tokens`). When supplied it is used instead
            of the char-ratio estimator.

    Returns:
        ``True`` when the estimated token count is at or below the
        effective budget, ``False`` otherwise.
    """
    return estimate_tokens(
        text, char_token_ratio, token_counter=token_counter
    ) <= effective_budget(budget_tokens, safety_margin)


def _build_probe_sub_metadata(parent: BaseNode) -> dict[str, object]:
    """Build the metadata dict a prospective sub-node will carry.

    Mirrors the shape produced by :func:`_build_sub_node` so callers can
    estimate the per-sub-node ``MetadataMode.EMBED`` overhead (which is
    dominated by the serialized metadata block, not the text payload).
    ``split_part_index`` and ``split_total_parts`` are set to pessimistic
    stand-ins so the rendered key/value widths match the real sub-node
    to within a character or two.

    Args:
        parent: The oversize parent whose metadata will be inherited by
            every sub-node.

    Returns:
        A metadata dict with the parent's metadata merged with the
        split-tracking markers every sub-node receives.
    """
    parent_level = int(parent.metadata.get("hier.level", 2))
    probe: dict[str, object] = dict(parent.metadata)
    probe.update(
        {
            "hier.parent_id": parent.node_id,
            "hier.level": parent_level + 1,
            "docint_hier_type": "fine",
            "embedding_split": True,
            "split_part_index": 0,
            "split_total_parts": 9999,
        }
    )
    return probe


def _estimate_sub_node_metadata_tokens(
    parent: BaseNode,
    *,
    char_token_ratio: float,
    token_counter: TokenCounter | None = None,
) -> int:
    """Estimate the ``MetadataMode.EMBED`` token overhead of a prospective sub-node.

    Real sub-nodes built by :func:`_build_sub_node` set
    ``excluded_embed_metadata_keys`` to every inherited metadata key, so
    their ``MetadataMode.EMBED`` rendering contains only the chunk text
    with no metadata block. This probe mirrors that exclusion, and in
    practice the returned overhead is ~0 tokens (llama_index may still
    add a few template separators).

    The function is retained as a forward-compatible guard: if a future
    sub-node construction path re-enables metadata in embed mode, this
    measurement starts returning a non-zero value and the caller's
    ``metadata_overhead_tokens >= effective`` short-circuit in
    :func:`_split_parent_text` catches the case where metadata alone
    exhausts the budget.

    Args:
        parent: The oversize parent node.
        char_token_ratio: Characters per token estimator (matches
            :func:`estimate_tokens`).

    Returns:
        Estimated metadata overhead in tokens (≈0 under the current
        exclusion contract).
    """
    probe_metadata = _build_probe_sub_metadata(parent)
    probe = TextNode(
        text="",
        metadata=probe_metadata,
        excluded_embed_metadata_keys=list(probe_metadata.keys()),
    )
    return estimate_tokens(
        probe.get_content(metadata_mode=MetadataMode.EMBED),
        char_token_ratio,
        token_counter=token_counter,
    )


def _sub_node_fits_budget(
    chunk: str,
    parent: BaseNode,
    *,
    budget_tokens: int,
    char_token_ratio: float,
    safety_margin: float,
    token_counter: TokenCounter | None = None,
) -> bool:
    """Return whether a prospective sub-node's embed payload fits the budget.

    Builds a probe :class:`TextNode` with the same metadata shape a real
    sub-node will carry and measures its ``MetadataMode.EMBED`` rendering
    — the same payload the embedding client sends to the provider — so
    the fit check matches what the provider will see rather than the raw
    chunk text alone.

    Args:
        chunk: Sub-chunk text payload.
        parent: The oversize parent node being split.
        budget_tokens: Raw context window size in tokens.
        char_token_ratio: Characters per token estimator.
        safety_margin: Budget reservation fraction.

    Returns:
        ``True`` when the rendered embed payload fits the effective
        budget, ``False`` otherwise.
    """
    probe_metadata = _build_probe_sub_metadata(parent)
    probe = TextNode(
        text=chunk,
        metadata=probe_metadata,
        excluded_embed_metadata_keys=list(probe_metadata.keys()),
    )
    return fits_budget(
        probe.get_content(metadata_mode=MetadataMode.EMBED),
        budget_tokens=budget_tokens,
        char_token_ratio=char_token_ratio,
        safety_margin=safety_margin,
        token_counter=token_counter,
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
        excluded_embed_metadata_keys=list(sub_metadata.keys()),
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
    token_counter: TokenCounter | None = None,
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
        token_counter: Optional authoritative token counter (see
            :func:`estimate_tokens`). When supplied, it is used in
            place of *char_token_ratio* for every fit check and is
            passed as ``tokenizer`` to the inner
            :class:`SentenceSplitter` so chunk sizing and fit checking
            share one token definition.

    Returns:
        A list of text chunks that each fit the effective budget.

    Raises:
        EmbeddingInputTooLongError: When no pass can produce all
            in-budget chunks (e.g. a single whitespace-free token
            stream longer than the budget).
    """
    effective = effective_budget(budget_tokens, safety_margin)
    parent_raw_text = parent.get_content()
    parent_embed_text = parent.get_content(metadata_mode=MetadataMode.EMBED)
    parent_token_estimate = estimate_tokens(
        parent_embed_text, char_token_ratio, token_counter=token_counter
    )

    if not _has_word_boundaries(parent_raw_text):
        # Use the raw-text estimate (not the embed-mode one) so the
        # diagnostic reflects the stream the whitespace guard actually
        # inspected. The embed-mode estimate can be an order of magnitude
        # larger for heavy-metadata nodes, which would mislead operators.
        raw_token_estimate = estimate_tokens(
            parent_raw_text, char_token_ratio, token_counter=token_counter
        )
        raise EmbeddingInputTooLongError(
            f"node_id={parent.node_id} estimated_tokens={raw_token_estimate} "
            f"budget={effective} — content is a single token stream larger than "
            f"the embedding budget; chunk your input further or raise EMBED_CTX_TOKENS."
        )

    metadata_overhead_tokens = _estimate_sub_node_metadata_tokens(
        parent,
        char_token_ratio=char_token_ratio,
        token_counter=token_counter,
    )
    if metadata_overhead_tokens >= effective:
        raise EmbeddingInputTooLongError(
            f"node_id={parent.node_id} metadata_tokens={metadata_overhead_tokens} "
            f"budget={effective} — parent metadata alone exceeds the embedding "
            f"budget; reduce node metadata or raise EMBED_CTX_TOKENS."
        )
    chunk_budget_tokens = max(1, effective - metadata_overhead_tokens)
    candidate_sizes = [chunk_budget_tokens, max(1, chunk_budget_tokens // 2)]
    last_chunks: list[str] = [parent_raw_text]
    for chunk_size in candidate_sizes:
        splitter_kwargs: dict[str, object] = {
            "chunk_size": chunk_size,
            "chunk_overlap": 0,
        }
        if token_counter is not None:
            splitter_kwargs["tokenizer"] = token_counter
        try:
            splitter = SentenceSplitter(**splitter_kwargs)
        except Exception:
            if sentence_splitter is None:
                raise
            splitter = sentence_splitter
        chunks = splitter.split_text(parent_raw_text)
        last_chunks = chunks
        if not chunks:
            continue
        if all(
            _sub_node_fits_budget(
                chunk,
                parent,
                budget_tokens=budget_tokens,
                char_token_ratio=char_token_ratio,
                safety_margin=safety_margin,
                token_counter=token_counter,
            )
            for chunk in chunks
        ):
            return chunks

    raise EmbeddingInputTooLongError(
        f"node_id={parent.node_id} estimated_tokens={parent_token_estimate} "
        f"budget={effective} metadata_tokens={metadata_overhead_tokens} — "
        f"content is a single token stream larger than the embedding budget; "
        f"chunk your input further or raise EMBED_CTX_TOKENS "
        f"(last_pass_chunks={len(last_chunks)})."
    )


def resplit_nodes_for_embedding(
    nodes: list[BaseNode],
    *,
    budget_tokens: int,
    char_token_ratio: float = 3.5,
    safety_margin: float = 0.95,
    sentence_splitter: SentenceSplitter | None = None,
    token_counter: TokenCounter | None = None,
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
        embed_payload = node.get_content(metadata_mode=MetadataMode.EMBED)
        if fits_budget(
            embed_payload,
            budget_tokens=budget_tokens,
            char_token_ratio=char_token_ratio,
            safety_margin=safety_margin,
            token_counter=token_counter,
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
            token_counter=token_counter,
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
