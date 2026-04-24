"""RED tests for the pre-embed re-chunking contract.

These tests pin the public contract of ``docint.utils.embed_chunking``
before the implementation is written. They MUST fail on ``HEAD``
(the module does not yet exist) — that red state is the TDD signal
for the implementer to satisfy the contract.
"""

from __future__ import annotations

import pytest
from llama_index.core.schema import BaseNode, MetadataMode, TextNode

from docint.utils.embed_chunking import (
    effective_budget,
    estimate_tokens,
    fits_budget,
    resplit_nodes_for_embedding,
)
from docint.utils.openai_cfg import EmbeddingInputTooLongError


def test_estimate_tokens_conservative_for_mixed_language() -> None:
    """``estimate_tokens`` should stay conservative for multilingual text.

    The fix-branch contract: the char/token ratio default of ``3.5`` is
    picked so that mixed English + CJK content is never *under*-counted.
    A safe estimator predicts at most ``len(text) / 3.5`` tokens (plus a
    small slack for rounding). An estimator that over-counts on this
    input is still safe — the concern is never *under*-counting.
    """
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
        assert (
            estimate_tokens(sub.get_content(metadata_mode=MetadataMode.EMBED))
            <= effective
        )


def test_resplit_sub_nodes_carry_parent_id_and_markers() -> None:
    """Sub-nodes must carry the parent-context contract metadata keys.

    Every sub-node must link back to its parent via ``hier.parent_id``,
    carry a contiguous ``split_part_index`` starting at 0, share the
    same ``split_total_parts``, and declare ``embedding_split=True``,
    ``docint_hier_type='fine'``, ``hier.level`` one deeper than the
    parent's.
    """
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


def _build_heavy_metadata_table_row_node(
    *,
    raw_text: str,
    node_id: str,
    reference_keys: int,
    reference_value_chars: int,
    extra_keys: int,
    extra_value_chars: int,
) -> TextNode:
    """Build a TextNode mirroring a heavy-metadata table-row payload.

    Constructs a node whose ``metadata`` carries both a nested
    ``reference_metadata`` dict and a flat band of long string keys.
    This matches the real-world shape of a table-row chunk emitted by
    the social-table reader, where every column header surfaces as a
    metadata key. The combined block, once serialised by
    :class:`MetadataMode.EMBED`, dwarfs the raw text payload.

    Args:
        raw_text: The node's body text.
        node_id: Stable ``node_id`` for assertion targeting.
        reference_keys: Count of keys inside the nested
            ``reference_metadata`` dict.
        reference_value_chars: Length of each value inside the nested
            ``reference_metadata`` dict.
        extra_keys: Count of additional flat top-level metadata keys.
        extra_value_chars: Length of each flat metadata value.

    Returns:
        TextNode: A node whose embed-mode payload exceeds its raw payload
        by an order of magnitude.
    """
    reference_metadata: dict[str, str] = {
        f"colvalue_long_name_{i}": "V" * reference_value_chars
        for i in range(reference_keys)
    }
    metadata: dict[str, object] = {"reference_metadata": reference_metadata}
    for i in range(extra_keys):
        metadata[f"col_long_name_{i}"] = "V" * extra_value_chars
    return TextNode(text=raw_text, id_=node_id, metadata=metadata)


def test_resplit_splits_node_whose_embed_payload_exceeds_budget_even_though_raw_text_fits() -> (
    None
):
    """Heavy-metadata nodes whose raw text fits MUST still be re-split when their EMBED payload overflows.

    Pins the bug at ``embed_chunking.py`` where the
    ``resplit_nodes_for_embedding`` fits-check measures
    ``node.get_content()`` (default ``MetadataMode.NONE``, raw text only)
    while the actual embed call in ``rag.py`` sends
    ``node.get_content(metadata_mode=MetadataMode.EMBED)`` (text plus the
    rendered metadata block). Heavy-metadata nodes — table rows, Nextext
    transcript segments — short-circuit the fits-check on raw text and
    are then handed whole to the embedder, which overflows the
    provider's context window. The re-splitter must measure the same
    payload the embed call sees, so heavy-metadata nodes are split into
    sub-nodes whose EMBED payload fits the budget.
    """
    budget_tokens = 8192
    parent = _build_heavy_metadata_table_row_node(
        raw_text="word " * 1000,
        node_id="heavy-meta-parent-1",
        reference_keys=40,
        reference_value_chars=600,
        extra_keys=15,
        extra_value_chars=400,
    )

    raw_payload = parent.get_content()
    embed_payload = parent.get_content(metadata_mode=MetadataMode.EMBED)
    assert fits_budget(raw_payload, budget_tokens=budget_tokens), (
        "test fixture invariant: raw text must fit so the bug's fits-check passes"
    )
    assert estimate_tokens(embed_payload) > effective_budget(budget_tokens), (
        "test fixture invariant: embed payload must exceed budget so the bug bites"
    )

    vector_nodes, docstore_nodes = resplit_nodes_for_embedding(
        [parent], budget_tokens=budget_tokens
    )

    vector_ids = [n.node_id for n in vector_nodes]
    assert len(vector_nodes) >= 1, (
        "heavy-metadata oversize node must be replaced by at least one sub-node "
        "(sub-nodes exclude inherited metadata from embed payload, so a small "
        "raw text may yield a single sub-node whose vector represents only the text)"
    )
    assert parent.node_id not in vector_ids, (
        "oversize parent must not be embedded — sub-nodes replace it in the vector view"
    )
    assert parent in docstore_nodes, (
        "oversize parent must remain in the docstore for parent-context reconstruction"
    )

    effective = effective_budget(budget_tokens)
    for sub in vector_nodes:
        sub_embed_payload = sub.get_content(metadata_mode=MetadataMode.EMBED)
        assert estimate_tokens(sub_embed_payload) <= effective, (
            f"sub-node {sub.node_id} embed payload "
            f"({estimate_tokens(sub_embed_payload)} tokens) exceeds budget ({effective})"
        )
        assert sub.metadata["hier.parent_id"] == parent.node_id


def test_resplit_sub_nodes_embed_payload_fits_budget_when_parent_has_heavy_metadata() -> (
    None
):
    """Sub-nodes inherit the parent's heavy metadata; their EMBED payload MUST still fit the budget.

    Even when the parent is split because its raw text obviously
    overflows, the sub-nodes copy parent metadata wholesale via
    ``_build_sub_node``. If the splitter sizes chunks against the raw
    budget alone, every sub-node ends up with the parent's metadata
    block plus a chunk of text — and the embed-mode payload still
    overflows. The contract this test pins is: every sub-node's
    ``MetadataMode.EMBED`` payload must fit the effective budget so the
    embed call cannot blow past the provider's context window.
    """
    budget_tokens = 8192
    parent = _build_heavy_metadata_table_row_node(
        raw_text="word " * 8000,  # ~40_000 chars, raw text clearly above budget
        node_id="heavy-meta-parent-2",
        reference_keys=20,
        reference_value_chars=600,
        extra_keys=15,
        extra_value_chars=200,
    )

    # Sanity: raw text overflows AND there is meaningful metadata overhead.
    raw_payload = parent.get_content()
    embed_payload = parent.get_content(metadata_mode=MetadataMode.EMBED)
    assert estimate_tokens(raw_payload) > effective_budget(budget_tokens)
    assert len(embed_payload) - len(raw_payload) >= 3000, (
        "fixture invariant: parent must carry a heavy metadata block"
    )

    vector_nodes, _ = resplit_nodes_for_embedding([parent], budget_tokens=budget_tokens)

    effective = effective_budget(budget_tokens)
    assert len(vector_nodes) >= 2
    for sub in vector_nodes:
        sub_embed_tokens = estimate_tokens(
            sub.get_content(metadata_mode=MetadataMode.EMBED)
        )
        assert sub_embed_tokens <= effective, (
            f"sub-node {sub.node_id} embed payload ({sub_embed_tokens} tokens) "
            f"exceeds budget ({effective}) — splitter ignored metadata overhead"
        )


def test_resplit_passthrough_requires_embed_payload_under_budget() -> None:
    """Regression guard: small nodes with small metadata pass through both lists unchanged.

    Not a RED test. Ensures the pre-embed re-splitter does not
    over-trigger on routine within-budget nodes. A node whose EMBED
    payload comfortably fits the budget must appear in both
    ``vector_nodes`` and ``docstore_nodes`` unchanged, with no
    ``embedding_split`` marker. This guards against a regression where a
    fix that switches the fits-check to ``MetadataMode.EMBED`` mistakenly
    splits trivially-small nodes.
    """
    budget_tokens = 8192
    small_node = TextNode(
        text="short text body",
        id_="small-passthrough-1",
        metadata={"chunk_id": "cid", "source_ref": "s3://bucket/d.pdf"},
    )

    # Sanity: this fixture really does fit comfortably under the budget.
    embed_payload = small_node.get_content(metadata_mode=MetadataMode.EMBED)
    assert estimate_tokens(embed_payload) <= effective_budget(budget_tokens)

    vector_nodes, docstore_nodes = resplit_nodes_for_embedding(
        [small_node], budget_tokens=budget_tokens
    )

    assert vector_nodes == [small_node]
    assert docstore_nodes == [small_node]
    assert "embedding_split" not in small_node.metadata


def test_estimate_tokens_prefers_token_counter_when_supplied() -> None:
    """``estimate_tokens`` must use ``token_counter`` authoritatively when provided.

    When a tokenizer-backed counter is supplied, the character ratio
    is ignored: the counter's output length is returned as-is. This
    lets bge-m3's real tokenizer override the conservative
    char-ratio estimator for multilingual content where the ratio
    under-counts or over-counts. When no counter is supplied, the
    char-ratio fallback is retained so offline/cache-miss paths keep
    working.
    """
    counter = lambda text: [0] * (len(text) // 2)  # noqa: E731

    # Counter reports 50 tokens for 100 chars; char-ratio would say 10.
    assert (
        estimate_tokens("x" * 100, char_token_ratio=10.0, token_counter=counter) == 50
    )
    # Without the counter, fallback is the char-ratio estimator.
    assert estimate_tokens("x" * 100, char_token_ratio=10.0) == 10


def test_fits_budget_uses_token_counter_for_accurate_measurement() -> None:
    """``fits_budget`` must honour a strict ``token_counter`` over the char ratio.

    Pins the bug where multilingual content slips past the fits-check
    because the char-ratio estimator under-counts. Crafted fixture: a
    24_000-char payload estimates at ``ceil(24000/3.5) = 6858`` tokens,
    well under the 7782-token effective budget for ``budget_tokens=8192``,
    so the char-ratio check admits it. A strict counter that reports
    9000 tokens (as bge-m3 would for CJK-heavy content) must flip the
    result to ``False``.

    The no-counter fallback is also pinned: the same payload still
    passes the char-ratio check to prove the ``token_counter`` kwarg
    is the only thing changing the verdict.
    """
    text = "x" * 24_000
    counter = lambda _text: [0] * 9000  # noqa: E731

    # Without counter -> char-ratio estimate (6858 tokens) fits 7782 budget.
    assert fits_budget(text, budget_tokens=8192) is True, (
        "fixture invariant: char-ratio estimate must admit the payload"
    )
    # With counter -> 9000 tokens exceed 7782 effective budget.
    assert fits_budget(text, budget_tokens=8192, token_counter=counter) is False


def test_resplit_honors_token_counter_for_dense_tokenization() -> None:
    """``resplit_nodes_for_embedding`` must split when the counter signals overflow.

    Fake counter simulates ~2 chars/token (CJK-like density). A parent
    node of 20_000 chars reports 10_000 tokens — over the 7782-token
    effective budget — even though the default 3.5-chars/token
    estimator would say ``ceil(20000/3.5) = 5715`` and admit the node.
    The re-splitter must honour the counter: produce at least two
    sub-nodes, each with a counter-measured token count at or below
    the effective budget.
    """
    dense_counter = lambda text: [0] * (len(text) // 2)  # noqa: E731
    parent = TextNode(text="word " * 4000, id_="dense-parent", metadata={})

    budget_tokens = 8192
    vector_nodes, _ = resplit_nodes_for_embedding(
        [parent], budget_tokens=budget_tokens, token_counter=dense_counter
    )

    effective = effective_budget(budget_tokens)
    assert len(vector_nodes) >= 2, (
        "dense tokenization must trigger a split even when raw char-ratio "
        "estimate admits the node"
    )
    for sub in vector_nodes:
        sub_embed = sub.get_content(metadata_mode=MetadataMode.EMBED)
        assert len(dense_counter(sub_embed)) <= effective, (
            f"sub-node {sub.node_id} has "
            f"{len(dense_counter(sub_embed))} counter-measured tokens, "
            f"exceeds effective budget {effective}"
        )


def test_resplit_passes_token_counter_to_sentence_splitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``SentenceSplitter`` must receive ``tokenizer=token_counter`` in its kwargs.

    Chunk sizing inside ``SentenceSplitter`` uses its ``tokenizer``
    callable to decide where to cut. If the splitter and the fit
    check use different token definitions, the splitter can produce
    chunks the fit check rejects, looping until the fallback raise.
    Pass the counter into the splitter so both sides measure the
    same way.
    """
    import docint.utils.embed_chunking as embed_chunking

    recorded_kwargs: list[dict[str, object]] = []
    RealSentenceSplitter = embed_chunking.SentenceSplitter

    class SpySentenceSplitter(RealSentenceSplitter):  # type: ignore[misc,valid-type]
        """Spy that records init kwargs and delegates to the real splitter."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Record kwargs and forward to the real splitter.

            Args:
                *args: Positional args forwarded to ``SentenceSplitter``.
                **kwargs: Keyword args forwarded; also recorded.
            """
            recorded_kwargs.append(dict(kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(embed_chunking, "SentenceSplitter", SpySentenceSplitter)

    counter = lambda text: [0] * (len(text) // 2)  # noqa: E731
    oversize_node = TextNode(text="word " * 4000, id_="spy-parent", metadata={})

    embed_chunking.resplit_nodes_for_embedding(
        [oversize_node], budget_tokens=8192, token_counter=counter
    )

    assert any(kwargs.get("tokenizer") is counter for kwargs in recorded_kwargs), (
        "SentenceSplitter was never constructed with tokenizer=token_counter; "
        f"observed kwargs: {recorded_kwargs!r}"
    )
