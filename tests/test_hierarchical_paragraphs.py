r"""Tests for paragraph-aware chunking in :class:`HierarchicalNodeParser`.

These cover two coupled defaults that combined to squash multi-paragraph
plain-text documents into a single fine node:

1. ``fine_chunk_size`` defaulted to 8192 tokens, so any document under roughly
   30KB fit in one fine chunk regardless of internal structure.
2. The underlying ``SentenceSplitter`` used llama-index's default
   ``paragraph_separator="\n\n\n"``, so after ``basic_clean`` (which
   collapses ``\n{3,}`` → ``\n\n``) the splitter could not see paragraph
   boundaries at all and was forced to break mid-paragraph.
"""

from __future__ import annotations

from llama_index.core import Document

from docint.core.storage.hierarchical import HierarchicalNodeParser
from docint.utils.env_cfg import load_ingestion_env

# Six paragraphs each large enough to exceed the new default ``fine_chunk_size``
# when combined. Each paragraph is ~400 tokens of identical filler so the
# splitter cannot find a "more natural" cut than the inter-paragraph break.
_PARAGRAPH = "Paragraph body. " + "This sentence pads the paragraph to a respectable length. " * 30
_SIX_PARAGRAPH_DOC = "\n\n".join(f"Paragraph {i}.\n{_PARAGRAPH}" for i in range(1, 7))


def test_default_fine_chunk_size_splits_multi_paragraph_text() -> None:
    """Default config must split a multi-paragraph doc into multiple fine nodes.

    Otherwise the hierarchy degenerates to a single chunk and retrieval
    cannot localize to a paragraph.
    """
    cfg = load_ingestion_env()
    parser = HierarchicalNodeParser(
        coarse_chunk_size=cfg.coarse_chunk_size,
        fine_chunk_size=cfg.fine_chunk_size,
        fine_chunk_overlap=cfg.fine_chunk_overlap,
    )

    nodes = parser.get_nodes_from_documents([Document(text=_SIX_PARAGRAPH_DOC)])
    fine_nodes = [n for n in nodes if n.metadata.get("hier.level") == 2]

    assert len(fine_nodes) >= 2, (
        f"expected multi-paragraph doc to split into ≥2 fine nodes "
        f"with default fine_chunk_size={cfg.fine_chunk_size}, got {len(fine_nodes)}"
    )


def test_hierarchical_parser_respects_double_newline_paragraph_breaks() -> None:
    r"""Fine nodes must align with ``\n\n`` paragraph breaks in the document.

    No fine node may straddle a paragraph boundary in the middle of its
    body — that break is the natural cut point.
    """
    # chunk_size just over one paragraph: forces a split between paragraphs
    # but never inside one, so any straddling boundary is a bug in the
    # splitter configuration.
    parser = HierarchicalNodeParser(
        coarse_chunk_size=8192,
        fine_chunk_size=500,
        fine_chunk_overlap=0,
    )

    nodes = parser.get_nodes_from_documents([Document(text=_SIX_PARAGRAPH_DOC)])
    fine_nodes = [n for n in nodes if n.metadata.get("hier.level") == 2]

    assert len(fine_nodes) >= 2
    for n in fine_nodes:
        body = n.get_content().strip()
        assert "\n\n" not in body, (
            f"fine node straddles a paragraph boundary (paragraph_separator override missing?): {body[:200]!r}…"
        )
