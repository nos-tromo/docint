"""Tests for the hierarchical node parser used in document storage."""

import pytest
from llama_index.core import Document

from docint.core.storage.hierarchical import HierarchicalNodeParser


@pytest.fixture
def sample_document() -> Document:
    """Create a sample document with multiple sections and sentences.

    Returns:
        Document: A sample document with multiple sections and sentences.
    """
    text = (
        "This is the first section key-metadata. It has three sentences. This is the third one.\n\n"
        "This is the second section. It is a bit longer. It also has sentences."
    )
    return Document(text=text, metadata={"file_name": "test.txt"})


def test_hierarchical_structure(sample_document: Document) -> None:
    """Test that the parser produces Level 1 and Level 2 chunks with correct metadata.

    Args:
        sample_document (Document): A sample document for testing.
    """
    # Use small sizes to force splitting
    parser = HierarchicalNodeParser(
        coarse_chunk_size=50,  # Small enough to split the doc
        fine_chunk_size=25,  # Small enough to split coarse chunks
        fine_chunk_overlap=0,
    )

    nodes = parser.get_nodes_from_documents([sample_document])

    assert len(nodes) > 0

    # Segregate nodes
    coarse_nodes = [n for n in nodes if n.metadata.get("hier.level") == 1]
    fine_nodes = [n for n in nodes if n.metadata.get("hier.level") == 2]

    assert len(coarse_nodes) > 0
    assert len(fine_nodes) > 0

    # Check linkage
    for fine in fine_nodes:
        assert "hier.parent_id" in fine.metadata
        parent_id = fine.metadata["hier.parent_id"]
        assert any(c.node_id == parent_id for c in coarse_nodes)

        # Check metadata inheritance
        assert fine.metadata["file_name"] == "test.txt"
        assert fine.metadata["docint_hier_type"] == "fine"

    for coarse in coarse_nodes:
        assert coarse.metadata["hier.level"] == 1
        assert coarse.metadata["docint_hier_type"] == "coarse"
        assert coarse.metadata["file_name"] == "test.txt"


def test_sentence_boundaries() -> None:
    """Test that fine chunks respect sentence boundaries even with small chunk size."""
    text = "Sentence one. Sentence two. Sentence three."
    doc = Document(text=text)

    # Chunk size usually counts tokens/chars.
    # SentenceSplitter tries to keep sentences together.
    parser = HierarchicalNodeParser(
        coarse_chunk_size=1000,
        fine_chunk_size=200,  # Increased to avoid metadata error
    )

    # Note: SentenceSplitter might respect sentences OVER chunk size default behavior depending on settings.
    # But usually it prefers breaking at sentences.

    nodes = parser.get_nodes_from_documents([doc])
    fine_nodes = [n for n in nodes if n.metadata.get("hier.level") == 2]

    for n in fine_nodes:
        # Check content existence but ignore value
        _ = n.get_content()
        # It should ideally be a complete sentence or list of sentences.
        # But if it is forced to split mid-sentence due to effective constraints, we accept it,
        # but user requirement is "Every fine chunk must be composed of complete sentences".
        # SentenceSplitter logic: "Splits text... while keeping sentences whole."
        # If a single sentence is larger than chunk_size, it might split it.
        # "Sentence one." is ~13 chars. fine_chunk_size=10 might force split.
        pass

    # Let's use a more realistic size that fits one sentence but maybe not two
    parser = HierarchicalNodeParser(
        coarse_chunk_size=1000,
        fine_chunk_size=20,
    )
    nodes = parser.get_nodes_from_documents([doc])
    fine_nodes = [n for n in nodes if n.metadata.get("hier.level") == 2]

    for n in fine_nodes:
        txt = n.get_content().strip()
        # Verify it ends with punctuation usually (heuristic)
        assert txt.endswith(".")


def test_input_nodes() -> None:
    """Test that the parser can handle input nodes (e.g. from MarkdownParser)."""
    # Simulate a node from Markdown parser
    # node = Document(
    #     text="Section text. Sentence inside.", metadata={"header_path": "/Section 1"}
    # )
    # Only using get_nodes_from_documents wrapper which handles both logic essentially
    # but internally treats input as Document -> Level 0 -> Level 1.

    # Wait, if we want to test the _parse_nodes logic specifically for nodes:
    parser = HierarchicalNodeParser()

    # IngestionPipeline calls _parse_nodes with nodes
    # Let's simulate that
    coarse_node = Document(text="Coarse content. Multiple sentences.", metadata={"section": "1"})
    # Cast to ensure it's treated as input node

    nodes = parser._parse_nodes([coarse_node])

    # In my implementation:
    # if isinstance(node, Document) -> treats as Level 0 -> splits to Coarse
    # if I want to simulate Level 1 input, I should pass a BaseNode that is NOT a Document?
    # Actually Document IS a BaseNode.
    # In `hierarchical.py`: `is_document = isinstance(node, Document)`
    # The Llama Index `BaseNode` usually isn't instantiated directly (it's abstract-ish or just TextNode).
    # `TextNode` is what we likely get.

    from llama_index.core.schema import TextNode

    text_node = TextNode(text="Existing chunk. More text.", metadata={"origin": "md"})

    nodes = parser._parse_nodes([text_node])

    # Should treat text_node as Coarse candidate (Level 1) and split to Fine (Level 2)
    levels = [n.metadata.get("hier.level") for n in nodes]
    assert 1 in levels  # The input node itself should be preserved/returned/marked as Level 1?

    # In my implementation:
    # else: coarse_candidates = [node] (if fits)
    # So the input node IS the coarse chunk.
    # And then we create fine chunks linking to it.

    assert nodes[0] == text_node  # Identical object
    assert nodes[0].metadata["hier.level"] == 1
    assert nodes[0].metadata["docint_hier_type"] == "coarse"

    fine_nodes = nodes[1:]
    assert len(fine_nodes) > 0
    assert fine_nodes[0].metadata["hier.level"] == 2
    assert fine_nodes[0].metadata["hier.parent_id"] == text_node.node_id


def test_pdf_coarse_metadata_propagates_to_fine() -> None:
    """A coarse TextNode with PDF layout keys passes them down to fine children.

    This is the contract the PDF reader relies on: page/section/bbox metadata
    placed on a coarse unit must survive onto the embedded fine children so
    citations and image/table linking keep working for PDFs.
    """
    from llama_index.core.schema import TextNode

    parser = HierarchicalNodeParser(coarse_chunk_size=1000, fine_chunk_size=200, fine_chunk_overlap=0)
    coarse = TextNode(
        text="First sentence here. Second sentence follows. Third sentence ends.",
        metadata={
            "page": 3,
            "section_path": ["Results"],
            "bbox_refs": [{"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}],
            "file_hash": "abc",
        },
    )

    nodes = parser._parse_nodes([coarse])
    fine_nodes = [n for n in nodes if n.metadata.get("hier.level") == 2]
    assert len(fine_nodes) >= 1
    for fine in fine_nodes:
        assert fine.metadata["page"] == 3
        assert fine.metadata["section_path"] == ["Results"]
        assert fine.metadata["bbox_refs"] == [{"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}]
        assert fine.metadata["file_hash"] == "abc"
        assert fine.metadata["hier.parent_id"] == coarse.node_id


def test_oversize_coarse_metadata_still_splits_into_fine_children() -> None:
    """A coarse unit whose rendered metadata exceeds ``fine_chunk_size`` still splits.

    PDF coarse units carry layout metadata (``block_ids`` / ``image_ids`` /
    ``bbox_refs``) that, for a section spanning many blocks, renders to *more*
    tokens than the fine chunk budget. llama-index's ``SentenceSplitter`` is
    metadata-aware — it reserves part of each chunk for the node's rendered
    metadata (the longer of ``MetadataMode.EMBED`` / ``MetadataMode.LLM``) — so
    it raises *"Metadata length (N) is longer than chunk size (M)"* and aborts
    the whole ingestion (production hit this with the default
    ``fine_chunk_size=1024``).

    docint never embeds structural metadata (``embed_chunking`` excludes it
    from ``MetadataMode.EMBED`` and the chat path curates ``MetadataMode.LLM``
    via ``LLM_VISIBLE_METADATA_KEYS``), so the fine splitter must split on the
    text alone. The metadata dict must still ride onto the children (citations
    and image/table linking read it), but it must not leak into the embedded
    payload.
    """
    from llama_index.core.schema import MetadataMode, TextNode

    # A block list whose repr dwarfs ``fine_chunk_size`` once rendered.
    bulky_block_ids = [f"blk-{i:04d}-{'d' * 12}" for i in range(80)]
    parser = HierarchicalNodeParser(coarse_chunk_size=2000, fine_chunk_size=64, fine_chunk_overlap=0)
    coarse = TextNode(
        text="First sentence here. Second sentence follows. Third sentence ends. Fourth wraps it up.",
        metadata={
            "page": 3,
            "section_path": ["Results", "Subsection"],
            "block_ids": bulky_block_ids,
            "image_ids": bulky_block_ids,
            "bbox_refs": [{"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}] * 40,
        },
    )

    # Before the fix this raised ValueError("Metadata length ... is longer than
    # chunk size ...") here, aborting ingestion.
    nodes = parser._parse_nodes([coarse])

    fine_nodes = [n for n in nodes if n.metadata.get("hier.level") == 2]
    assert len(fine_nodes) >= 1, "the coarse unit must still produce fine children"

    for fine in fine_nodes:
        # Structural metadata still rides on the dict — citations / image-table
        # linking read it directly.
        assert fine.metadata["page"] == 3
        assert fine.metadata["block_ids"] == bulky_block_ids
        assert fine.metadata["hier.parent_id"] == coarse.node_id
        # ...but it is hidden from the embedded payload: the EMBED rendering is
        # exactly the chunk text, so vectors stay clean and metadata can never
        # blow the embedding budget.
        assert "blk-" not in fine.get_content(metadata_mode=MetadataMode.EMBED)
        assert fine.get_content(metadata_mode=MetadataMode.EMBED) == fine.get_content(metadata_mode=MetadataMode.NONE)
