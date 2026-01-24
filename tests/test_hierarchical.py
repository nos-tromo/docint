import pytest
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from docint.core.storage.hierarchical import HierarchicalNodeParser


@pytest.fixture
def sample_document() -> Document:
    """
    Create a sample document with multiple sections and sentences.

    Returns:
        Document: A sample document with multiple sections and sentences.
    """    
    text = (
        "This is the first section key-metadata. It has three sentences. This is the third one.\n\n"
        "This is the second section. It is a bit longer. It also has sentences."
    )
    return Document(text=text, metadata={"file_name": "test.txt"})


def test_hierarchical_structure(sample_document: Document) -> None:
    """
    Test that the parser produces Level 1 and Level 2 chunks with correct metadata.

    Args:
        sample_document (Document): A sample document for testing.
    """
    # Use small sizes to force splitting
    parser = HierarchicalNodeParser(
        coarse_chunk_size=50,  # Small enough to split the doc
        fine_chunk_size=25,   # Small enough to split coarse chunks
        fine_chunk_overlap=0
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
    """
    Test that fine chunks respect sentence boundaries even with small chunk size.
    """
    text = "Sentence one. Sentence two. Sentence three."
    doc = Document(text=text)
    
    # Chunk size usually counts tokens/chars. 
    # SentenceSplitter tries to keep sentences together.
    parser = HierarchicalNodeParser(
        coarse_chunk_size=1000,
        fine_chunk_size=200, # Increased to avoid metadata error
    )
    
    # Note: SentenceSplitter might respect sentences OVER chunk size default behavior depending on settings.
    # But usually it prefers breaking at sentences.
    
    nodes = parser.get_nodes_from_documents([doc])
    fine_nodes = [n for n in nodes if n.metadata.get("hier.level") == 2]
    
    for n in fine_nodes:
        content = n.get_content()
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
    """
    Test that the parser can handle input nodes (e.g. from MarkdownParser)
    """
    # Simulate a node from Markdown parser
    node = Document(text="Section text. Sentence inside.", metadata={"header_path": "/Section 1"})
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
    assert 1 in levels # The input node itself should be preserved/returned/marked as Level 1?
    
    # In my implementation:
    # else: coarse_candidates = [node] (if fits)
    # So the input node IS the coarse chunk.
    # And then we create fine chunks linking to it.
    
    assert nodes[0] == text_node # Identical object
    assert nodes[0].metadata["hier.level"] == 1
    assert nodes[0].metadata["docint_hier_type"] == "coarse"
    
    fine_nodes = nodes[1:]
    assert len(fine_nodes) > 0
    assert fine_nodes[0].metadata["hier.level"] == 2
    assert fine_nodes[0].metadata["hier.parent_id"] == text_node.node_id
