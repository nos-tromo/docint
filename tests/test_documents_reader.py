"""Tests for the CorePDFPipelineReader document hash and node-building logic."""

from pathlib import Path
from typing import Any, cast

from llama_index.core.schema import TextNode

from docint.core.readers.documents import CorePDFPipelineReader
from docint.core.storage.hierarchical import HierarchicalNodeParser


def _long_pdf_chunk() -> dict[str, Any]:
    """Return a coarse-unit chunk payload long enough to split into >1 fine node."""
    body = " ".join(f"Sentence number {i} is complete and clear." for i in range(200))
    return {"chunk_id": "c1", "text": body, "page_range": [0], "section_path": ["Intro"]}


def test_build_nodes_attaches_file_hash_as_doc_id(tmp_path: Path) -> None:
    """The doc_id should appear as file_hash in every node's metadata.

    Args:
        tmp_path (Path): Temporary directory path for the test.
    """
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    chunks = [
        {
            "chunk_id": "c1",
            "text": "First chunk.",
            "page_range": [0],
        },
        {
            "chunk_id": "c2",
            "text": "Second chunk.",
            "page_range": [1],
        },
    ]

    docs, nodes = CorePDFPipelineReader._build_nodes(
        file_path=pdf_path,
        doc_id="abc123",
        pipeline_version="1.0.0",
        chunks=chunks,
    )

    assert len(docs) == 2
    assert len(nodes) == 2
    for node in nodes:
        assert node.metadata["file_hash"] == "abc123"


def test_build_nodes_skips_empty_text_chunks(tmp_path: Path) -> None:
    """Chunks with empty or whitespace-only text should be omitted."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    chunks = [
        {"chunk_id": "c1", "text": "Real content.", "page_range": [0]},
        {"chunk_id": "c2", "text": "", "page_range": [1]},
        {"chunk_id": "c3", "text": "   ", "page_range": [2]},
    ]

    _docs, nodes = CorePDFPipelineReader._build_nodes(
        file_path=pdf_path,
        doc_id="hash1",
        pipeline_version="1.0.0",
        chunks=chunks,
    )

    assert len(nodes) == 1
    assert cast(TextNode, nodes[0]).text == "Real content."


def test_iter_pdf_files_single_file(tmp_path: Path) -> None:
    """When data_dir is a single PDF, only that file is returned."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    txt = tmp_path / "readme.txt"
    txt.write_text("not a pdf")

    assert CorePDFPipelineReader._iter_pdf_files(pdf) == [pdf]
    assert CorePDFPipelineReader._iter_pdf_files(txt) == []


def test_iter_pdf_files_directory(tmp_path: Path) -> None:
    """When data_dir is a directory, all nested PDFs are found."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (sub / "b.pdf").write_bytes(b"%PDF")
    (tmp_path / "c.txt").write_text("nope")

    result = CorePDFPipelineReader._iter_pdf_files(tmp_path)
    names = [p.name for p in result]
    assert "a.pdf" in names
    assert "b.pdf" in names
    assert "c.txt" not in names


def test_build_nodes_flat_without_parser(tmp_path: Path) -> None:
    """Without a parser the reader keeps the legacy flat one-node-per-chunk shape."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    chunks = [{"chunk_id": "c1", "text": "Flat content.", "page_range": [0]}]
    _docs, nodes = CorePDFPipelineReader._build_nodes(
        file_path=pdf_path,
        doc_id="hash1",
        pipeline_version="2.0.0",
        chunks=chunks,
    )

    assert len(nodes) == 1
    assert "docint_hier_type" not in nodes[0].metadata


def test_build_nodes_emits_coarse_and_fine_with_parser(tmp_path: Path) -> None:
    """With a parser the reader emits coarse parents plus sentence-clean fine children."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    parser = HierarchicalNodeParser(coarse_chunk_size=100_000, fine_chunk_size=1024, fine_chunk_overlap=0)
    _docs, nodes = CorePDFPipelineReader._build_nodes(
        file_path=pdf_path,
        doc_id="hashX",
        pipeline_version="2.0.0",
        chunks=[_long_pdf_chunk()],
        hierarchical_node_parser=parser,
    )

    coarse = [n for n in nodes if n.metadata.get("docint_hier_type") == "coarse"]
    fine = [n for n in nodes if n.metadata.get("docint_hier_type") == "fine"]
    assert len(coarse) == 1
    assert len(fine) > 1

    coarse_ids = {c.node_id for c in coarse}
    for f in fine:
        # Fine children link to a coarse parent for query-time expansion.
        assert f.metadata["hier.parent_id"] in coarse_ids
        # PDF locator/section metadata is inherited by the embedded children.
        assert f.metadata["file_hash"] == "hashX"
        assert f.metadata["page"] == 1
        assert f.metadata["section_path"] == ["Intro"]
        # Regression: fine chunks begin at a sentence boundary, never mid-sentence.
        assert cast(TextNode, f).text.strip().startswith("Sentence number")


def test_build_nodes_coarse_excluded_from_vector_set(tmp_path: Path) -> None:
    """Coarse parents are filtered out of the vector set but kept for the docstore."""
    from docint.core.rag import RAG

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    parser = HierarchicalNodeParser(coarse_chunk_size=100_000, fine_chunk_size=1024, fine_chunk_overlap=0)
    _docs, nodes = CorePDFPipelineReader._build_nodes(
        file_path=pdf_path,
        doc_id="hashY",
        pipeline_version="2.0.0",
        chunks=[_long_pdf_chunk()],
        hierarchical_node_parser=parser,
    )

    vector_nodes = RAG._select_vector_nodes(nodes)
    vector_ids = {id(n) for n in vector_nodes}
    coarse = [n for n in nodes if n.metadata.get("docint_hier_type") == "coarse"]
    fine = [n for n in nodes if n.metadata.get("docint_hier_type") == "fine"]
    # Coarse parents never reach the vector store (they go to the docstore via
    # _docstore_batch_for_persist's non-vector-candidate path); fine children do.
    assert coarse and all(id(c) not in vector_ids for c in coarse)
    assert fine and all(id(f) in vector_ids for f in fine)
