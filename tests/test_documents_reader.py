"""Tests for the CorePDFPipelineReader document hash and node-building logic."""

from pathlib import Path

from docint.core.readers.documents import CorePDFPipelineReader


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

    docs, nodes = CorePDFPipelineReader._build_nodes(
        file_path=pdf_path,
        doc_id="hash1",
        pipeline_version="1.0.0",
        chunks=chunks,
    )

    assert len(nodes) == 1
    assert nodes[0].text == "Real content."


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
