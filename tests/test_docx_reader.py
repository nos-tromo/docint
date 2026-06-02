"""Tests for the docx reader."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Never, cast

import pytest

from docint.core.ingest.ingestion_pipeline import DocumentIngestionPipeline
from docint.core.readers.docx import DocxReader
from docint.utils.clean_text import basic_clean
from docint.utils.env_cfg import load_ingestion_env

# Minimal OOXML ``.docx`` parts (a Heading-1 + one prose paragraph) assembled
# with the stdlib ``zipfile`` module. This keeps the suite free of
# ``python-docx`` and of any committed binary fixture; docling parses these
# pure-XML parts with no model load (verified: convert() runs offline in ms).
_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    "</Types>"
)
_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    'Target="word/document.xml"/></Relationships>'
)
_DOCUMENT = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Chapter One</w:t></w:r></w:p>'
    "<w:p><w:r><w:t>It was the best of times, it was the worst of times.</w:t></w:r></w:p>"
    "</w:body></w:document>"
)

_PROSE = "best of times"


def _write_docx(path: Path) -> Path:
    """Write a minimal valid ``.docx`` (OOXML zip) to ``path`` and return it.

    Args:
        path: Destination ``.docx`` path inside a temp directory.

    Returns:
        Path: The same ``path``, now holding a readable docx.
    """
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _CONTENT_TYPES)
        archive.writestr("_rels/.rels", _RELS)
        archive.writestr("word/document.xml", _DOCUMENT)
    return path


def _raise_conversion_error(*_args: object, **_kwargs: object) -> Never:
    """Stand in for a docling converter whose ``convert`` always fails."""
    raise RuntimeError("conversion failed")


class _DictlessDoc:
    """Fake DoclingDocument whose JSON export fails but markdown succeeds."""

    def export_to_dict(self) -> dict[str, object]:
        """Raise to force the reader onto its markdown fallback."""
        raise ValueError("no dict")

    def export_to_markdown(self) -> str:
        """Return markdown so the reader emits non-JSON text."""
        return "# Chapter One\n\nIt was the best of times."


def test_docx_reader_emits_text_not_zip_bytes(tmp_path: Path) -> None:
    """Emit one Document whose text is the parsed prose, never raw ZIP bytes.

    Reproduces the reported bug: the docx ZIP container (``PK`` headers,
    ``[Content_Types].xml``, ``word/document.xml``) used to leak verbatim into
    the embedded text.
    """
    docx_path = _write_docx(tmp_path / "word_doc.docx")

    docs = DocxReader().load_data(docx_path)

    assert len(docs) == 1
    text = docs[0].text
    assert "[Content_Types]" not in text
    assert "PK\x03\x04" not in text
    assert "word/document.xml" not in text
    assert _PROSE in text


def test_docx_reader_emits_valid_docling_json(tmp_path: Path) -> None:
    """The emitted text is JSON-parseable so it routes to the DoclingNodeParser.

    ``_create_nodes_without_enrichment`` sends JSON-parseable document text to
    the Docling parser and everything else to the Markdown parser.
    """
    docx_path = _write_docx(tmp_path / "word_doc.docx")

    text = DocxReader().load_data(docx_path)[0].text
    payload = json.loads(text)

    assert isinstance(payload, dict)
    assert _PROSE in text


def test_docx_reader_sets_document_metadata(tmp_path: Path) -> None:
    """Set the standard ingestion metadata keys with ``source='document'``."""
    docx_path = _write_docx(tmp_path / "word_doc.docx")

    meta = DocxReader().load_data(docx_path)[0].metadata

    assert meta["file_path"] == str(docx_path)
    assert meta["file_name"] == "word_doc.docx"
    assert meta["filename"] == "word_doc.docx"
    assert meta["source"] == "document"
    assert isinstance(meta["file_hash"], str) and meta["file_hash"]


def test_docx_reader_honours_extra_info_file_hash(tmp_path: Path) -> None:
    """Reuse a pre-computed ``file_hash`` from ``extra_info`` instead of recomputing."""
    docx_path = _write_docx(tmp_path / "word_doc.docx")

    docs = DocxReader().load_data(
        docx_path,
        extra_info={
            "file_path": str(docx_path),
            "file_name": docx_path.name,
            "filename": docx_path.name,
            "file_hash": "deadbeef",
        },
    )

    assert docs[0].metadata["file_hash"] == "deadbeef"


def test_docx_reader_iter_matches_load_data(tmp_path: Path) -> None:
    """``iter_documents`` and ``load_data`` return identical content."""
    docx_path = _write_docx(tmp_path / "word_doc.docx")

    reader = DocxReader()
    eager = reader.load_data(docx_path)
    streamed = list(reader.iter_documents(docx_path))

    assert len(eager) == len(streamed) == 1
    assert eager[0].text == streamed[0].text


def test_docx_reader_survives_basic_clean(tmp_path: Path) -> None:
    """The emitted JSON stays valid after ``basic_clean``.

    The pipeline runs ``basic_clean`` over every document before parsing; if
    cleaning broke JSON validity the doc would silently miss the Docling path.
    """
    docx_path = _write_docx(tmp_path / "word_doc.docx")

    text = DocxReader().load_data(docx_path)[0].text
    json.loads(basic_clean(text))  # must not raise


def test_docx_reader_skips_on_conversion_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Yield nothing (never raw bytes) when docling conversion raises."""
    docx_path = _write_docx(tmp_path / "word_doc.docx")
    reader = DocxReader()
    monkeypatch.setattr(reader._converter, "convert", _raise_conversion_error)

    assert reader.load_data(docx_path) == []


def test_docx_reader_falls_back_to_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fall back to Markdown text when Docling-JSON serialization fails."""
    docx_path = _write_docx(tmp_path / "word_doc.docx")
    reader = DocxReader()
    monkeypatch.setattr(
        reader._converter,
        "convert",
        lambda *_a, **_k: SimpleNamespace(document=_DictlessDoc()),
    )

    docs = reader.load_data(docx_path)

    assert len(docs) == 1
    text = docs[0].text
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)  # markdown, not JSON
    assert _PROSE in text


def test_all_binary_supported_filetypes_have_a_reader(tmp_path: Path) -> None:
    """Every binary supported filetype must have a dedicated ``file_extractor`` reader.

    Both ``.rtf`` and ``.docx`` regressed by being *declared* supported while
    missing from ``file_extractor``: ``SimpleDirectoryReader`` then decoded the
    binary file as UTF-8 and embedded the raw bytes. This cross-cutting guard
    fails if a new binary type is added to ``supported_filetypes`` without a
    reader, preventing the whole bug class from recurring.
    """
    # SimpleDirectoryReader rejects an empty input dir at construction.
    (tmp_path / "seed.txt").write_text("seed", encoding="utf-8")

    pipeline = DocumentIngestionPipeline(data_dir=tmp_path, ner_model=None, progress_callback=None)
    # Avoid the real ImageIngestionService (it probes the remote CLIP service at
    # construction); _load_doc_readers short-circuits on a truthy stub.
    pipeline.image_ingestion_service = cast(Any, SimpleNamespace())
    pipeline._load_doc_readers()

    assert pipeline.dir_reader is not None
    registered = set(pipeline.dir_reader.file_extractor or {})
    supported = set(load_ingestion_env().supported_filetypes)

    # Types intentionally absent from file_extractor:
    #   .md / .txt -> plain UTF-8 text is the correct representation
    #   .pdf       -> owned by the dedicated CorePDFPipelineReader, not here
    handled_elsewhere = {".md", ".txt", ".pdf"}
    missing = supported - registered - handled_elsewhere

    assert not missing, f"binary supported filetypes with no reader: {sorted(missing)}"
    assert ".docx" in registered
    assert ".rtf" in registered
