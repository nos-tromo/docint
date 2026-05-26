"""Tests for the RTF reader."""

from __future__ import annotations

from pathlib import Path

from docint.core.readers.rtf import RTFReader

# Minimal valid RTF document containing a single line of plain text. The header
# matches what TextEdit / WordPad emit so the test exercises a realistic input.
_RTF_SAMPLE = (
    r"{\rtf1\ansi\ansicpg1252\cocoartf2869"
    "\n"
    r"{\fonttbl\f0\fswiss\fcharset0 Helvetica;}"
    "\n"
    r"{\colortbl;\red255\green255\blue255;}"
    "\n"
    r"\f0\fs24 \cf0 It was the best of times.}"
)


def test_rtf_reader_strips_markup_and_returns_plain_text(tmp_path: Path) -> None:
    r"""Return a single Document whose text is the stripped RTF body.

    None of the control words (``\rtf1``, ``\fonttbl``, ``\cocoartf``)
    should leak into the embedded text.
    """
    rtf_path = tmp_path / "sample.rtf"
    rtf_path.write_text(_RTF_SAMPLE, encoding="cp1252")

    docs = RTFReader().load_data(rtf_path)

    assert len(docs) == 1
    text = docs[0].text
    assert "It was the best of times." in text
    assert "\\rtf1" not in text
    assert "\\fonttbl" not in text
    assert "\\cocoartf" not in text


def test_rtf_reader_sets_file_metadata(tmp_path: Path) -> None:
    """Set file_path, file_name, and file_type on the emitted Document.

    The file_type lets the ingestion pipeline dispatcher route the document
    through the text branch.
    """
    rtf_path = tmp_path / "sample.rtf"
    rtf_path.write_text(_RTF_SAMPLE, encoding="cp1252")

    docs = RTFReader().load_data(rtf_path)

    meta = docs[0].metadata
    assert meta["file_path"] == str(rtf_path)
    assert meta["file_name"] == "sample.rtf"
    assert meta["filename"] == "sample.rtf"
    # File hash is populated either by the reader or by the pipeline's
    # file_metadata callback. The reader path must populate it on its own
    # because callers (e.g. tests, ad-hoc invocation) won't supply extra_info.
    assert isinstance(meta["file_hash"], str) and meta["file_hash"]


def test_rtf_reader_honours_extra_info_file_hash(tmp_path: Path) -> None:
    """Reuse a pre-computed file_hash from ``extra_info`` instead of recomputing.

    The hash must round-trip through the reader so the docstore can dedupe
    correctly.
    """
    rtf_path = tmp_path / "sample.rtf"
    rtf_path.write_text(_RTF_SAMPLE, encoding="cp1252")

    docs = RTFReader().load_data(
        rtf_path,
        extra_info={
            "file_path": str(rtf_path),
            "file_name": rtf_path.name,
            "filename": rtf_path.name,
            "file_hash": "deadbeef",
        },
    )

    assert docs[0].metadata["file_hash"] == "deadbeef"
