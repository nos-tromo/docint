from types import SimpleNamespace
from pathlib import Path

import pytest

from docint.core.readers.documents import CustomDoclingReader


def test_file_hash_computed_once_when_no_extra_info(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Test that the file hash is computed only once when no extra_info is provided.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): The temporary path fixture.
    """
    calls = {"count": 0}

    def fake_compute_file_hash(path):
        calls["count"] += 1
        return "deadbeef"

    # Patch compute_file_hash used by the reader module
    monkeypatch.setattr(
        "docint.core.readers.documents.compute_file_hash",
        fake_compute_file_hash,
    )

    # Mock DoclingReader
    class DummyDoclingReader:
        class ExportType:
            JSON = "json"
            MARKDOWN = "markdown"

        def __init__(self, export_type=None):
            pass

        def load_data(self, path):
            return [
                SimpleNamespace(text="page1", metadata={"page": 1}, id_="doc1"),
                SimpleNamespace(text="page2", metadata={"page": 2}, id_="doc2"),
            ]

    monkeypatch.setattr(
        "docint.core.readers.documents.DoclingReader",
        DummyDoclingReader,
    )

    # Create a dummy file to represent the PDF
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"dummy pdf content")

    r = CustomDoclingReader()
    docs = r.load_data(p)

    # Ensure file_hash was computed once and attached to both pages
    assert calls["count"] == 1
    assert len(docs) == 2
    assert docs[0].metadata["file_hash"] == "deadbeef"
    assert docs[1].metadata["file_hash"] == "deadbeef"
    assert docs[0].text == "page1"
    assert docs[1].text == "page2"


def test_file_hash_respects_extra_info_and_not_recomputed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Test that the file hash from extra_info is used and not recomputed.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): The temporary path fixture.
    """
    calls = {"count": 0}

    def fake_compute_file_hash(path):
        calls["count"] += 1
        return "noop"

    monkeypatch.setattr(
        "docint.core.readers.documents.compute_file_hash",
        fake_compute_file_hash,
    )

    # Mock DoclingReader
    class DummyDoclingReader:
        class ExportType:
            JSON = "json"
            MARKDOWN = "markdown"

        def __init__(self, export_type=None):
            pass

        def load_data(self, path):
            return [SimpleNamespace(text="only", metadata={"page": 1}, id_="doc3")]

    monkeypatch.setattr(
        "docint.core.readers.documents.DoclingReader",
        DummyDoclingReader,
    )

    p = tmp_path / "doc2.pdf"
    p.write_bytes(b"another content")

    r = CustomDoclingReader()
    docs = r.load_data(p, extra_info={"file_hash": "explicit-hash"})

    # compute_file_hash should not be called because extra_info supplied it
    assert calls["count"] == 0
    assert docs[0].metadata["file_hash"] == "explicit-hash"
    assert docs[0].text == "only"
