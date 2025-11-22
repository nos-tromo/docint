from types import SimpleNamespace
from pathlib import Path


from docint.core.readers.documents import HybridPDFReader


def _make_dummy_page(text: str, page: int = 1, id_: str | None = None):
    return SimpleNamespace(text=text, metadata={"page": page}, id_=(id_ or f"p{page}"))


def test_file_hash_computed_once_when_no_extra_info(
    monkeypatch, tmp_path: Path
) -> None:
    calls = {"count": 0}

    def fake_compute_file_hash(path):
        calls["count"] += 1
        return "deadbeef"

    # Patch compute_file_hash used by the reader module
    monkeypatch.setattr(
        "docint.core.readers.documents.compute_file_hash",
        fake_compute_file_hash,
    )

    # Patch the PyMuPDF reader to return two pages
    class DummyReader:
        def load_data(self, path):
            return [
                _make_dummy_page("page1", page=1),
                _make_dummy_page("page2", page=2),
            ]

    monkeypatch.setattr(
        "docint.core.readers.documents.pymupdf4llm.LlamaMarkdownReader",
        DummyReader,
    )

    # Create a dummy file to represent the PDF
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"dummy pdf content")

    r = HybridPDFReader()
    docs = r.load_data(p)

    # Ensure file_hash was computed once and attached to both pages
    assert calls["count"] == 1
    assert len(docs) == 2
    assert docs[0].metadata["file_hash"] == "deadbeef"
    assert docs[1].metadata["file_hash"] == "deadbeef"


def test_file_hash_respects_extra_info_and_not_recomputed(
    monkeypatch, tmp_path: Path
) -> None:
    calls = {"count": 0}

    def fake_compute_file_hash(path):
        calls["count"] += 1
        return "noop"

    monkeypatch.setattr(
        "docint.core.readers.documents.compute_file_hash",
        fake_compute_file_hash,
    )

    class DummyReader:
        def load_data(self, path):
            return [_make_dummy_page("only", page=1)]

    monkeypatch.setattr(
        "docint.core.readers.documents.pymupdf4llm.LlamaMarkdownReader",
        DummyReader,
    )

    p = tmp_path / "doc2.pdf"
    p.write_bytes(b"another content")

    r = HybridPDFReader()
    docs = r.load_data(p, extra_info={"file_hash": "explicit-hash"})

    # compute_file_hash should not be called because extra_info supplied it
    assert calls["count"] == 0
    assert docs[0].metadata["file_hash"] == "explicit-hash"
