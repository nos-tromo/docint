"""Tests for the ``extract`` CLI entry point.

Every fixture is synthetic: invented hashes, filenames and collection names.
"""

from __future__ import annotations

import json
import types
import zipfile
from pathlib import Path
from typing import Any

import pytest

import docint.cli.extract as cli

_MAIN = [
    (
        "p1",
        {
            "file_hash": "a1b2c3d4",
            "file_name": "report.pdf",
            "page": 1,
            "_node_content": json.dumps({"text": "the body", "start_char_idx": 0}),
        },
    )
]


class _FakeRAG:
    """Stand-in for :class:`~docint.core.rag.RAG` that touches no Qdrant."""

    def __init__(self, *, qdrant_collection: str) -> None:
        """Record the collection and report it as already existing."""
        self.collection = qdrant_collection
        self.unloaded = False
        self.qdrant_client = types.SimpleNamespace(collection_exists=lambda collection_name: True)

    def _image_collection_name(self, collection: str | None = None) -> str:
        """Return the companion name for a collection."""
        return f"{collection or self.collection}_images"

    def unload_models(self) -> None:
        """Record that the CLI released its models."""
        self.unloaded = True


@pytest.fixture(autouse=True)
def _stub_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the RAG class and the Qdrant scan for fakes."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    monkeypatch.setattr(cli, "RAG", _FakeRAG)
    monkeypatch.setattr(
        cli,
        "scroll_collection",
        lambda client, collection, image_collection, source_id=None: (_MAIN, []),
    )


def test_writes_a_bundle_named_for_the_collection(tmp_path: Path) -> None:
    """The archive lands in the output directory under a datestamped name."""
    path = cli.build_extract("mydocs", target=None, out_dir=tmp_path, with_pdf=False)
    assert path.parent == tmp_path
    assert path.name.startswith("mydocs-extract-")
    with zipfile.ZipFile(path) as archive:
        assert any(name.endswith("documents/report.pdf-a1b2c3d4/extract.md") for name in archive.namelist())


def test_no_pdf_skips_the_engine_entirely(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--no-pdf`` must not even probe WeasyPrint."""
    called: dict[str, bool] = {}

    def _probe(document: str) -> bytes:
        called["probed"] = True
        return b"%PDF"

    monkeypatch.setattr(cli, "html_to_pdf", _probe)
    cli.build_extract("mydocs", target=None, out_dir=tmp_path, with_pdf=False)
    assert "probed" not in called


def test_a_missing_pdf_engine_still_writes_the_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The written files are the deliverable; the PDF is a convenience."""
    from docint.core.state.report_render import PdfEngineUnavailableError

    def _unavailable(document: str) -> bytes:
        raise PdfEngineUnavailableError("no pango")

    monkeypatch.setattr(cli, "html_to_pdf", _unavailable)
    path = cli.build_extract("mydocs", target=None, out_dir=tmp_path, with_pdf=True)
    with zipfile.ZipFile(path) as archive:
        assert not any(name.endswith(".pdf") for name in archive.namelist())


def test_a_target_renders_only_that_source(tmp_path: Path) -> None:
    """``--target`` narrows the bundle to one source."""
    path = cli.build_extract("mydocs", target="a1b2c3d4", out_dir=tmp_path, with_pdf=False)
    with zipfile.ZipFile(path) as archive:
        assert any("documents/report.pdf-a1b2c3d4" in name for name in archive.namelist())


def test_an_unknown_target_exits_rather_than_writing_an_empty_bundle(tmp_path: Path) -> None:
    """An empty archive would read as a real but empty result."""
    with pytest.raises(SystemExit):
        cli.build_extract("mydocs", target="nope", out_dir=tmp_path, with_pdf=False)
    assert list(tmp_path.iterdir()) == []


def test_models_are_released_even_when_the_scan_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed run must not strand a loaded model."""
    created: list[_FakeRAG] = []

    class _Recording(_FakeRAG):
        def __init__(self, *, qdrant_collection: str) -> None:
            super().__init__(qdrant_collection=qdrant_collection)
            created.append(self)

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("qdrant boom")

    monkeypatch.setattr(cli, "RAG", _Recording)
    monkeypatch.setattr(cli, "scroll_collection", _boom)
    with pytest.raises(RuntimeError):
        cli.build_extract("mydocs", target=None, out_dir=tmp_path, with_pdf=False)
    assert created[0].unloaded is True


def test_parse_args_defaults() -> None:
    """The flags the runbook documents are the flags the parser accepts."""
    args = cli.parse_args(["mydocs", "--target", "abc", "--no-pdf"])
    assert (args.collection, args.target, args.no_pdf) == ("mydocs", "abc", True)
    assert cli.parse_args([]).collection is None


def test_parse_args_takes_the_appendix_chrome() -> None:
    """Offline there is no report to inherit from, so the flags supply it."""
    args = cli.parse_args(["mydocs", "--reference-number", "AZ-12/26", "--operator", "A. Analyst"])
    assert (args.reference_number, args.operator) == ("AZ-12/26", "A. Analyst")
    assert cli.parse_args(["mydocs"]).reference_number is None
