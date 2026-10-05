"""Tests for the on-disk store of rendered report PDFs."""

from __future__ import annotations

import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from docint.core.state.report_pdf_store import ReportPdfRecord, ReportPdfStore
from docint.utils.env_cfg import load_path_env

_NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
_CREATED = "2026-01-01T09:00:00"
_UPDATED = "2026-01-02T03:00:00"


def _store(tmp_path: Path) -> ReportPdfStore:
    return ReportPdfStore(tmp_path / "report-pdfs")


def _write(
    store: ReportPdfStore, report_id: int = 7, *, pdf: bytes = b"%PDF-1.7 one", created: str = _CREATED
) -> ReportPdfRecord:
    return store.write(
        report_id,
        pdf,
        report_created_at=created,
        report_updated_at=_UPDATED,
        filename=f"report-{report_id}-Case Alpha.pdf",
        pages=3,
        now=_NOW,
    )


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    """A stored render comes back with what the status route reports."""
    store = _store(tmp_path)
    record = _write(store)

    assert record == {
        "report_id": 7,
        "filename": "report-7-Case Alpha.pdf",
        "size": len(b"%PDF-1.7 one"),
        "created_at": "2026-01-02T03:04:05+00:00",
        "report_created_at": _CREATED,
        "report_updated_at": _UPDATED,
        "pages": 3,
    }
    assert store.get(7, _CREATED) == record
    assert store.path(7, _CREATED).read_bytes() == b"%PDF-1.7 one"


def test_a_new_render_replaces_the_last_and_leaves_no_temp_file(tmp_path: Path) -> None:
    """One PDF per report: rendering again overwrites it in place."""
    store = _store(tmp_path)
    _write(store, pdf=b"%PDF old")
    _write(store, pdf=b"%PDF new")

    assert store.path(7, _CREATED).read_bytes() == b"%PDF new"
    assert sorted(path.suffix for path in (tmp_path / "report-pdfs").iterdir()) == [".json", ".pdf"]


def test_a_reused_report_id_never_serves_the_old_reports_pdf(tmp_path: Path) -> None:
    """SQLite hands a deleted report's id to the next one, so a PDF is matched by creation time too."""
    store = _store(tmp_path)
    _write(store, created=_CREATED)

    assert store.get(7, "2026-03-03T00:00:00") is None


def test_an_unknown_report_has_no_pdf(tmp_path: Path) -> None:
    """Nothing rendered, nothing to download."""
    assert _store(tmp_path).get(8, _CREATED) is None


def test_a_sidecar_whose_pdf_is_gone_reads_as_missing(tmp_path: Path) -> None:
    """A record must never offer a download that would 404."""
    store = _store(tmp_path)
    _write(store)
    store.path(7, _CREATED).unlink()

    assert store.get(7, _CREATED) is None


def test_delete_removes_the_pdf_and_its_sidecar(tmp_path: Path) -> None:
    """A deleted report leaves nothing rendered from it behind."""
    store = _store(tmp_path)
    _write(store)

    assert store.delete(7) is True
    assert list((tmp_path / "report-pdfs").iterdir()) == []
    assert store.delete(7) is False


def test_a_guarded_delete_spares_the_pdf_of_a_report_that_reused_the_id(tmp_path: Path) -> None:
    """Cleaning up after a deleted report must not remove a newer report's render."""
    store = _store(tmp_path)
    _write(store, created="2026-03-03T00:00:00")

    assert store.delete(7, report_created_at=_CREATED) is False
    assert store.get(7, "2026-03-03T00:00:00") is not None


def test_two_reports_sharing_an_id_never_share_a_file(tmp_path: Path) -> None:
    """A path checked for one report holds only that report's PDF, whatever is rendered under the reused id.

    Were the files keyed by id alone, a render for the report that inherited
    the id could replace the PDF between a download's check and its read, and
    serve one owner another owner's evidence.
    """
    store = _store(tmp_path)
    _write(store, pdf=b"%PDF the deleted report", created=_CREATED)
    _write(store, pdf=b"%PDF the new report", created="2026-03-03T00:00:00")

    assert store.path(7, _CREATED).read_bytes() == b"%PDF the deleted report"
    assert store.path(7, "2026-03-03T00:00:00").read_bytes() == b"%PDF the new report"


def test_an_unguarded_delete_removes_every_render_under_the_id(tmp_path: Path) -> None:
    """Deleting a report clears whatever was ever rendered under its id, orphans included."""
    store = _store(tmp_path)
    _write(store, created=_CREATED)
    _write(store, created="2026-03-03T00:00:00")

    assert store.delete(7) is True
    assert list((tmp_path / "report-pdfs").iterdir()) == []


def test_the_store_is_private_to_the_process_user(tmp_path: Path) -> None:
    """Rendered evidence sits in scratch space other local users must not read."""
    store = _store(tmp_path)
    _write(store)

    assert stat.S_IMODE((tmp_path / "report-pdfs").stat().st_mode) == 0o700
    for path in (tmp_path / "report-pdfs").iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, path.name


def test_the_root_defaults_to_scratch_space(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PDF can always be rendered again, so it lives in the temp directory, not on the data volume."""
    monkeypatch.delenv("REPORT_PDF_DIR", raising=False)

    assert load_path_env().report_pdfs == Path(tempfile.gettempdir()) / "docint-report-pdfs"


def test_report_pdf_dir_relocates_the_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``REPORT_PDF_DIR`` moves the store, like ``EXTRACT_DIR`` moves extracts."""
    monkeypatch.setenv("REPORT_PDF_DIR", str(tmp_path / "r"))

    assert load_path_env().report_pdfs == tmp_path / "r"
