"""Tests for the read-only ``retention-report`` CLI."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from docint.cli import retention_report
from docint.core.retention import ExpiringItem, RetentionPeriodState
from docint.core.state.collection_owner_manager import CollectionOwnerManager
from docint.core.state.document_retention_manager import DocumentRetentionManager
from docint.core.state.report_manager import ReportManager

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
DAY = timedelta(days=1)
LONG_AGO = NOW - 400 * DAY


def _item(kind: str, name: str, due: datetime | None, *, owner: str | None = "alice") -> ExpiringItem:
    return ExpiringItem(kind, "u1__docs", "docs", owner, name, name, NOW - 300 * DAY, due)  # type: ignore[arg-type]


def _line(lines: list[str], name: str) -> str:
    return next(line for line in lines if f" {name} " in f" {line} ")


def test_only_a_valid_period_is_accepted() -> None:
    """A typo is refused here rather than silently evaluated as something else."""
    for bad in ("0", "3651", "12m", "abc"):
        with pytest.raises(SystemExit):
            retention_report.parse_args(["--days", bad])
    assert retention_report.parse_args(["--days", "180"]).days == 180


def test_report_cli_lists_documents_sessions_reports() -> None:
    """Each item is listed with its kind, name, start and deadline; overdue ones are called out."""
    lines = retention_report.render_report(
        [
            _item("document", "a.pdf", NOW - DAY),
            _item("session", "s1", NOW + 10 * DAY),
            _item("report", "Weekly", NOW + 100 * DAY),
        ],
        period_days=365,
        notice_days=30,
        set_at=LONG_AGO,
        hypothetical=False,
        now=NOW,
        unowned=[],
    )

    assert any(line.startswith("OWNER") and "KIND" in line and "DUE" in line for line in lines)
    assert "document" in _line(lines, "a.pdf") and "due now" in _line(lines, "a.pdf")
    assert "within 30 days" in _line(lines, "s1")
    assert "2027-02-13" in _line(lines, "Weekly")
    assert any("3 item(s): 1 due now, 1 within 30 days" in line for line in lines)


def test_report_cli_hypothetical_days_applies_grace() -> None:
    """A period evaluated as if switched on now holds every deletion for the notice period."""
    period = RetentionPeriodState(days=180, set_at=NOW)
    items = retention_report.hypothetical_items(
        [_row("old", ingested=LONG_AGO), _row("new", ingested=NOW - 10 * DAY)], period, notice_days=30
    )
    grace_end = (NOW + 30 * DAY).date().isoformat()

    lines = retention_report.render_report(
        items, period_days=180, notice_days=30, set_at=NOW, hypothetical=True, now=NOW, unowned=[]
    )

    assert grace_end in _line(lines, "old")
    assert (NOW + 170 * DAY).date().isoformat() in _line(lines, "new")
    assert any(f"nothing is deleted before {grace_end}" in line for line in lines)
    assert any("follow the earliest of their sources" in line for line in lines)


def test_with_retention_off_nothing_is_deleted() -> None:
    """The report says so, and how to evaluate a period."""
    lines = retention_report.render_report(
        [], period_days=None, notice_days=30, set_at=NOW, hypothetical=False, now=NOW, unowned=[]
    )

    assert any("nothing is deleted" in line for line in lines)
    assert any("--days" in line for line in lines)


def test_unowned_collections_are_named() -> None:
    """A collection without an owner expires like any other, but no one is notified."""
    lines = retention_report.render_report(
        [], period_days=365, notice_days=30, set_at=LONG_AGO, hypothetical=False, now=NOW, unowned=["cli-batch"]
    )

    assert any("cli-batch" in line and "no one is notified" in line for line in lines)


def _row(file_hash: str, *, ingested: datetime) -> Any:
    return SimpleNamespace(
        physical="u1__docs",
        logical="docs",
        owner="alice",
        file_hash=file_hash,
        filename=f"{file_hash}.pdf",
        ingested_at=ingested,
        expires_at=None,
    )


class _StubRAG:
    """Just what the CLI needs from ``RAG``: the session store."""

    def __init__(self, session_store: str) -> None:
        self.session_store = session_store
        self.qdrant_collection = ""
        self.unloaded = False

    def unload_models(self) -> None:
        self.unloaded = True


def test_the_cli_reads_the_store_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Running the report records no period and moves no deadline."""
    rag: Any = _StubRAG(f"sqlite:///{tmp_path / 'sessions.sqlite3'}")
    owners = CollectionOwnerManager(rag=rag)
    physical = owners.register("alice", "docs")
    owners.record_retention_period(365, now=LONG_AGO)
    state = owners.retention_period()
    assert state is not None
    DocumentRetentionManager(rag=rag).record_ingest(
        physical, "docs", "alice", "h1", "a.pdf", ["h1"], now=NOW - 360 * DAY, period=state, notice_days=30
    )
    DocumentRetentionManager(rag=rag).record_ingest(
        "cli-batch", "cli-batch", None, "h2", "b.pdf", ["h2"], now=NOW - DAY, period=state, notice_days=30
    )
    report_id = int(ReportManager(rag=rag).create_report(title="Weekly", owner="alice", collection_name="docs")["id"])
    ReportManager(rag=rag).set_report_expiry(report_id, NOW + 5 * DAY)
    monkeypatch.setattr(retention_report, "RAG", lambda **_kw: rag)
    monkeypatch.delenv("DOCUMENT_RETENTION_DAYS", raising=False)

    retention_report.main([])

    out = capsys.readouterr().out
    assert "h1" in out and "alice" in out and str(report_id) in out
    # Not audited, so references only: no file name, no title.
    assert "a.pdf" not in out and "Weekly" not in out
    assert "cli-batch" in out
    after = CollectionOwnerManager(rag=rag).retention_period()
    assert after is not None and after.set_at == state.set_at
    assert rag.unloaded is True

    retention_report.main(["--days", "30"])

    assert "as if switched on now" in capsys.readouterr().out
    after = CollectionOwnerManager(rag=rag).retention_period()
    assert after is not None and after.days == 365
