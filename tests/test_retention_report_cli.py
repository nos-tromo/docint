"""Tests for the read-only ``retention-report`` CLI."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from docint.cli import retention_report
from docint.core.retention import NOTICE_DAYS
from docint.core.state.collection_owner_manager import CollectionActivity, CollectionOwnerManager
from docint.core.state.collection_ownership import CollectionOwnership

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
LONG_AGO = datetime(2020, 1, 1, tzinfo=UTC)


def _row(logical: str, stamp: datetime | None, owner: str = "alice") -> CollectionActivity:
    return CollectionActivity(owner=owner, logical=logical, physical=f"u1__{logical}", last_activity_at=stamp)


def _line(lines: list[str], name: str) -> str:
    return next(line for line in lines if f" {name} " in f" {line} ")


def test_only_the_supported_windows_are_accepted() -> None:
    """A typo is refused here rather than silently evaluated as ``off``."""
    with pytest.raises(SystemExit):
        retention_report.parse_args(["--window", "7m"])
    assert retention_report.parse_args(["--window", "12m"]).window == "12m"


def test_each_collection_is_listed_with_its_last_activity_and_deadline() -> None:
    """Long after the window was set, the deadline is the last activity plus the window."""
    stamp = datetime(2026, 5, 2, tzinfo=UTC)

    lines = retention_report.render_report(
        [_row("alpha", stamp)], window="6m", months=6, window_set_at=LONG_AGO, hypothetical=False, now=NOW, unowned=[]
    )

    assert "2026-05-02" in _line(lines, "alpha")
    assert "2026-11-02" in _line(lines, "alpha")


def test_switching_on_now_holds_every_deletion_for_the_notice_period() -> None:
    """The report shows what the grace period does, not a deletion on the next sweep."""
    grace_end = (NOW + timedelta(days=NOTICE_DAYS)).date().isoformat()

    lines = retention_report.render_report(
        [_row("stale", LONG_AGO)], window="6m", months=6, window_set_at=NOW, hypothetical=True, now=NOW, unowned=[]
    )

    assert grace_end in _line(lines, "stale")
    assert any(f"nothing is deleted before {grace_end}" in line for line in lines)


def test_an_overdue_collection_is_called_out() -> None:
    """Past its deadline means the next sweep deletes it."""
    lines = retention_report.render_report(
        [_row("stale", LONG_AGO), _row("fresh", NOW)],
        window="6m",
        months=6,
        window_set_at=LONG_AGO,
        hypothetical=False,
        now=NOW,
        unowned=[],
    )

    assert "due now" in _line(lines, "stale")
    assert "due now" not in _line(lines, "fresh")
    assert any("1 due now" in line for line in lines)


def test_with_retention_off_there_are_no_deadlines() -> None:
    """The report still shows the clock, and says how to evaluate a window."""
    lines = retention_report.render_report(
        [_row("alpha", LONG_AGO)], window="off", months=0, window_set_at=NOW, hypothetical=False, now=NOW, unowned=[]
    )

    assert "2020-01-01" in _line(lines, "alpha")
    assert any("--window" in line for line in lines)
    assert not any("due now" in line for line in lines)


def test_a_collection_without_activity_never_expires() -> None:
    """``NULL`` is listed as never expiring, not as overdue."""
    lines = retention_report.render_report(
        [_row("alpha", None)], window="6m", months=6, window_set_at=LONG_AGO, hypothetical=False, now=NOW, unowned=[]
    )

    assert "never" in _line(lines, "alpha")


def test_unowned_collections_are_named_as_outside_retention() -> None:
    """A collection with no owner row has no clock, so the operator must hear about it."""
    lines = retention_report.render_report(
        [], window="6m", months=6, window_set_at=LONG_AGO, hypothetical=False, now=NOW, unowned=["cli-batch"]
    )

    assert any("cli-batch" in line and "never" in line for line in lines)


class _StubRAG:
    """Just what the CLI needs from ``RAG``: the session store and Qdrant's collections."""

    def __init__(self, session_store: str, collections: list[str]) -> None:
        self.session_store = session_store
        self._collections = collections
        self.unloaded = False

    def list_collections(self) -> list[str]:
        return self._collections

    def unload_models(self) -> None:
        self.unloaded = True


def test_the_cli_reads_the_store_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Running the report must neither move a clock nor start a grace period."""
    store = f"sqlite:///{tmp_path / 'sessions.sqlite3'}"
    rag = _StubRAG(store, ["cli-batch"])
    seed = CollectionOwnerManager(rag=rag)  # type: ignore[arg-type]
    physical = seed.register("alice", "alpha")
    with seed._session_scope() as s:
        s.query(CollectionOwnership).update({CollectionOwnership.last_activity_at: LONG_AGO})
        s.commit()
    rag._collections.append(physical)
    monkeypatch.setattr(retention_report, "RAG", lambda **_kw: rag)
    monkeypatch.delenv("COLLECTION_RETENTION", raising=False)

    retention_report.main(["--window", "6m"])

    out = capsys.readouterr().out
    assert "alpha" in out and "alice" in out
    assert "cli-batch" in out
    check = CollectionOwnerManager(rag=rag)  # type: ignore[arg-type]
    assert check.retention_window() is None
    [row] = check.list_activity("alice")
    assert row.last_activity_at == LONG_AGO
    assert rag.unloaded is True


def test_the_cli_uses_the_configured_window_and_its_recorded_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without ``--window`` the report shows exactly what the running API will act on."""
    store = f"sqlite:///{tmp_path / 'sessions.sqlite3'}"
    rag: Any = _StubRAG(store, [])
    seed = CollectionOwnerManager(rag=rag)
    seed.register("alice", "alpha")
    seed.record_retention_window("12m", now=LONG_AGO)
    with seed._session_scope() as s:
        s.query(CollectionOwnership).update({CollectionOwnership.last_activity_at: LONG_AGO})
        s.commit()
    monkeypatch.setattr(retention_report, "RAG", lambda **_kw: rag)
    monkeypatch.setenv("COLLECTION_RETENTION", "12m")

    retention_report.main([])

    out = capsys.readouterr().out
    assert "12m" in out
    assert "2021-01-01" in out  # LONG_AGO + 12 months: the recorded start is long past, so no grace applies
