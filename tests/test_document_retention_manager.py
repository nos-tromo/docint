"""Tests for the document retention index: every document's ingest time and deadline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from docint.core.retention import RetentionPeriodState
from docint.core.state.base import Base
from docint.core.state.document_retention_manager import DocumentRetentionManager, PayloadDocument

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
DAY = timedelta(days=1)
PERIOD = RetentionPeriodState(days=365, set_at=NOW - 400 * DAY)


class _Stub:
    """Minimal RAG stand-in exposing only the session-store URL."""

    session_store = "sqlite://"


@pytest.fixture
def mgr() -> DocumentRetentionManager:
    """A manager backed by a shared in-memory SQLite DB (StaticPool)."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    m = DocumentRetentionManager(rag=cast(Any, _Stub()))
    m._SessionMaker = sessionmaker(bind=engine)
    return m


def _record(
    mgr: DocumentRetentionManager, file_hash: str, *, at: datetime, physical: str = "c1", owner: str = "alice"
) -> None:
    mgr.record_ingest(
        physical, "docs", owner, file_hash, f"{file_hash}.pdf", [file_hash], now=at, period=PERIOD, notice_days=30
    )


def test_ingest_records_one_row_per_document(mgr: DocumentRetentionManager) -> None:
    """A completed ingest is indexed with its time, its deadline and every hash its points carry."""
    mgr.record_ingest("c1", "docs", "alice", "h1", "a.pdf", ["h1", "clip"], now=NOW, period=PERIOD, notice_days=30)
    (row,) = mgr.rows_for("c1")
    assert (row.file_hash, row.filename, row.owner, row.logical) == ("h1", "a.pdf", "alice", "docs")
    assert sorted(row.point_hashes) == ["clip", "h1"]
    assert (row.ingested_at, row.expires_at) == (NOW, NOW + 365 * DAY)


def test_reingest_of_identical_content_keeps_the_clock(mgr: DocumentRetentionManager) -> None:
    """Re-ingesting the same file into the same collection does not move its deadline."""
    _record(mgr, "h1", at=NOW - 100 * DAY)
    _record(mgr, "h1", at=NOW)
    (row,) = mgr.rows_for("c1")
    assert row.ingested_at == NOW - 100 * DAY


def test_same_file_in_two_collections_has_two_clocks(mgr: DocumentRetentionManager) -> None:
    """The clock is per collection."""
    _record(mgr, "h1", at=NOW - 100 * DAY, physical="c1")
    _record(mgr, "h1", at=NOW, physical="c2")
    assert mgr.rows_for("c1")[0].ingested_at != mgr.rows_for("c2")[0].ingested_at


def test_off_records_the_time_but_no_deadline(mgr: DocumentRetentionManager) -> None:
    """With retention off the ingest time is still kept; the deadline is set once a period is in force."""
    mgr.record_ingest(
        "c1", "docs", None, "h1", "a.pdf", ["h1"], now=NOW, period=RetentionPeriodState(None, NOW), notice_days=30
    )
    (row,) = mgr.rows_for("c1")
    assert (row.ingested_at, row.expires_at) == (NOW, None)
    assert mgr.recompute(PERIOD, notice_days=30) == 1
    assert mgr.rows_for("c1")[0].expires_at == NOW + 365 * DAY


def test_due_warning_oldest_and_expiry_of(mgr: DocumentRetentionManager) -> None:
    """The queries the sweep, the notice and the inheritance rely on."""
    _record(mgr, "old", at=NOW - 370 * DAY)
    _record(mgr, "soon", at=NOW - 350 * DAY)
    _record(mgr, "young", at=NOW - 10 * DAY)
    _record(mgr, "bob", at=NOW - 370 * DAY, owner="bob")
    assert sorted(row.file_hash for row in mgr.due(NOW, limit=10)) == ["bob", "old"]
    assert sorted(row.file_hash for row in mgr.warning_for("alice", NOW, 30)) == ["old", "soon"]
    oldest = mgr.oldest_live("c1")
    assert oldest is not None and oldest.file_hash in {"old", "bob"}
    expiry = mgr.expiry_of("c1", ["young", "missing"])
    assert set(expiry) == {"young"}
    mgr.delete("c1", "old")
    assert "old" not in {row.file_hash for row in mgr.rows_for("c1")}


def test_repair_uses_payload_time_then_manifest_then_set_at_never_now(mgr: DocumentRetentionManager) -> None:
    """Documents Qdrant holds and the index lacks get rows; a restore never earns a fresh clock."""
    _record(mgr, "known", at=NOW - 5 * DAY)
    docs = [
        PayloadDocument("known", frozenset({"known"}), "k.pdf", NOW - 300 * DAY),
        PayloadDocument("stamped", frozenset({"stamped"}), "s.pdf", NOW - 300 * DAY),
        PayloadDocument("manifest", frozenset({"manifest"}), "m.pdf", None),
        PayloadDocument("bare", frozenset({"bare"}), "b.pdf", None),
    ]
    added = mgr.repair_from_payload(
        "c1", "docs", "alice", docs, manifest_times={"manifest": NOW - 200 * DAY}, period=PERIOD, notice_days=30
    )
    assert added == 3
    rows = {row.file_hash: row for row in mgr.rows_for("c1")}
    assert rows["known"].ingested_at == NOW - 5 * DAY
    assert rows["stamped"].ingested_at == NOW - 300 * DAY
    assert rows["manifest"].ingested_at == NOW - 200 * DAY
    assert rows["bare"].ingested_at == PERIOD.set_at


def test_repair_merges_new_point_hashes_and_a_missing_name(mgr: DocumentRetentionManager) -> None:
    """A row the index already has gets the hashes and name Qdrant knows, never a new clock."""
    mgr.record_ingest("c1", "docs", "alice", "h1", "", ["h1"], now=NOW - 5 * DAY, period=PERIOD, notice_days=30)
    docs = [PayloadDocument("h1", frozenset({"h1", "clip"}), "a.pdf", NOW - 300 * DAY)]
    assert mgr.repair_from_payload("c1", "docs", "alice", docs, manifest_times={}, period=PERIOD, notice_days=30) == 0
    (row,) = mgr.rows_for("c1")
    assert (sorted(row.point_hashes), row.filename, row.ingested_at) == (["clip", "h1"], "a.pdf", NOW - 5 * DAY)
