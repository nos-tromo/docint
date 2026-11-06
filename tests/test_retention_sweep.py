"""Tests for the daily retention sweep: due documents and the derived items that inherit their deadlines."""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import docint.core.api as api_module
from docint.core import audit
from docint.core.rag import DocumentDeletion
from docint.core.retention import RetentionPeriodState
from docint.core.state.base import Base
from docint.core.state.collection_owner_manager import CollectionOwnerManager
from docint.core.state.conversation import Conversation
from docint.core.state.document_retention_manager import DocumentRetentionManager
from docint.core.state.report import Report
from docint.core.state.report_manager import ReportManager
from docint.core.state.session_manager import SessionManager
from docint.utils.env_cfg import DocumentRetentionConfig

NOW = datetime.now(UTC)
DAY = timedelta(days=1)
STATE = RetentionPeriodState(days=365, set_at=NOW - 400 * DAY)
CFG = DocumentRetentionConfig(days=365, notice_days=30)


class _SweepRAG:
    """A RAG stand-in with real stores on one in-memory database and a spy document delete."""

    def __init__(self, tmp_path: Path) -> None:
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        maker = sessionmaker(bind=engine)
        self.session_store = "sqlite://"
        self.qdrant_collection = ""
        self.owners = CollectionOwnerManager(rag=cast(Any, self))
        self.owners._SessionMaker = maker
        self.retention = DocumentRetentionManager(rag=cast(Any, self))
        self.retention._SessionMaker = maker
        self.reports = ReportManager(rag=cast(Any, self))
        self.reports._SessionMaker = maker
        self.sessions = SessionManager(rag=cast(Any, self))
        self.sessions._SessionMaker = maker
        self.existing: set[str] = set()
        self.deleted_docs: list[tuple[str, list[str]]] = []
        self.deleted_collections: list[str] = []
        self.empty_after = False
        self.entities_dropped = False
        self.resolved_in: list[str] = []
        self.scope: str = ""
        self.fail_delete = False
        self.repaired = 0
        self.path_config = type("P", (), {"extracts": tmp_path / "extracts", "topics": tmp_path / "topics"})()

    def ensure_collection_owner_manager(self) -> CollectionOwnerManager:
        return self.owners

    def ensure_document_retention_manager(self) -> DocumentRetentionManager:
        return self.retention

    def ensure_report_manager(self) -> ReportManager:
        return self.reports

    def ensure_session_manager(self) -> SessionManager:
        return self.sessions

    def list_collections(self) -> list[str]:
        return sorted(self.existing)

    def repair_retention_index(
        self, period: RetentionPeriodState, *, notice_days: int, skip: Callable[[str], bool] | None = None
    ) -> int:
        self.repaired += 1
        return 0

    def delete_documents(self, physical: str, rows: Any) -> DocumentDeletion:
        if self.fail_delete:
            raise RuntimeError("qdrant down")
        self.deleted_docs.append((physical, sorted(row.file_hash for row in rows)))
        if not self.empty_after:
            for row in rows:
                self.retention.delete(physical, row.file_hash)
        return DocumentDeletion(
            points=len(rows), entities_dropped=self.entities_dropped, collection_empty=self.empty_after
        )

    @contextmanager
    def collection_scope(self, physical: str) -> Iterator[None]:
        self.scope = physical
        try:
            yield
        finally:
            self.scope = ""

    def resolve_entities(self) -> Any:
        self.resolved_in.append(self.scope)
        return type("S", (), {"processed": 1, "minted": 1, "attached": 0, "skipped": 0, "entities_touched": 1})()

    def delete_collection(self, name: str) -> None:
        self.existing.discard(name)
        self.deleted_collections.append(name)
        self.retention.delete_collection(name)


class _IdleJobs:
    async def any_active_for(self, physical: str) -> bool:
        return physical == "busy"


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Generator[_SweepRAG, None, None]:
    """Install the stand-in as the API's RAG.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: Pytest-provided temporary directory.

    Yields:
        The stand-in.
    """
    instance = _SweepRAG(tmp_path)
    monkeypatch.setattr(api_module, "rag", instance)
    monkeypatch.setattr(api_module, "_discard_report_pdfs", lambda ids: None)
    monkeypatch.setattr(api_module, "_report_pdf_store", lambda: api_module.ReportPdfStore(tmp_path / "pdfs"))
    monkeypatch.setattr(api_module, "_extract_store", lambda: api_module.ExtractStore(tmp_path / "extracts"))
    yield instance


def _doc(
    fake: _SweepRAG, file_hash: str, *, days_ago: int, physical: str = "phys", owner: str | None = "alice"
) -> None:
    fake.existing.add(physical)
    fake.retention.record_ingest(
        physical,
        "docs",
        owner,
        file_hash,
        f"{file_hash}.pdf",
        [file_hash],
        now=NOW - days_ago * DAY,
        period=STATE,
        notice_days=30,
    )


def _sweep(state: RetentionPeriodState = STATE) -> Any:
    return anyio.run(api_module._retention_sweep, CFG, state, cast(Any, _IdleJobs()))


def _events(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_due_document_is_deleted_and_recorded_before(fake: _SweepRAG, audit_stream: io.StringIO) -> None:
    """A document past its deadline goes; its record names it by collection and hash."""
    _doc(fake, "old", days_ago=370)
    _doc(fake, "young", days_ago=10)
    report = _sweep()
    assert fake.deleted_docs == [("phys", ["old"])]
    assert report.documents == 1
    (event,) = [e for e in _events(audit_stream) if e["operation"] == "retention.document"]
    assert event["actor"] == {"user": None, "via": "system"}
    assert event["target"]["ids"] == ["old"] and event["target"]["container"] == "docs"
    assert event["criteria"] == {"owner": "alice", "period_days": 365}
    assert "purpose" not in event and "request_id" not in event


def test_not_due_during_the_grace_period(fake: _SweepRAG) -> None:
    """A period that took effect today deletes nothing for the notice period."""
    _doc(fake, "old", days_ago=370)
    fake.retention.recompute(RetentionPeriodState(days=365, set_at=NOW), notice_days=30)
    _sweep(RetentionPeriodState(days=365, set_at=NOW))
    assert fake.deleted_docs == []


def test_busy_collection_is_skipped(fake: _SweepRAG) -> None:
    """A collection a job works on is left for the next sweep."""
    _doc(fake, "old", days_ago=370, physical="busy")
    report = _sweep()
    assert fake.deleted_docs == [] and report.skipped_busy == ["busy"]


def test_last_document_takes_the_collection_and_its_sessions(fake: _SweepRAG, audit_stream: io.StringIO) -> None:
    """When the collection is empty afterwards, it is deleted with the existing cascade, recorded."""
    physical = fake.owners.register("alice", "docs")
    _doc(fake, "old", days_ago=370, physical=physical)
    with fake.sessions._session_scope() as s:
        s.add(Conversation(id="s-pinned", owner="alice", collection_name=physical, expires_at=NOW + 300 * DAY))
        s.commit()
    fake.empty_after = True
    report = _sweep()
    assert fake.deleted_collections == [physical]
    with fake.sessions._session_scope() as s:
        assert s.query(Conversation).count() == 0
    assert report.collections == 1
    assert [e["operation"] for e in _events(audit_stream) if e["action"] == "delete"] == [
        "retention.document",
        "retention.document.removed",
        "retention.collection",
    ]


def test_a_failed_delete_is_recorded_as_error_and_retried(fake: _SweepRAG, audit_stream: io.StringIO) -> None:
    """The record is written first; a delete that then fails adds an error event and keeps the row."""
    _doc(fake, "old", days_ago=370)
    fake.fail_delete = True
    report = _sweep()
    assert report.failed == ["phys"]
    outcomes = [e["outcome"] for e in _events(audit_stream) if e["operation"].startswith("retention.document")]
    assert outcomes == ["ok", "error"]
    assert [row.file_hash for row in fake.retention.rows_for("phys")] == ["old"]


def test_unowned_collection_expires_too(fake: _SweepRAG) -> None:
    """A CLI-ingested collection without an owner is swept like any other."""
    _doc(fake, "old", days_ago=370, physical="cli-coll", owner=None)
    _sweep()
    assert fake.deleted_docs == [("cli-coll", ["old"])]


def test_the_sweep_repairs_the_index_first(fake: _SweepRAG) -> None:
    """Restored or unindexed documents get rows before anything is judged due."""
    _sweep()
    assert fake.repaired == 1


def test_due_session_and_report_go_with_their_records(fake: _SweepRAG, audit_stream: io.StringIO) -> None:
    """Sessions and reports past their inherited deadline are deleted with their existing paths."""
    with fake.sessions._session_scope() as s:
        s.add(Conversation(id="s-old", owner="alice", collection_name="phys", expires_at=NOW - DAY))
        s.add(Conversation(id="s-young", owner="alice", collection_name="phys", expires_at=NOW + DAY))
        s.commit()
    old = fake.reports.create_report(title="r", owner="alice", collection_name="docs")
    with fake.reports._session_scope() as s:
        row = s.get(Report, int(old["id"]))
        assert row is not None
        row.expires_at = NOW - DAY
        s.commit()
    report = _sweep()
    with fake.sessions._session_scope() as s:
        assert {c.id for c in s.query(Conversation).all()} == {"s-young"}
    assert fake.reports.get_report(int(old["id"]), "alice") is None
    assert (report.sessions, report.reports) == (1, 1)
    operations = [e["operation"] for e in _events(audit_stream) if e["action"] == "delete"]
    assert "retention.session" in operations and "retention.report" in operations


def test_due_extract_goes(fake: _SweepRAG, tmp_path: Path) -> None:
    """A stored extract past its deadline is deleted."""
    store = api_module.ExtractStore(tmp_path / "extracts")
    fake.existing.add("phys")
    store.write(
        "phys", zip_bytes=b"zip", meta={"collection": "docs", "expires_at": (NOW - DAY).isoformat()}, now=NOW - 2 * DAY
    )
    store.write("phys", zip_bytes=b"zip", meta={"collection": "docs", "expires_at": (NOW + DAY).isoformat()}, now=NOW)
    report = _sweep()
    assert report.extracts == 1
    assert len(store.list("phys")) == 1


def test_a_failed_record_deletes_nothing(
    fake: _SweepRAG, audit_stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed: without its audit record no document and no session goes."""
    _doc(fake, "old", days_ago=370)
    with fake.sessions._session_scope() as s:
        s.add(Conversation(id="s-old", owner="alice", collection_name="phys", expires_at=NOW - DAY))
        s.commit()

    def refuse(self: Any, **_: Any) -> None:
        raise audit.AuditWriteError("trail unwritable")

    monkeypatch.setattr(api_module.Operation, "record", refuse)
    report = _sweep()
    assert fake.deleted_docs == []
    assert [row.file_hash for row in fake.retention.rows_for("phys")] == ["old"]
    with fake.sessions._session_scope() as s:
        assert s.query(Conversation).count() == 1
    assert report.failed == ["phys", "sessions"]


def test_dropped_entities_are_resolved_again(fake: _SweepRAG, audit_stream: io.StringIO) -> None:
    """No canonical entity survives a deleted document: the collection is resolved again, by the system."""
    _doc(fake, "old", days_ago=370)
    _doc(fake, "young", days_ago=10)
    fake.entities_dropped = True
    _sweep()
    assert fake.resolved_in == ["phys"]
    (event,) = [e for e in _events(audit_stream) if e["operation"] == "entities.resolve"]
    assert (event["action"], event["actor"]["via"]) == ("combine", "system")


def test_resolution_failure_never_fails_the_sweep(fake: _SweepRAG, monkeypatch: pytest.MonkeyPatch) -> None:
    """Entity views stay unresolved until the next resolve; the deletion itself stands."""
    _doc(fake, "old", days_ago=370)
    _doc(fake, "young", days_ago=10)
    fake.entities_dropped = True
    monkeypatch.setattr(_SweepRAG, "resolve_entities", lambda self: (_ for _ in ()).throw(RuntimeError("llm down")))
    report = _sweep()
    assert report.documents == 1 and report.failed == []


def _session_expiry(fake: _SweepRAG, session_id: str) -> datetime | None:
    with fake.sessions._session_scope() as s:
        convo = s.get(Conversation, session_id)
        assert convo is not None
        value = convo.expires_at
    return None if value is None else value.replace(tzinfo=UTC)


def _report_expiry(fake: _SweepRAG, report_id: int) -> datetime | None:
    with fake.reports._session_scope() as s:
        row = s.get(Report, report_id)
        assert row is not None
        value = row.expires_at
    return None if value is None else value.replace(tzinfo=UTC)


def test_a_shortened_period_moves_derived_deadlines_forward(fake: _SweepRAG) -> None:
    """Chats and reports go as much sooner as their sources do, never inside the new notice period."""
    fake.owners.record_retention_period(365, now=NOW - 400 * DAY)
    with fake.sessions._session_scope() as s:
        s.add(Conversation(id="far", owner="alice", collection_name="phys", expires_at=NOW + 300 * DAY))
        s.add(Conversation(id="near", owner="alice", collection_name="phys", expires_at=NOW + 5 * DAY))
        s.commit()
    rid = int(fake.reports.create_report(title="r", owner="alice", collection_name="docs")["id"])
    fake.reports.set_report_expiry(rid, NOW + 300 * DAY)

    api_module._record_retention_period(DocumentRetentionConfig(days=365, notice_days=30))
    assert _session_expiry(fake, "far") == NOW + 300 * DAY

    api_module._record_retention_period(DocumentRetentionConfig(days=180, notice_days=30))
    assert _session_expiry(fake, "far") == NOW + 115 * DAY
    assert _report_expiry(fake, rid) == NOW + 115 * DAY
    assert _session_expiry(fake, "near") == NOW + 5 * DAY

    api_module._record_retention_period(DocumentRetentionConfig(days=730, notice_days=30))
    assert _session_expiry(fake, "far") == NOW + 115 * DAY


def test_switching_retention_off_clears_derived_deadlines(fake: _SweepRAG) -> None:
    """Off deletes nothing; switching on again stamps every chat and report afresh."""
    fake.owners.record_retention_period(365, now=NOW - 400 * DAY)
    with fake.sessions._session_scope() as s:
        s.add(Conversation(id="s1", owner="alice", collection_name="phys", expires_at=NOW + 300 * DAY))
        s.commit()

    api_module._record_retention_period(DocumentRetentionConfig(days=None, notice_days=30))

    assert [sid for sid, *_ in fake.sessions.sessions_without_deadline()] == ["s1"]


def test_a_failing_floor_read_never_stops_the_sweep(fake: _SweepRAG, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the collection floors the sweep still deletes due documents and derived items."""
    _doc(fake, "old", days_ago=370)
    monkeypatch.setattr(
        DocumentRetentionManager, "oldest_live", lambda self, physical: (_ for _ in ()).throw(RuntimeError("db"))
    )
    report = _sweep()
    assert report.documents == 1


def test_collection_wide_items_follow_a_shortened_period(fake: _SweepRAG, tmp_path: Path) -> None:
    """An extract stamped under a longer period still goes with the collection's earliest document."""
    _doc(fake, "old", days_ago=370)
    _doc(fake, "young", days_ago=10)
    store = api_module.ExtractStore(tmp_path / "extracts")
    store.write(
        "phys", zip_bytes=b"zip", meta={"collection": "docs", "expires_at": (NOW + 200 * DAY).isoformat()}, now=NOW
    )
    report = _sweep()
    assert report.documents == 1
    assert report.extracts == 1


def test_items_with_nothing_to_inherit_from_get_their_own_period(fake: _SweepRAG) -> None:
    """A chat or report tied to no collection, or to one without documents, still goes a period after stamping."""
    with fake.sessions._session_scope() as s:
        s.add(Conversation(id="loose", owner="alice", collection_name=None))
        s.add(Conversation(id="emptycoll", owner="alice", collection_name="nodocs"))
        s.commit()
    rid = int(fake.reports.create_report(title="r", owner="alice", collection_name=None)["id"])
    _sweep()
    for session_id in ("loose", "emptycoll"):
        stamped = _session_expiry(fake, session_id)
        assert stamped is not None and abs(stamped - (NOW + 365 * DAY)) < timedelta(minutes=5)
    stamped = _report_expiry(fake, rid)
    assert stamped is not None and abs(stamped - (NOW + 365 * DAY)) < timedelta(minutes=5)


def test_a_failed_collection_cascade_is_retried(fake: _SweepRAG, monkeypatch: pytest.MonkeyPatch) -> None:
    """The last document's row stays until the collection is gone, so the next sweep finishes the cascade."""
    physical = fake.owners.register("alice", "docs")
    _doc(fake, "old", days_ago=370, physical=physical)
    fake.empty_after = True
    real = api_module._purge_collection_held
    calls: list[str] = []

    def flaky(owner: str | None, logical: str, phys: str) -> object:
        calls.append(phys)
        if len(calls) == 1:
            raise RuntimeError("store offline")
        return real(owner, logical, phys)

    monkeypatch.setattr(api_module, "_purge_collection_held", flaky)
    first = _sweep()
    assert first.failed == [physical] and fake.deleted_collections == []
    second = _sweep()
    assert second.collections == 1 and fake.deleted_collections == [physical]
    assert fake.retention.rows_for(physical) == []


def test_every_collection_with_due_documents_is_visited(fake: _SweepRAG, monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-sweep bound applies per collection: one large collection never starves the others."""
    monkeypatch.setattr(api_module, "_RETENTION_SWEEP_LIMIT", 1)
    _doc(fake, "a1", days_ago=380, physical="a")
    _doc(fake, "a2", days_ago=380, physical="a")
    _doc(fake, "b1", days_ago=370, physical="b")
    report = _sweep()
    assert [physical for physical, _ in fake.deleted_docs] == ["a", "b"]
    assert report.documents == 2


def test_one_unreadable_session_never_blocks_the_others(fake: _SweepRAG, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stamping goes item by item: a session that fails is skipped, the rest and every report still get dates."""
    with fake.sessions._session_scope() as s:
        s.add(Conversation(id="bad", owner="alice", collection_name="phys"))
        s.add(Conversation(id="good", owner="alice", collection_name="phys"))
        s.commit()
    rid = int(fake.reports.create_report(title="r", owner="alice", collection_name=None)["id"])
    real = SessionManager.set_session_expiry

    def picky(self: SessionManager, session_id: str, expires_at: datetime) -> None:
        if session_id == "bad":
            raise RuntimeError("locked")
        real(self, session_id, expires_at)

    monkeypatch.setattr(SessionManager, "set_session_expiry", picky)
    _sweep()
    assert _session_expiry(fake, "good") is not None
    assert _report_expiry(fake, rid) is not None


def test_the_busy_check_runs_under_the_purge_guard(fake: _SweepRAG) -> None:
    """No job can start between the busy check and the delete: the guard is taken first."""
    _doc(fake, "old", days_ago=370)
    seen: list[bool] = []

    class _WatchingJobs:
        async def any_active_for(self, physical: str) -> bool:
            seen.append(physical in api_module._purging)
            return False

    anyio.run(api_module._retention_sweep, CFG, STATE, cast(Any, _WatchingJobs()))
    assert seen == [True]
    assert fake.deleted_docs == [("phys", ["old"])]


def test_a_document_purge_names_itself_in_the_409() -> None:
    """Work refused while documents are deleted says so, not that the whole collection goes."""
    from fastapi import HTTPException

    with api_module._purge_guard("phys-x", detail="Documents of this collection are being deleted."):
        with pytest.raises(HTTPException) as caught:
            api_module._refuse_while_purging("phys-x")
    assert caught.value.detail == "Documents of this collection are being deleted."
    with api_module._purge_guard("phys-y"):
        with pytest.raises(HTTPException) as caught:
            api_module._refuse_while_purging("phys-y")
    assert caught.value.detail == "Collection is being deleted."


def test_a_pdf_whose_report_is_gone_is_deleted(
    fake: _SweepRAG, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A render left behind by a report delete that failed to remove it goes with the next sweep."""
    store = api_module.ReportPdfStore(tmp_path / "pdfs")
    monkeypatch.setattr(api_module, "_report_pdf_store", lambda: store)
    live = fake.reports.create_report(title="r", owner="alice", collection_name="docs")
    for report_id in (int(live["id"]), 9999):
        store.write(
            report_id,
            b"%PDF-1.7",
            report_created_at="2026-01-01T00:00:00",
            report_updated_at="2026-01-01T00:00:00",
            filename="r.pdf",
            pages=1,
            now=NOW,
        )
    report = _sweep()
    assert store.ids() == {int(live["id"])}
    assert report.report_pdfs == 1


def test_what_a_document_delete_removed_is_recorded(fake: _SweepRAG, audit_stream: io.StringIO) -> None:
    """After the delete, a child event counts what went: points, images, nodes, files."""
    _doc(fake, "old", days_ago=370)
    _sweep()
    events = _events(audit_stream)
    (primary,) = [e for e in events if e["operation"] == "retention.document"]
    (removed,) = [e for e in events if e["operation"] == "retention.document.removed"]
    assert removed["parent_event_id"] == primary["event_id"]
    assert removed["outcome"] == "ok"
    assert removed["target"]["ids"] == ["old"]
    assert removed["criteria"]["points"] == 1
    assert {"image_points", "image_occurrences", "nodes", "source_files", "artifacts"} <= set(removed["criteria"])
