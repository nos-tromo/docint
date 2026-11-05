"""Tests for derived items inheriting the earliest deadline of the documents they draw on."""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from docint.core.retention import RetentionPeriodState
from docint.core.retention_sources import report_item_sources, source_expiry
from docint.core.state.base import Base
from docint.core.state.conversation import Conversation
from docint.core.state.document_retention_manager import DocumentRetentionManager
from docint.core.state.session_manager import SessionManager

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
DAY = timedelta(days=1)
PERIOD = RetentionPeriodState(days=365, set_at=NOW - 400 * DAY)


@pytest.fixture
def stores() -> Generator[tuple[MagicMock, DocumentRetentionManager, SessionManager], None, None]:
    """A RAG mock whose retention index and session store share one in-memory database.

    Yields:
        The RAG mock, the retention manager and a session manager.
    """
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    rag = MagicMock()
    rag.index = None
    rag.qdrant_collection = "phys"
    rag._infer_collection_profile.return_value = {"coverage_unit": "documents"}
    cast(Any, rag.get_source_by_node_id).return_value = None
    retention = DocumentRetentionManager(rag=cast(Any, rag))
    retention._SessionMaker = sessionmaker(bind=engine)
    rag.ensure_document_retention_manager.return_value = retention
    sessions = SessionManager(rag=rag)
    sessions._SessionMaker = sessionmaker(bind=engine)
    for file_hash, days_ago in (("old", 300), ("mid", 200), ("new", 10)):
        retention.record_ingest(
            "phys",
            "docs",
            "alice",
            file_hash,
            f"{file_hash}.pdf",
            [file_hash],
            now=NOW - days_ago * DAY,
            period=PERIOD,
            notice_days=30,
        )
    yield rag, retention, sessions
    engine.dispose()


def _response(file_hashes: list[str]) -> MagicMock:
    resp = MagicMock()
    resp.metadata = cast(dict[str, Any], {})
    nodes = []
    for index, file_hash in enumerate(file_hashes):
        node = MagicMock()
        node.node_id = f"n{index}"
        node.metadata = {"filename": f"{file_hash}.pdf", "file_hash": file_hash, "source": "document"}
        source = MagicMock()
        source.node = node
        source.score = 0.5
        nodes.append(source)
    resp.source_nodes = nodes
    return resp


def _expiry(sessions: SessionManager, session_id: str) -> datetime | None:
    with sessions._session_scope() as s:
        convo = s.get(Conversation, session_id)
        assert convo is not None
        value = convo.expires_at
    return None if value is None else value.replace(tzinfo=UTC)


def _turn(sessions: SessionManager, session_id: str, cited: list[str]) -> None:
    sessions._persist_turn(session_id, "q", _response(cited), {"response": "a", "reasoning": None}, owner="alice")


def test_source_expiry_takes_the_earliest_cited_document(
    stores: tuple[MagicMock, DocumentRetentionManager, SessionManager],
) -> None:
    """The earliest deadline among the named documents wins."""
    rag, _, _ = stores
    assert source_expiry(rag, "phys", ["mid", "new"], now=NOW) == NOW - 200 * DAY + 365 * DAY


def test_a_source_already_gone_falls_back_to_the_oldest(
    stores: tuple[MagicMock, DocumentRetentionManager, SessionManager],
) -> None:
    """A cited document no longer indexed counts as the collection's oldest one, never as 'never'."""
    rag, _, _ = stores
    assert source_expiry(rag, "phys", ["new", "gone"], now=NOW) == NOW - 300 * DAY + 365 * DAY


def test_no_sources_take_the_oldest(stores: tuple[MagicMock, DocumentRetentionManager, SessionManager]) -> None:
    """Collection-wide items inherit from the collection's oldest live document."""
    rag, _, _ = stores
    assert source_expiry(rag, "phys", [], now=NOW) == NOW - 300 * DAY + 365 * DAY


def test_a_collection_without_documents_gets_its_own_period(
    stores: tuple[MagicMock, DocumentRetentionManager, SessionManager],
) -> None:
    """With nothing indexed to inherit from, the item still expires a period after it was made."""
    rag, _, _ = stores
    assert source_expiry(rag, "empty", [], now=NOW, period_days=365) == NOW + 365 * DAY


def test_turn_inherits_its_cited_documents(stores: tuple[MagicMock, DocumentRetentionManager, SessionManager]) -> None:
    """A chat answer citing two documents expires with the earlier of them."""
    _, _, sessions = stores
    _turn(sessions, "s1", ["mid", "new"])
    assert _expiry(sessions, "s1") == NOW - 200 * DAY + 365 * DAY


def test_uncited_turn_takes_the_oldest_document(
    stores: tuple[MagicMock, DocumentRetentionManager, SessionManager],
) -> None:
    """An answer without citations may have drawn on anything in the collection."""
    _, _, sessions = stores
    _turn(sessions, "s1", [])
    assert _expiry(sessions, "s1") == NOW - 300 * DAY + 365 * DAY


def test_scoped_or_focused_session_takes_the_oldest(
    stores: tuple[MagicMock, DocumentRetentionManager, SessionManager],
) -> None:
    """A session with a scope or a topic focus counts as collection-wide."""
    _, _, sessions = stores
    _turn(sessions, "s1", ["new"])
    with sessions._session_scope() as s:
        convo = s.get(Conversation, "s1")
        assert convo is not None
        convo.topic_focus = "topic-1"
        convo.expires_at = None
        s.commit()
    _turn(sessions, "s1", ["new"])
    assert _expiry(sessions, "s1") == NOW - 300 * DAY + 365 * DAY


def test_later_turn_never_raises_the_expiry(stores: tuple[MagicMock, DocumentRetentionManager, SessionManager]) -> None:
    """Once a session drew on an early document, a later turn citing only new ones keeps the early deadline."""
    _, _, sessions = stores
    _turn(sessions, "s1", ["mid"])
    _turn(sessions, "s1", ["new"])
    assert _expiry(sessions, "s1") == NOW - 200 * DAY + 365 * DAY


def test_report_item_sources() -> None:
    """Chat answers name their sources; findings their chunks; summaries and topics are collection-wide."""
    answer = {"sources": [{"file_hash": "a"}, {"image_id": "img"}, {"file_hash": "b"}]}
    assert report_item_sources("chat_answer", answer) == ({"a", "b"}, set(), True)
    assert report_item_sources("chat_answer", {"sources": [{"file_hash": "a"}]}) == ({"a"}, set(), False)
    assert report_item_sources("entity_finding", {"chunk_id": "p1"}) == (set(), {"p1"}, False)
    assert report_item_sources("hate_speech_finding", {"chunk_id": ""}) == (set(), set(), True)
    assert report_item_sources("summary", {"collection": "docs"}) == (set(), set(), True)
    assert report_item_sources("topics", {"run_id": "r"}) == (set(), set(), True)


def _reports(rag: MagicMock, sessions: SessionManager) -> Any:
    from docint.core.state.report_manager import ReportManager

    rag.ensure_collection_owner_manager.return_value.resolve.return_value = "phys"
    point = MagicMock()
    point.payload = {"file_hash": "mid"}
    rag.qdrant_client.retrieve.return_value = [point]
    reports = ReportManager(rag=rag)
    reports._SessionMaker = sessions._SessionMaker
    return reports


def _report_expiry(reports: Any, report_id: int) -> datetime | None:
    from docint.core.state.report import Report

    with reports._session_scope() as s:
        value = s.get(Report, report_id).expires_at
    return None if value is None else value.replace(tzinfo=UTC)


def test_report_item_inherits_its_sources(stores: tuple[MagicMock, DocumentRetentionManager, SessionManager]) -> None:
    """A chat answer item takes its sources' earliest deadline; a finding's chunk resolves to its document."""
    rag, _, sessions = stores
    reports = _reports(rag, sessions)
    report = reports.create_report(title="r", owner="alice", collection_name="docs")
    report_id = int(report["id"])
    reports.add_item(
        report_id, "alice", artifact_type="chat_answer", dedupe_key="a:1", snapshot={"sources": [{"file_hash": "new"}]}
    )
    assert _report_expiry(reports, report_id) == NOW - 10 * DAY + 365 * DAY
    reports.add_item(report_id, "alice", artifact_type="entity_finding", dedupe_key="e:1", snapshot={"chunk_id": "p1"})
    assert _report_expiry(reports, report_id) == NOW - 200 * DAY + 365 * DAY


def test_summary_item_and_overview_take_the_oldest(
    stores: tuple[MagicMock, DocumentRetentionManager, SessionManager],
) -> None:
    """Collection-wide items and the collection overview inherit from the oldest live document."""
    rag, _, sessions = stores
    reports = _reports(rag, sessions)
    first = int(reports.create_report(title="r1", owner="alice", collection_name="docs")["id"])
    reports.add_items(
        first, "alice", items=[{"artifact_type": "summary", "dedupe_key": "s:1", "snapshot": {"collection": "docs"}}]
    )
    assert _report_expiry(reports, first) == NOW - 300 * DAY + 365 * DAY
    second = int(reports.create_report(title="r2", owner="alice", collection_name="docs")["id"])
    reports.set_collection_overview_snapshot(second, "alice", {"documents": []})
    assert _report_expiry(reports, second) == NOW - 300 * DAY + 365 * DAY


def test_extract_takes_the_earlier_of_its_two_deadlines(
    stores: tuple[MagicMock, DocumentRetentionManager, SessionManager],
) -> None:
    """An extract goes after its own short retention, or with the oldest document if that comes first."""
    from docint.core.retention_sources import extract_expiry

    rag, _, _ = stores
    assert extract_expiry(rag, "phys", now=NOW, retention_days=7) == NOW + 7 * DAY
    assert extract_expiry(rag, "phys", now=NOW + 60 * DAY, retention_days=7) == NOW - 300 * DAY + 365 * DAY


def test_topic_run_records_its_expiry(stores: tuple[MagicMock, DocumentRetentionManager, SessionManager]) -> None:
    """The RAG hands the topic service the collection's deadline, written into run.json."""
    from docint.core.retention_sources import collection_expiry_iso

    rag, _, _ = stores
    assert collection_expiry_iso(rag, "phys", now=NOW) == (NOW - 300 * DAY + 365 * DAY).isoformat()
