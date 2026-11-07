"""Tests for how a change of the retention period moves deadlines: what is already due or announced stays."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from docint.core.retention import RetentionPeriodState, deadline_under, next_period_state
from docint.core.state.base import Base, _ensure_derived_expiry_columns, _ensure_retention_state_columns
from docint.core.state.collection_owner_manager import CollectionOwnerManager
from docint.core.state.document_retention_manager import DocumentRetentionManager

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
DAY = timedelta(days=1)


def _settled(days: int) -> RetentionPeriodState:
    """A period in force for long, its notice long over."""
    return RetentionPeriodState(days, NOW - 400 * DAY, NOW - 370 * DAY)


def test_first_period_waits_the_notice() -> None:
    """A period seen for the first time deletes nothing for the notice."""
    state = next_period_state(None, 365, 30, NOW)
    assert (state.set_at, state.grace_until, state.in_force_days) == (NOW, NOW + 30 * DAY, None)
    assert deadline_under(NOW - 400 * DAY, state, 30) == NOW + 30 * DAY


def test_shortening_spares_only_what_falls_due_sooner() -> None:
    """What the period in force made due keeps its date; what the shorter one makes due waits the notice."""
    state = next_period_state(_settled(365), 30, 30, NOW)
    assert deadline_under(NOW - 400 * DAY, state, 30) == NOW - 35 * DAY
    assert deadline_under(NOW - 200 * DAY, state, 30) == NOW + 30 * DAY
    assert deadline_under(NOW - 10 * DAY, state, 30) == NOW + 30 * DAY


def test_lengthening_takes_effect_at_once() -> None:
    """A longer period never pauses deletion: what is older than it is due at once."""
    state = next_period_state(_settled(30), 365, 30, NOW)
    assert deadline_under(NOW - 400 * DAY, state, 30) == NOW
    assert deadline_under(NOW - 100 * DAY, state, 30) == NOW + 265 * DAY


def test_changing_back_and_forth_postpones_nothing() -> None:
    """365 → 364 → 365 within the notice: the 365 days still decide on time."""
    first = next_period_state(_settled(365), 364, 30, NOW)
    back = next_period_state(first, 365, 30, NOW + DAY)
    assert deadline_under(NOW - 400 * DAY, back, 30) == NOW + DAY


def test_raising_then_lowering_gives_the_notice() -> None:
    """30 → 365 → 30: the 365 days were in force, so a document they keep gets the full notice."""
    raised = next_period_state(_settled(30), 365, 30, NOW)
    lowered = next_period_state(raised, 30, 30, NOW + 10 * DAY)
    assert deadline_under(NOW - 100 * DAY, lowered, 30) == NOW + 40 * DAY


def test_a_change_during_the_first_notice_keeps_its_end() -> None:
    """Changing the period before the first notice ended never moves that end later."""
    first = next_period_state(None, 365, 30, NOW)
    changed = next_period_state(first, 30, 30, NOW + 20 * DAY)
    assert deadline_under(NOW - 400 * DAY, changed, 30) == NOW + 30 * DAY


def test_a_lower_notice_never_ends_a_running_notice_sooner() -> None:
    """Changing the period and lowering the notice together keeps the notice already running."""
    first = next_period_state(None, 365, 30, NOW)
    changed = next_period_state(first, 30, 7, NOW + DAY)
    assert changed.grace_until == NOW + 30 * DAY


def test_off_then_on_gives_a_fresh_notice() -> None:
    """Switching off and on again starts a notice of its own."""
    off = next_period_state(_settled(365), None, 30, NOW)
    assert off.days is None
    on = next_period_state(off, 365, 30, NOW + DAY)
    assert (on.grace_until, on.in_force_days) == (NOW + 31 * DAY, None)


@pytest.fixture
def owners() -> CollectionOwnerManager:
    """An ownership manager on an in-memory store."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    manager = CollectionOwnerManager(rag=cast(Any, type("R", (), {"session_store": "sqlite://"})()))
    manager._SessionMaker = sessionmaker(bind=engine)
    return manager


def test_a_notice_change_alone_moves_nothing(owners: CollectionOwnerManager) -> None:
    """Lowering RETENTION_NOTICE_DAYS during the notice keeps the end it announced."""
    first = owners.record_retention_period(365, notice_days=30, now=NOW)
    again = owners.record_retention_period(365, notice_days=7, now=NOW + 10 * DAY)
    assert again.grace_until == first.grace_until == NOW + 30 * DAY


def test_lowering_the_notice_never_makes_documents_due_now(owners: CollectionOwnerManager) -> None:
    """The recomputed deadlines keep the notice already announced."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    documents = DocumentRetentionManager(rag=cast(Any, type("R", (), {"session_store": "sqlite://"})()))
    documents._SessionMaker = sessionmaker(bind=engine)
    state = owners.record_retention_period(365, notice_days=30, now=NOW)
    documents.record_ingest(
        "c1", "docs", "alice", "h1", "a.pdf", ["h1"], now=NOW - 400 * DAY, period=state, notice_days=30
    )
    later = owners.record_retention_period(365, notice_days=7, now=NOW + 20 * DAY)
    documents.recompute(later, notice_days=7)
    (row,) = documents.rows_for("c1")
    assert row.expires_at == NOW + 30 * DAY


def test_a_legacy_row_is_upgraded_with_its_notice_end(owners: CollectionOwnerManager) -> None:
    """A period recorded by the release before gets its notice end written once."""
    with owners._session_scope() as s:
        s.execute(
            text("INSERT INTO retention_state (id, window, window_set_at) VALUES (1, '365d', :at)"),
            {"at": NOW - 5 * DAY},
        )
        s.commit()
    state = owners.record_retention_period(365, notice_days=30, now=NOW)
    assert state.grace_until == NOW + 25 * DAY
    assert owners.record_retention_period(365, notice_days=1, now=NOW + DAY).grace_until == NOW + 25 * DAY


def test_older_stores_get_the_new_columns_and_the_expiry_index() -> None:
    """Migrating a store from before adds the period columns and the conversations' expiry index."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE retention_state (id INTEGER PRIMARY KEY, window VARCHAR NOT NULL,"
                " window_set_at DATETIME NOT NULL)"
            )
        )
        conn.execute(text("CREATE TABLE conversations (id VARCHAR PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE reports (id INTEGER PRIMARY KEY)"))
    _ensure_retention_state_columns(engine)
    _ensure_derived_expiry_columns(engine)
    inspector = inspect(engine)
    columns = {col["name"] for col in inspector.get_columns("retention_state")}
    assert {"grace_until", "in_force_days", "in_force_grace"} <= columns
    assert "ix_conversations_expires_at" in {index["name"] for index in inspector.get_indexes("conversations")}


def test_a_clock_stepped_back_keeps_what_is_due() -> None:
    """A state dated a few seconds ahead after a change is capped at now; the notice never restarts."""
    lengthened = next_period_state(_settled(30), 365, 30, NOW)
    stepped = next_period_state(lengthened, 365, 30, NOW - timedelta(seconds=5))
    assert stepped.grace_until == NOW
    assert deadline_under(NOW - 400 * DAY, stepped, 30) == NOW


def test_the_off_warning_is_logged_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retention off warns once per process, not on every chat turn that reads the setting."""
    from loguru import logger

    from docint.utils import env_cfg

    monkeypatch.setenv("DOCUMENT_RETENTION_DAYS", "off")
    monkeypatch.setattr(env_cfg, "_RETENTION_OFF_WARNED", False, raising=False)
    lines: list[str] = []
    sink = logger.add(lines.append, level="WARNING")
    try:
        env_cfg.load_document_retention_env()
        env_cfg.load_document_retention_env()
    finally:
        logger.remove(sink)
    assert sum("DOCUMENT_RETENTION_DAYS=off" in line for line in lines) == 1
