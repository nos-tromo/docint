"""Tests for the document retention period: its settings, its rule, and when it took effect."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from docint.core.retention import RetentionPeriodState, document_expires_at, inherit, is_due, is_warning_within
from docint.core.state.base import Base
from docint.core.state.collection_owner_manager import CollectionOwnerManager
from docint.core.state.retention_state import RetentionState
from docint.utils.env_cfg import load_document_retention_env

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
DAY = timedelta(days=1)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the variables so an ambient ``.env`` cannot skew the defaults."""
    monkeypatch.delenv("DOCUMENT_RETENTION_DAYS", raising=False)
    monkeypatch.delenv("RETENTION_NOTICE_DAYS", raising=False)


class _Stub:
    """Minimal RAG stand-in exposing only the session-store URL."""

    session_store = "sqlite://"


@pytest.fixture
def mgr() -> CollectionOwnerManager:
    """A manager backed by a shared in-memory SQLite DB (StaticPool)."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    m = CollectionOwnerManager(rag=cast(Any, _Stub()))
    m._SessionMaker = sessionmaker(bind=engine)
    return m


def test_default_period_is_365_days_and_notice_30() -> None:
    """On by default: documents go a year after ingest; a new or changed period waits 30 days."""
    cfg = load_document_retention_env()
    assert (cfg.days, cfg.notice_days, cfg.enabled) == (365, 30, True)


def test_off_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """``off`` (development only) switches retention off."""
    monkeypatch.setenv("DOCUMENT_RETENTION_DAYS", " OFF ")
    cfg = load_document_retention_env()
    assert (cfg.days, cfg.enabled) == (None, False)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DOCUMENT_RETENTION_DAYS", "12m"),
        ("DOCUMENT_RETENTION_DAYS", "0"),
        ("DOCUMENT_RETENTION_DAYS", "3651"),
        ("DOCUMENT_RETENTION_DAYS", "abc"),
        ("RETENTION_NOTICE_DAYS", "0"),
        ("RETENTION_NOTICE_DAYS", "366"),
        ("RETENTION_NOTICE_DAYS", "off"),
    ],
)
def test_garbage_stops_startup(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    """A value that is no period never falls back to a period or to off."""
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        load_document_retention_env()


def test_deadline_is_ingest_plus_period() -> None:
    """Long after the period took effect, a document is due exactly the period after its ingest."""
    ingested = NOW - 10 * DAY
    deadline = document_expires_at(ingested, 365, set_at=NOW - 400 * DAY, notice_days=30)
    assert deadline == ingested + 365 * DAY


def test_grace_after_switch_on() -> None:
    """Ingested 400 days ago, the period set today: due in 30 days, not at once."""
    deadline = document_expires_at(NOW - 400 * DAY, 365, set_at=NOW, notice_days=30)
    assert deadline == NOW + 30 * DAY


def test_naive_times_are_refused() -> None:
    """A timestamp without a timezone is a bug, never a guess."""
    with pytest.raises(ValueError):
        document_expires_at(datetime(2026, 1, 1), 365, set_at=NOW, notice_days=30)


def test_inherit_takes_the_earliest_and_never_raises() -> None:
    """A derived item's expiry is the earliest of its own and its sources'; None sources do not count."""
    assert inherit(None) is None
    assert inherit(NOW, None) == NOW
    assert inherit(NOW, NOW - DAY, NOW + DAY) == NOW - DAY
    assert inherit(NOW - DAY, NOW + 5 * DAY) == NOW - DAY
    assert inherit(None, NOW + DAY) == NOW + DAY


def test_due_and_warning() -> None:
    """Due once the deadline is reached; flagged from the notice period before it."""
    assert is_due(NOW, NOW) and not is_due(NOW + DAY, NOW) and not is_due(None, NOW)
    assert is_warning_within(NOW + 30 * DAY, NOW, 30)
    assert not is_warning_within(NOW + 31 * DAY, NOW, 30)
    assert not is_warning_within(None, NOW, 30)


def test_period_change_rewrites_set_at_same_period_keeps_it(mgr: CollectionOwnerManager) -> None:
    """The same period keeps when it took effect; another takes effect now, keeping the rule in force."""
    first = mgr.record_retention_period(365, now=NOW)
    assert first == RetentionPeriodState(days=365, set_at=NOW, grace_until=NOW + 30 * DAY)
    assert mgr.record_retention_period(365, now=NOW + DAY) == first
    shorter = RetentionPeriodState(
        days=180, set_at=NOW + 2 * DAY, grace_until=NOW + 32 * DAY, in_force_days=365, in_force_grace=NOW + 30 * DAY
    )
    assert mgr.record_retention_period(180, now=NOW + 2 * DAY) == shorter
    assert mgr.retention_period() == shorter
    assert mgr.record_retention_period(None, now=NOW + 3 * DAY) == RetentionPeriodState(days=None, set_at=NOW + 3 * DAY)


def test_legacy_window_row_counts_as_a_change(mgr: CollectionOwnerManager) -> None:
    """A row the idle-window rule left (``6m``) is no period: the first record restarts the grace."""
    with mgr._session_scope() as s:
        s.add(RetentionState(id=1, window="6m", window_set_at=NOW - 400 * DAY))
        s.commit()
    assert mgr.retention_period() is None
    assert mgr.record_retention_period(365, now=NOW) == RetentionPeriodState(
        days=365, set_at=NOW, grace_until=NOW + 30 * DAY
    )
