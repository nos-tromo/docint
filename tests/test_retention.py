"""Tests for the collection retention rules in ``docint.core.retention``."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from docint.core.retention import (
    NOTICE_DAYS,
    RetentionClock,
    SweepReport,
    add_months,
    expires_at,
    is_expired,
    is_warning,
    run_retention_loop,
    sweep_once,
)

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
LONG_AGO = datetime(2020, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("start", "months", "expected"),
    [
        (datetime(2026, 1, 15, 10, 30, tzinfo=UTC), 6, datetime(2026, 7, 15, 10, 30, tzinfo=UTC)),
        (datetime(2026, 11, 5, tzinfo=UTC), 6, datetime(2027, 5, 5, tzinfo=UTC)),
        (datetime(2026, 8, 31, tzinfo=UTC), 6, datetime(2027, 2, 28, tzinfo=UTC)),
        (datetime(2027, 8, 31, tzinfo=UTC), 6, datetime(2028, 2, 29, tzinfo=UTC)),
        (datetime(2024, 2, 29, tzinfo=UTC), 24, datetime(2026, 2, 28, tzinfo=UTC)),
        (datetime(2026, 12, 31, tzinfo=UTC), 12, datetime(2027, 12, 31, tzinfo=UTC)),
    ],
    ids=["same-day", "year-wrap", "clamp-to-feb", "leap-feb", "two-years-from-leap-day", "full-year"],
)
def test_add_months_is_calendar_months_clamped_to_the_month_end(
    start: datetime, months: int, expected: datetime
) -> None:
    """Aug 31 + 6 months is the end of February, not an error or March 3."""
    assert add_months(start, months) == expected


def test_a_collection_without_a_stamp_never_expires() -> None:
    """No recorded activity is not the same as no activity."""
    assert expires_at(None, 6, window_set_at=LONG_AGO) is None


def test_nothing_expires_while_retention_is_off() -> None:
    """A zero-month window is ``off``."""
    assert expires_at(LONG_AGO, 0, window_set_at=LONG_AGO) is None


def test_expiry_is_last_activity_plus_the_window() -> None:
    """Long after the window was set, only the last activity counts."""
    stamp = datetime(2026, 3, 10, 8, 0, tzinfo=UTC)
    assert expires_at(stamp, 6, window_set_at=LONG_AGO) == datetime(2026, 9, 10, 8, 0, tzinfo=UTC)


def test_the_grace_period_holds_back_a_long_idle_collection() -> None:
    """Switching retention on never deletes before a full notice period has passed."""
    set_at = NOW - timedelta(days=1)
    assert expires_at(LONG_AGO, 6, window_set_at=set_at) == set_at + timedelta(days=NOTICE_DAYS)


def test_a_naive_timestamp_is_refused() -> None:
    """Comparing a naive and an aware datetime is a crash waiting for the sweep."""
    with pytest.raises(ValueError, match="timezone"):
        expires_at(datetime(2026, 3, 10), 6, window_set_at=LONG_AGO)


@pytest.mark.parametrize(
    ("deadline", "expected"),
    [
        (NOW + timedelta(days=NOTICE_DAYS + 1), False),
        (NOW + timedelta(days=NOTICE_DAYS), True),
        (NOW + timedelta(hours=1), True),
        (NOW - timedelta(days=3), True),
        (None, False),
    ],
    ids=["outside-notice", "notice-edge", "imminent", "overdue", "never"],
)
def test_warning_covers_the_notice_period_and_anything_overdue(deadline: datetime | None, expected: bool) -> None:
    """The SPA flags a collection from ``NOTICE_DAYS`` before deletion onwards."""
    assert is_warning(deadline, NOW) is expected


@pytest.mark.parametrize(
    ("deadline", "expected"),
    [(NOW + timedelta(seconds=1), False), (NOW, True), (NOW - timedelta(days=1), True), (None, False)],
    ids=["future", "exactly-now", "past", "never"],
)
def test_expired_means_the_deadline_has_passed(deadline: datetime | None, expected: bool) -> None:
    """A deadline of exactly now is due."""
    assert is_expired(deadline, NOW) is expected


# --- The sweep ---


@pytest.fixture
def anyio_backend() -> str:
    """Run the async tests on asyncio only (trio is not a dependency)."""
    return "asyncio"


@dataclass(frozen=True)
class _Row:
    """One collection's clock, shaped like the ownership manager's rows."""

    logical: str
    last_activity_at: datetime | None
    owner: str | None = "alice"

    @property
    def physical(self) -> str:
        return f"u1__{self.logical}"


STALE = NOW - timedelta(days=400)


async def _sweep(
    rows: list[_Row],
    *,
    busy: frozenset[str] = frozenset(),
    touched: frozenset[str] = frozenset(),
    broken: frozenset[str] = frozenset(),
    window_set_at: datetime = LONG_AGO,
) -> tuple[SweepReport, list[str]]:
    """Run one sweep over ``rows`` with a recording purge; return the report and what was purged."""
    purged: list[str] = []

    async def is_busy(physical: str) -> bool:
        return physical in {f"u1__{name}" for name in busy}

    async def purge(row: RetentionClock) -> bool:
        if row.logical in broken:
            raise RuntimeError("qdrant down")
        if row.logical in touched:
            return False
        purged.append(row.logical)
        return True

    report = await sweep_once(rows, months=6, window_set_at=window_set_at, now=NOW, is_busy=is_busy, purge=purge)
    return report, purged


@pytest.mark.anyio
async def test_the_sweep_deletes_only_what_is_past_its_deadline() -> None:
    """A fresh collection and one with no recorded activity are never touched."""
    report, purged = await _sweep([_Row("alt", STALE), _Row("frisch", NOW), _Row("ohne", None)])

    assert purged == ["alt"]
    assert (report.scanned, report.expired, report.deleted) == (3, 1, ["alt"])


@pytest.mark.anyio
async def test_the_sweep_leaves_a_collection_a_job_is_working_on() -> None:
    """Deleting under a running ingest would let the job re-create the collection."""
    report, purged = await _sweep([_Row("alt", STALE)], busy=frozenset({"alt"}))

    assert purged == []
    assert report.skipped_busy == ["alt"]


@pytest.mark.anyio
async def test_the_sweep_spares_a_collection_touched_since_the_scan() -> None:
    """The purge re-reads the clock; activity after the scan wins."""
    report, purged = await _sweep([_Row("alt", STALE)], touched=frozenset({"alt"}))

    assert purged == []
    assert report.skipped_active == ["alt"]


@pytest.mark.anyio
async def test_one_failed_delete_does_not_strand_the_rest() -> None:
    """A collection that fails to delete is retried on the next sweep; the others still go."""
    report, purged = await _sweep([_Row("kaputt", STALE), _Row("alt", STALE)], broken=frozenset({"kaputt"}))

    assert purged == ["alt"]
    assert report.failed == ["kaputt"]
    assert report.deleted == ["alt"]


@pytest.mark.anyio
async def test_the_sweep_honours_the_grace_period() -> None:
    """Right after retention is switched on, nothing is due however long it sat idle."""
    report, purged = await _sweep([_Row("alt", STALE)], window_set_at=NOW - timedelta(days=1))

    assert purged == []
    assert report.expired == 0


@pytest.mark.anyio
async def test_the_loop_waits_first_then_runs_on_its_interval() -> None:
    """The first sweep runs a little after boot, then once per interval."""
    waits: list[float] = []
    runs: list[int] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) > 3:
            raise asyncio.CancelledError

    async def sweep() -> None:
        runs.append(len(waits))

    with pytest.raises(asyncio.CancelledError):
        await run_retention_loop(sweep, first_delay=300, interval=86_400, sleep=sleep)

    assert waits == [300, 86_400, 86_400, 86_400]
    assert runs == [1, 2, 3]


@pytest.mark.anyio
async def test_the_loop_survives_a_failed_sweep() -> None:
    """One bad run must not end retention for the rest of the process's life."""
    waits: list[float] = []
    runs: list[int] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) > 2:
            raise asyncio.CancelledError

    async def sweep() -> None:
        runs.append(len(waits))
        if len(runs) == 1:
            raise RuntimeError("sessions store locked")

    with pytest.raises(asyncio.CancelledError):
        await run_retention_loop(sweep, first_delay=1, interval=2, sleep=sleep)

    assert runs == [1, 2]


@pytest.mark.anyio
async def test_cancelling_during_a_sweep_stops_the_loop() -> None:
    """Shutdown cancels mid-sweep; the failure handler must not swallow that."""
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) > 3:
            raise AssertionError("the loop swallowed the cancellation")

    async def sweep() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_retention_loop(sweep, first_delay=0, interval=0, sleep=sleep)
