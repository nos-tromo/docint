"""Collection retention: when an idle collection is due for deletion.

A collection expires once it has seen no activity for the configured window
(``COLLECTION_RETENTION``, see ``docs/retention.md``). The rule lives here and
nowhere else, so the API listing, the operator report and the sweep can never
disagree about a date.

The module holds no docint domain imports: callers hand in timestamps, the
busy check and the purge, so the rule and the sweep are testable without a
database, Qdrant, or a clock.
"""

from __future__ import annotations

import asyncio
import calendar
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

from loguru import logger

#: Days of notice every user gets: the SPA flags a collection this long before
#: it goes, and switching retention on (or changing the window) holds every
#: deletion back by the same span, so the warning is never shorter than this.
NOTICE_DAYS = 30


def add_months(moment: datetime, months: int) -> datetime:
    """Add calendar months, clamping the day to the target month's length.

    Args:
        moment (datetime): The starting point.
        months (int): Calendar months to add.

    Returns:
        datetime: The same wall-clock time ``months`` later; Aug 31 + 6 months
            is the last day of February.
    """
    index = moment.month - 1 + months
    year, month = moment.year + index // 12, index % 12 + 1
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def expires_at(last_activity: datetime | None, months: int, *, window_set_at: datetime) -> datetime | None:
    """Return when a collection is due for deletion.

    Args:
        last_activity (datetime | None): The collection's last recorded
            activity, timezone-aware.
        months (int): The retention window; ``0`` means retention is off.
        window_set_at (datetime): When the window in force was set; no
            deadline falls earlier than ``NOTICE_DAYS`` after it.

    Returns:
        datetime | None: The deadline, or ``None`` when the collection never
            expires — retention is off, or no activity was ever recorded.

    Raises:
        ValueError: When either timestamp is naive.
    """
    if months <= 0 or last_activity is None:
        return None
    if last_activity.tzinfo is None or window_set_at.tzinfo is None:
        raise ValueError("Retention timestamps must carry a timezone.")
    return max(add_months(last_activity, months), window_set_at + timedelta(days=NOTICE_DAYS))


def is_expired(deadline: datetime | None, now: datetime) -> bool:
    """Whether a deadline has passed.

    Args:
        deadline (datetime | None): From :func:`expires_at`.
        now (datetime): The current time, timezone-aware.

    Returns:
        bool: ``True`` once ``now`` has reached the deadline.
    """
    return deadline is not None and deadline <= now


def is_warning(deadline: datetime | None, now: datetime) -> bool:
    """Whether a collection is within its notice period (or overdue).

    Args:
        deadline (datetime | None): From :func:`expires_at`.
        now (datetime): The current time, timezone-aware.

    Returns:
        bool: ``True`` from ``NOTICE_DAYS`` before the deadline onwards.
    """
    return deadline is not None and deadline - now <= timedelta(days=NOTICE_DAYS)


class RetentionClock(Protocol):
    """One collection's retention clock, as the sweep reads it."""

    @property
    def owner(self) -> str | None:
        """The owning principal."""
        ...

    @property
    def logical(self) -> str:
        """The user-visible collection name."""
        ...

    @property
    def physical(self) -> str:
        """The Qdrant collection name."""
        ...

    @property
    def last_activity_at(self) -> datetime | None:
        """The last recorded activity, timezone-aware; ``None`` never expires."""
        ...


@dataclass
class SweepReport:
    """What one sweep did, by logical collection name.

    Attributes:
        scanned (int): Collections looked at.
        expired (int): Collections past their deadline at the scan.
        deleted (list[str]): Collections deleted.
        skipped_busy (list[str]): Expired, but a job was working on them.
        skipped_active (list[str]): Expired at the scan, active again by the purge.
        failed (list[str]): Collections whose delete raised; retried next sweep.
    """

    scanned: int = 0
    expired: int = 0
    deleted: list[str] = field(default_factory=list)
    skipped_busy: list[str] = field(default_factory=list)
    skipped_active: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


async def sweep_once(
    rows: Sequence[RetentionClock],
    *,
    months: int,
    window_set_at: datetime,
    now: datetime,
    is_busy: Callable[[str], Awaitable[bool]],
    purge: Callable[[RetentionClock], Awaitable[bool]],
) -> SweepReport:
    """Delete every collection past its deadline, one at a time.

    A collection a job is still working on is left for the next sweep. The
    purge re-reads the clock before it deletes and answers ``False`` when the
    collection was used since the scan; one that raises is counted and the
    sweep moves on, so a single failure never strands the rest.

    Args:
        rows (Sequence[RetentionClock]): Every collection's clock at the scan.
        months (int): The retention window.
        window_set_at (datetime): When the window was set (the grace anchor).
        now (datetime): The scan time, timezone-aware.
        is_busy (Callable[[str], Awaitable[bool]]): Whether work is in
            flight on a physical collection.
        purge (Callable[[RetentionClock], Awaitable[bool]]): Deletes a
            collection if it is still expired; ``False`` when it no longer is.

    Returns:
        SweepReport: What was deleted, skipped and failed.
    """
    report = SweepReport(scanned=len(rows))
    for row in rows:
        if not is_expired(expires_at(row.last_activity_at, months, window_set_at=window_set_at), now):
            continue
        report.expired += 1
        if await is_busy(row.physical):
            logger.info("Retention postponed collection '{}': a job is still working on it.", row.logical)
            report.skipped_busy.append(row.logical)
            continue
        try:
            purged = await purge(row)
        except Exception:
            logger.exception("Retention could not delete collection '{}'; the next sweep retries.", row.logical)
            report.failed.append(row.logical)
            continue
        if purged:
            logger.info("Retention deleted collection '{}' (last activity {}).", row.logical, row.last_activity_at)
            report.deleted.append(row.logical)
        else:
            report.skipped_active.append(row.logical)
    return report


async def run_retention_loop(
    sweep: Callable[[], Awaitable[object]],
    *,
    first_delay: float,
    interval: float,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> None:
    """Run ``sweep`` after ``first_delay`` seconds, then every ``interval``, until cancelled.

    A sweep that raises is logged and the loop carries on: one bad run must
    never end retention for the rest of the process's life. Cancellation
    propagates, which is how shutdown stops it.

    Args:
        sweep (Callable[[], Awaitable[object]]): One full sweep.
        first_delay (float): Seconds before the first sweep.
        interval (float): Seconds between sweeps.
        sleep (Callable[[float], Awaitable[object]]): The wait; injectable so
            tests need not sleep.
    """
    delay = first_delay
    while True:
        await sleep(delay)
        delay = interval
        try:
            await sweep()
        except Exception:
            logger.exception("Retention sweep failed; the next one runs in {} s.", interval)
