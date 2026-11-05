"""Document retention: when a stored item is due for deletion.

Every document expires a fixed period (``DOCUMENT_RETENTION_DAYS``) after its
ingest completed; whatever draws on documents inherits the earliest deadline
among them (``docs/retention.md``). The rule lives here and nowhere else, so
the API listing, the operator report and the sweep can never disagree about a
date.

The module holds no docint domain imports: callers hand in timestamps, so the
rule is testable without a database, Qdrant, or a clock.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from loguru import logger


@dataclass(frozen=True, slots=True)
class RetentionPeriodState:
    """The document retention period in force and when it took effect.

    Attributes:
        days (int | None): The period; ``None`` is off.
        set_at (datetime): Aware UTC; anchors the grace period.
    """

    days: int | None
    set_at: datetime


def document_expires_at(ingested_at: datetime, days: int, *, set_at: datetime, notice_days: int) -> datetime:
    """Return when a document is due: its ingest plus the period, never before the grace ends.

    Args:
        ingested_at (datetime): When its ingest completed, timezone-aware.
        days (int): The retention period.
        set_at (datetime): When the period took effect.
        notice_days (int): The grace after ``set_at``.

    Returns:
        datetime: The deadline.

    Raises:
        ValueError: When either timestamp is naive.
    """
    if ingested_at.tzinfo is None or set_at.tzinfo is None:
        raise ValueError("Retention timestamps must carry a timezone.")
    return max(ingested_at + timedelta(days=days), set_at + timedelta(days=notice_days))


def inherit(current: datetime | None, *sources: datetime | None) -> datetime | None:
    """Return a derived item's expiry: the earliest of its own and its sources'.

    Args:
        current (datetime | None): The item's expiry so far.
        *sources (datetime | None): Its sources' expiries; ``None`` ones do not count.

    Returns:
        datetime | None: The minimum, or ``None`` when every value is ``None``.
    """
    known = [value for value in (current, *sources) if value is not None]
    return min(known) if known else None


def is_due(deadline: datetime | None, now: datetime) -> bool:
    """Whether a deadline has been reached.

    Args:
        deadline (datetime | None): The deadline; ``None`` never.
        now (datetime): The current time.

    Returns:
        bool: ``True`` from the deadline on.
    """
    return deadline is not None and deadline <= now


def is_warning_within(deadline: datetime | None, now: datetime, notice_days: int) -> bool:
    """Whether an item is within its notice period before deletion (or overdue).

    Args:
        deadline (datetime | None): The deadline; ``None`` never.
        now (datetime): The current time.
        notice_days (int): The notice lead time.

    Returns:
        bool: ``True`` from ``notice_days`` before the deadline on.
    """
    return deadline is not None and deadline - now <= timedelta(days=notice_days)


@dataclass(frozen=True, slots=True)
class ExpiringItem:
    """One stored item with its deadline, as the notice and the operator report list it.

    Attributes:
        kind (Literal["document", "session", "report"]): What it is.
        physical (str | None): Its Qdrant collection, where the store keeps that.
        logical (str | None): Its collection's user-visible name, where the store keeps that.
        owner (str | None): Its owner; ``None`` for an unowned collection.
        id (str): The document's file hash, the session id, or the report id.
        name (str): The file name, the session's first question, or the report title.
        since (datetime | None): When it was ingested or created.
        expires_at (datetime | None): Its deadline; ``None`` while retention is off.
    """

    kind: Literal["document", "session", "report"]
    physical: str | None
    logical: str | None
    owner: str | None
    id: str
    name: str
    since: datetime | None
    expires_at: datetime | None


@dataclass
class SweepReport:
    """What one sweep did.

    Attributes:
        skipped_busy (list[str]): Collections with due documents a job was working on.
        failed (list[str]): Collections, or kinds of derived item, whose delete raised; retried next sweep.
        documents (int): Documents deleted.
        collections (int): Collections deleted because their last document went.
        sessions (int): Chat sessions deleted.
        reports (int): Reports deleted.
        extracts (int): Stored extracts deleted.
        topic_runs (int): Topic runs deleted.
        repaired (int): Documents the index repair added.
    """

    skipped_busy: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    documents: int = 0
    collections: int = 0
    sessions: int = 0
    reports: int = 0
    extracts: int = 0
    topic_runs: int = 0
    repaired: int = 0


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
