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
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Literal

from loguru import logger


@dataclass(frozen=True, slots=True)
class RetentionPeriodState:
    """The document retention period in force and when it took effect.

    A changed period never deletes sooner than announced: a longer one
    applies at once, and a shorter one gives the notice only to what it makes
    due sooner while the rule in force before it keeps deleting
    (:func:`next_period_state`).

    Attributes:
        days (int | None): The period; ``None`` is off.
        set_at (datetime): Aware UTC; when the period took effect.
        grace_until (datetime | None): Until when the period does not delete
            on its own; ``None`` (a row of the release before) is ``set_at``
            plus the notice.
        in_force_days (int | None): The period in force before a change,
            deleting meanwhile.
        in_force_grace (datetime | None): Since when that period deletes.
    """

    days: int | None
    set_at: datetime
    grace_until: datetime | None = None
    in_force_days: int | None = None
    in_force_grace: datetime | None = None

    def grace_end(self, notice_days: int) -> datetime:
        """When the period's notice ends.

        Args:
            notice_days (int): The notice, for a row of the release before.

        Returns:
            datetime: The end.
        """
        return self.grace_until if self.grace_until is not None else self.set_at + timedelta(days=notice_days)


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


def deadline_under(ingested_at: datetime, state: RetentionPeriodState, notice_days: int) -> datetime | None:
    """Return when a document is due under the period in force.

    Args:
        ingested_at (datetime): When its ingest completed, timezone-aware.
        state (RetentionPeriodState): The period, its notice, and the rule in force before it.
        notice_days (int): The notice, for a row of the release before.

    Returns:
        datetime | None: The deadline; ``None`` while retention is off.

    Raises:
        ValueError: When a timestamp is naive.
    """
    if state.days is None:
        return None
    grace = state.grace_end(notice_days)
    if ingested_at.tzinfo is None or grace.tzinfo is None:
        raise ValueError("Retention timestamps must carry a timezone.")
    own = ingested_at + timedelta(days=state.days)
    if state.in_force_days is None or state.in_force_grace is None:
        return max(own, grace)
    in_force = max(ingested_at + timedelta(days=state.in_force_days), state.in_force_grace)
    return max(own, min(grace, in_force))


def next_period_state(
    previous: RetentionPeriodState | None, days: int | None, notice_days: int, now: datetime
) -> RetentionPeriodState:
    """Return the period state after a startup with ``days`` configured.

    The same period keeps its dates (a row of the release before gets its
    notice end). A period that is new, or on again after off, waits the notice.
    A longer one applies at once (within a running first notice, at its end),
    also to what was due under the shorter one but not yet deleted. A shorter
    one gives the notice only to what it makes due sooner, never ending before
    a notice already running; the rule actually in force before the change
    keeps deleting meanwhile. Changing the notice alone moves nothing.

    Args:
        previous (RetentionPeriodState | None): The state recorded so far.
        days (int | None): The period configured now; ``None`` is off.
        notice_days (int): The notice configured now.
        now (datetime): The startup time, timezone-aware.

    Returns:
        RetentionPeriodState: The state to record.
    """
    if days is None:
        return previous if previous is not None and previous.days is None else RetentionPeriodState(None, now)
    fresh_grace = now + timedelta(days=notice_days)
    fresh = RetentionPeriodState(days, now, fresh_grace)
    if previous is None or previous.days is None:
        return fresh
    recorded: int = previous.days
    grace = previous.grace_end(notice_days)
    if previous.set_at > now:
        # A clock set back: keep the rule, its dates never more than a notice ahead.
        grace = min(grace, now + timedelta(days=notice_days))
        in_force_grace = previous.in_force_grace
        previous = replace(
            previous,
            set_at=now,
            grace_until=grace,
            in_force_grace=None if in_force_grace is None else min(in_force_grace, grace),
        )
    if recorded == days:
        return replace(previous, grace_until=grace)
    if now >= grace or previous.in_force_days is None or previous.in_force_grace is None:
        effective, effective_grace = recorded, grace
    else:
        effective, effective_grace = max(previous.in_force_days, recorded), previous.in_force_grace
    if days >= effective:
        return RetentionPeriodState(days, now, max(now, effective_grace))
    return RetentionPeriodState(days, now, max(fresh_grace, grace), effective, effective_grace)


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
