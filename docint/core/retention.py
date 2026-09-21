"""Collection retention: when an idle collection is due for deletion.

A collection expires once it has seen no activity for the configured window
(``COLLECTION_RETENTION``, see ``docs/retention.md``). The rule lives here and
nowhere else, so the API listing, the operator report and the sweep can never
disagree about a date.

The module holds no docint domain imports: callers hand in timestamps, so the
rule is testable without a database, Qdrant, or a clock.
"""

from __future__ import annotations

import calendar
from datetime import datetime, timedelta

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
