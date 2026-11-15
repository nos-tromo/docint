"""Record what a chat turn hands to workbench, before it is handed over.

The chat agent's ``run_python`` tool sends passages and counts of the turn's evidence to
workbench. That is a ``disclose`` in the data-protection audit trail, and there is no hand-over
without its record: the tool records each call here with :func:`record_disclosure` *before* it
sends, and a call that cannot be recorded is not sent. The turn's route binds a
:class:`DisclosureLedger` around the turn with :func:`binding`; whatever was recorded is taken
back when the turn ends and written as its ``disclose`` events.

The ledger is carried in a :class:`contextvars.ContextVar`, like the transfer meter
(``docint.utils.audit_meter``), so it follows the work into anyio worker threads and into the
search threads of a round, which run in a copy of the turn's context. A plain thread does not
copy the caller's context; a call made there finds no ledger and is refused.

A disclosure is recorded when the call starts and settled when it ends: :meth:`Disclosure.answered`
once the answer names its workbench job, :meth:`Disclosure.unanswered` when no answer came.
Taking the ledger writes every disclosure as it stands: one still in flight is written as an
incomplete error, and a later settlement of it changes nothing. :meth:`DisclosureLedger.seal`
takes and closes the ledger in one step, so a call that starts after the turn ended is refused
and sends nothing.

The ledger holds references and counts only: point ids, numbers, a job id. Never code, data or
output.

Stdlib only, so every client can import it.
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "Disclosure",
    "DisclosureLedger",
    "DisclosureRecord",
    "DisclosureRefused",
    "binding",
    "current_ledger",
    "record_disclosure",
]


class DisclosureRefused(RuntimeError):
    """A hand-over could not be recorded, so it must not be sent.

    Raised when no ledger is bound to the work, or the bound one is sealed because its turn ended.
    """


@dataclass(frozen=True)
class DisclosureRecord:
    """One hand-over, as the turn's audit trail writes it.

    Attributes:
        point_ids (tuple[str, ...]): The points whose passages were handed over, by reference.
        passages (int): Passages sent.
        keyword_checks (int): Keyword checks sent.
        topic_listings (int): Topic listings sent.
        data_bytes (int): Bytes of data sent.
        job_id (str | None): The workbench job the answer named; ``None`` when no answer did.
        outcome (Literal["ok", "error"]): ``ok`` once an answer came, ``error`` otherwise.
        incomplete (bool): Whether the call was still in flight when the disclosure was written,
            so its outcome is not known.
    """

    point_ids: tuple[str, ...]
    passages: int
    keyword_checks: int
    topic_listings: int
    data_bytes: int
    job_id: str | None
    outcome: Literal["ok", "error"]
    incomplete: bool


_FLYING = "flying"
_ANSWERED = "answered"
_UNANSWERED = "unanswered"


@dataclass
class _Entry:
    """One recorded call and how far it got; read and written under its ledger's lock.

    Its record is a snapshot taken when the call is written, so a settlement after that reaches
    nothing.
    """

    point_ids: tuple[str, ...]
    passages: int
    keyword_checks: int
    topic_listings: int
    data_bytes: int
    state: str = _FLYING
    job_id: str | None = None

    def record(self) -> DisclosureRecord:
        """Describe the call as it stands.

        Returns:
            DisclosureRecord: ``ok`` when answered, ``error`` when unanswered, and an incomplete
            ``error`` with no job when still in flight.
        """
        return DisclosureRecord(
            point_ids=self.point_ids,
            passages=self.passages,
            keyword_checks=self.keyword_checks,
            topic_listings=self.topic_listings,
            data_bytes=self.data_bytes,
            job_id=self.job_id,
            outcome="ok" if self.state == _ANSWERED else "error",
            incomplete=self.state == _FLYING,
        )


class Disclosure:
    """The handle of one recorded call, to settle it when the call ends.

    The first settlement stands, and none changes a disclosure that was already written.
    """

    def __init__(self, ledger: DisclosureLedger, entry: _Entry) -> None:
        """Bind the handle to its ledger and entry.

        Args:
            ledger (DisclosureLedger): The ledger that holds the entry.
            entry (_Entry): The recorded call.
        """
        self._ledger = ledger
        self._entry = entry

    def answered(self, job_id: str | None) -> None:
        """Say that an answer came.

        Does nothing once the disclosure was written or settled.

        Args:
            job_id (str | None): The workbench job the answer names, if it names one.
        """
        self._ledger._settle(self._entry, _ANSWERED, job_id)

    def unanswered(self) -> None:
        """Say that no answer came.

        Does nothing once the disclosure was written or settled.
        """
        self._ledger._settle(self._entry, _UNANSWERED, None)


class DisclosureLedger:
    """Thread-safe list of what one turn handed to workbench, until it is written and sealed."""

    def __init__(self) -> None:
        """Start empty and open."""
        self._lock = threading.Lock()
        self._pending: list[_Entry] = []
        self._sealed = False

    @property
    def sealed(self) -> bool:
        """Whether the ledger was sealed and refuses every later disclosure.

        Returns:
            bool: ``True`` once :meth:`seal` ran.
        """
        with self._lock:
            return self._sealed

    def add(
        self, *, point_ids: Sequence[str], passages: int, keyword_checks: int, topic_listings: int, data_bytes: int
    ) -> Disclosure:
        """Record a call that is about to send data.

        Args:
            point_ids (Sequence[str]): The points whose passages are handed over.
            passages (int): Passages sent.
            keyword_checks (int): Keyword checks sent.
            topic_listings (int): Topic listings sent.
            data_bytes (int): Bytes of data sent.

        Returns:
            Disclosure: The handle to settle the call with.

        Raises:
            DisclosureRefused: When the ledger is sealed.
        """
        entry = _Entry(tuple(point_ids), passages, keyword_checks, topic_listings, data_bytes)
        with self._lock:
            if self._sealed:
                raise DisclosureRefused("The turn's disclosures were already written, so nothing more is recorded.")
            self._pending.append(entry)
        return Disclosure(self, entry)

    def take(self) -> list[DisclosureRecord]:
        """Write every disclosure not yet written, as it stands.

        Returns:
            list[DisclosureRecord]: One record per disclosure, in the order they were added; a
            disclosure is returned once, and a later settlement of it changes nothing.
        """
        with self._lock:
            return self._take_locked()

    def seal(self) -> list[DisclosureRecord]:
        """Take what is left and refuse every later disclosure, in one step.

        Returns:
            list[DisclosureRecord]: What :meth:`take` would return; none is added between that and
            the refusal. A second seal returns an empty list.
        """
        with self._lock:
            records = self._take_locked()
            self._sealed = True
            return records

    def _take_locked(self) -> list[DisclosureRecord]:
        """Write the pending disclosures; the caller holds the lock.

        Returns:
            list[DisclosureRecord]: One record per pending disclosure, in order.
        """
        pending = self._pending
        self._pending = []
        return [entry.record() for entry in pending]

    def _settle(self, entry: _Entry, state: str, job_id: str | None) -> None:
        """Settle a call, unless it was already settled.

        A call that was written has left the ledger, so settling it reaches no record.

        Args:
            entry (_Entry): The recorded call.
            state (str): ``answered`` or ``unanswered``.
            job_id (str | None): The job an answer named.
        """
        with self._lock:
            if entry.state != _FLYING:
                return
            entry.state = state
            entry.job_id = job_id


_CURRENT: contextvars.ContextVar[DisclosureLedger | None] = contextvars.ContextVar(
    "docint_disclosure_ledger", default=None
)


@contextmanager
def binding(ledger: DisclosureLedger) -> Iterator[DisclosureLedger]:
    """Make this ledger the one every disclosure in this block records into.

    Args:
        ledger (DisclosureLedger): The turn's ledger.

    Yields:
        DisclosureLedger: The bound ledger.
    """
    token = _CURRENT.set(ledger)
    try:
        yield ledger
    finally:
        _CURRENT.reset(token)


def current_ledger() -> DisclosureLedger | None:
    """Return the ledger bound to this context.

    Returns:
        DisclosureLedger | None: The bound ledger, or ``None`` when none is bound.
    """
    return _CURRENT.get()


def record_disclosure(
    *, point_ids: Sequence[str], passages: int, keyword_checks: int, topic_listings: int, data_bytes: int
) -> Disclosure:
    """Record a hand-over in the ledger bound to this context, before it is sent.

    Args:
        point_ids (Sequence[str]): The points whose passages are handed over.
        passages (int): Passages sent.
        keyword_checks (int): Keyword checks sent.
        topic_listings (int): Topic listings sent.
        data_bytes (int): Bytes of data sent.

    Returns:
        Disclosure: The handle to settle the call with.

    Raises:
        DisclosureRefused: When no ledger is bound, or the bound one is sealed. The caller must
            then send nothing.
    """
    ledger = _CURRENT.get()
    if ledger is None:
        raise DisclosureRefused("No disclosure ledger is bound to this work, so nothing is recorded.")
    return ledger.add(
        point_ids=point_ids,
        passages=passages,
        keyword_checks=keyword_checks,
        topic_listings=topic_listings,
        data_bytes=data_bytes,
    )
