"""The disclosure ledger: what a chat turn hands to workbench is recorded before it is sent.

The ledger rides a ``ContextVar`` like the transfer meter, so a call made on a search thread
(which runs in a copy of the turn's context) records into the turn's own ledger. The turn's end
writes what is on it and seals it, after which no later call may send anything.
"""

from __future__ import annotations

import contextvars
import sys
import threading
import time
from dataclasses import FrozenInstanceError, dataclass

import pytest

from docint.utils.audit_disclose import (
    Disclosure,
    DisclosureLedger,
    DisclosureRecord,
    DisclosureRefused,
    binding,
    current_ledger,
    record_disclosure,
)

JOB = "0" * 32


def _hand_over(ledger: DisclosureLedger, *point_ids: str, data_bytes: int = 100) -> Disclosure:
    """Add a disclosure of the given points straight to a ledger.

    Args:
        ledger (DisclosureLedger): The ledger to add to.
        *point_ids (str): The points handed over.
        data_bytes (int): The bytes of data sent.

    Returns:
        Disclosure: The handle of the new disclosure.
    """
    return ledger.add(
        point_ids=point_ids, passages=len(point_ids), keyword_checks=0, topic_listings=0, data_bytes=data_bytes
    )


def test_a_disclosure_is_taken_as_it_stands() -> None:
    """Answered is ok with its job, unanswered is an error, and one still in flight is an incomplete error."""
    ledger = DisclosureLedger()
    with binding(ledger):
        done = record_disclosure(
            point_ids=["p1", "p2"], passages=2, keyword_checks=1, topic_listings=0, data_bytes=2048
        )
        lost = record_disclosure(point_ids=["p3"], passages=1, keyword_checks=0, topic_listings=0, data_bytes=900)
        record_disclosure(point_ids=["p4"], passages=1, keyword_checks=0, topic_listings=0, data_bytes=800)
    done.answered(JOB)
    lost.unanswered()

    taken = ledger.take()

    assert [(r.outcome, r.job_id, r.incomplete) for r in taken] == [
        ("ok", JOB, False),
        ("error", None, False),
        ("error", None, True),
    ]
    assert ledger.take() == []


def test_an_answer_that_names_no_job_is_still_ok() -> None:
    """An answer came, so the hand-over is ok; it names no workbench job, so the record names none."""
    ledger = DisclosureLedger()
    _hand_over(ledger, "p1").answered(None)

    (record,) = ledger.take()

    assert (record.outcome, record.job_id, record.incomplete) == ("ok", None, False)


@pytest.mark.parametrize("point_ids", ["p1", b"p1"], ids=["str", "bytes"])
def test_a_bare_string_is_no_list_of_point_ids(point_ids: str | bytes) -> None:
    """A single id passed as text would become one id per character, so it is refused and nothing is recorded."""
    ledger = DisclosureLedger()

    with pytest.raises(TypeError):
        ledger.add(point_ids=point_ids, passages=1, keyword_checks=0, topic_listings=0, data_bytes=10)  # type: ignore[arg-type]
    with binding(ledger), pytest.raises(TypeError):
        record_disclosure(point_ids=point_ids, passages=1, keyword_checks=0, topic_listings=0, data_bytes=10)  # type: ignore[arg-type]

    assert ledger.take() == []


def test_a_record_holds_what_was_handed_over() -> None:
    """The point ids and the counts reach the record as they were added, in a form the caller cannot change."""
    ledger = DisclosureLedger()
    ids = ["p1", "p2"]
    handle = ledger.add(point_ids=ids, passages=2, keyword_checks=1, topic_listings=3, data_bytes=2048)
    ids.append("p9")
    handle.answered(JOB)

    (record,) = ledger.take()

    assert record == DisclosureRecord(
        point_ids=("p1", "p2"),
        passages=2,
        keyword_checks=1,
        topic_listings=3,
        data_bytes=2048,
        job_id=JOB,
        outcome="ok",
        incomplete=False,
    )
    with pytest.raises(FrozenInstanceError):
        record.passages = 5  # type: ignore[misc]


def test_a_disclosure_is_written_once_and_in_order() -> None:
    """Take hands out what is unwritten, in the order it was added, and never hands one out twice."""
    ledger = DisclosureLedger()
    first = _hand_over(ledger, "p1")
    first.answered(JOB)
    assert [r.point_ids for r in ledger.take()] == [("p1",)]

    _hand_over(ledger, "p2")
    _hand_over(ledger, "p3")

    assert [r.point_ids for r in ledger.take()] == [("p2",), ("p3",)]
    assert ledger.take() == []


def test_the_seal_takes_what_is_left_and_refuses_every_later_one() -> None:
    """Seal writes what take would, then nothing more can be added, on the ledger or through the context."""
    ledger = DisclosureLedger()
    done = _hand_over(ledger, "p1")
    _hand_over(ledger, "p2")
    done.answered(JOB)
    assert ledger.sealed is False

    sealed = ledger.seal()

    assert [(r.point_ids, r.outcome, r.incomplete) for r in sealed] == [
        (("p1",), "ok", False),
        (("p2",), "error", True),
    ]
    assert ledger.sealed is True
    with pytest.raises(DisclosureRefused):
        _hand_over(ledger, "p3")
    with binding(ledger), pytest.raises(DisclosureRefused):
        record_disclosure(point_ids=["p3"], passages=1, keyword_checks=0, topic_listings=0, data_bytes=10)
    assert ledger.take() == []
    assert ledger.seal() == []


def test_nothing_is_recorded_after_the_seal() -> None:
    """A late call that finds the turn's ledger sealed is refused, so it sends nothing."""
    ledger = DisclosureLedger()
    with binding(ledger):
        ledger.seal()

        with pytest.raises(DisclosureRefused):
            record_disclosure(point_ids=["p1"], passages=1, keyword_checks=0, topic_listings=0, data_bytes=10)


def test_nothing_is_recorded_without_a_ledger() -> None:
    """With no ledger bound there is nowhere to record, so the call is refused."""
    assert current_ledger() is None

    with pytest.raises(DisclosureRefused):
        record_disclosure(point_ids=["p1"], passages=1, keyword_checks=0, topic_listings=0, data_bytes=10)


def test_a_refusal_is_a_runtime_error() -> None:
    """The refusal is a RuntimeError, so a caller that does not know it still cannot mistake it for a result."""
    assert issubclass(DisclosureRefused, RuntimeError)


def test_an_answer_after_the_write_changes_nothing() -> None:
    """A call still in flight at the seal is written as incomplete; its answer arriving later cannot rewrite that."""
    ledger = DisclosureLedger()
    flying = _hand_over(ledger, "p1")
    (written,) = ledger.seal()

    flying.answered(JOB)
    flying.unanswered()

    assert (written.outcome, written.job_id, written.incomplete) == ("error", None, True)
    assert ledger.take() == []


def test_an_unanswer_after_the_write_changes_nothing() -> None:
    """An answered disclosure already written stays ok when a later failure reports it unanswered."""
    ledger = DisclosureLedger()
    done = _hand_over(ledger, "p1")
    done.answered(JOB)
    (written,) = ledger.take()

    done.unanswered()

    assert (written.outcome, written.job_id) == ("ok", JOB)
    assert ledger.take() == []


def test_the_first_settlement_stands() -> None:
    """A disclosure is answered or not once: a later word before the write does not turn it round."""
    ledger = DisclosureLedger()
    answered = _hand_over(ledger, "p1")
    answered.answered(JOB)
    answered.unanswered()
    unanswered = _hand_over(ledger, "p2")
    unanswered.unanswered()
    unanswered.answered(JOB)

    assert [(r.outcome, r.job_id, r.incomplete) for r in ledger.take()] == [("ok", JOB, False), ("error", None, False)]


def test_the_ledger_follows_the_work_into_another_thread() -> None:
    """A search thread running in a copy of the turn's context records into the turn's ledger."""
    ledger = DisclosureLedger()
    handles: list[Disclosure] = []

    def work() -> None:
        handles.append(
            record_disclosure(point_ids=["p1"], passages=1, keyword_checks=0, topic_listings=0, data_bytes=10)
        )

    with binding(ledger):
        context = contextvars.copy_context()
        thread = threading.Thread(target=context.run, args=(work,))
        thread.start()
        thread.join()

    handles[0].answered(JOB)
    assert [(r.point_ids, r.job_id) for r in ledger.take()] == [(("p1",), JOB)]


def test_a_thread_outside_the_context_records_nothing() -> None:
    """A plain thread does not inherit the ledger, so a call it makes is refused, never sent unrecorded."""
    ledger = DisclosureLedger()
    refused: list[bool] = []

    def work() -> None:
        try:
            record_disclosure(point_ids=["p1"], passages=1, keyword_checks=0, topic_listings=0, data_bytes=10)
        except DisclosureRefused:
            refused.append(True)

    with binding(ledger):
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()

    assert refused == [True]
    assert ledger.take() == []


def test_binding_is_undone_on_exit_and_nests() -> None:
    """The ledger is bound only inside the block, and an inner block puts the outer one back when it ends."""
    outer, inner = DisclosureLedger(), DisclosureLedger()
    assert current_ledger() is None

    with binding(outer) as bound:
        assert bound is outer
        with binding(inner):
            assert current_ledger() is inner
        assert current_ledger() is outer

    assert current_ledger() is None


def test_binding_is_undone_when_the_block_raises() -> None:
    """A turn that fails does not leave its ledger bound to whatever runs next in that context."""
    with pytest.raises(ValueError, match="boom"), binding(DisclosureLedger()):
        raise ValueError("boom")

    assert current_ledger() is None


class _CountingLock:
    """A lock that counts how many times it was entered."""

    def __init__(self) -> None:
        """Start unentered."""
        self._lock = threading.Lock()
        self.entered = 0

    def __enter__(self) -> _CountingLock:
        """Take the lock and count the entry.

        Returns:
            _CountingLock: This lock.
        """
        self._lock.acquire()
        self.entered += 1
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the lock.

        Args:
            *exc (object): The exception, if any; unused.
        """
        self._lock.release()


def test_the_seal_takes_and_closes_in_one_critical_section(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take and refusal happen under one hold of the lock, so no call can be accepted between them."""
    ledger = DisclosureLedger()
    _hand_over(ledger, "p1")
    lock = _CountingLock()
    monkeypatch.setattr(ledger, "_lock", lock)

    ledger.seal()

    assert lock.entered == 1


#: How many disclosures a racing worker may try before it gives up unrefused, which is a failure.
_RACE_TRIES = 5_000
#: Seconds the race may take, from the workers' start to the last join, before it is a failure.
_RACE_SECONDS = 20.0


@dataclass
class _Race:
    """What a race between worker threads and the seal left behind.

    Attributes:
        accepted (list[str]): The points the ledger accepted.
        written (list[str]): The points written by take and seal.
        after (list[DisclosureRecord]): What a take after the seal returned.
        refused (list[int]): The workers that stopped because the ledger refused them.
        stuck (int): Workers still running at the deadline.
    """

    accepted: list[str]
    written: list[str]
    after: list[DisclosureRecord]
    refused: list[int]
    stuck: int


def _race_the_seal(workers: int, enough: int) -> _Race:
    """Let worker threads record disclosures while the main thread takes and then seals the ledger.

    Every wait is bounded, and a worker that is never refused stops after ``_RACE_TRIES``, so a
    ledger that accepts after its seal fails the race instead of hanging it.

    Args:
        workers (int): Threads that record until one is refused.
        enough (int): Disclosures to accept before the seal.

    Returns:
        _Race: What the race left behind.
    """
    ledger = DisclosureLedger()
    deadline = time.monotonic() + _RACE_SECONDS
    started = threading.Barrier(workers + 1, timeout=_RACE_SECONDS)
    accepted: list[str] = []
    refused: list[int] = []
    guard = threading.Lock()

    def work(worker: int) -> None:
        started.wait()
        for number in range(_RACE_TRIES):
            point = f"w{worker}-{number}"
            try:
                handle = _hand_over(ledger, point)
            except DisclosureRefused:
                with guard:
                    refused.append(worker)
                return
            handle.answered(JOB)
            with guard:
                accepted.append(point)

    threads = [threading.Thread(target=work, args=(worker,), daemon=True) for worker in range(workers)]
    for thread in threads:
        thread.start()
    started.wait()
    written: list[str] = []
    while len(accepted) < enough and time.monotonic() < deadline:
        written.extend(point for record in ledger.take() for point in record.point_ids)
    written.extend(point for record in ledger.seal() for point in record.point_ids)
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    return _Race(accepted, written, ledger.take(), refused, stuck=sum(thread.is_alive() for thread in threads))


def test_calls_racing_the_seal_are_written_once_or_refused() -> None:
    """Under contention every accepted call is written exactly once, and every worker ends refused."""
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(10):
            race = _race_the_seal(workers=6, enough=100)

            assert race.stuck == 0
            assert sorted(race.refused) == list(range(6))
            assert race.after == []
            assert sorted(race.written) == sorted(race.accepted)
    finally:
        sys.setswitchinterval(interval)
