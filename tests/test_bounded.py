"""Bounded work started without waiting for it. Every value here is invented."""

from __future__ import annotations

import concurrent.futures
import contextvars
import threading
import time
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from typing import TypeVar

import pytest
from assessment_fakes import join_threads
from typing_extensions import override

from docint.agents.assessment.bounded import NoSlot, start_bounded

T = TypeVar("T")

#: Name of every worker thread these tests start, so a held one can be found and joined.
WORKER = "bounded-test-worker"


@pytest.fixture
def release() -> Iterator[threading.Event]:
    """Hold work until the test lets it go; teardown lets go and joins the workers, so none outlives the test.

    Yields:
        threading.Event: The event held work waits on.
    """
    event = threading.Event()
    yield event
    event.set()
    join_threads(WORKER)


def _start(work: Callable[[], T], slots: threading.Semaphore, slot_wait: float = 1.0) -> concurrent.futures.Future[T]:
    """Start ``work`` under the tests' worker name.

    Args:
        work (Callable[[], T]): The work to start.
        slots (threading.Semaphore): The slots to take one from.
        slot_wait (float): The longest the work waits for a slot.

    Returns:
        concurrent.futures.Future[T]: The started work's future.
    """
    return start_bounded(work, slots=slots, clock=time.monotonic, name=WORKER, slot_wait=slot_wait)


def test_start_bounded_returns_without_waiting(release: threading.Event) -> None:
    """The caller gets the future while the work is still running, and the work finishes on its own.

    Args:
        release (threading.Event): Holds the work.
    """
    entered = threading.Event()

    def _held() -> str:
        """Say it has begun, wait for the release, then finish.

        Returns:
            str: The invented result.
        """
        entered.set()
        release.wait(timeout=10)
        return "finished"

    future = _start(_held, threading.BoundedSemaphore(1))
    still_running = not future.done()
    began = entered.wait(timeout=5)
    release.set()

    assert (still_running, began) == (True, True)
    assert future.result(timeout=5) == "finished"


def test_the_slot_is_released_when_abandoned_work_ends(release: threading.Event) -> None:
    """Work nobody waits for holds its slot until it ends and gives it back then, its future never read.

    Args:
        release (threading.Event): Holds the work.
    """
    slots = threading.BoundedSemaphore(1)

    def _held() -> None:
        """Wait for the release."""
        release.wait(timeout=10)

    future = _start(_held, slots)

    # The caller moves on without waiting for the work or reading its future.
    held_while_running = slots.acquire(blocking=False) is False
    release.set()
    join_threads(WORKER)
    free_once_ended = slots.acquire(blocking=False)

    assert (held_while_running, free_once_ended) == (True, True)
    assert future.done()
    slots.release()


class _NoFreeSlot(threading.BoundedSemaphore):
    """Slots that record how long a caller would wait for one and find them all taken."""

    def __init__(self) -> None:
        """Start with one slot and no waits recorded."""
        super().__init__(1)
        self.waits: list[float | None] = []

    @override
    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        """Record the wait and report no free slot.

        Args:
            blocking (bool): Whether the caller would block.
            timeout (float | None): How long it would wait.

        Returns:
            bool: Always ``False``.
        """
        self.waits.append(timeout)
        return False


@pytest.mark.parametrize("slot_wait", [0.0, 1.0, 5.0])
def test_no_slot_raises_after_the_wait(slot_wait: float, monkeypatch: pytest.MonkeyPatch) -> None:
    """With every slot taken the call waits ``slot_wait`` for one, then raises; the work never runs, no thread starts.

    Args:
        slot_wait (float): The longest the call may wait for a slot.
        monkeypatch (pytest.MonkeyPatch): Records thread starts.
    """
    slots = _NoFreeSlot()
    called: list[str] = []
    started: list[str] = []
    real_start = threading.Thread.start

    def _record_start(thread: threading.Thread) -> None:
        """Record a worker thread's start, then start any thread.

        Args:
            thread (threading.Thread): The thread.
        """
        if thread.name == WORKER:
            started.append(thread.name)
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", _record_start)

    with pytest.raises(NoSlot):
        _start(lambda: called.append("ran"), slots, slot_wait)

    assert slots.waits == [slot_wait]
    assert (called, started) == ([], [])


class _Watched(threading.BoundedSemaphore):
    """One slot that notes whether the work's future had resolved when the slot came back."""

    def __init__(self) -> None:
        """Start with one slot, no future and nothing noted."""
        super().__init__(1)
        self.future: concurrent.futures.Future[None] | None = None
        self.resolved_at_release: list[bool] = []

    @override
    def release(self, n: int = 1) -> None:
        """Note whether the future had resolved, then give the slot back.

        Args:
            n (int): How many slots to give back.
        """
        self.resolved_at_release.append(self.future is not None and self.future.done())
        super().release(n)


def test_the_slot_is_given_back_before_the_future_resolves(release: threading.Event) -> None:
    """A caller that sees its work finished never finds the work's own slot still taken.

    Args:
        release (threading.Event): Holds the work until the future is noted.
    """
    slots = _Watched()

    def _held() -> None:
        """Wait for the release."""
        release.wait(timeout=10)

    slots.future = future = _start(_held, slots)
    release.set()
    future.result(timeout=5)

    assert slots.resolved_at_release == [False]
    assert slots.acquire(blocking=False) is True
    slots.release()


def test_the_result_lands_on_the_future_and_the_slot_comes_back() -> None:
    """A finished piece of work is read from its future, with its slot free again."""
    slots = threading.BoundedSemaphore(1)

    future = _start(lambda: "finished", slots)

    assert future.result(timeout=5) == "finished"
    assert slots.acquire(blocking=False) is True
    slots.release()


class _Abort(BaseException):
    """Ends work the way an interrupt would, past ``except Exception``."""


@pytest.mark.parametrize("failure", [RuntimeError("invented failure"), _Abort()], ids=["exception", "base-exception"])
def test_a_failure_lands_on_the_future_and_the_slot_comes_back(failure: BaseException) -> None:
    """Work that raises, even past ``except Exception``, hands the failure to its future and frees its slot.

    Args:
        failure (BaseException): What the work raises.
    """
    slots = threading.BoundedSemaphore(1)

    def _failing() -> None:
        """Raise the failure.

        Raises:
            BaseException: Always.
        """
        raise failure

    future = _start(_failing, slots)
    done, _ = concurrent.futures.wait([future], timeout=5)

    assert done == {future}
    assert future.exception(timeout=0) is failure
    assert slots.acquire(blocking=False) is True
    slots.release()


_SCOPE: ContextVar[str | None] = ContextVar("invented_bounded_scope", default=None)


def test_the_work_runs_on_a_named_daemon_thread_in_a_copy_of_the_callers_context() -> None:
    """The work sees the request's scope, which lives in a context variable, and cannot change the caller's."""
    seen: list[tuple[str, bool, str | None]] = []

    def _record() -> None:
        """Record the thread and the scope the work runs under, then change the scope."""
        thread = threading.current_thread()
        seen.append((thread.name, thread.daemon, _SCOPE.get()))
        _SCOPE.set("changed by the work")

    token = _SCOPE.set("invented-collection")
    try:
        _start(_record, threading.BoundedSemaphore(1)).result(timeout=5)
        scope_after = _SCOPE.get()
    finally:
        _SCOPE.reset(token)

    assert seen == [(WORKER, True, "invented-collection")]
    assert scope_after == "invented-collection"


@pytest.mark.parametrize("failure", ["context", "thread", "start"])
def test_a_start_that_fails_gives_the_slot_back(failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever fails between taking the slot and the thread running gives the slot back and raises.

    Args:
        failure (str): The failing step: copying the context, building the thread, or starting it.
        monkeypatch (pytest.MonkeyPatch): Makes that step fail.
    """
    slots = threading.BoundedSemaphore(1)
    called: list[str] = []
    real_start = threading.Thread.start

    def _abort(*args: object, **kwargs: object) -> None:
        """Fail the way the selected step would.

        Args:
            *args (object): Ignored.
            **kwargs (object): Ignored.

        Raises:
            _Abort: Always.
        """
        raise _Abort

    def _start_or_abort(thread: threading.Thread) -> None:
        """Fail a worker thread's start as a process out of threads does, and start any other thread.

        Args:
            thread (threading.Thread): The thread.

        Raises:
            _Abort: For a worker thread.
        """
        if thread.name == WORKER:
            raise _Abort
        real_start(thread)

    if failure == "context":
        monkeypatch.setattr(contextvars, "copy_context", _abort)
    if failure == "thread":
        monkeypatch.setattr(threading.Thread, "__init__", _abort)
    if failure == "start":
        monkeypatch.setattr(threading.Thread, "start", _start_or_abort)

    with pytest.raises(_Abort):
        _start(lambda: called.append("ran"), slots)

    assert called == []
    assert slots.acquire(blocking=False) is True
    slots.release()


def test_a_started_future_cannot_be_cancelled(release: threading.Event) -> None:
    """The work is under way once its future is handed out, so a cancel changes nothing and the outcome still lands.

    Args:
        release (threading.Event): Holds the work.
    """

    def _held() -> str:
        """Wait for the release, then finish.

        Returns:
            str: The invented result.
        """
        release.wait(timeout=10)
        return "finished"

    future = _start(_held, threading.BoundedSemaphore(1))
    cancelled = future.cancel()
    release.set()

    assert cancelled is False
    assert future.result(timeout=5) == "finished"
