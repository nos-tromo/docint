"""Run work on its own thread, in a free slot, and carry on without waiting for it.

The process-wide slot pools live here, so every caller that searches or calls the model
competes for the same slots: a second pool would double the bound.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import threading
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

#: Searches that may run at once in this process, abandoned ones included.
SEARCH_SLOTS = 8
#: Longest a search waits for a free slot; the wait comes out of its turn's time budget.
SEARCH_SLOT_WAIT_S = 5.0
#: The process-wide slots every search takes; a resource cap, not session state.
SEARCH_SLOT_POOL = threading.BoundedSemaphore(SEARCH_SLOTS)
#: Model calls that may run at once in this process, abandoned ones included.
MODEL_SLOTS = 8
#: Longest a model call waits for a free slot; the wait comes out of its turn's time budget.
MODEL_SLOT_WAIT_S = 5.0
#: The process-wide slots every model call takes; a resource cap, not session state.
MODEL_SLOT_POOL = threading.BoundedSemaphore(MODEL_SLOTS)


class NoSlot(Exception):
    """Every slot stayed taken for as long as the work could wait."""


def _launch(work: Callable[[], T], *, slots: threading.Semaphore, name: str) -> concurrent.futures.Future[T]:
    """Run ``work`` on a named daemon thread, in a slot the caller already holds.

    From here the slot is the thread's: its worker gives it back when the work
    actually ends, before it resolves the future. When the thread cannot be
    started, the slot is given back here and the failure propagates.

    Args:
        work (Callable[[], T]): The work to run.
        slots (threading.Semaphore): The slots the held one belongs to.
        name (str): The worker thread's name, so abandoned work can be found.

    Returns:
        concurrent.futures.Future[T]: The work's future, running from the start.

    Raises:
        Exception: Whatever stops the work from starting, its thread included,
            once its slot has been given back.
    """
    try:
        future: concurrent.futures.Future[T] = concurrent.futures.Future()
        # Running from the start, so a caller's cancel() cannot land on a future the worker resolves.
        future.set_running_or_notify_cancel()
        context = contextvars.copy_context()

        def _run() -> None:
            """Run the work, give its slot back, then hand its outcome to the future."""
            try:
                try:
                    outcome = context.run(work)
                finally:
                    slots.release()
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(outcome)

        threading.Thread(target=_run, name=name, daemon=True).start()
    except BaseException:
        # Until its thread runs, the slot is this call's to give back.
        slots.release()
        raise
    return future


def start_bounded(
    work: Callable[[], T],
    *,
    slots: threading.Semaphore,
    name: str,
    slot_wait: float,
) -> concurrent.futures.Future[T]:
    """Start ``work`` on its own thread, in a free slot, and return without waiting for it.

    The call waits for a slot at most ``slot_wait``. The worker gives the slot back
    when the work actually ends, before it resolves the future, so work its caller
    abandons holds its slot until it finishes and the caller never races its own
    slot on its next call; abandoned work is thus capped at the slot count. No
    deadline is passed into the work: the work's own timeouts end it, and so free
    its slot.

    The work runs in a copy of the caller's context, since request state such as
    the active collection lives in context variables. Its outcome, result or
    exception, lands on the returned future, which is running from the start and so
    cannot be cancelled. Wait on it with :func:`concurrent.futures.wait` and read
    it only once done: the work's own ``TimeoutError`` must not read as the
    deadline passing.

    Args:
        work (Callable[[], T]): The work to run.
        slots (threading.Semaphore): The slots to take one from.
        name (str): The worker thread's name, so abandoned work can be found.
        slot_wait (float): The longest the call waits for a slot; the semaphore times
            it. A caller with a deadline passes no more than the time it has left.

    Returns:
        concurrent.futures.Future[T]: The running work's future, holding its result
        or its exception once the work ends.

    Raises:
        NoSlot: When no slot frees up within ``slot_wait``; the work is never called
            and no thread is started.
        Exception: Whatever stops the work from starting, its thread included,
            once its slot has been given back.
    """
    if not slots.acquire(timeout=slot_wait):
        raise NoSlot
    return _launch(work, slots=slots, name=name)
