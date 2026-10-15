"""Run one piece of work on its own thread, in a free slot, and wait for it no longer than a deadline."""

from __future__ import annotations

import concurrent.futures
import contextvars
import threading
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class NoSlot(Exception):
    """Every slot stayed taken for as long as the work could wait."""


def run_bounded(
    work: Callable[[], T],
    timeout: float,
    *,
    slots: threading.Semaphore,
    clock: Callable[[], float],
    name: str,
    slot_wait: float,
) -> concurrent.futures.Future[T] | None:
    """Run ``work`` on its own thread, in a free slot, and wait for it no longer than ``timeout``.

    The work first waits for a slot, at most ``min(timeout, slot_wait)``, and that
    wait comes out of ``timeout``. The worker gives the slot back when the work
    actually ends, before it resolves the future, so abandoned work holds its slot
    until it finishes and the caller never races its own slot on its next call;
    abandoned work is thus capped at the slot count. The deadline is not passed
    into the work: the work's own timeouts end it, and so free its slot.

    The work runs in a copy of the caller's context, since request state such as
    the active collection lives in context variables. Its outcome, result or
    exception, lands on a future that is only waited on, never asked for a result
    with a timeout: the work's own ``TimeoutError`` must not read as the deadline
    passing.

    Args:
        work (Callable[[], T]): The work to run.
        timeout (float): Seconds the slot wait and the work may take together.
        slots (threading.Semaphore): The slots to take one from.
        clock (Callable[[], float]): Monotonic clock the slot wait is measured with.
        name (str): The worker thread's name, so abandoned work can be found.
        slot_wait (float): The longest the work waits for a slot.

    Returns:
        concurrent.futures.Future[T] | None: The finished work, holding its result
        or its exception; ``None`` when the slot wait left no time (the work never
        starts) or the work outlasts the timeout, in which case it finishes in the
        background, unread, still in its slot.

    Raises:
        NoSlot: When no slot frees up in time; the work is never called and no
            thread is started.
        Exception: Whatever stops the work from starting, its thread included,
            once its slot has been given back.
    """
    began = clock()
    if not slots.acquire(timeout=min(timeout, slot_wait)):
        raise NoSlot
    try:
        left = timeout - (clock() - began)
        if left <= 0:
            slots.release()
            return None
        future: concurrent.futures.Future[T] = concurrent.futures.Future()
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
    done, _ = concurrent.futures.wait([future], timeout=left)
    return future if done else None
