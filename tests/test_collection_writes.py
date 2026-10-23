"""The registry of the collections this process is writing: its windows, its epochs and the closers that end them.

Every collection name here is invented.
"""

from __future__ import annotations

import sys
import threading

import pytest

from docint.core import collection_writes
from docint.core.collection_writes import WriteState, begin, state, touch, writing

#: An invented physical collection.
DOCS = "u0123456789ab__docs"


def test_a_collection_nothing_has_written_is_idle_at_epoch_zero() -> None:
    """A collection no window has opened on, and nothing has touched, stands at epoch 0 and is not written."""
    assert state(DOCS) == WriteState(0, False)


def test_a_window_keeps_its_collection_written_until_it_closes() -> None:
    """Opening a window moves the epoch and marks the collection written; closing it moves the epoch again."""
    close = begin(DOCS)
    opened = state(DOCS)
    close()

    assert opened == WriteState(1, True)
    assert state(DOCS) == WriteState(2, False)


def test_a_collection_stays_written_until_its_last_window_closes() -> None:
    """With two writes at once, the collection stays written until both have ended."""
    first = begin(DOCS)
    second = begin(DOCS)
    first()
    one_left = state(DOCS)
    second()

    assert one_left == WriteState(3, True)
    assert state(DOCS) == WriteState(4, False)


def test_closing_a_window_twice_closes_it_once() -> None:
    """A closer called again does nothing, so it cannot end a window another write still holds open."""
    first = begin(DOCS)
    begin(DOCS)
    first()
    first()

    assert state(DOCS) == WriteState(3, True)


def test_the_writing_block_closes_its_window_when_its_body_raises() -> None:
    """A write that fails still ends its window, so a failed run never leaves its collection written."""
    inside: list[WriteState] = []

    with pytest.raises(RuntimeError, match="the write failed"), writing(DOCS):
        inside.append(state(DOCS))
        raise RuntimeError("the write failed")

    assert inside == [WriteState(1, True)]
    assert state(DOCS) == WriteState(2, False)


def test_a_touch_moves_the_epoch_and_opens_no_window() -> None:
    """A change made in one step moves the epoch, so a read kept before it no longer holds, and opens no window."""
    touch(DOCS)

    assert state(DOCS) == WriteState(1, False)


def test_a_window_leaves_every_other_collection_alone() -> None:
    """A window, its close and a touch move their own collection only: another one and its companion stay idle."""
    others = ("u0123456789ab__hafen", f"{DOCS}_images")
    close = begin(DOCS)
    during = [state(name) for name in others]
    close()
    touch(DOCS)

    assert during == [WriteState(0, False)] * len(others)
    assert [state(name) for name in others] == [WriteState(0, False)] * len(others)


def test_a_window_closes_on_the_registry_it_was_opened_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """A closer ends its window where it was opened, so a registry swapped in meanwhile is left as it was.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    close = begin(DOCS)
    opened_on = collection_writes._registry
    monkeypatch.setattr(collection_writes, "_registry", collection_writes._Registry())
    close()

    assert state(DOCS) == WriteState(0, False)
    assert opened_on.state(DOCS) == WriteState(2, False)


def test_windows_opened_and_closed_across_threads_all_close() -> None:
    """Eight threads each opening and closing 200 windows lose no move of the epoch and leave no window open.

    The threads are switched every microsecond, so a registry without its lock would lose updates.
    """
    start = threading.Barrier(8, timeout=5)

    def _write() -> None:
        """Open and close 200 windows, starting together with the other threads."""
        start.wait()
        for _ in range(200):
            close = begin(DOCS)
            close()

    threads = [threading.Thread(target=_write, daemon=True) for _ in range(8)]
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
    finally:
        sys.setswitchinterval(interval)

    assert not any(thread.is_alive() for thread in threads)
    assert state(DOCS) == WriteState(3200, False)
