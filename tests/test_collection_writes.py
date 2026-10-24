"""The registry of the collections this process is writing: its windows, its epochs and the closers that end them.

Every collection name here is invented.
"""

from __future__ import annotations

import sys
import threading

import pytest

from docint.core import collection_writes
from docint.core.collection_writes import WriteState, begin, state, touch, watch, writing

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


def test_a_watch_over_no_window_sees_no_write() -> None:
    """A watch over a stretch in which no window on its collection stood open reports no write."""
    written = watch(DOCS)

    assert written() is False


def test_a_watch_sees_a_window_open_when_it_began_though_it_closed_since() -> None:
    """A window already open when a watch began is a write to it, though the window closed before it was asked."""
    close = begin(DOCS)
    written = watch(DOCS)
    close()

    assert written() is True


def test_a_watch_sees_a_window_open_when_it_is_asked() -> None:
    """A window that opened after a watch began, and is still open when the watch is asked, is a write to it."""
    written = watch(DOCS)

    with writing(DOCS):
        during = written()

    assert during is True


def test_a_watch_sees_a_window_that_opened_and_closed_in_between() -> None:
    """A window that opened and closed between a watch's start and its question is a write to it.

    No window is open at either end, and the epoch moved as a touch would have moved it.
    """
    written = watch(DOCS)
    close = begin(DOCS)
    close()

    assert written() is True
    assert state(DOCS) == WriteState(2, False)


def test_a_touch_alone_is_no_write_to_a_watch() -> None:
    """A touch moves the epoch and opens no window, so it is no write to a watch.

    A keyword check's own index ensure touches its collection, which must not flag the check.
    """
    written = watch(DOCS)
    touch(DOCS)

    assert written() is False
    assert state(DOCS) == WriteState(1, False)


def test_a_watch_sees_only_its_own_collection() -> None:
    """A window on another collection, or on the watched one's image companion, is no write to the watch.

    It is none while it stands open, nor once it has closed.
    """
    others = ("u0123456789ab__hafen", f"{DOCS}_images")
    seen: list[tuple[bool, bool]] = []
    for name in others:
        written = watch(DOCS)
        with writing(name):
            during = written()
        seen.append((during, written()))

    assert seen == [(False, False)] * len(others)


def test_a_watch_asks_the_registry_it_began_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """A watch asks the registry it began on, so a window on a registry swapped in meanwhile is no write to it.

    Once its own registry is back, a window opened there is one.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    began_on = collection_writes._registry
    written = watch(DOCS)
    monkeypatch.setattr(collection_writes, "_registry", collection_writes._Registry())
    with writing(DOCS):
        elsewhere = written()
    monkeypatch.setattr(collection_writes, "_registry", began_on)
    with writing(DOCS):
        here = written()

    assert (elsewhere, here) == (False, True)
