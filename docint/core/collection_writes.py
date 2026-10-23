"""The physical collections this process is writing, so that no cached read of one is trusted across a write.

A write in this process is a window on the physical collection it writes, open from before its first
change until after its last, however the write ends. Each collection carries an epoch, which moves
whenever a window on it opens or closes and whenever a change made in one step touches it. A cached
read of a collection is valid only for the epoch it was read at, and never while a window on it is
open.

The registry is keyed by physical collection name and holds nothing else: no user, no session and no
active collection, so it never selects what a request works on. Standard library only.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class WriteState:
    """Where a collection's writes stand.

    Attributes:
        epoch (int): How often a window on the collection has opened or closed, or it has been
            touched; 0 until then. A cached read is valid only for the epoch it was read at.
        writing (bool): Whether a window on the collection is open.
    """

    epoch: int
    writing: bool


class _Registry:
    """The open windows and the epoch of every collection a window or a touch has reached.

    Its lock is a leaf: nothing is called while it is held. A collection's count of open windows
    is dropped once it reaches 0; its epoch is never dropped, so no epoch is ever handed out twice.
    """

    def __init__(self) -> None:
        """Start with no window open and every epoch at 0."""
        self._lock = threading.Lock()
        self._open: dict[str, int] = {}
        self._epochs: dict[str, int] = {}

    def open(self, collection: str) -> None:
        """Open a window on a collection, moving its epoch.

        Args:
            collection (str): The physical collection.
        """
        with self._lock:
            self._epochs[collection] = self._epochs.get(collection, 0) + 1
            self._open[collection] = self._open.get(collection, 0) + 1

    def close(self, collection: str) -> None:
        """Close a window on a collection, moving its epoch.

        Args:
            collection (str): The physical collection.
        """
        with self._lock:
            self._epochs[collection] = self._epochs.get(collection, 0) + 1
            left = self._open.get(collection, 0) - 1
            if left > 0:
                self._open[collection] = left
            else:
                self._open.pop(collection, None)

    def touch(self, collection: str) -> None:
        """Move a collection's epoch, opening no window.

        Args:
            collection (str): The physical collection.
        """
        with self._lock:
            self._epochs[collection] = self._epochs.get(collection, 0) + 1

    def state(self, collection: str) -> WriteState:
        """Return where a collection's writes stand.

        Args:
            collection (str): The physical collection.

        Returns:
            WriteState: Its epoch, and whether a window on it is open.
        """
        with self._lock:
            return WriteState(epoch=self._epochs.get(collection, 0), writing=collection in self._open)


_registry = _Registry()


def begin(collection: str) -> Callable[[], None]:
    """Open a write window on a collection, and return the call that closes it.

    The closer closes the window once, on the registry it was opened on, however often and from
    whichever thread it is called; a call after the first does nothing. Opening and closing each
    move the collection's epoch.

    Args:
        collection (str): The physical collection.

    Returns:
        Callable[[], None]: The closer.
    """
    registry = _registry
    # The first close takes it and never gives it back.
    once = threading.Lock()

    def close() -> None:
        """Close the window, unless it is closed already."""
        if once.acquire(blocking=False):
            registry.close(collection)

    # Opened last, so nothing that can fail is left to strand the window.
    registry.open(collection)
    return close


@contextmanager
def writing(collection: str) -> Iterator[None]:
    """Hold a write window on a collection for the block, closing it however the block ends.

    Args:
        collection (str): The physical collection.

    Yields:
        None: Control, while the window is open.
    """
    close = begin(collection)
    try:
        yield
    finally:
        close()


def touch(collection: str) -> None:
    """Move a collection's epoch without opening a window, for a change made in one step.

    Args:
        collection (str): The physical collection.
    """
    _registry.touch(collection)


def state(collection: str) -> WriteState:
    """Return where a collection's writes stand in this process.

    Args:
        collection (str): The physical collection.

    Returns:
        WriteState: Its epoch, and whether a window on it is open.
    """
    return _registry.state(collection)
