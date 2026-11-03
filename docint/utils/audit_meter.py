"""Count what an audited operation sends to inference, stage by stage.

A batch operation (an ingest run, a summary build, entity resolution, a topics
run) hands collection content to the inference services in thousands of
calls. The audit trail records each such transfer once per stage and
operation, with counts, instead of once per call. The inference clients report
every call here with :func:`count_transfer`, and the operation that owns the
work binds a :class:`TransferMeter` around it with :func:`metering`; whatever
was counted is read back when the operation ends and written as its
``ai_call`` events.

The meter is carried in a :class:`contextvars.ContextVar`, so it follows the
work into anyio worker threads by itself. A plain thread pool does not copy
the caller's context: submit through :func:`in_context` so the task counts
into the meter of whoever submitted it. A call made with no meter bound —
a chat turn's own retrieval, a health probe — is not counted here; the
request records its model call itself.

Work the preprocessing pool starts when a file is uploaded belongs to no run
yet. It counts into the collection's *staged* meter (:func:`staged_meter`),
which the next ingest run of that collection takes over with
:func:`take_staged`, since that run is what consumes the results.

Stdlib only, so every client can import it.
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")


@dataclass(frozen=True, order=True)
class TransferKey:
    """One kind of transfer: which stage sent what to which model.

    Attributes:
        stage: The stage that made the calls, e.g. ``embed`` or ``caption``.
        recipient: Where the data went, e.g. ``inference:embed``.
        model: The model the calls asked for.
        kind: What one counted item is: ``text``, ``image`` or ``media``.
    """

    stage: str
    recipient: str
    model: str
    kind: str


@dataclass
class TransferTally:
    """How much of one transfer kind an operation sent.

    Attributes:
        items: Items sent (texts, images, media files).
        requests: Calls made to carry them.
    """

    items: int = 0
    requests: int = 0


class TransferMeter:
    """Thread-safe tally of one operation's transfers, by :class:`TransferKey`."""

    def __init__(self) -> None:
        """Start with nothing counted."""
        self._lock = threading.Lock()
        self._tallies: dict[TransferKey, TransferTally] = {}

    def add(self, key: TransferKey, items: int, requests: int = 1) -> None:
        """Count one call (or several) that carried ``items`` items.

        Args:
            key (TransferKey): The transfer kind.
            items (int): Items the call carried.
            requests (int): Calls counted.
        """
        with self._lock:
            tally = self._tallies.setdefault(key, TransferTally())
            tally.items += items
            tally.requests += requests

    def merge(self, other: TransferMeter) -> None:
        """Add another meter's counts to this one.

        Args:
            other (TransferMeter): The meter to fold in; left unchanged.
        """
        for key, tally in other.snapshot():
            self.add(key, tally.items, tally.requests)

    def snapshot(self) -> list[tuple[TransferKey, TransferTally]]:
        """Return the counts so far, in a stable order.

        Returns:
            list[tuple[TransferKey, TransferTally]]: One entry per transfer kind.
        """
        with self._lock:
            return [(key, TransferTally(tally.items, tally.requests)) for key, tally in sorted(self._tallies.items())]

    def __bool__(self) -> bool:
        """Whether anything was counted.

        Returns:
            bool: ``True`` once a transfer was counted.
        """
        with self._lock:
            return bool(self._tallies)


_CURRENT: contextvars.ContextVar[TransferMeter | None] = contextvars.ContextVar("docint_transfer_meter", default=None)


def count_transfer(stage: str, recipient: str, model: str | None, items: int = 1, *, kind: str = "text") -> None:
    """Count one inference call against the meter bound to this context.

    Never raises and does nothing when no meter is bound, so a client can
    call it on every request.

    Args:
        stage (str): The stage making the call.
        recipient (str): Where the data goes, ``inference:<task>``.
        model (str | None): The model asked for; ``unknown`` when not known.
        items (int): Items the call carries.
        kind (str): What one item is.
    """
    meter = _CURRENT.get()
    if meter is None or items <= 0:
        return
    meter.add(TransferKey(stage, recipient, str(model or "unknown"), kind), items)


@contextmanager
def metering(meter: TransferMeter | None = None) -> Iterator[TransferMeter]:
    """Count every transfer made in this context into one meter.

    A nested block counts into its own meter only, so a sub-operation
    (entity resolution after an ingest) keeps its transfers apart.

    Args:
        meter (TransferMeter | None): The meter to count into; a new one when
            omitted.

    Yields:
        TransferMeter: The bound meter.
    """
    bound = meter if meter is not None else TransferMeter()
    token = _CURRENT.set(bound)
    try:
        yield bound
    finally:
        _CURRENT.reset(token)


def in_context(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    """Bind a callable to a copy of the current context, for another thread.

    Call it once per submission: a context can only be entered by one thread
    at a time.

    Args:
        fn (Callable[_P, _T]): The callable to run elsewhere.

    Returns:
        Callable[_P, _T]: The callable, running in the caller's context.
    """
    context = contextvars.copy_context()

    def run(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        return context.run(fn, *args, **kwargs)

    return run


_staged_lock = threading.Lock()
_staged: dict[str, TransferMeter] = {}


def staged_meter(collection: str) -> TransferMeter:
    """Return the meter upload-time work on a collection counts into.

    Args:
        collection (str): The physical collection.

    Returns:
        TransferMeter: The collection's staged meter, created on first use.
    """
    with _staged_lock:
        return _staged.setdefault(collection, TransferMeter())


def take_staged(collection: str) -> TransferMeter | None:
    """Remove and return a collection's staged meter.

    Args:
        collection (str): The physical collection.

    Returns:
        TransferMeter | None: What upload-time work counted, or ``None``.
    """
    with _staged_lock:
        return _staged.pop(collection, None)
