"""Helpers for turning a job's progress stream into readable log lines.

A long ingest reports progress per chunk — a 2000-chunk collection emits
two thousand ``Extracting entities: n/2000 chunks processed`` messages,
and another two thousand for hate-speech detection. Those messages are
written for a client that renders the latest one and discards the rest;
copied verbatim into the log they would bury everything else.

:class:`ProgressLogThrottle` decides which of them an operator sees. It
holds no reference to jobs, loguru, or docint domain code: it takes
messages and returns the ones worth logging, so it can be exercised
directly with a fake clock.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable

#: Matches the first ``n/m`` counter in a progress message.
_COUNTER_RE = re.compile(r"(\d+)\s*/\s*(\d+)")

#: Collapses every run of digits, so messages that differ only by their
#: counters share a key.
_DIGITS_RE = re.compile(r"\d+")


def progress_key(message: str) -> str:
    """Reduce a progress message to a key identifying its stage.

    Every run of digits collapses to ``#``, which splits the message
    stream exactly where it needs splitting. A per-file message keeps the
    filename that makes it unique, so each file announces itself:

        ``Core pipeline processing PDF (1/3): alpha.pdf``
        -> ``Core pipeline processing PDF (#/#): alpha.pdf``

    A per-chunk counter has nothing left to distinguish one tick from the
    next, so the whole stage shares one key and heartbeats:

        ``Extracting entities: 840/2000 chunks processed``
        -> ``Extracting entities: #/# chunks processed``

    This holds only while per-tick messages carry no other varying token.
    No current message does; one that did would defeat the throttle and
    log every tick.

    Args:
        message (str): A progress message.

    Returns:
        str: The stage key for ``message``.
    """
    return _DIGITS_RE.sub("#", message).strip()


def parse_counter(message: str) -> tuple[int, int] | None:
    """Extract the first ``n/m`` counter from a progress message.

    Args:
        message (str): A progress message.

    Returns:
        tuple[int, int] | None: ``(n, m)`` if the message carries a
        counter, otherwise ``None``.
    """
    match = _COUNTER_RE.search(message)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


class ProgressLogThrottle:
    """Collapse a repeating progress message down to a heartbeat.

    Thread-safe: the ingestion pipeline reports progress from a
    ``ThreadPoolExecutor``, so ticks arrive concurrently.

    A message is logged when it is the first of a new stage, when its
    counter reaches its total, or when ``interval_s`` has passed since
    this stage last logged. Anything else is held as *pending* and
    released by the next stage change or by :meth:`flush`, so a stage
    that stops short of its total still leaves its last observed value on
    the record rather than trailing off mid-count.
    """

    def __init__(
        self,
        interval_s: float,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialise the throttle.

        Args:
            interval_s (float): Seconds between heartbeat lines for one
                stage. ``0`` disables throttling entirely — every message
                is logged, which is the debug escape hatch.
            time_fn (Callable[[], float], optional): Monotonic clock,
                injectable for tests. Defaults to ``time.monotonic``.
        """
        self._interval_s = interval_s
        self._time_fn = time_fn
        self._lock = threading.Lock()
        self._last_key: str | None = None
        self._last_logged_at: float | None = None
        self._pending: str | None = None

    def observe(self, message: str) -> list[str]:
        """Record one progress message and return what should be logged.

        Args:
            message (str): The progress message the runner just emitted.

        Returns:
            list[str]: Zero, one, or two messages to log, in order. Two
            happens when a new stage starts while the previous stage had
            an unlogged tick: the stranded tick is released first so the
            stage it belongs to does not end mid-count.
        """
        with self._lock:
            key = progress_key(message)
            now = self._time_fn()

            if key != self._last_key:
                stranded = self._pending
                self._last_key = key
                self._pending = None
                self._last_logged_at = now
                return [stranded, message] if stranded is not None else [message]

            if self._interval_s <= 0:
                self._last_logged_at = now
                return [message]

            counter = parse_counter(message)
            if counter is not None and counter[0] >= counter[1]:
                self._pending = None
                self._last_logged_at = now
                return [message]

            if self._last_logged_at is None or (now - self._last_logged_at) >= self._interval_s:
                self._pending = None
                self._last_logged_at = now
                return [message]

            self._pending = message
            return []

    def flush(self) -> list[str]:
        """Release a held message, if any.

        Called on a job's terminal paths so the last thing a stage said
        is on the record even when it never reached its total.

        Returns:
            list[str]: The pending message, or an empty list.
        """
        with self._lock:
            pending = self._pending
            self._pending = None
            return [pending] if pending is not None else []
