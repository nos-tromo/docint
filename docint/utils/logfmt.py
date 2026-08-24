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

import math
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

#: Unit labels for :func:`format_bytes`, mirroring the SPA's ``formatBytes``.
_BYTE_UNITS: tuple[str, ...] = ("KB", "MB", "GB", "TB")

#: Extension reported for a file that has none.
_UNKNOWN_KIND = "none"

#: Matches the first ``n/m`` counter in a progress message.
_COUNTER_RE = re.compile(r"(\d+)\s*/\s*(\d+)")

#: Collapses every run of digits, so messages that differ only by their
#: counters share a key.
_DIGITS_RE = re.compile(r"\d+")


def format_bytes(n: float) -> str:
    """Render a byte count the way the SPA's ingest card renders it.

    Deliberately a port of ``formatBytes`` in
    ``frontend/src/lib/ingestStatus.ts``, down to the truncation: binary
    sizing with decimal-style labels, one decimal place, floored rather
    than rounded so ``1.499 MB`` reads as ``1.4 MB``. The log and the card
    describe the same upload, so they must not disagree about its size.

    Args:
        n (float): Byte count.

    Returns:
        str: A human-readable size such as ``"0 B"``, ``"1023 B"``, or
        ``"1.4 MB"``.
    """
    if not math.isfinite(n) or n <= 0:
        return "0 B"
    if n < 1024:
        return f"{int(n)} B"
    value = n / 1024
    unit_idx = 0
    while value >= 1024 and unit_idx < len(_BYTE_UNITS) - 1:
        value /= 1024
        unit_idx += 1
    truncated = math.floor(value * 10) / 10
    return f"{truncated:.1f} {_BYTE_UNITS[unit_idx]}"


@dataclass(frozen=True)
class InputFile:
    """One staged file, as the run-start banner describes it.

    Attributes:
        name: Path relative to the batch directory, so nesting stays visible.
        kind: Lowercased extension without the dot, or ``"none"``.
        size_bytes: Size on disk.
    """

    name: str
    kind: str
    size_bytes: int


@dataclass(frozen=True)
class InputInventory:
    """What a job is about to ingest.

    Attributes:
        files: The listed files, capped at the caller's limit.
        total_files: Every file found, including those beyond the cap.
        total_bytes: Total size of every file found.
        by_type: ``(extension, count)`` pairs, most frequent first, over
            every file found.
        omitted: How many files the cap left out.
    """

    files: tuple[InputFile, ...]
    total_files: int
    total_bytes: int
    by_type: tuple[tuple[str, int], ...]
    omitted: int


def _iter_files(root: Path) -> Iterator[os.DirEntry[str]]:
    """Walk ``root`` recursively, yielding file entries.

    Uses ``os.scandir`` so each entry's ``stat`` is served from the
    directory read where the platform allows it. Unreadable directories
    are skipped rather than raising — an inventory is a log line, and must
    not be able to fail a run.

    Args:
        root (Path): Directory to walk.

    Yields:
        os.DirEntry[str]: One entry per regular file found.
    """
    stack = [str(root)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            yield entry
                    except OSError:
                        continue
        except OSError:
            continue


def describe_inputs(root: Path, limit: int = 50) -> InputInventory:
    """Inventory the files staged under ``root``.

    This is the first thing in the ingest path that asks the filesystem how
    big anything is — nothing else computes a size, so the banner cannot
    borrow one. One ``stat`` per file is the whole cost.

    Args:
        root (Path): The job's batch directory.
        limit (int, optional): Most files to list individually. Totals and
            ``by_type`` still cover every file found. Defaults to 50.

    Returns:
        InputInventory: The staged inventory; empty if ``root`` is missing
        or unreadable.
    """
    found: list[InputFile] = []
    total_bytes = 0
    kinds: Counter[str] = Counter()

    for entry in _iter_files(root):
        try:
            size = entry.stat(follow_symlinks=False).st_size
        except OSError:
            size = 0
        name = os.path.relpath(entry.path, str(root))
        kind = Path(entry.name).suffix.lstrip(".").lower() or _UNKNOWN_KIND
        found.append(InputFile(name=name, kind=kind, size_bytes=size))
        total_bytes += size
        kinds[kind] += 1

    found.sort(key=lambda f: f.name)
    capped = max(0, limit)
    return InputInventory(
        files=tuple(found[:capped]),
        total_files=len(found),
        total_bytes=total_bytes,
        by_type=tuple(kinds.most_common()),
        omitted=max(0, len(found) - capped),
    )


def format_override(value: bool | None) -> str:
    """Render a tri-state job override for the run banner.

    ``hybrid`` / ``ner`` / ``hate_speech`` are per-request *overrides*: ``None``
    means the request specified nothing and the configured default applies. Left
    to ``str(None).lower()`` that prints as ``none``, which in a line an operator
    greps reads as the feature being off — and ``none`` is already taken on the
    same line, where ``by_type`` uses it for files with no extension.

    Args:
        value (bool | None): The override, or ``None`` when unspecified.

    Returns:
        str: ``"true"``, ``"false"``, or ``"default"``.
    """
    if value is None:
        return "default"
    return "true" if value else "false"


def format_by_type(by_type: tuple[tuple[str, int], ...]) -> str:
    """Render a by-extension rollup for the banner header.

    Args:
        by_type (tuple[tuple[str, int], ...]): ``(extension, count)`` pairs.

    Returns:
        str: A compact rollup such as ``"pdf:2,docx:1"``, or ``"none"``
        when there is nothing to describe.
    """
    if not by_type:
        return "none"
    return ",".join(f"{kind}:{count}" for kind, count in by_type)


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
    *that stage* last logged. Anything else is held as pending and
    released by :meth:`flush`, so a stage that stops short of its total
    still leaves its last observed value on the record rather than
    trailing off mid-count.

    State is kept **per stage key**, not for one "current" stage. NER and
    hate-speech detection run concurrently over the same chunks, so their
    messages interleave: with a single current-stage cursor every message
    differs from the one before it, every message reads as a stage change,
    and nothing throttles at all. That is not hypothetical — it is what a
    live run did before this was keyed per stage.
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
        self._last_logged_at: dict[str, float] = {}
        self._pending: dict[str, str] = {}

    def observe(self, message: str) -> list[str]:
        """Record one progress message and return what should be logged.

        Args:
            message (str): The progress message the runner just emitted.

        Returns:
            list[str]: The message, or an empty list if it was held. A
            list rather than an optional so callers iterate one way.
        """
        with self._lock:
            key = progress_key(message)
            now = self._time_fn()

            last = self._last_logged_at.get(key)
            if last is None:
                # First sighting of this stage — always worth announcing.
                self._last_logged_at[key] = now
                return [message]

            if self._interval_s <= 0:
                self._last_logged_at[key] = now
                return [message]

            counter = parse_counter(message)
            reached_total = counter is not None and counter[0] >= counter[1]
            if reached_total or (now - last) >= self._interval_s:
                self._pending.pop(key, None)
                self._last_logged_at[key] = now
                return [message]

            self._pending[key] = message
            return []

    def flush(self) -> list[str]:
        """Release every held message.

        Called on a job's terminal paths so the last thing each stage said
        is on the record even when it never reached its total — which is
        most useful precisely when a run died partway through one.

        Returns:
            list[str]: The pending messages, ordered by stage key so the
            output is deterministic, or an empty list.
        """
        with self._lock:
            pending = [self._pending[key] for key in sorted(self._pending)]
            self._pending.clear()
            return pending
