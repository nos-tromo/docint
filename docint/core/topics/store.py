"""On-disk store for topic runs.

One directory per physical collection holding one subdirectory per run
(``centres.npy`` and ``run.json``), the ``active.json`` pointer and a ``.lock``
file. The run id and the collection name both arrive from a request, so the id
is pattern-checked before any path join and a collection only becomes a
directory as itself when it is a plain file name; any other is hashed.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
from loguru import logger

from docint.core.topics.model import TopicInfo, TopicTable

__all__ = ["RUN_ID_PATTERN", "TopicLock", "TopicRun", "TopicStore"]

RUN_ID_PATTERN = re.compile(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}")
#: A collection name that is its own directory; any other is hashed (see ``TopicStore._dir``).
_COLLECTION_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,128}")
_TORN_RETRY_S = 0.05


class _TornRun(ValueError):
    """A run's centres and record disagree, as they do mid-save."""


@dataclass(frozen=True)
class TopicRun:
    """One stored topic run.

    Attributes:
        run_id: The run id.
        record: ``run.json`` without its ``topics`` key.
        table: The topics and their centres.
    """

    run_id: str
    record: Mapping[str, Any]
    table: TopicTable


class TopicLock:
    """An exclusive per-collection lock; a context manager."""

    def __init__(self, descriptor: int) -> None:
        """Wrap a locked file descriptor.

        Args:
            descriptor (int): Descriptor holding the ``flock``.
        """
        self._descriptor: int | None = descriptor

    def release(self) -> None:
        """Unlock and close; calling it again does nothing."""
        descriptor, self._descriptor = self._descriptor, None
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> TopicLock:
        """Return the lock.

        Returns:
            TopicLock: This lock.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the lock."""
        self.release()


def _write_private(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically, readable by the process user only."""
    descriptor, staging = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(staging, path)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise


def _check_run_id(run_id: str) -> str:
    """Return ``run_id`` when it is a well-formed run id.

    Args:
        run_id (str): The candidate.

    Returns:
        str: The same id.

    Raises:
        ValueError: When it does not match :data:`RUN_ID_PATTERN`.
    """
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"Unsafe topic run id: {run_id!r}")
    return run_id


class TopicStore:
    """Read and write topic runs under one root directory."""

    def __init__(self, root: Path) -> None:
        """Create the store.

        Args:
            root (Path): Directory holding one subdirectory per collection.
        """
        self._root = Path(root)
        self._root_prefix = os.path.join(os.path.realpath(self._root), "")

    def _contained(self, candidate: Path) -> Path:
        """Return a path once it is proven to resolve under the store root.

        Args:
            candidate (Path): The path about to be read or written.

        Returns:
            Path: The normalized path.

        Raises:
            ValueError: When the path resolves outside the root.
        """
        resolved = os.path.realpath(candidate)
        if not resolved.startswith(self._root_prefix):
            raise ValueError(f"Topic path escapes the store root: {candidate!r}")
        return Path(resolved)

    def _dir(self, physical: str) -> Path:
        """Return a collection's directory, whatever the collection is called.

        Args:
            physical (str): Owner-namespaced Qdrant collection name.

        Returns:
            Path: The directory (not created).

        Raises:
            ValueError: When the directory resolves outside the root.
        """
        plain = physical not in {".", ".."} and _COLLECTION_PATTERN.fullmatch(physical)
        name = physical if plain else "~" + hashlib.sha256(physical.encode("utf-8")).hexdigest()
        return self._contained(self._root / name)

    def _run_dir(self, physical: str, run_id: str) -> Path:
        """Return a run's directory after checking the id.

        Args:
            physical (str): Collection name.
            run_id (str): The run id.

        Returns:
            Path: The directory (not created).
        """
        return self._contained(self._dir(physical) / _check_run_id(run_id))

    def new_run_id(self, now: datetime) -> str:
        """Mint a run id.

        Args:
            now (datetime): The build time.

        Returns:
            str: ``YYYYMMDD-HHMMSS-<6 hex>``.
        """
        return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"

    def try_lock(self, physical: str) -> TopicLock | None:
        """Take the collection's exclusive lock without waiting.

        Args:
            physical (str): Collection name.

        Returns:
            TopicLock | None: The lock, ``None`` when another holder has it.
        """
        directory = self._dir(physical)
        directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(directory / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            return None
        except BaseException:
            os.close(descriptor)
            raise
        return TopicLock(descriptor)

    def save(self, physical: str, run: TopicRun) -> None:
        """Write a run: the centres first, ``run.json`` last.

        A run may be re-saved under its id with updated metadata, or with rows
        appended to its centres (topics discovered later); existing rows never
        change. A lock-free reader can meet the new centres beside the old
        ``run.json`` between the two writes, which :meth:`load` reads again.

        Args:
            physical (str): Collection name.
            run (TopicRun): The run.
        """
        directory = self._run_dir(physical, run.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        buffer = _npy_bytes(np.asarray(run.table.centres, dtype=np.float32))
        _write_private(directory / "centres.npy", buffer)
        record = {**run.record, "run_id": run.run_id, "topics": [t.to_json() for t in run.table.topics]}
        _write_private(directory / "run.json", json.dumps(record).encode("utf-8"))

    def load(self, physical: str, run_id: str) -> TopicRun | None:
        """Read a run back.

        Args:
            physical (str): Collection name.
            run_id (str): The run id.

        A shape mismatch is read once more after a short pause, since a save in
        progress produces one; only a second mismatch counts as unreadable.

        Returns:
            TopicRun | None: The run; ``None`` when a file is missing, corrupt
                or the centres do not match the record.

        Raises:
            ValueError: When the run id is unsafe.
        """
        directory = self._run_dir(physical, run_id)
        for attempt in range(2):
            try:
                return self._read(directory, run_id)
            except FileNotFoundError:
                return None
            except _TornRun:
                if attempt == 0:
                    time.sleep(_TORN_RETRY_S)
                    continue
                logger.warning("Topic run '{}' centres do not match its record", run_id)
                return None
            except Exception as exc:
                logger.warning("Unreadable topic run '{}' ({})", run_id, type(exc).__name__)
                return None
        return None

    def _read(self, directory: Path, run_id: str) -> TopicRun:
        """Read a run's two files once.

        Args:
            directory (Path): The run's directory.
            run_id (str): The run id.

        Returns:
            TopicRun: The run.

        Raises:
            _TornRun: When the centres' shape disagrees with the record.
        """
        record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        topics = tuple(TopicInfo.from_json(t) for t in record.pop("topics"))
        centres = np.load(directory / "centres.npy", allow_pickle=False)
        if centres.shape != (len(topics), int(record["dense"]["dim"])):
            raise _TornRun("centres shape does not match the record")
        return TopicRun(
            run_id=run_id, record=record, table=TopicTable(topics=topics, centres=centres.astype(np.float64))
        )

    def activate(self, physical: str, run_id: str) -> None:
        """Point the collection at a run.

        Args:
            physical (str): Collection name.
            run_id (str): The run id.
        """
        _check_run_id(run_id)
        directory = self._dir(physical)
        directory.mkdir(parents=True, exist_ok=True)
        _write_private(directory / "active.json", json.dumps({"run_id": run_id}).encode("utf-8"))

    def active_id(self, physical: str) -> str | None:
        """Return the active run id.

        Args:
            physical (str): Collection name.

        Returns:
            str | None: The id, ``None`` when none is set or the pointer is unusable.
        """
        try:
            stored = json.loads((self._dir(physical) / "active.json").read_text(encoding="utf-8"))
            run_id = stored["run_id"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
            return None
        return run_id

    def active(self, physical: str, *, owner_created_at: str | None) -> TopicRun | None:
        """Return the active run unless it belongs to a recreated collection.

        Args:
            physical (str): Collection name.
            owner_created_at (str | None): The ownership row's creation time now.

        Returns:
            TopicRun | None: The run; ``None`` when absent, unreadable or stale.
                Nothing is deleted.
        """
        run_id = self.active_id(physical)
        if run_id is None:
            return None
        run = self.load(physical, run_id)
        if run is None or run.record.get("owner_created_at") != owner_created_at:
            return None
        return run

    def discard_stale(self, physical: str, *, owner_created_at: str | None) -> bool:
        """Delete the active run when it belongs to a recreated collection.

        Takes no lock; call it under the collection lock (:meth:`try_lock`).

        Args:
            physical (str): Collection name.
            owner_created_at (str | None): The ownership row's creation time now.

        Returns:
            bool: Whether a stale run was deleted.
        """
        run_id = self.active_id(physical)
        if run_id is None:
            return False
        run = self.load(physical, run_id)
        if run is not None and run.record.get("owner_created_at") == owner_created_at:
            return False
        self.delete_run(physical, run_id)
        (self._dir(physical) / "active.json").unlink(missing_ok=True)
        return True

    def delete_run(self, physical: str, run_id: str) -> None:
        """Remove one run; a missing one is ignored.

        Args:
            physical (str): Collection name.
            run_id (str): The run id.
        """
        shutil.rmtree(self._run_dir(physical, run_id), ignore_errors=True)

    def prune(self, physical: str, *, keep: str) -> None:
        """Remove every run but one.

        Does not read ``active.json``: if ``keep`` is not the active run, the
        active run's directory is deleted and the pointer is left dangling.
        Call it under the collection lock, passing the run just activated.

        Args:
            physical (str): Collection name.
            keep (str): The run id to keep.
        """
        _check_run_id(keep)
        directory = self._dir(physical)
        if not directory.is_dir():
            return
        for child in directory.iterdir():
            if child.is_dir() and child.name != keep and RUN_ID_PATTERN.fullmatch(child.name):
                shutil.rmtree(child, ignore_errors=True)

    def delete_collection(self, physical: str) -> None:
        """Remove every run of a collection; a missing one is ignored.

        This is the collection-deletion cascade. It takes no lock and removes
        ``.lock`` with the rest. A build still running against the deleted
        collection is refused by its ``owner_created_at`` checks before it
        activates, and a discover before it completes.

        Args:
            physical (str): Collection name.
        """
        shutil.rmtree(self._dir(physical), ignore_errors=True)


def _npy_bytes(array: np.ndarray) -> bytes:
    """Serialise an array as ``.npy`` without pickle.

    Args:
        array (np.ndarray): The array.

    Returns:
        bytes: The file contents.
    """
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()
