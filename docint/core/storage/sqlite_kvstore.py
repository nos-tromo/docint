"""SQLite-backed key-value store implementing the LlamaIndex BaseKVStore interface.

Uses a single SQLite database with a ``kv_data`` table keyed by
``(collection, key)`` and a JSON-encoded value column.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from llama_index.core.storage.kvstore.types import DEFAULT_COLLECTION, BaseKVStore
from loguru import logger

T = TypeVar("T")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv_data (
    collection TEXT    NOT NULL,
    key        TEXT    NOT NULL,
    val        TEXT    NOT NULL,
    PRIMARY KEY (collection, key)
);
"""


def _is_locked_db_error(exc: sqlite3.OperationalError) -> bool:
    """Return whether *exc* is a transient ``database is locked`` error.

    SQLite raises :class:`sqlite3.OperationalError` with a message of either
    ``"database is locked"`` or ``"database is busy"`` when another
    connection (possibly in a different process) is holding a write lock.
    These errors are retryable.

    Args:
        exc: The exception raised by SQLite.

    Returns:
        ``True`` if the error is a transient lock/busy condition.
    """
    msg = str(exc).lower()
    return "database is locked" in msg or "database is busy" in msg


class SQLiteKVStore(BaseKVStore):
    """A key-value store backed by a local SQLite database.

    Concurrency model:
        The store holds a single long-lived :class:`sqlite3.Connection` shared
        across threads (``check_same_thread=False``) and serialises all access
        through a :class:`threading.Lock` to prevent cursor races.  Transient
        ``database is locked`` errors from cross-process contention are
        retried with exponential backoff.

    Args:
        db_path: Path to the SQLite database file.  Created automatically if it
            does not exist.
        batch_size: Number of rows per batch in ``put_all``.
        max_retries: Maximum retries for transient ``database is locked``
            errors.  Set to ``0`` to disable retries.
        retry_backoff_seconds: Initial backoff in seconds between retries.
        retry_backoff_max_seconds: Maximum backoff in seconds between retries.
    """

    def __init__(
        self,
        db_path: str | Path,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.25,
        retry_backoff_max_seconds: float = 2.0,
    ) -> None:
        """Initialize the SQLiteKVStore.

        Args:
            db_path: Filesystem path for the SQLite database.
            batch_size: Batch size for bulk operations.
            max_retries: Maximum retries for locked-DB errors.
            retry_backoff_seconds: Initial retry backoff in seconds.
            retry_backoff_max_seconds: Maximum retry backoff in seconds.
        """
        self.db_path = str(db_path)
        self.batch_size = batch_size
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.retry_backoff_max_seconds = max(0.0, retry_backoff_max_seconds)

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        logger.debug("SQLiteKVStore initialised at {}", self.db_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_locked(self, operation: str, fn: Callable[[], T]) -> T:
        """Run *fn* under the instance lock with retries on locked-DB errors.

        The instance lock serialises in-process access to the shared
        connection.  A small exponential backoff retries transient
        ``database is locked`` errors caused by cross-process contention
        (e.g. another process holding the WAL write lock).

        Args:
            operation: Human-readable operation name used for log messages.
            fn: Callable that performs the underlying SQLite operation.

        Returns:
            The value returned by *fn*.

        Raises:
            sqlite3.OperationalError: If retries are exhausted or the error
                is not a transient lock/busy condition.
        """
        attempts = max(1, self.max_retries + 1)
        attempt = 1
        with self._lock:
            while True:
                try:
                    return fn()
                except sqlite3.OperationalError as exc:
                    if not _is_locked_db_error(exc) or attempt >= attempts:
                        raise
                    delay = min(
                        self.retry_backoff_seconds * (2 ** (attempt - 1)),
                        self.retry_backoff_max_seconds,
                    )
                    logger.warning(
                        "SQLite KV operation '{}' hit locked DB on attempt "
                        "{}/{}: {}. Retrying in {:.2f}s",
                        operation,
                        attempt,
                        attempts,
                        exc,
                        delay,
                    )
                    if delay > 0:
                        time.sleep(delay)
                    attempt += 1

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def put(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """Store a key-value pair.

        Args:
            key: The key.
            val: The value to store.
            collection: Logical collection name.
        """

        def _do_put() -> None:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv_data (collection, key, val) VALUES (?, ?, ?)",
                (collection, key, json.dumps(val)),
            )
            self._conn.commit()

        self._execute_locked("put", _do_put)

    def put_all(
        self,
        kv_pairs: list[tuple[str, dict[str, Any]]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int | None = None,
    ) -> None:
        """Store multiple key-value pairs.

        Args:
            kv_pairs: List of ``(key, value)`` tuples.
            collection: Logical collection name.
            batch_size: Batch size override.
        """
        effective_batch_size = batch_size or self.batch_size
        rows = [(collection, k, json.dumps(v)) for k, v in kv_pairs]

        def _do_put_all() -> None:
            for i in range(0, len(rows), effective_batch_size):
                batch = rows[i : i + effective_batch_size]
                self._conn.executemany(
                    "INSERT OR REPLACE INTO kv_data (collection, key, val) VALUES (?, ?, ?)",
                    batch,
                )
            self._conn.commit()

        self._execute_locked("put_all", _do_put_all)

    def get(
        self,
        key: str,
        collection: str = DEFAULT_COLLECTION,
    ) -> dict | None:
        """Retrieve a value by key.

        Args:
            key: The key.
            collection: Logical collection name.

        Returns:
            The stored dict, or ``None`` if the key does not exist.
        """

        def _do_get() -> dict | None:
            row = self._conn.execute(
                "SELECT val FROM kv_data WHERE collection = ? AND key = ?",
                (collection, key),
            ).fetchone()
            if row is None:
                return None
            val = json.loads(row[0])
            return val if isinstance(val, dict) else None

        return self._execute_locked("get", _do_get)

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> dict[str, Any]:
        """Retrieve all key-value pairs for *collection*.

        Args:
            collection: Logical collection name.

        Returns:
            Dictionary mapping keys to their stored values.
        """

        def _do_get_all() -> dict[str, Any]:
            rows = self._conn.execute(
                "SELECT key, val FROM kv_data WHERE collection = ?",
                (collection,),
            ).fetchall()
            return {k: json.loads(v) for k, v in rows}

        return self._execute_locked("get_all", _do_get_all)

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Delete a key-value pair.

        Args:
            key: The key.
            collection: Logical collection name.

        Returns:
            ``True`` after the operation completes.
        """

        def _do_delete() -> None:
            self._conn.execute(
                "DELETE FROM kv_data WHERE collection = ? AND key = ?",
                (collection, key),
            )
            self._conn.commit()

        self._execute_locked("delete", _do_delete)
        return True

    # ------------------------------------------------------------------
    # Async API — delegates to sync (SQLite is local, latency is trivial)
    # ------------------------------------------------------------------

    async def aput(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """Async wrapper for :meth:`put`.

        Args:
            key: The key.
            val: The value.
            collection: Logical collection name.
        """
        return self.put(key, val, collection)

    async def aput_all(
        self,
        kv_pairs: list[tuple[str, dict[str, Any]]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int | None = None,
    ) -> None:
        """Async wrapper for :meth:`put_all`.

        Args:
            kv_pairs: Key-value pairs.
            collection: Logical collection name.
            batch_size: Batch size override.
        """
        return self.put_all(kv_pairs, collection, batch_size)

    async def aget(
        self,
        key: str,
        collection: str = DEFAULT_COLLECTION,
    ) -> dict | None:
        """Async wrapper for :meth:`get`."""
        return self.get(key, collection)

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> dict[str, Any]:
        """Async wrapper for :meth:`get_all`."""
        return self.get_all(collection)

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Async wrapper for :meth:`delete`."""
        return self.delete(key, collection)
