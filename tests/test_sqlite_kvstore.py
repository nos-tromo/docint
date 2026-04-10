"""Tests for SQLiteKVStore."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pytest

from docint.core.storage.sqlite_kvstore import SQLiteKVStore


@pytest.fixture
def store(tmp_path: Path) -> SQLiteKVStore:
    """Return a fresh SQLiteKVStore backed by a temporary database file.

    Args:
        tmp_path: Pytest-provided temporary directory unique to the test
            invocation.

    Returns:
        A new ``SQLiteKVStore`` instance pointing at a fresh database.
    """
    return SQLiteKVStore(db_path=tmp_path / "test_kv.db")


def test_put_and_get(store: SQLiteKVStore) -> None:
    """Round-trip a single key-value pair through put and get.

    Verifies that a value stored with ``put`` is returned unchanged by
    ``get`` for the same key.

    Args:
        store: A fresh ``SQLiteKVStore`` fixture instance.
    """
    store.put("k1", {"a": 1})
    assert store.get("k1") == {"a": 1}


def test_get_missing_key(store: SQLiteKVStore) -> None:
    """Return ``None`` when getting a key that was never stored.

    Args:
        store: A fresh ``SQLiteKVStore`` fixture instance.
    """
    assert store.get("nonexistent") is None


def test_put_overwrites(store: SQLiteKVStore) -> None:
    """Verify that a second put for the same key replaces the previous value.

    Args:
        store: A fresh ``SQLiteKVStore`` fixture instance.
    """
    store.put("k1", {"v": 1})
    store.put("k1", {"v": 2})
    assert store.get("k1") == {"v": 2}


def test_put_all_and_get_all(store: SQLiteKVStore) -> None:
    """Bulk-insert multiple pairs and retrieve all of them at once.

    Verifies that ``put_all`` stores every provided pair and that
    ``get_all`` returns a dict containing exactly those pairs.

    Args:
        store: A fresh ``SQLiteKVStore`` fixture instance.
    """
    pairs = [("a", {"v": 1}), ("b", {"v": 2}), ("c", {"v": 3})]
    store.put_all(pairs)
    result = store.get_all()
    assert result == {"a": {"v": 1}, "b": {"v": 2}, "c": {"v": 3}}


def test_delete(store: SQLiteKVStore) -> None:
    """Verify that ``delete`` removes the key and returns ``True``.

    After deletion ``get`` must return ``None`` for the same key.

    Args:
        store: A fresh ``SQLiteKVStore`` fixture instance.
    """
    store.put("k1", {"v": 1})
    assert store.delete("k1") is True
    assert store.get("k1") is None


def test_collection_isolation(store: SQLiteKVStore) -> None:
    """Verify that separate collection namespaces do not share data.

    The same key stored in two different collections must resolve to the
    value that was written to that specific collection.

    Args:
        store: A fresh ``SQLiteKVStore`` fixture instance.
    """
    store.put("k1", {"v": "coll_a"}, collection="a")
    store.put("k1", {"v": "coll_b"}, collection="b")
    assert store.get("k1", collection="a") == {"v": "coll_a"}
    assert store.get("k1", collection="b") == {"v": "coll_b"}
    assert store.get_all(collection="a") == {"k1": {"v": "coll_a"}}


def test_get_all_empty(store: SQLiteKVStore) -> None:
    """Return an empty dict when the store contains no entries.

    Args:
        store: A fresh ``SQLiteKVStore`` fixture instance.
    """
    assert store.get_all() == {}


def test_put_all_batching(store: SQLiteKVStore) -> None:
    """Verify ``put_all`` with a small batch size stores all entries correctly.

    Uses a batch size smaller than the number of pairs to exercise the
    batching logic and confirm no entries are dropped or corrupted.

    Args:
        store: A fresh ``SQLiteKVStore`` fixture instance.
    """
    pairs = [(f"k{i}", {"v": i}) for i in range(10)]
    store.put_all(pairs, batch_size=3)
    result = store.get_all()
    assert len(result) == 10
    assert result["k9"] == {"v": 9}


def test_concurrent_put_and_get_are_thread_safe(store: SQLiteKVStore) -> None:
    """Serialise concurrent put/get calls through the instance lock.

    Without the instance lock, sharing a single ``sqlite3.Connection`` across
    threads raises cursor/recursive-use errors under contention.  This test
    fires many concurrent ``put`` and ``get`` calls from a thread pool and
    asserts that every write is eventually observable via ``get`` with no
    exceptions.

    Args:
        store: A fresh ``SQLiteKVStore`` fixture instance.
    """
    keys = [f"k{i}" for i in range(200)]

    def writer(i: int) -> None:
        store.put(keys[i], {"v": i})

    def reader(i: int) -> None:
        # Reads may legitimately return ``None`` if the writer has not yet
        # committed — we only assert that no exception is raised.
        store.get(keys[i])

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = []
        for i in range(len(keys)):
            futures.append(pool.submit(writer, i))
            futures.append(pool.submit(reader, i))
        for fut in as_completed(futures):
            fut.result()  # re-raise any worker exception

    final = store.get_all()
    assert final == {keys[i]: {"v": i} for i in range(len(keys))}


class _ConnProxy:
    """Forwarding proxy around a real ``sqlite3.Connection`` for tests.

    Exposes ``execute``, ``executemany`` and ``commit`` so the test can
    override the underlying behaviour (which cannot be done directly on
    :class:`sqlite3.Connection` because its methods are read-only C
    attributes).  Every other attribute is looked up on the wrapped
    connection.

    Args:
        real: The real SQLite connection to wrap.
        execute_fn: Override for ``execute``.  ``None`` delegates to ``real``.
        executemany_fn: Override for ``executemany``.  ``None`` delegates.
    """

    def __init__(
        self,
        real: sqlite3.Connection,
        execute_fn: Callable[..., Any] | None = None,
        executemany_fn: Callable[..., Any] | None = None,
    ) -> None:
        """Initialise the connection proxy.

        Args:
            real (sqlite3.Connection): The real SQLite connection to wrap.
            execute_fn (Callable[..., Any] | None, optional): Override for ``execute``. Defaults to None.
            executemany_fn (Callable[..., Any] | None, optional): Override for ``executemany``. Defaults to None.
        """
        self._real = real
        self._execute_fn = execute_fn
        self._executemany_fn = executemany_fn

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the override or the real connection's ``execute``.

        Args:
            sql: The SQL statement.
            *args: Positional arguments forwarded downstream.
            **kwargs: Keyword arguments forwarded downstream.

        Returns:
            The cursor returned by the override or the real connection.
        """
        if self._execute_fn is not None:
            return self._execute_fn(sql, *args, **kwargs)
        return self._real.execute(sql, *args, **kwargs)

    def executemany(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the override or the real connection's ``executemany``.

        Args:
            sql: The SQL statement.
            *args: Positional arguments forwarded downstream.
            **kwargs: Keyword arguments forwarded downstream.

        Returns:
            The cursor returned by the override or the real connection.
        """
        if self._executemany_fn is not None:
            return self._executemany_fn(sql, *args, **kwargs)
        return self._real.executemany(sql, *args, **kwargs)

    def commit(self) -> None:
        """Forward ``commit`` to the wrapped connection."""
        self._real.commit()


def test_put_retries_locked_db_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry transient ``database is locked`` errors during ``put``.

    Replaces ``store._conn`` with a proxy whose ``execute`` raises a
    locked-DB error once and then delegates, and asserts that ``put``
    succeeds with exactly one retry backoff recorded.

    Args:
        tmp_path: Pytest-provided temporary directory for the database.
        monkeypatch: Pytest monkeypatch fixture used to stub ``time.sleep``.
    """
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "docint.core.storage.sqlite_kvstore.time.sleep",
        lambda delay: sleep_calls.append(delay),
    )

    store = SQLiteKVStore(
        db_path=tmp_path / "retry.db",
        max_retries=2,
        retry_backoff_seconds=0.1,
        retry_backoff_max_seconds=0.5,
    )

    real_conn = store._conn
    calls = {"n": 0}

    def flaky_execute(sql: str, *args: Any, **kwargs: Any) -> Any:
        """Raise a locked-DB error once on writes, then delegate.

        Args:
            sql: The SQL statement.
            *args: Positional arguments forwarded to the real connection.
            **kwargs: Keyword arguments forwarded to the real connection.

        Returns:
            The cursor returned by the real connection's ``execute``.
        """
        if sql.startswith("INSERT") and calls["n"] == 0:
            calls["n"] += 1
            raise sqlite3.OperationalError("database is locked")
        return real_conn.execute(sql, *args, **kwargs)

    store._conn = _ConnProxy(real_conn, execute_fn=flaky_execute)  # type: ignore[assignment]

    store.put("k1", {"v": 1})

    store._conn = real_conn
    assert store.get("k1") == {"v": 1}
    assert sleep_calls == [0.1]


def test_put_raises_non_retryable_sqlite_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Propagate non-locked ``OperationalError`` without retrying.

    Args:
        tmp_path: Pytest-provided temporary directory for the database.
        monkeypatch: Pytest monkeypatch fixture.
    """
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "docint.core.storage.sqlite_kvstore.time.sleep",
        lambda delay: sleep_calls.append(delay),
    )

    store = SQLiteKVStore(
        db_path=tmp_path / "noretry.db",
        max_retries=3,
        retry_backoff_seconds=0.1,
    )

    def broken_execute(*_args: Any, **_kwargs: Any) -> Any:
        """Raise a non-retryable ``OperationalError``.

        Args:
            *_args: Ignored.
            **_kwargs: Ignored.

        Returns:
            Never returns — always raises an exception.

        Raises:
            sqlite3.OperationalError: Always raised to simulate a broken database.
        """
        raise sqlite3.OperationalError("no such table: kv_data")

    store._conn = _ConnProxy(store._conn, execute_fn=broken_execute)  # type: ignore[assignment]

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        store.put("k1", {"v": 1})

    assert sleep_calls == []
