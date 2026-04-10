"""Tests for SQLiteKVStore."""

from __future__ import annotations

from pathlib import Path

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
