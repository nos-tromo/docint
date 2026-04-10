"""SQLite-backed key-value store implementing the LlamaIndex BaseKVStore interface.

Replaces the QdrantKVStore workaround that stored dummy vectors just to use
Qdrant as a KV store.  Uses a single SQLite database with a ``kv_data`` table
keyed by ``(collection, key)`` and a JSON-encoded value column.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from llama_index.core.storage.kvstore.types import DEFAULT_COLLECTION, BaseKVStore
from loguru import logger

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv_data (
    collection TEXT    NOT NULL,
    key        TEXT    NOT NULL,
    val        TEXT    NOT NULL,
    PRIMARY KEY (collection, key)
);
"""


class SQLiteKVStore(BaseKVStore):
    """A key-value store backed by a local SQLite database.

    Args:
        db_path: Path to the SQLite database file.  Created automatically if it
            does not exist.
        batch_size: Number of rows per batch in ``put_all``.
    """

    def __init__(
        self,
        db_path: str | Path,
        batch_size: int = 100,
    ) -> None:
        """Initialize the SQLiteKVStore.

        Args:
            db_path: Filesystem path for the SQLite database.
            batch_size: Batch size for bulk operations.
        """
        self.db_path = str(db_path)
        self.batch_size = batch_size

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        logger.debug("SQLiteKVStore initialised at {}", self.db_path)

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
        self._conn.execute(
            "INSERT OR REPLACE INTO kv_data (collection, key, val) VALUES (?, ?, ?)",
            (collection, key, json.dumps(val)),
        )
        self._conn.commit()

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
        batch_size = batch_size or self.batch_size
        rows = [(collection, k, json.dumps(v)) for k, v in kv_pairs]
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            self._conn.executemany(
                "INSERT OR REPLACE INTO kv_data (collection, key, val) VALUES (?, ?, ?)",
                batch,
            )
        self._conn.commit()

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
        row = self._conn.execute(
            "SELECT val FROM kv_data WHERE collection = ? AND key = ?",
            (collection, key),
        ).fetchone()
        if row is None:
            return None
        val = json.loads(row[0])
        return val if isinstance(val, dict) else None

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> dict[str, Any]:
        """Retrieve all key-value pairs for *collection*.

        Args:
            collection: Logical collection name.

        Returns:
            Dictionary mapping keys to their stored values.
        """
        rows = self._conn.execute(
            "SELECT key, val FROM kv_data WHERE collection = ?",
            (collection,),
        ).fetchall()
        return {k: json.loads(v) for k, v in rows}

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Delete a key-value pair.

        Args:
            key: The key.
            collection: Logical collection name.

        Returns:
            ``True`` after the operation completes.
        """
        self._conn.execute(
            "DELETE FROM kv_data WHERE collection = ? AND key = ?",
            (collection, key),
        )
        self._conn.commit()
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
