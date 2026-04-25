"""SQLite-backed manifest of in-flight, completed, and failed file ingestions.

Phase 3 of the streaming ingestion generalisation introduced this module
to give operators visibility into partial ingestion state after a SIGINT,
OOM, or container kill.

**Resume correctness does not depend on this manifest.** The ingestion
pipeline produces deterministic ``uuid.uuid5(NAMESPACE_URL, doc_id +
chunk_id)`` point IDs (see :mod:`docint.core.readers.documents.reader`)
and writes through ``allow_update=True`` upserts, so re-ingesting a
partially-ingested file is a content-identity overwrite. The manifest's
job is:

1. Skip files that completed in a prior run via a fast SQLite lookup,
   complementing the existing Qdrant ``file_hash`` scroll
   (:meth:`docint.core.rag.RAG._get_existing_file_hashes`).
2. Surface "what was in flight when we crashed" so operators can
   triage and so future Phase 6 per-file isolation can record
   structured failure reasons.

A no-op stub is returned when ``INGEST_MANIFEST_ENABLED=false`` so the
behaviour can be reverted without code changes.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

from docint.utils.retry import retry_with_backoff

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_manifest (
    collection      TEXT NOT NULL,
    file_hash       TEXT NOT NULL,
    status          TEXT NOT NULL,
    started_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    error_message   TEXT,
    PRIMARY KEY (collection, file_hash)
);
CREATE INDEX IF NOT EXISTS idx_manifest_status
    ON ingest_manifest(collection, status);
"""

STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

_ERROR_MESSAGE_MAX_LENGTH = 512


def _is_locked_manifest_error(exc: BaseException) -> bool:
    """Predicate matching transient SQLite locked-DB errors for the manifest.

    Mirrors :func:`docint.core.storage.sqlite_kvstore._is_retryable_sqlite_error`
    but lives separately so future evolution of the manifest's transient
    classifier (e.g. adding I/O errors when the database is on a flaky
    network mount) does not destabilise the docstore predicate.

    Args:
        exc: The exception raised by a manifest SQLite operation.

    Returns:
        True for transient ``database is locked`` / ``database is busy``
        errors raised as :class:`sqlite3.OperationalError`; False for
        every other error.
    """
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    msg = str(exc).lower()
    return "database is locked" in msg or "database is busy" in msg


def _truncate_error(message: str) -> str:
    """Truncate operator-visible error messages to a sane upper bound.

    Defends against unbounded log payloads or accidental data leakage
    from third-party exceptions that embed stack traces or sample
    document content.

    Args:
        message: Raw error message string.

    Returns:
        A trimmed message of at most :data:`_ERROR_MESSAGE_MAX_LENGTH`
        characters, with a trailing ``…`` marker when truncation
        occurred.
    """
    if len(message) <= _ERROR_MESSAGE_MAX_LENGTH:
        return message
    return message[: _ERROR_MESSAGE_MAX_LENGTH - 1] + "…"


class IngestManifest:
    """SQLite manifest of file-level ingestion state per collection.

    Concurrency model:
        Holds a single long-lived :class:`sqlite3.Connection` shared
        across threads (``check_same_thread=False``) and serialises all
        access through a :class:`threading.Lock` to prevent cursor
        races. Transient ``database is locked`` errors from
        cross-process contention are retried via
        :func:`docint.utils.retry.retry_with_backoff` with the same
        exponential schedule the SQLite docstore uses.

    Args:
        db_path: Filesystem path for the manifest database. Created
            automatically if absent.
        max_retries: Maximum retries for locked-DB errors. ``0`` disables.
        retry_backoff_seconds: Initial retry backoff in seconds.
        retry_backoff_max_seconds: Maximum retry backoff in seconds.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.25,
        retry_backoff_max_seconds: float = 2.0,
    ) -> None:
        self.db_path = str(db_path)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.retry_backoff_max_seconds = max(0.0, float(retry_backoff_max_seconds))

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        logger.debug("IngestManifest initialised at {}", self.db_path)

    @property
    def enabled(self) -> bool:
        """Whether the manifest performs real writes (always True for this class)."""
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute(self, operation: str, fn: Any) -> Any:
        """Run *fn* under the instance lock with locked-DB retries.

        Args:
            operation: Human-readable operation name for log messages.
            fn: Callable performing the underlying SQLite operation.

        Returns:
            Whatever *fn* returns.

        Raises:
            sqlite3.OperationalError: If retries are exhausted or the
                error is not a transient lock/busy condition.
        """
        return retry_with_backoff(
            f"ingest_manifest.{operation}",
            fn,
            max_retries=self.max_retries,
            initial_backoff=self.retry_backoff_seconds,
            max_backoff=self.retry_backoff_max_seconds,
            is_retryable=_is_locked_manifest_error,
            lock=self._lock,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mark_started(self, collection: str, file_hash: str) -> None:
        """Record that ingestion of *file_hash* began on *collection*.

        Idempotent: re-marking an already-started file refreshes
        ``updated_at`` but preserves the original ``started_at``. Files
        already in ``completed`` status are not regressed back to
        ``in_progress``.

        Args:
            collection: Logical Qdrant collection name.
            file_hash: SHA-256 (or equivalent) of the source file.
        """
        if not collection or not file_hash:
            return
        now = time.time()

        def _do() -> None:
            self._conn.execute(
                """
                INSERT INTO ingest_manifest
                    (collection, file_hash, status, started_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(collection, file_hash) DO UPDATE SET
                    status = CASE
                        WHEN ingest_manifest.status = 'completed' THEN 'completed'
                        ELSE excluded.status
                    END,
                    updated_at = excluded.updated_at,
                    error_message = NULL
                """,
                (collection, file_hash, STATUS_IN_PROGRESS, now, now),
            )
            self._conn.commit()

        self._execute("mark_started", _do)

    def mark_completed(self, collection: str, file_hash: str) -> None:
        """Mark *file_hash* as fully ingested for *collection*.

        Args:
            collection: Logical Qdrant collection name.
            file_hash: SHA-256 of the source file.
        """
        if not collection or not file_hash:
            return
        now = time.time()

        def _do() -> None:
            self._conn.execute(
                """
                INSERT INTO ingest_manifest
                    (collection, file_hash, status, started_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(collection, file_hash) DO UPDATE SET
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    error_message = NULL
                """,
                (collection, file_hash, STATUS_COMPLETED, now, now),
            )
            self._conn.commit()

        self._execute("mark_completed", _do)

    def mark_failed(
        self,
        collection: str,
        file_hash: str,
        error_message: str | None,
    ) -> None:
        """Mark *file_hash* as failed for *collection* with a truncated reason.

        Args:
            collection: Logical Qdrant collection name.
            file_hash: SHA-256 of the source file.
            error_message: Operator-visible failure description; truncated
                to :data:`_ERROR_MESSAGE_MAX_LENGTH` characters before
                persistence to bound row size and avoid leaking unbounded
                content from third-party exceptions.
        """
        if not collection or not file_hash:
            return
        now = time.time()
        truncated = _truncate_error(error_message) if error_message else None

        def _do() -> None:
            self._conn.execute(
                """
                INSERT INTO ingest_manifest
                    (collection, file_hash, status, started_at,
                     updated_at, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(collection, file_hash) DO UPDATE SET
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    error_message = excluded.error_message
                """,
                (collection, file_hash, STATUS_FAILED, now, now, truncated),
            )
            self._conn.commit()

        self._execute("mark_failed", _do)

    def pending_files(self, collection: str) -> dict[str, float]:
        """Return ``{file_hash: started_at}`` for files in flight.

        Args:
            collection: Logical Qdrant collection name.

        Returns:
            Dict mapping each in-progress file hash to its ``started_at``
            wall-clock timestamp. Empty when nothing is in flight.
        """
        if not collection:
            return {}

        def _do() -> dict[str, float]:
            rows = self._conn.execute(
                """
                SELECT file_hash, started_at
                FROM ingest_manifest
                WHERE collection = ? AND status = ?
                """,
                (collection, STATUS_IN_PROGRESS),
            ).fetchall()
            return {file_hash: started_at for file_hash, started_at in rows}

        return self._execute("pending_files", _do)

    def completed_files(self, collection: str) -> set[str]:
        """Return the set of file hashes already completed in *collection*.

        Args:
            collection: Logical Qdrant collection name.

        Returns:
            Set of completed file hashes. Empty when nothing is done.
        """
        if not collection:
            return set()

        def _do() -> set[str]:
            rows = self._conn.execute(
                """
                SELECT file_hash
                FROM ingest_manifest
                WHERE collection = ? AND status = ?
                """,
                (collection, STATUS_COMPLETED),
            ).fetchall()
            return {row[0] for row in rows}

        return self._execute("completed_files", _do)

    def failed_files(self, collection: str) -> dict[str, str | None]:
        """Return ``{file_hash: error_message}`` for files marked failed.

        Args:
            collection: Logical Qdrant collection name.

        Returns:
            Dict mapping each failed file hash to its truncated error
            message (or ``None`` when no message was recorded).
        """
        if not collection:
            return {}

        def _do() -> dict[str, str | None]:
            rows = self._conn.execute(
                """
                SELECT file_hash, error_message
                FROM ingest_manifest
                WHERE collection = ? AND status = ?
                """,
                (collection, STATUS_FAILED),
            ).fetchall()
            return {file_hash: error for file_hash, error in rows}

        return self._execute("failed_files", _do)

    def close(self) -> None:
        """Close the underlying SQLite connection.

        Idempotent — safe to call multiple times.
        """
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.exception("IngestManifest close failed at {}", self.db_path)


class NullIngestManifest:
    """No-op stub returned when ``INGEST_MANIFEST_ENABLED=false``.

    Mirrors the public surface of :class:`IngestManifest` so call sites
    can use the manifest unconditionally without a None-check. All write
    methods are silent; all read methods return empty results.
    """

    @property
    def enabled(self) -> bool:
        """Always ``False`` — the stub performs no I/O."""
        return False

    def mark_started(self, collection: str, file_hash: str) -> None:
        """No-op."""
        _ = (collection, file_hash)

    def mark_completed(self, collection: str, file_hash: str) -> None:
        """No-op."""
        _ = (collection, file_hash)

    def mark_failed(
        self,
        collection: str,
        file_hash: str,
        error_message: str | None,
    ) -> None:
        """No-op."""
        _ = (collection, file_hash, error_message)

    def pending_files(self, collection: str) -> dict[str, float]:
        """Return an empty dict."""
        _ = collection
        return {}

    def completed_files(self, collection: str) -> set[str]:
        """Return an empty set."""
        _ = collection
        return set()

    def failed_files(self, collection: str) -> dict[str, str | None]:
        """Return an empty dict."""
        _ = collection
        return {}

    def close(self) -> None:
        """No-op."""
