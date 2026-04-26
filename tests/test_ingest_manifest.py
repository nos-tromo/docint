"""Tests for the SQLite-backed ingestion manifest.

Pins the resume/visibility contract introduced by Phase 3 of the streaming
ingestion generalisation: which file hashes are in flight, completed, or
failed for a given collection.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from docint.core.storage.ingest_manifest import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_IN_PROGRESS,
    IngestManifest,
    NullIngestManifest,
    _truncate_error,
)


def test_mark_started_inserts_row(tmp_path: Path) -> None:
    """``mark_started`` should record an in-progress row for the given file."""
    manifest = IngestManifest(tmp_path / "m.db")
    try:
        manifest.mark_started("docs", "hash-1")
        pending = manifest.pending_files("docs")
        assert set(pending.keys()) == {"hash-1"}
        assert pending["hash-1"] > 0.0
        # status is in_progress
        with sqlite3.connect(tmp_path / "m.db") as conn:
            row = conn.execute(
                "SELECT status FROM ingest_manifest WHERE collection=? AND file_hash=?",
                ("docs", "hash-1"),
            ).fetchone()
        assert row == (STATUS_IN_PROGRESS,)
    finally:
        manifest.close()


def test_mark_completed_replaces_in_progress(tmp_path: Path) -> None:
    """``mark_completed`` should flip the row to completed without re-inserting."""
    manifest = IngestManifest(tmp_path / "m.db")
    try:
        manifest.mark_started("docs", "hash-1")
        manifest.mark_completed("docs", "hash-1")

        assert manifest.pending_files("docs") == {}
        assert manifest.completed_files("docs") == {"hash-1"}
    finally:
        manifest.close()


def test_mark_started_does_not_regress_completed(tmp_path: Path) -> None:
    """A late ``mark_started`` after ``mark_completed`` should be a no-op."""
    manifest = IngestManifest(tmp_path / "m.db")
    try:
        manifest.mark_started("docs", "hash-1")
        manifest.mark_completed("docs", "hash-1")
        manifest.mark_started("docs", "hash-1")  # late start

        assert manifest.completed_files("docs") == {"hash-1"}
        assert manifest.pending_files("docs") == {}
    finally:
        manifest.close()


def test_mark_failed_records_error_message(tmp_path: Path) -> None:
    """``mark_failed`` should persist the file hash with a truncated error message."""
    manifest = IngestManifest(tmp_path / "m.db")
    try:
        manifest.mark_failed("docs", "hash-1", "boom: connection refused")

        failed = manifest.failed_files("docs")
        assert failed == {"hash-1": "boom: connection refused"}
        # also ensure it's not in pending or completed
        assert manifest.pending_files("docs") == {}
        assert manifest.completed_files("docs") == set()
    finally:
        manifest.close()


def test_mark_failed_truncates_long_error_message(tmp_path: Path) -> None:
    """Long error messages should be truncated to bound row size."""
    manifest = IngestManifest(tmp_path / "m.db")
    long_msg = "A" * 2000
    try:
        manifest.mark_failed("docs", "hash-1", long_msg)
        stored = manifest.failed_files("docs")["hash-1"]
        assert stored is not None
        assert len(stored) <= 512
        assert stored.endswith("…")
    finally:
        manifest.close()


def test_mark_failed_handles_none_error_message(tmp_path: Path) -> None:
    """A ``None`` error message should be persisted as ``NULL``."""
    manifest = IngestManifest(tmp_path / "m.db")
    try:
        manifest.mark_failed("docs", "hash-1", None)
        assert manifest.failed_files("docs") == {"hash-1": None}
    finally:
        manifest.close()


def test_pending_completed_failed_isolation_per_collection(tmp_path: Path) -> None:
    """Manifest rows should be scoped per collection."""
    manifest = IngestManifest(tmp_path / "m.db")
    try:
        manifest.mark_started("alpha", "h1")
        manifest.mark_completed("alpha", "h2")
        manifest.mark_failed("alpha", "h3", "err")
        manifest.mark_started("beta", "h1")

        assert set(manifest.pending_files("alpha").keys()) == {"h1"}
        assert manifest.completed_files("alpha") == {"h2"}
        assert manifest.failed_files("alpha") == {"h3": "err"}

        assert set(manifest.pending_files("beta").keys()) == {"h1"}
        assert manifest.completed_files("beta") == set()
        assert manifest.failed_files("beta") == {}
    finally:
        manifest.close()


def test_empty_collection_or_hash_is_a_no_op(tmp_path: Path) -> None:
    """Empty collection or file_hash should silently no-op rather than crash."""
    manifest = IngestManifest(tmp_path / "m.db")
    try:
        manifest.mark_started("", "h1")
        manifest.mark_started("docs", "")
        manifest.mark_completed("", "h1")
        manifest.mark_failed("docs", "", "err")

        # Nothing was persisted.
        with sqlite3.connect(tmp_path / "m.db") as conn:
            count = conn.execute("SELECT COUNT(*) FROM ingest_manifest").fetchone()[0]
        assert count == 0
    finally:
        manifest.close()


def test_pending_files_empty_collection_returns_empty(tmp_path: Path) -> None:
    """Reads against an empty collection name should return empty results."""
    manifest = IngestManifest(tmp_path / "m.db")
    try:
        assert manifest.pending_files("") == {}
        assert manifest.completed_files("") == set()
        assert manifest.failed_files("") == {}
    finally:
        manifest.close()


def test_close_is_idempotent(tmp_path: Path) -> None:
    """``close`` should be safe to call multiple times."""
    manifest = IngestManifest(tmp_path / "m.db")
    manifest.close()
    # Second close — must not raise.
    manifest.close()


class _ConnProxy:
    """Minimal connection wrapper that lets tests inject a flaky ``execute``."""

    def __init__(
        self,
        real: sqlite3.Connection,
        execute_fn: Any = None,
    ) -> None:
        self._real = real
        self._execute_fn = execute_fn

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        if self._execute_fn is not None:
            return self._execute_fn(sql, *args, **kwargs)
        return self._real.execute(sql, *args, **kwargs)

    def executemany(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        return self._real.executemany(sql, *args, **kwargs)

    def commit(self) -> None:
        self._real.commit()

    def close(self) -> None:
        self._real.close()


def test_retries_locked_db_via_shared_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest should retry transient locked-DB errors via the shared helper."""
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "docint.utils.retry.time.sleep",
        lambda delay: sleep_calls.append(delay),
    )

    manifest = IngestManifest(
        tmp_path / "m.db",
        max_retries=2,
        retry_backoff_seconds=0.1,
        retry_backoff_max_seconds=0.5,
    )
    try:
        real_conn = manifest._conn
        calls = {"n": 0}

        def flaky_execute(sql: str, *args: Any, **kwargs: Any) -> Any:
            if sql.lstrip().upper().startswith("INSERT") and calls["n"] == 0:
                calls["n"] += 1
                raise sqlite3.OperationalError("database is locked")
            return real_conn.execute(sql, *args, **kwargs)

        manifest._conn = _ConnProxy(real_conn, execute_fn=flaky_execute)  # type: ignore[assignment]
        manifest.mark_started("docs", "hash-1")
        manifest._conn = real_conn  # type: ignore[assignment]

        assert manifest.completed_files("docs") == set()
        assert "hash-1" in manifest.pending_files("docs")
        assert sleep_calls == [0.1]
    finally:
        manifest.close()


def test_truncate_error_short_string_unchanged() -> None:
    """Short messages should pass through ``_truncate_error`` unchanged."""
    assert _truncate_error("short") == "short"


def test_truncate_error_long_string_clipped_with_ellipsis() -> None:
    """Long messages should be clipped and end with the ellipsis marker."""
    long_msg = "x" * 1000
    truncated = _truncate_error(long_msg)
    assert len(truncated) == 512
    assert truncated.endswith("…")


# ---------------------------------------------------------------------------
# NullIngestManifest stub
# ---------------------------------------------------------------------------


def test_null_manifest_is_no_op() -> None:
    """The null stub should accept all writes and return empty reads."""
    null = NullIngestManifest()
    assert null.enabled is False
    null.mark_started("c", "h")
    null.mark_completed("c", "h")
    null.mark_failed("c", "h", "err")
    assert null.pending_files("c") == {}
    assert null.completed_files("c") == set()
    assert null.failed_files("c") == {}
    null.close()


def test_null_manifest_status_constants_exposed() -> None:
    """Status constants should be importable for callers that branch on them."""
    assert STATUS_IN_PROGRESS == "in_progress"
    assert STATUS_COMPLETED == "completed"
    assert STATUS_FAILED == "failed"
