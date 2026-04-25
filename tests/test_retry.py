"""Tests for the shared exponential-backoff retry helpers.

The retry engine in :mod:`docint.utils.retry` underpins both the SQLite KV
store (``database is locked`` retries from PR #116) and the Qdrant vector
inserts (added when generalising the streaming pattern across the
ingestion pipeline). These tests pin the schedule, the predicate-based
classification, and the async variant.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from typing import Any

import pytest

from docint.utils.retry import (
    aretry_with_backoff,
    is_transient_qdrant_error,
    retry_with_backoff,
)


def _always(_exc: BaseException) -> bool:
    """Predicate returning True for every exception (retry everything)."""
    return True


def _never(_exc: BaseException) -> bool:
    """Predicate returning False for every exception (retry nothing)."""
    return False


def test_retry_succeeds_on_first_attempt() -> None:
    """A successful call should not invoke sleep or trigger a retry."""
    sleeps: list[float] = []

    def fn() -> str:
        return "ok"

    result = retry_with_backoff(
        "noop",
        fn,
        max_retries=3,
        initial_backoff=0.1,
        max_backoff=1.0,
        is_retryable=_always,
        sleep=sleeps.append,
    )

    assert result == "ok"
    assert sleeps == []


def test_retry_succeeds_after_transient_failures() -> None:
    """Two transient failures followed by success returns the success value."""
    sleeps: list[float] = []
    attempts: list[int] = []

    def fn() -> str:
        attempts.append(len(attempts) + 1)
        if len(attempts) <= 2:
            raise ConnectionError("flaky")
        return "ok"

    result = retry_with_backoff(
        "flaky",
        fn,
        max_retries=3,
        initial_backoff=0.1,
        max_backoff=2.0,
        is_retryable=_always,
        sleep=sleeps.append,
    )

    assert result == "ok"
    assert attempts == [1, 2, 3]
    assert sleeps == [0.1, 0.2]


def test_retry_exhausts_and_reraises_last_exception() -> None:
    """When all attempts fail with retryable errors, the final exception escapes."""
    sleeps: list[float] = []

    def fn() -> str:
        raise ConnectionError("never recovers")

    with pytest.raises(ConnectionError, match="never recovers"):
        retry_with_backoff(
            "doomed",
            fn,
            max_retries=2,
            initial_backoff=0.1,
            max_backoff=1.0,
            is_retryable=_always,
            sleep=sleeps.append,
        )

    # 3 attempts total (initial + 2 retries) → 2 sleeps before final failure.
    assert sleeps == [0.1, 0.2]


def test_retry_non_retryable_raises_immediately() -> None:
    """Non-retryable errors should propagate without sleeping."""
    sleeps: list[float] = []

    def fn() -> str:
        raise ValueError("hard error")

    with pytest.raises(ValueError, match="hard error"):
        retry_with_backoff(
            "hard",
            fn,
            max_retries=3,
            initial_backoff=0.1,
            max_backoff=1.0,
            is_retryable=_never,
            sleep=sleeps.append,
        )

    assert sleeps == []


def test_retry_backoff_caps_at_max_backoff() -> None:
    """Exponential growth should clamp at ``max_backoff``."""
    sleeps: list[float] = []

    def fn() -> str:
        raise ConnectionError("flaky")

    with pytest.raises(ConnectionError):
        retry_with_backoff(
            "doomed",
            fn,
            max_retries=5,
            initial_backoff=1.0,
            max_backoff=3.5,
            is_retryable=_always,
            sleep=sleeps.append,
        )

    # Schedule: 1.0, 2.0, 3.5 (clamped from 4.0), 3.5, 3.5
    assert sleeps == [1.0, 2.0, 3.5, 3.5, 3.5]


def test_retry_with_lock_serialises_calls() -> None:
    """The optional lock should be held across the retry loop."""
    lock = threading.Lock()
    seen_locked: list[bool] = []

    def fn() -> str:
        seen_locked.append(lock.locked())
        return "done"

    result = retry_with_backoff(
        "locked",
        fn,
        max_retries=0,
        initial_backoff=0.0,
        max_backoff=0.0,
        is_retryable=_never,
        lock=lock,
        sleep=lambda _: None,
    )

    assert result == "done"
    assert seen_locked == [True]
    assert lock.locked() is False


def test_retry_max_retries_zero_means_single_attempt() -> None:
    """``max_retries=0`` should allow exactly one attempt."""
    attempts: list[int] = []

    def fn() -> str:
        attempts.append(1)
        raise ConnectionError("fail")

    with pytest.raises(ConnectionError):
        retry_with_backoff(
            "single",
            fn,
            max_retries=0,
            initial_backoff=0.1,
            max_backoff=1.0,
            is_retryable=_always,
            sleep=lambda _: None,
        )

    assert attempts == [1]


def test_retry_clamps_negative_backoff_values() -> None:
    """Negative backoff parameters should clamp to zero and skip the sleep call."""
    sleeps: list[float] = []
    attempts: list[int] = []

    def fn() -> str:
        attempts.append(1)
        raise ConnectionError("fail")

    with pytest.raises(ConnectionError):
        retry_with_backoff(
            "negative",
            fn,
            max_retries=2,
            initial_backoff=-1.0,
            max_backoff=-5.0,
            is_retryable=_always,
            sleep=sleeps.append,
        )

    # 3 attempts (initial + 2 retries) with zero-delay backoff means
    # the sleep callable is never invoked (the helper short-circuits on
    # ``delay <= 0``), but the retries still happen.
    assert len(attempts) == 3
    assert sleeps == []


def test_aretry_succeeds_after_transient_failure() -> None:
    """Async variant should mirror sync retry semantics."""
    sleeps: list[float] = []
    attempts: list[int] = []

    async def fn() -> str:
        attempts.append(len(attempts) + 1)
        if len(attempts) <= 1:
            raise ConnectionError("flaky")
        return "ok"

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    result = asyncio.run(
        aretry_with_backoff(
            "async-flaky",
            fn,
            max_retries=2,
            initial_backoff=0.1,
            max_backoff=1.0,
            is_retryable=_always,
            sleep=sleep,
        )
    )

    assert result == "ok"
    assert attempts == [1, 2]
    assert sleeps == [0.1]


def test_aretry_exhausts_and_reraises() -> None:
    """Async variant should propagate the final exception after exhaustion."""
    sleeps: list[float] = []

    async def fn() -> str:
        raise TimeoutError("perma-fail")

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    with pytest.raises(TimeoutError, match="perma-fail"):
        asyncio.run(
            aretry_with_backoff(
                "async-doomed",
                fn,
                max_retries=1,
                initial_backoff=0.05,
                max_backoff=1.0,
                is_retryable=_always,
                sleep=sleep,
            )
        )

    assert sleeps == [0.05]


def test_aretry_non_retryable_raises_immediately() -> None:
    """Async variant should not sleep for non-retryable errors."""
    sleeps: list[float] = []

    async def fn() -> str:
        raise ValueError("hard")

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    with pytest.raises(ValueError):
        asyncio.run(
            aretry_with_backoff(
                "async-hard",
                fn,
                max_retries=3,
                initial_backoff=0.1,
                max_backoff=1.0,
                is_retryable=_never,
                sleep=sleep,
            )
        )

    assert sleeps == []


# ---------------------------------------------------------------------------
# Predicate: is_transient_qdrant_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionError("connection reset by peer"),
        ConnectionResetError("reset"),
        TimeoutError("timeout"),
    ],
)
def test_is_transient_qdrant_error_catches_builtin_transports(
    exc: BaseException,
) -> None:
    """Built-in transport errors must classify as transient."""
    assert is_transient_qdrant_error(exc) is True


@pytest.mark.parametrize(
    "message",
    [
        "503 Service Unavailable",
        "504 Gateway Timeout",
        "Connection refused",
        "broken pipe",
        "request timed out",
        "Service Unavailable upstream",
    ],
)
def test_is_transient_qdrant_error_message_substring(message: str) -> None:
    """Generic exceptions whose message matches a transient token should retry."""
    assert is_transient_qdrant_error(RuntimeError(message)) is True


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("bad payload"),
        RuntimeError("auth failed"),
        sqlite3.OperationalError("syntax error"),
        Exception("schema mismatch"),
    ],
)
def test_is_transient_qdrant_error_rejects_application_errors(
    exc: BaseException,
) -> None:
    """Application-level errors without transient markers should not retry."""
    assert is_transient_qdrant_error(exc) is False


def test_is_transient_qdrant_error_handles_httpx_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``httpx.TransportError`` should classify as transient.

    Args:
        monkeypatch: Pytest monkeypatch fixture (unused; kept for future
            extensibility when stubbing optional imports).
    """
    _ = monkeypatch
    httpx = pytest.importorskip("httpx")
    exc: Any = httpx.ConnectError("nope")
    assert is_transient_qdrant_error(exc) is True
