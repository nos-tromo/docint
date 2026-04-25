"""Exponential-backoff retry helpers shared across ingestion subsystems.

PR #116 introduced exponential-backoff retries for the SQLite docstore
(``database is locked`` errors) but left Qdrant vector inserts unprotected.
This module hoists the retry engine into a generic helper that the SQLite
KV store, the Qdrant vector path, and any future ingestion stage can
share.

Two invariants:

* The engine is generic over the exception predicate. SQLite, Qdrant, and
  per-file ingestion failures supply their own classifier via
  ``is_retryable``.
* The async variant uses :func:`asyncio.sleep` so it never blocks the
  event loop; the sync variant uses :func:`time.sleep` and may be guarded
  by a :class:`threading.Lock` for connection-level serialisation.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import AbstractContextManager
from typing import Awaitable, Callable, TypeVar

from loguru import logger

T = TypeVar("T")

RetryPredicate = Callable[[BaseException], bool]
SyncSleep = Callable[[float], None]
AsyncSleep = Callable[[float], Awaitable[None]]


_QDRANT_TRANSIENT_TOKENS: tuple[str, ...] = (
    "503",
    "504",
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "connection refused",
    "broken pipe",
    "remoteprotocolerror",
    "readerror",
    "service unavailable",
)


def _matches_transient_substring(exc: BaseException) -> bool:
    """Return True if ``str(exc)`` contains a known transient-failure substring.

    Last-resort classifier for transports that wrap their underlying
    transient cause in a generic exception type. The substring set is
    deliberately conservative to avoid false positives.

    Args:
        exc: The exception to inspect.

    Returns:
        True if the stringified exception contains a transient marker.
    """
    msg = str(exc).lower()
    return any(token in msg for token in _QDRANT_TRANSIENT_TOKENS)


def is_transient_qdrant_error(exc: BaseException) -> bool:
    """Return True if *exc* is a transient Qdrant transport failure.

    Used by :meth:`docint.core.rag.RAG._persist_node_batches` and
    :meth:`docint.core.rag.RAG._apersist_node_batches` to wrap
    ``index.insert_nodes`` / ``index.ainsert_nodes`` calls. The Qdrant
    client has its own inner retry layer; this is the outer envelope
    for cases the inner layer did not (or was configured not to)
    handle, so operators see end-to-end resilience even with the inner
    retry disabled.

    Args:
        exc: The exception raised by a Qdrant or HTTP transport call.

    Returns:
        True for connection resets, timeouts, 503/504 responses, and
        ``ResponseHandlingException``-wrapped transient causes; False
        for application-level errors that should fail fast (auth,
        schema mismatch, RuntimeError without a transient marker).
    """
    if isinstance(exc, (ConnectionError, ConnectionResetError, TimeoutError)):
        return True

    try:
        from qdrant_client.http.exceptions import (  # type: ignore[import-not-found]
            ResponseHandlingException,
            UnexpectedResponse,
        )

        if isinstance(exc, ResponseHandlingException):
            return True
        if isinstance(exc, UnexpectedResponse):
            return _matches_transient_substring(exc)
    except ImportError:  # pragma: no cover - qdrant is a hard dependency
        pass

    try:
        import httpx  # type: ignore[import-not-found]

        if isinstance(
            exc, (httpx.TransportError, httpx.RemoteProtocolError, httpx.TimeoutException)
        ):
            return True
    except ImportError:  # pragma: no cover - httpx ships with qdrant-client
        pass

    return _matches_transient_substring(exc)


def retry_with_backoff(
    operation: str,
    fn: Callable[[], T],
    *,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    is_retryable: RetryPredicate,
    lock: AbstractContextManager[object] | None = None,
    sleep: SyncSleep | None = None,
) -> T:
    """Run *fn* with exponential-backoff retries on retryable exceptions.

    Implements the same backoff formula as the SQLite KV store
    introduced in PR #116:
    ``delay = min(initial_backoff * 2**(attempt-1), max_backoff)``.

    Args:
        operation: Human-readable name used in retry log messages.
        fn: Callable that performs the underlying operation. Must take
            no arguments; closures are the typical idiom.
        max_retries: Maximum number of retry attempts after the first
            failure. Total attempts = ``max_retries + 1``. Values
            below 0 are clamped to 0 (no retries).
        initial_backoff: Initial delay between retries, in seconds.
            Doubled on each subsequent attempt up to ``max_backoff``.
        max_backoff: Maximum delay between retries, in seconds.
        is_retryable: Predicate that classifies an exception as
            transient (return ``True`` to retry) or fatal (return
            ``False`` to re-raise).
        lock: Optional context manager held for the duration of the
            retry loop. Used by :class:`SQLiteKVStore` to serialise
            access to a shared connection across threads. Pass
            ``None`` (the default) when no lock is required.
        sleep: Sleep callable, parameterised for testability. Defaults
            to a late-binding ``time.sleep`` lookup so monkeypatching
            ``time.sleep`` in tests is observed.

    Returns:
        The value returned by *fn* on the first successful attempt.

    Raises:
        Exception: Re-raises the last exception after retries are
            exhausted, or immediately if ``is_retryable`` returns
            ``False`` for a given exception.
    """
    attempts = max(1, int(max_retries) + 1)
    initial = max(0.0, float(initial_backoff))
    cap = max(0.0, float(max_backoff))

    def _run() -> T:
        attempt = 1
        while True:
            try:
                return fn()
            except Exception as exc:
                if not is_retryable(exc) or attempt >= attempts:
                    raise
                delay = min(initial * (2 ** (attempt - 1)), cap)
                logger.warning(
                    "Operation '{}' hit retryable error on attempt {}/{}: "
                    "{!r}. Retrying in {:.2f}s",
                    operation,
                    attempt,
                    attempts,
                    exc,
                    delay,
                )
                if delay > 0:
                    sleep_fn = sleep if sleep is not None else time.sleep
                    sleep_fn(delay)
                attempt += 1

    if lock is not None:
        with lock:
            return _run()
    return _run()


async def aretry_with_backoff(
    operation: str,
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    is_retryable: RetryPredicate,
    sleep: AsyncSleep | None = None,
) -> T:
    """Async twin of :func:`retry_with_backoff`.

    Args:
        operation: Human-readable operation name.
        fn: Callable returning an awaitable to retry.
        max_retries: Maximum retry attempts after the first failure.
        initial_backoff: Initial delay between retries, in seconds.
        max_backoff: Maximum delay between retries, in seconds.
        is_retryable: Predicate classifying transient vs. fatal errors.
        sleep: Async sleep callable, parameterised for testability.

    Returns:
        The awaited value from *fn* on the first successful attempt.

    Raises:
        Exception: Re-raises the last exception after retries are
            exhausted, or immediately for non-retryable errors.
    """
    attempts = max(1, int(max_retries) + 1)
    initial = max(0.0, float(initial_backoff))
    cap = max(0.0, float(max_backoff))

    attempt = 1
    while True:
        try:
            return await fn()
        except Exception as exc:
            if not is_retryable(exc) or attempt >= attempts:
                raise
            delay = min(initial * (2 ** (attempt - 1)), cap)
            logger.warning(
                "Async operation '{}' hit retryable error on attempt "
                "{}/{}: {!r}. Retrying in {:.2f}s",
                operation,
                attempt,
                attempts,
                exc,
                delay,
            )
            if delay > 0:
                sleep_fn = sleep if sleep is not None else asyncio.sleep
                await sleep_fn(delay)
            attempt += 1
