"""Bounded producer/consumer overlap for the streaming ingestion pipeline.

Phase 4 of the streaming ingestion generalisation. Today the ingestion
loop runs ``read → enrich → embed → persist`` sequentially: a single
yield from :meth:`DocumentIngestionPipeline.build_streaming` blocks the
HTTP path until persistence completes. With this utility, a producer
thread runs the generator and pushes yields onto a bounded queue while
the consumer (main thread) drains the queue and persists. Embedder and
docstore I/O overlap with NER / hate-speech enrichment.

Three invariants:

* **Bounded queue** — capped at ``INGEST_BATCH_SIZE × 2`` by default to
  bound memory under back-pressure.
* **Single consumer** — Qdrant insert ordering matters less than
  contention; one consumer thread drains the queue serially.
* **Sentinel-based shutdown** — the producer puts a sentinel on the
  queue before exiting (clean or error path) so the consumer can
  terminate without timeouts.

Threading is preferred over asyncio because: (a) the sync ingestion
path is canonical and converting to async would cascade through the
CLI / Streamlit UI / tests; (b) embedder calls are HTTP-bound and
release the GIL inside ``httpx``; (c) NER inference releases the GIL
via PyTorch C++.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Generic, Iterable, Iterator, TypeVar

from loguru import logger

T = TypeVar("T")


class _Sentinel:
    """Marker placed on the queue by the producer to signal end-of-stream."""

    __slots__ = ()


_END = _Sentinel()


class ProducerConsumer(Generic[T]):
    """Thread-based producer/consumer with a bounded queue.

    Use as a context manager::

        def producer() -> Iterable[tuple[...]]:
            yield from pipeline.build_streaming(processed_hashes)

        with ProducerConsumer(producer, queue_max_size=10) as pc:
            for batch in pc.consume():
                # ... persist + manifest hooks ...

    The producer runs on a background daemon thread. Exceptions
    raised by the producer surface to the consumer when the queue
    drains — the consumer sees a clean ``StopIteration`` only on a
    successful run.

    Args:
        producer_fn: Zero-argument callable returning an iterable.
            Typically a closure that calls
            :meth:`DocumentIngestionPipeline.build_streaming`.
        queue_max_size: Maximum number of yielded items buffered
            ahead of the consumer. Defaults to 4.
    """

    def __init__(
        self,
        producer_fn: Callable[[], Iterable[T]],
        *,
        queue_max_size: int = 4,
    ) -> None:
        self._producer_fn = producer_fn
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, queue_max_size))
        self._thread: threading.Thread | None = None
        self._exception: BaseException | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> ProducerConsumer[T]:
        """Start the producer thread."""
        self._thread = threading.Thread(
            target=self._run,
            name="docint-ingest-producer",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Drain the queue and join the producer.

        If the consumer exited early (e.g. via an exception), the
        queue may still have items the producer is trying to push.
        Drain whatever is buffered so the producer can put its
        sentinel and exit cleanly.
        """
        if self._thread is None:
            return
        # Drain any remaining items so the producer (which is blocked on
        # queue.put under back-pressure) can complete and place its
        # sentinel. We do not raise from here — context-manager exits
        # should not mask the original exception.
        while self._thread.is_alive():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                self._thread.join(timeout=0.1)
        # Flush any trailing items (the sentinel we placed in _run).
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Producer thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Drive the producer iterable; relay exceptions to the consumer."""
        try:
            for item in self._producer_fn():
                self._queue.put(item)
        except BaseException as exc:  # noqa: BLE001 - re-raised on consume()
            self._exception = exc
        finally:
            self._queue.put(_END)

    # ------------------------------------------------------------------
    # Consumer interface
    # ------------------------------------------------------------------

    def consume(self) -> Iterator[T]:
        """Yield items produced on the background thread.

        Yields:
            Items from the producer iterable, in production order.

        Raises:
            BaseException: If the producer raised, the same exception
                is re-raised on the consumer side after the queue
                drains.
        """
        while True:
            item = self._queue.get()
            if isinstance(item, _Sentinel):
                if self._exception is not None:
                    exc = self._exception
                    self._exception = None
                    raise exc
                return
            yield item


def overlapped(
    producer_fn: Callable[[], Iterable[T]],
    *,
    queue_max_size: int = 4,
) -> Iterator[T]:
    """Convenience wrapper: run *producer_fn* in a background thread.

    Equivalent to::

        with ProducerConsumer(producer_fn, queue_max_size=qs) as pc:
            yield from pc.consume()

    Useful for one-shot consumption inside a ``for`` loop without an
    explicit ``with`` block.

    Args:
        producer_fn: Zero-argument callable returning an iterable.
        queue_max_size: Maximum number of buffered items.

    Yields:
        Items from the producer iterable.
    """
    pc = ProducerConsumer(producer_fn, queue_max_size=queue_max_size)
    with pc:
        yield from pc.consume()


__all__ = ["ProducerConsumer", "overlapped"]


def _diagnose_queue_health() -> None:  # pragma: no cover - operator hook
    """Reserved hook for future telemetry on queue saturation patterns."""
    logger.debug("ProducerConsumer queue diagnostics not yet implemented.")
