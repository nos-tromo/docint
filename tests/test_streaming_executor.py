"""Tests for the bounded producer/consumer ingestion executor.

Phase 4 of the streaming ingestion generalisation. Pins:

* Successful production yields all items in input order.
* Producer exceptions surface to the consumer side after the queue
  drains (not silently swallowed).
* The queue caps memory under back-pressure — a slow consumer cannot
  cause unbounded buffering.
* Sentinel-based shutdown lets the consumer terminate without
  timeouts.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Iterator

import pytest

from docint.core.ingest.streaming_executor import (
    ProducerConsumer,
    overlapped,
)


def _slow_producer(items: list[Any], delay_seconds: float = 0.0) -> Any:
    """Yield *items* with an optional ``delay_seconds`` between each."""

    def _gen() -> Iterator[Any]:
        for item in items:
            if delay_seconds:
                time.sleep(delay_seconds)
            yield item

    return _gen


def test_producer_consumer_yields_items_in_order() -> None:
    """All items produced should reach the consumer in input order."""
    pc: ProducerConsumer[int] = ProducerConsumer(
        _slow_producer([1, 2, 3, 4, 5]), queue_max_size=2
    )
    with pc:
        result: list[int] = list(pc.consume())
    assert result == [1, 2, 3, 4, 5]


def test_overlapped_helper_short_form() -> None:
    """The :func:`overlapped` helper should mirror context-manager usage."""
    items: list[str] = list(
        overlapped(_slow_producer(["a", "b", "c"]), queue_max_size=1)
    )
    assert items == ["a", "b", "c"]


def test_producer_exception_surfaces_to_consumer() -> None:
    """An exception raised by the producer should re-raise on consume()."""

    def _producer() -> Iterator[int]:
        yield 1
        yield 2
        raise RuntimeError("producer-blew-up")

    pc: ProducerConsumer[int] = ProducerConsumer(_producer, queue_max_size=2)
    received: list[int] = []
    with pytest.raises(RuntimeError, match="producer-blew-up"):
        with pc:
            for item in pc.consume():
                received.append(item)
    # The two pre-failure items still reached the consumer.
    assert received == [1, 2]


def test_queue_bound_keeps_producer_blocked_until_consumer_drains() -> None:
    """A bounded queue should backpressure a fast producer."""

    produced: list[int] = []
    drain_event = threading.Event()

    def _producer() -> Iterator[int]:
        for i in range(10):
            produced.append(i)
            yield i

    pc: ProducerConsumer[int] = ProducerConsumer(_producer, queue_max_size=2)
    with pc:
        # Pull the first item, then wait briefly to let the producer fill
        # the queue. Since the queue caps at 2 items, the producer should
        # have produced at most 3 items (one yielded + two buffered)
        # before being blocked on queue.put.
        consumer: Iterator[int] = pc.consume()
        first = next(consumer)
        assert first == 0
        time.sleep(0.05)
        # The producer can sit at most: 1 (consumer-taken) + 2 (queue
        # capacity) + 1 (in-flight ``yield``) = 4 ahead of consumption.
        assert len(produced) <= 4, f"producer ran ahead of bounded queue: {produced}"
        # Drain the rest.
        rest: list[int] = list(consumer)
    drain_event.set()  # not strictly used; pin sequencing for future
    assert rest == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_empty_producer_terminates_cleanly() -> None:
    """An empty producer should produce nothing and not deadlock."""
    pc: ProducerConsumer[Any] = ProducerConsumer(_slow_producer([]), queue_max_size=1)
    with pc:
        result: list[Any] = list(pc.consume())
    assert result == []


def test_consumer_can_break_early_without_deadlock() -> None:
    """Breaking out of the consumer loop should not hang the producer."""

    def _producer() -> Iterator[int]:
        for i in range(100):
            yield i

    pc: ProducerConsumer[int] = ProducerConsumer(_producer, queue_max_size=2)
    received: list[int] = []
    with pc:
        for item in pc.consume():
            received.append(item)
            if len(received) >= 3:
                break
    assert received == [0, 1, 2]
    # Context exit completed — no deadlock observed.


def test_overlap_runs_producer_on_background_thread() -> None:
    """The producer should execute on a thread distinct from the consumer."""

    main_thread = threading.current_thread()
    producer_thread: list[threading.Thread] = []

    def _producer() -> Iterator[int]:
        producer_thread.append(threading.current_thread())
        yield 1

    pc: ProducerConsumer[int] = ProducerConsumer(_producer, queue_max_size=1)
    with pc:
        list(pc.consume())

    assert producer_thread, "producer never ran"
    assert producer_thread[0] is not main_thread
