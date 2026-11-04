"""Model calls count against the audit meter of the operation that caused them.

The meter rides a ``ContextVar``: a model call made on another thread counts
only when that thread runs in the submitter's context. These tests pin the
three shapes the code uses (a raw producer thread, a pool task, a cached
client) — the meter itself is covered in ``test_audit_core.py``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import docint.utils.translate_client as tc
from docint.core.ingest.preprocess import PreprocessPool
from docint.core.ingest.streaming_executor import ProducerConsumer
from docint.utils.audit_meter import count_transfer, metering


def test_the_ingest_producer_counts_against_the_runs_meter() -> None:
    """The producer thread's model calls land on the meter of the run that started it."""

    def _produce() -> Iterator[int]:
        for item in range(3):
            count_transfer("caption", "inference:vision", "vision-model", kind="image")
            yield item

    with metering() as meter:
        pc: ProducerConsumer[int] = ProducerConsumer(_produce, queue_max_size=1)
        with pc:
            assert list(pc.consume()) == [0, 1, 2]

    assert [(key.stage, key.kind, tally.items) for key, tally in meter.snapshot()] == [("caption", "image", 3)]


def test_a_preprocess_task_counts_against_its_submitters_meter() -> None:
    """Upload-time work runs on the pool's threads but counts where it was submitted."""
    pool = PreprocessPool(max_workers=1)
    try:
        with metering() as meter:
            pool.submit("k", lambda: count_transfer("ocr", "inference:ocr", "ocr-model", kind="image")).result(
                timeout=5
            )
    finally:
        pool.shutdown()

    assert [(key.stage, key.kind, tally.items) for key, tally in meter.snapshot()] == [("ocr", "image", 1)]


class _Pipeline:
    """A chat pipeline that answers every translation the same way."""

    def call_chat(self, text: str, *, system_prompt: str, model: str) -> str:
        """Return a fixed translation.

        Args:
            text (str): The text to translate.
            system_prompt (str): The translation prompt.
            model (str): The model asked.

        Returns:
            str: The translation.
        """
        return "Hello"


def test_a_translation_counts_only_a_cache_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached translation sends nothing: two identical calls are one transfer.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setattr(tc, "_pipeline", lambda: _Pipeline())
    tc._translate_cached.cache_clear()
    try:
        with metering() as meter:
            assert tc._translate_cached("Hallo", "en", "chat-model") == "Hello"
            assert tc._translate_cached("Hallo", "en", "chat-model") == "Hello"
    finally:
        tc._translate_cached.cache_clear()

    assert [(key.stage, key.model, tally.requests) for key, tally in meter.snapshot()] == [
        ("translate", "chat-model", 1)
    ]
