"""Tests for Phase 6 per-file failure isolation.

Validates the two-mode contract:

* ``INGEST_FAIL_FAST=true`` → first persistence failure aborts the run
  (the original PR #116 behaviour).
* ``INGEST_FAIL_FAST=false`` (default) → the outer ingest loop logs the
  failure under ``failed_ingest_batch``, marks the affected file hashes
  failed in the manifest, and continues with the next batch. A
  ``failed_ingest_batch`` summary log lands in ``finally``.

The failure-classification predicates in :mod:`docint.utils.retry` are
covered separately by :mod:`tests.test_retry`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, cast

import pytest
from llama_index.core.schema import TextNode
from loguru import logger as _loguru_logger

from docint.core.rag import RAG
from docint.utils.retry import (
    is_hard_ingest_error,
    is_transient_ingest_error,
)


def _capture_loguru(caplog: pytest.LogCaptureFixture) -> Any:
    """Forward loguru records into pytest's caplog at WARNING+."""
    sink_id = _loguru_logger.add(
        lambda message: caplog.records.append(
            logging.LogRecord(
                name="loguru",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg=str(message),
                args=None,
                exc_info=None,
            )
        ),
        level="WARNING",
        format="{message}",
    )

    def _cleanup() -> None:
        _loguru_logger.remove(sink_id)

    return _cleanup


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        OSError("disk full"),
        IOError("io"),
        ConnectionError("transient"),
        TimeoutError("slow"),
    ],
)
def test_is_transient_ingest_error_classifies_io(exc: BaseException) -> None:
    """OS-level I/O errors should classify as transient at the ingest layer."""
    assert is_transient_ingest_error(exc) is True


def test_is_transient_ingest_error_includes_qdrant_transients() -> None:
    """Qdrant transient errors should also satisfy the ingest predicate."""
    assert is_transient_ingest_error(ConnectionError("connection reset")) is True


def test_is_transient_ingest_error_rejects_hard_errors() -> None:
    """Decoder/parser failures should not classify as transient."""
    assert is_transient_ingest_error(ValueError("bad header")) is False
    assert (
        is_transient_ingest_error(json.JSONDecodeError("bad", "x", 0))
        is False
    )


def test_is_hard_ingest_error_catches_decoder_failures() -> None:
    """JSON / unicode decode errors should classify as hard."""
    assert is_hard_ingest_error(json.JSONDecodeError("bad", "x", 0)) is True
    assert is_hard_ingest_error(UnicodeDecodeError("utf-8", b"\xff", 0, 1, "x")) is True


def test_is_hard_ingest_error_rejects_transient() -> None:
    """OS-level errors are not hard."""
    assert is_hard_ingest_error(OSError("disk")) is False
    assert is_hard_ingest_error(ConnectionError("net")) is False


# ---------------------------------------------------------------------------
# Outer-loop fail-fast vs. skip-and-continue
# ---------------------------------------------------------------------------


def _make_rag_with_streaming_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    yields: list[tuple[list[Any], list[TextNode], set[str]]],
    *,
    fail_fast: bool,
) -> tuple[RAG, list[Any], "Any"]:
    """Construct a RAG instance whose streaming pipeline yields *yields*.

    Returns (rag, persist_calls, fake_manifest). Each entry in
    persist_calls is the ``nodes`` argument that
    ``_persist_node_batches`` was invoked with.
    """
    persist_calls: list[Any] = []

    class FakeManifest:
        """Recording manifest that satisfies the IngestManifest interface."""

        def __init__(self) -> None:
            self.started: list[tuple[str, str]] = []
            self.completed: list[tuple[str, str]] = []
            self.failed: list[tuple[str, str, str | None]] = []
            self.closed = False

        def mark_started(self, c: str, h: str) -> None:
            self.started.append((c, h))

        def mark_completed(self, c: str, h: str) -> None:
            self.completed.append((c, h))

        def mark_failed(self, c: str, h: str, err: str | None) -> None:
            self.failed.append((c, h, err))

        def pending_files(self, c: str) -> dict[str, float]:
            return {}

        def completed_files(self, c: str) -> set[str]:
            return set()

        def failed_files(self, c: str) -> dict[str, str | None]:
            return {}

        def close(self) -> None:
            self.closed = True

    fake_manifest = FakeManifest()

    class FakePipeline:
        def __init__(self) -> None:
            self.entity_extractor = None
            self.ner_max_workers = 0
            self.image_ingestion_service = None
            self.dir_reader = None

        def build_streaming(self, processed_hashes: set[str]) -> Any:
            yield from yields

    rag = RAG(qdrant_collection="bench")
    rag.docstore_batch_size = 10
    rag.ingest_fail_fast = fail_fast
    rag.qdrant_collection = "bench"
    rag.data_dir = Path("/tmp/_bench_unused")

    monkeypatch.setattr(
        RAG, "_prepare_sources_dir", lambda self, p: p
    )
    monkeypatch.setattr(
        RAG, "_vector_store", lambda self: cast(Any, object())
    )
    monkeypatch.setattr(
        RAG, "_storage_context", lambda self, vs: cast(Any, object())
    )
    monkeypatch.setattr(
        RAG, "_build_ingestion_pipeline", lambda self, **kwargs: FakePipeline()
    )
    monkeypatch.setattr(
        RAG, "_build_ingest_manifest", lambda self, *a, **k: fake_manifest
    )
    monkeypatch.setattr(
        RAG, "_get_existing_file_hashes", lambda self: set()
    )

    class _StubCorePDFReader:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.discovered_hashes: set[str] = set()

        def build(self, *args: Any, **kwargs: Any) -> Any:
            return iter([])

    monkeypatch.setattr(
        "docint.core.rag.CorePDFPipelineReader",
        _StubCorePDFReader,
    )

    embed_model_marker = object()
    monkeypatch.setattr(
        type(rag),
        "embed_model",
        property(lambda self: embed_model_marker),
    )

    class _StubVectorStoreIndex:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(
        "docint.core.rag.VectorStoreIndex",
        _StubVectorStoreIndex,
    )
    monkeypatch.setattr(
        "docint.core.rag.qdrant_collection_exists",
        lambda *a, **k: True,
    )

    def _record_persist(self: RAG, nodes: list[Any]) -> None:
        persist_calls.append(list(nodes))
        if nodes and any(
            getattr(n, "node_id", "") == "fail-me" for n in nodes
        ):
            raise ConnectionError("connection reset")

    monkeypatch.setattr(RAG, "_persist_node_batches", _record_persist)

    monkeypatch.setattr(
        RAG, "create_query_engine", lambda self: None
    )
    monkeypatch.setattr(
        RAG, "reset_session_state", lambda self: None
    )
    monkeypatch.setattr(
        RAG, "_invalidate_ner_cache", lambda self, c: None
    )
    monkeypatch.setattr(
        RAG, "_bump_summary_revision", lambda self, c: None
    )

    return rag, persist_calls, fake_manifest


def _node(text: str, file_hash: str, node_id: str) -> TextNode:
    return TextNode(text=text, metadata={"file_hash": file_hash}, id_=node_id)


def test_skip_and_continue_logs_failed_ingest_batch_and_marks_failed(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single bad batch should be skipped and other batches still persist."""
    yields = [
        ([], [_node("ok-1", "h1", "ok-1")], {"h1"}),
        ([], [_node("ok-2", "h2", "fail-me")], set()),
        ([], [_node("ok-3", "h3", "ok-3")], {"h3"}),
    ]

    rag, persist_calls, fake = _make_rag_with_streaming_pipeline(
        monkeypatch, yields, fail_fast=False
    )

    cleanup = _capture_loguru(caplog)
    try:
        rag.ingest_docs(Path("/tmp/_unused"), build_query_engine=False)
    finally:
        cleanup()

    pass
    # All three yields attempted persistence.
    assert len(persist_calls) == 3
    # h1 and h3 completed; h2 was marked failed.
    assert ("bench", "h1") in fake.completed
    assert ("bench", "h3") in fake.completed
    assert any(
        c == "bench" and h == "h2" for c, h, _ in fake.failed
    )
    # Manifest was closed in the finally clause.
    assert fake.closed is True
    # Structured failure log emitted.
    combined = "\n".join(str(record.msg) for record in caplog.records)
    assert "failed_ingest_batch" in combined
    assert "h2" in combined
    # End-of-run summary present.
    assert "Ingest finished with 1 failed batch" in combined


def test_fail_fast_aborts_on_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ingest_fail_fast=True`` should re-raise on the first persistence failure."""
    yields = [
        ([], [_node("ok-1", "h1", "ok-1")], {"h1"}),
        ([], [_node("bad", "h2", "fail-me")], set()),
        ([], [_node("ok-3", "h3", "ok-3")], {"h3"}),
    ]

    rag, persist_calls, fake = _make_rag_with_streaming_pipeline(
        monkeypatch, yields, fail_fast=True
    )

    with pytest.raises(ConnectionError, match="connection reset"):
        rag.ingest_docs(Path("/tmp/_unused"), build_query_engine=False)

    # Only the first two yields hit persistence; the third never runs.
    assert len(persist_calls) == 2
    pass
    # The in-flight hash from the failing batch is marked failed.
    assert any(c == "bench" and h == "h2" for c, h, _ in fake.failed)


def test_skip_and_continue_async_path(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The async ingest path should mirror sync skip-and-continue semantics."""
    yields = [
        ([], [_node("ok-1", "h1", "ok-1")], {"h1"}),
        ([], [_node("bad", "h2", "fail-me")], set()),
    ]

    rag, persist_calls, fake = _make_rag_with_streaming_pipeline(
        monkeypatch, yields, fail_fast=False
    )

    apersist_calls: list[Any] = []

    async def _arecord(self: RAG, nodes: list[Any]) -> None:
        apersist_calls.append(list(nodes))
        if any(getattr(n, "node_id", "") == "fail-me" for n in nodes):
            raise ConnectionError("connection reset")

    monkeypatch.setattr(RAG, "_apersist_node_batches", _arecord)

    cleanup = _capture_loguru(caplog)
    try:
        asyncio.run(
            rag.asingest_docs(Path("/tmp/_unused"), build_query_engine=False)
        )
    finally:
        cleanup()

    assert len(apersist_calls) == 2
    pass
    assert ("bench", "h1") in fake.completed
    assert any(c == "bench" and h == "h2" for c, h, _ in fake.failed)
    combined = "\n".join(str(record.msg) for record in caplog.records)
    assert "failed_ingest_batch" in combined
    assert "Async ingest finished with 1 failed batch" in combined
    _ = persist_calls  # unused for the async path
