"""Concurrency / isolation tests for the stateless RAG read path (WS2).

These pin the multi-tenant invariant introduced in WS2: two requests for two
*different* physical collections, running concurrently on different threads,
must each see ONLY their own collection's retrieval results.

Before WS2 the active collection (``RAG.qdrant_collection``) and the derived
``index`` / ``query_engine`` were process-global singletons mutated by
``select_collection``; interleaved requests clobbered each other -- a
confidentiality bug. The fix backs ``qdrant_collection`` with a per-request
:class:`contextvars.ContextVar` (bound via :meth:`RAG.collection_scope`) and
caches ``index`` / ``query_engine`` per physical collection. Under the old
global model the ``run_query`` assertions below cross-contaminate and fail.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
from llama_index.core.base.response.schema import Response

from docint.core.rag import RAG


class _VectorStoreQueryModeStub:
    """Tiny enum-like stub with a ``.value`` attribute for the query-mode field."""

    class _Member:
        """Single member stand-in exposing ``.value``."""

        value = "default"

    DEFAULT = _Member()


def _patch_engine_factory(monkeypatch: pytest.MonkeyPatch, capture_barrier: threading.Barrier | None = None) -> None:
    """Patch the engine builder to capture the active collection at build time.

    The fake query engine records ``self.qdrant_collection`` when it is built
    and echoes it back from ``.query()``; the normalizer then surfaces that
    captured name as the single source's ``collection``. A cached engine that
    leaks across collections therefore returns the wrong name and fails the
    isolation assertions.

    When ``capture_barrier`` is supplied the build blocks on it immediately
    after reading the active collection, forcing every concurrent builder to
    capture *before* any of them returns (and thus before any request scope
    exits). Without this rendezvous the GIL lets each thread run its whole
    scoped section to completion in turn, so a global-state implementation
    would round-trip correctly by luck and the race would go undetected.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
        capture_barrier (threading.Barrier | None): Optional rendezvous awaited
            after the active collection is captured.
    """

    def _fake_build(self: RAG, **_kwargs: Any) -> Any:
        captured = self.qdrant_collection
        if capture_barrier is not None:
            capture_barrier.wait()

        class _Engine:
            def query(self, prompt: str) -> Response:
                return Response(response=captured, source_nodes=[])

        return _Engine()

    monkeypatch.setattr(RAG, "build_query_engine", _fake_build)
    monkeypatch.setattr(
        RAG,
        "_normalize_response_data",
        lambda self, prompt, result, **kw: {
            "response": result.response,
            "sources": [{"collection": result.response}],
        },
    )
    monkeypatch.setattr(
        RAG,
        "_resolve_runtime_retrieval_settings",
        lambda self, *a, **k: {
            "vector_store_query_mode": _VectorStoreQueryModeStub.DEFAULT,
            "label": "test",
            "parent_context_enabled": False,
        },
    )


def test_collection_scope_isolates_active_collection_per_thread() -> None:
    """``collection_scope`` overrides the active collection per thread only.

    Two threads each enter their own scope and rendezvous on a barrier so both
    scopes are simultaneously active. Each must read back its own collection,
    and the process default must be restored once both scopes exit.
    """
    rag = RAG(qdrant_collection="base")
    results: dict[str, str] = {}
    entered = threading.Barrier(2)
    read = threading.Barrier(2)

    def worker(name: str) -> None:
        with rag.collection_scope(name):
            entered.wait()  # both scopes are simultaneously active
            value = rag.qdrant_collection
            read.wait()  # both have read before either scope exits
            results[name] = value

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("collA", "collB")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == {"collA": "collA", "collB": "collB"}
    assert rag.qdrant_collection == "base"


def test_concurrent_run_query_returns_only_own_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interleaved queries on two collections each return only their own data.

    This is the headline WS2 guard. Each thread runs a query under its own
    ``collection_scope``; the engine builder blocks on a capture barrier right
    after reading the active collection, so both threads have captured their
    collection before either returns or exits its scope. Each result must
    reference only the collection that thread scoped. A global active collection
    would have both capture the last writer's value and fail this assertion.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
    """
    capture_barrier = threading.Barrier(2)
    _patch_engine_factory(monkeypatch, capture_barrier)
    rag = RAG(qdrant_collection="")
    results: dict[str, dict[str, Any]] = {}
    errors: dict[str, BaseException] = {}
    entered = threading.Barrier(2)

    def worker(physical: str) -> None:
        try:
            with rag.collection_scope(physical):
                entered.wait()  # both scopes active before either builds/queries
                results[physical] = rag.run_query("question")
        except BaseException as exc:
            errors[physical] = exc

    threads = [threading.Thread(target=worker, args=(p,)) for p in ("u_a__docs", "u_b__docs")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert results["u_a__docs"]["sources"] == [{"collection": "u_a__docs"}]
    assert results["u_b__docs"]["sources"] == [{"collection": "u_b__docs"}]


def test_query_engine_cached_per_physical_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lazily-built default engine is cached per physical collection.

    Switching collections must not reuse another collection's engine, and
    re-entering a collection must reuse its own cached engine.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
    """
    _patch_engine_factory(monkeypatch)
    rag = RAG(qdrant_collection="")

    with rag.collection_scope("collA"):
        rag.run_query("q")
        engine_a = rag.query_engine
    with rag.collection_scope("collB"):
        rag.run_query("q")
        engine_b = rag.query_engine

    assert engine_a is not None
    assert engine_b is not None
    assert engine_a is not engine_b
    with rag.collection_scope("collA"):
        assert rag.query_engine is engine_a
    # No scope active -> the empty process default, which never cached an engine.
    assert rag.query_engine is None
