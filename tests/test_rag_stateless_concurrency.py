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


def test_retrieval_handle_cache_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-collection handle cache evicts least-recently-used entries.

    An unbounded cache would pin a SQLite docstore connection per collection and
    leak file descriptors. With the bound set to 2, querying three collections
    must drop the oldest, keeping the cache within the bound.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
    """
    import docint.core.rag as rag_module

    _patch_engine_factory(monkeypatch)
    monkeypatch.setattr(rag_module, "_RETRIEVAL_HANDLE_CACHE_MAX", 2)
    rag = RAG(qdrant_collection="")

    for name in ("c1", "c2", "c3"):
        with rag.collection_scope(name):
            rag.run_query("q")

    assert len(rag._query_engine_cache) == 2
    assert set(rag._query_engine_cache.keys()) == {"c2", "c3"}  # c1 (oldest) evicted


def test_concurrent_query_requests_isolated_per_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two concurrent HTTP ``/query`` requests for two owners are fully isolated.

    Alice and Bob both query the *same* logical collection (``docs``) at the same
    time on different threads. The owner manager maps each to a distinct physical
    collection; the engine builder rendezvous on a barrier so both requests are
    in flight simultaneously. Each response must contain only its own owner's
    physical collection — the end-to-end proof that the read/query path carries
    no shared active-collection state. Under the old global model both responses
    would echo whichever request wrote the singleton last.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
    """
    from fastapi.testclient import TestClient

    import docint.core.api as api_module

    entered_barrier = threading.Barrier(2)
    read_barrier = threading.Barrier(2)

    def _fake_build(self: RAG, **_kwargs: Any) -> Any:
        rag_ref = self

        class _Engine:
            def query(self, prompt: str) -> Response:
                # Two rendezvous, both inside the request's collection scope:
                #   1. wait until BOTH requests have entered their scope, so a
                #      shared global already holds the last writer's value;
                #   2. read, then wait again so NEITHER request exits (and
                #      restores) its scope before both have read.
                # With the per-request ContextVar each read sees only its own
                # collection; with a shared global both see the last writer and
                # the assertions below fail.
                entered_barrier.wait(timeout=10)
                value = rag_ref.qdrant_collection
                read_barrier.wait(timeout=10)
                return Response(response=value, source_nodes=[])

        return _Engine()

    class _Owners:
        """Owner-manager stub mapping (owner, logical) to a per-owner physical name."""

        def resolve(self, owner: str | None, logical: str) -> str:
            return f"u_{owner}__{logical}"

    monkeypatch.setattr(RAG, "build_query_engine", _fake_build)
    monkeypatch.setattr(RAG, "create_index", lambda self: None)
    monkeypatch.setattr(RAG, "expand_query_with_graph_with_debug", lambda self, q: (q, {}))
    monkeypatch.setattr(
        RAG,
        "_normalize_response_data",
        lambda self, prompt, result, **kw: {"response": result.response, "sources": [{"collection": result.response}]},
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
    monkeypatch.setattr(RAG, "ensure_collection_owner_manager", lambda self: _Owners())

    rag = RAG(qdrant_collection="")
    monkeypatch.setattr(api_module, "rag", rag)
    monkeypatch.setattr(api_module, "_validation_payload", lambda **kwargs: {})
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")

    client = TestClient(api_module.app)
    results: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}

    def call(user: str) -> None:
        try:
            results[user] = client.post(
                "/query",
                json={"question": "q", "collection": "docs", "retrieval_mode": "stateless"},
                headers={"X-Auth-User": user},
            )
        except BaseException as exc:
            errors[user] = exc

    threads = [threading.Thread(target=call, args=(u,)) for u in ("alice", "bob")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert results["alice"].status_code == 200, results["alice"].text
    assert results["bob"].status_code == 200, results["bob"].text
    assert results["alice"].json()["sources"] == [{"collection": "u_alice__docs"}]
    assert results["bob"].json()["sources"] == [{"collection": "u_bob__docs"}]
