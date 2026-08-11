"""Endpoint tests for the ``/summarize`` cache-or-queue contract.

``/summarize`` no longer generates a summary synchronously: it serves the
cached tree-summary payload (``RAG.cached_collection_summary``) on a hit, or
queues a ``kind="summary"`` background job on a miss / explicit
``refresh=true``. ``/summarize/stream`` is removed outright -- there is no
token stream during a map-reduce job build, clients follow job progress on
``GET /ingest/jobs/events`` and refetch ``/summarize`` on the terminal event.

Fixtures mirror ``tests/test_ingest_jobs_api.py``: a private
:class:`~docint.core.jobs.IngestJobManager` is injected per test via
``app.dependency_overrides`` so no test observes another's in-flight jobs,
and the module's ``rag`` singleton is swapped for a minimal stub so these
tests do not depend on a real Qdrant / session-store backend. Collection
ownership is a no-op passthrough here (every logical name is "owned") --
real owner-gating behavior is covered by
``tests/test_api_collections_ownership.py``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docint.core import api as api_module
from docint.core.jobs import IngestJobManager, JobRunner


class _StubOwners:
    """Passthrough collection-ownership manager: every name is already owned."""

    def register(self, owner: str | None, logical: str) -> str:
        """Return ``logical`` unchanged, as if newly registered."""
        return logical

    def resolve(self, owner: str | None, logical: str) -> str | None:
        """Return ``logical`` unchanged, as if already owned."""
        return logical


class _StubRAG:
    """Minimal stand-in for :class:`~docint.core.rag.RAG` for endpoint tests.

    Provides just the surface ``/summarize`` touches: collection scoping,
    the ownership manager, and ``cached_collection_summary`` (overridden
    per test via ``monkeypatch``). ``build_tree_summary`` stands in for the
    real tree summarizer so a queued job completes cleanly under the
    default (non-blocking) test runner instead of failing with
    ``AttributeError``.
    """

    def __init__(self) -> None:
        """Initialize the stub with a default active collection."""
        self.qdrant_collection = "col"
        self.summarize_prompt = "Summarize this collection."
        self._owners = _StubOwners()

    def probe_qdrant(self) -> bool:
        """Satisfy the lifespan startup probe without touching the network."""
        return True

    def ensure_session_manager(self) -> SimpleNamespace:
        """Satisfy the lifespan's eager session-store init without a DB.

        Returns:
            SimpleNamespace: A stand-in whose ``init_session_store_if_needed``
            is a no-op.
        """
        return SimpleNamespace(init_session_store_if_needed=lambda: None)

    def reconcile_quantization(self) -> int:
        """Satisfy the lifespan quantization reconcile without touching Qdrant."""
        return 0

    def list_collections(self) -> list[str]:
        """Return the fixed single collection ``_require_active_collection`` validates against."""
        return ["col"]

    def ensure_collection_owner_manager(self) -> _StubOwners:
        """Return the passthrough ownership manager stub."""
        return self._owners

    @contextmanager
    def collection_scope(self, physical: str) -> Iterator[None]:
        """Mirror ``RAG.collection_scope``: bind then restore the active collection."""
        prev = self.qdrant_collection
        self.qdrant_collection = physical
        try:
            yield
        finally:
            self.qdrant_collection = prev

    def cached_collection_summary(self) -> dict[str, Any] | None:
        """Return ``None`` by default; tests override this via monkeypatch."""
        return None

    def build_tree_summary(self, progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        """Return a canned tree-summary payload, standing in for a real build."""
        if progress is not None:
            progress(1, 1)
        return {"response": "built summary", "sources": [], "summary_diagnostics": {}}


def _default_runner(state: Any, push: Any) -> dict[str, Any]:
    """Deterministic stand-in for ``_run_job`` that resolves immediately."""
    push("summary_progress", {"message": "working"})
    return {"empty": False, "resolution": None}


@pytest.fixture
def make_client(monkeypatch: pytest.MonkeyPatch) -> Generator[Callable[..., TestClient], None, None]:
    """Build a TestClient backed by a job manager and ``rag`` stub private to this test.

    Mirrors ``tests/test_ingest_jobs_api.py``'s fixture of the same name: each
    call constructs a fresh :class:`IngestJobManager` and injects it via
    ``app.dependency_overrides``, so no test can observe jobs left behind by
    another. Pass ``runner`` to control what the stub "job" does -- a
    blocking runner is how the 409-while-in-flight tests keep a job alive.
    """
    monkeypatch.setattr(api_module, "rag", _StubRAG())
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "test-operator")
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    clients: list[TestClient] = []

    def _make(runner: JobRunner = _default_runner) -> TestClient:
        manager = IngestJobManager(runner=runner)
        api_module.app.dependency_overrides[api_module.get_job_manager] = lambda: manager
        # Summary jobs run as a detached `asyncio.create_task` meant to outlive
        # the request that queued them (see test_ingest_jobs_api.py's fixture
        # for the full explanation) -- enter the client as a context manager
        # to keep one event-loop portal alive for the whole test.
        ctx = TestClient(api_module.app)
        clients.append(ctx)
        return ctx.__enter__()

    yield _make

    for ctx in clients:
        ctx.__exit__(None, None, None)
    api_module.app.dependency_overrides.clear()


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    """A TestClient whose queued summary jobs run a deterministic stub runner."""
    return make_client()


def test_summarize_cache_hit_returns_200(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached summary answers 200 with the stored payload, no job queued."""
    payload: dict[str, Any] = {
        "response": "cached summary",
        "sources": [],
        "summary_diagnostics": {
            "total_documents": 1,
            "covered_documents": 1,
            "coverage_ratio": 1.0,
            "uncovered_documents": [],
            "coverage_target": 0.7,
        },
    }
    monkeypatch.setattr(api_module.rag, "cached_collection_summary", lambda: payload)

    res = client.post("/summarize?collection=col")

    assert res.status_code == 200
    assert res.json()["summary"] == "cached summary"


def test_summarize_cache_miss_queues_job_202(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cache miss queues a summary job and answers 202 with its job id."""
    monkeypatch.setattr(api_module.rag, "cached_collection_summary", lambda: None)

    res = client.post("/summarize?collection=col")

    assert res.status_code == 202
    assert res.json()["job_id"]


def test_summarize_refresh_true_queues_even_with_cache(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """``refresh=true`` queues a rebuild job even when a cached summary exists."""
    monkeypatch.setattr(api_module.rag, "cached_collection_summary", lambda: {"response": "x"})

    res = client.post("/summarize?collection=col&refresh=true")

    assert res.status_code == 202
    assert res.json()["job_id"]


def test_summarize_second_call_409_with_job_id(
    make_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second summarize call while a build is in flight 409s with the existing job id."""
    gate = threading.Event()

    def _blocking(state: Any, push: Any) -> dict[str, Any]:
        gate.wait(timeout=5)
        return {"empty": False, "resolution": None}

    client = make_client(runner=_blocking)
    monkeypatch.setattr(api_module.rag, "cached_collection_summary", lambda: None)

    first = client.post("/summarize?collection=col")
    second = client.post("/summarize?collection=col")

    assert first.status_code == 202
    assert second.status_code == 409
    assert second.json()["detail"]["job_id"] == first.json()["job_id"]
    gate.set()


def test_summarize_stream_route_removed(client: TestClient) -> None:
    """``POST /summarize/stream`` no longer exists."""
    res = client.post("/summarize/stream?collection=col")

    assert res.status_code in (404, 405)


def test_summarize_queue_requires_explicit_collection(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Queuing a build without an explicit ``collection`` 400s.

    ``_resolve_request_collection`` falls back to the process-default active
    collection (``rag.qdrant_collection``) when ``collection`` is omitted --
    that value may be a physical, owner-namespaced Qdrant name, which must
    never be echoed into a job's ``logical_name`` snapshot or back to a
    client. The queue path therefore requires an explicit logical name
    rather than risk leaking the physical one; the cache-read path has no
    such requirement since it never creates a job.
    """
    monkeypatch.setattr(api_module.rag, "cached_collection_summary", lambda: None)

    res = client.post("/summarize")

    assert res.status_code == 400


def test_summarize_cache_hit_without_collection_uses_default(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache-read path still honors the process-default collection fallback."""
    payload: dict[str, Any] = {"response": "default-collection summary", "sources": [], "summary_diagnostics": None}
    monkeypatch.setattr(api_module.rag, "cached_collection_summary", lambda: payload)

    res = client.post("/summarize")

    assert res.status_code == 200
    assert res.json()["summary"] == "default-collection summary"
