"""Endpoint tests for the ingest job registry."""

from __future__ import annotations

import asyncio
import tempfile
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docint.core import api as api_module
from docint.core.jobs import IngestJobManager, IngestJobState, JobRunner


def _default_runner(state: Any, push: Any) -> dict[str, Any]:
    """Deterministic stand-in for the real ingestion pipeline."""
    push("ingestion_progress", {"message": "working"})
    return {"empty": False, "resolution": None}


@pytest.fixture
def make_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Generator[Callable[..., TestClient], None, None]:
    """Build a TestClient backed by a manager private to this test.

    Each call constructs a fresh :class:`IngestJobManager` and injects it via
    ``app.dependency_overrides``, so no test can observe jobs left behind by
    another and none has to reach into the manager's private state to reset
    it. Pass ``runner`` to control what the stub pipeline does.
    """
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    clients: list[Any] = []

    def _make(runner: JobRunner = _default_runner) -> TestClient:
        manager = IngestJobManager(runner=runner)
        api_module.app.dependency_overrides[api_module.get_job_manager] = lambda: manager
        # Ingest jobs run as a detached ``asyncio.create_task`` meant to outlive
        # the request that queued them. A bare ``TestClient(app)`` opens a
        # brand-new throwaway event loop for every single call (see starlette's
        # ``_portal_factory``), which orphans that task the instant the queuing
        # request returns — the job then never advances past "queued". Entering
        # the client as a context manager keeps one portal (and its background
        # event-loop thread) alive for the whole test, so the worker runs.
        ctx = TestClient(api_module.app)
        clients.append(ctx)
        return ctx.__enter__()

    yield _make

    for ctx in clients:
        ctx.__exit__(None, None, None)
    api_module.app.dependency_overrides.clear()


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    """A TestClient whose ingest jobs run a deterministic stub pipeline."""
    return make_client()


def _headers(user: str = "alice") -> dict[str, str]:
    return {"X-Auth-User": user}


def _await_terminal(client: TestClient, job_id: str, user: str = "alice") -> dict[str, Any]:
    """Poll a job's snapshot until it finishes, and return it.

    The worker runs on the test client's portal thread, so sleeping here lets
    it advance.
    """
    snapshot: dict[str, Any] = {}
    for _ in range(200):
        snapshot = client.get(f"/ingest/jobs/{job_id}", headers=_headers(user)).json()
        if snapshot["status"] in {"completed", "failed"}:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"job did not finish; last snapshot={snapshot}")


def _stage(client: TestClient, collection: str, user: str = "alice") -> None:
    client.post(
        "/ingest/upload",
        data={"collection": collection},
        files={"files": ("sample.txt", b"hello", "text/plain")},
        headers=_headers(user),
    )


def test_finalize_returns_a_job_id(client: TestClient) -> None:
    """`/ingest/finalize` queues a job and returns 202 with its id."""
    _stage(client, "mydocs")
    res = client.post("/ingest/finalize", json={"collection": "mydocs"}, headers=_headers())

    assert res.status_code == 202
    assert res.json()["job_id"]


def test_finalize_folds_the_client_upload_leg_into_the_run(client: TestClient) -> None:
    """The run's duration starts when the user did, not when the job was queued.

    The upload happens before any job exists, so the client reports how long
    it took and the snapshot's window has to cover it — otherwise the logged
    total undercounts exactly the stretch the ingest card already ticked
    through.
    """
    _stage(client, "mydocs")
    job_id = client.post(
        "/ingest/finalize",
        json={"collection": "mydocs", "upload_elapsed_ms": 30_000},
        headers=_headers(),
    ).json()["job_id"]

    snapshot = _await_terminal(client, job_id)
    assert snapshot["duration_ms"] >= 30_000
    assert snapshot["run_started_at"] < snapshot["created_at"]


def test_finalize_rejects_a_negative_upload_leg(client: TestClient) -> None:
    """A duration cannot be negative; the model refuses it before it reaches a log line."""
    _stage(client, "mydocs")
    res = client.post(
        "/ingest/finalize",
        json={"collection": "mydocs", "upload_elapsed_ms": -1},
        headers=_headers(),
    )

    assert res.status_code == 422


def test_finalize_409s_with_the_existing_job_when_already_running(
    make_client: Callable[..., TestClient],
) -> None:
    """A second finalize for the same collection 409s, carrying the in-flight job id."""
    import threading

    gate = threading.Event()

    def _blocking(state: Any, push: Any) -> dict[str, Any]:
        gate.wait(timeout=5)
        return {"empty": False, "resolution": None}

    client = make_client(runner=_blocking)
    _stage(client, "mydocs")
    first = client.post("/ingest/finalize", json={"collection": "mydocs"}, headers=_headers())
    second = client.post("/ingest/finalize", json={"collection": "mydocs"}, headers=_headers())

    assert second.status_code == 409
    assert second.json()["detail"]["job_id"] == first.json()["job_id"]
    gate.set()


def test_jobs_list_is_owner_scoped(client: TestClient) -> None:
    """`GET /ingest/jobs` lists only the caller's own jobs."""
    _stage(client, "mydocs", user="alice")
    client.post("/ingest/finalize", json={"collection": "mydocs"}, headers=_headers("alice"))

    assert len(client.get("/ingest/jobs", headers=_headers("alice")).json()["jobs"]) == 1
    assert client.get("/ingest/jobs", headers=_headers("bob")).json()["jobs"] == []


def test_job_snapshot_404s_cross_owner(client: TestClient) -> None:
    """`GET /ingest/jobs/{job_id}` 404s for a caller who does not own the job."""
    _stage(client, "mydocs", user="alice")
    job_id = client.post("/ingest/finalize", json={"collection": "mydocs"}, headers=_headers("alice")).json()["job_id"]

    assert client.get(f"/ingest/jobs/{job_id}", headers=_headers("alice")).status_code == 200
    assert client.get(f"/ingest/jobs/{job_id}", headers=_headers("bob")).status_code == 404


def test_events_route_is_not_shadowed_by_the_job_id_route() -> None:
    """`/ingest/jobs/events` must resolve to the events route, not the job-id one.

    ``/ingest/jobs/events`` never terminates on its own (it idles on a 15 s
    ping loop until the client disconnects), and every in-process ASGI
    transport available here — the synchronous ``fastapi.testclient.TestClient``
    *and* an async ``httpx.AsyncClient`` over ``ASGITransport`` — fully drains
    the app's response before handing anything back to the caller (confirmed
    against both transports' source: neither surfaces ``http.response.start``
    early). Driving this route through an actual request therefore deadlocks
    rather than failing fast, regardless of client choice.

    So this asserts the property directly against Starlette's router: for the
    path ``/ingest/jobs/events``, the first *fully* matching route must be the
    events endpoint, not ``get_ingest_job`` (which would otherwise treat
    "events" as a ``job_id`` path parameter — exactly the shadowing this test
    guards against, per route declaration order).
    """
    from fastapi.routing import APIRoute
    from starlette.routing import Match

    scope = {"type": "http", "method": "GET", "path": "/ingest/jobs/events"}
    for route in api_module.app.router.routes:
        match, _ = route.matches(scope)
        if match == Match.FULL:
            assert isinstance(route, APIRoute)
            assert route.endpoint is api_module.ingest_job_events
            return
    pytest.fail("No route matched GET /ingest/jobs/events")


def test_delete_409s_while_running(make_client: Callable[..., TestClient]) -> None:
    """`DELETE /ingest/jobs/{job_id}` refuses to dismiss a still-running job."""
    import threading

    gate = threading.Event()

    def _blocking(state: Any, push: Any) -> dict[str, Any]:
        gate.wait(timeout=5)
        return {"empty": False, "resolution": None}

    client = make_client(runner=_blocking)
    _stage(client, "mydocs")
    job_id = client.post("/ingest/finalize", json={"collection": "mydocs"}, headers=_headers()).json()["job_id"]

    assert client.delete(f"/ingest/jobs/{job_id}", headers=_headers()).status_code == 409
    gate.set()


def test_job_manager_is_injectable(client: TestClient) -> None:
    """Endpoints use the injected manager, never the application's own.

    This is what makes per-test isolation possible without reaching into the
    manager's private state to reset it between tests.
    """
    _stage(client, "mydocs")
    res = client.post("/ingest/finalize", json={"collection": "mydocs"}, headers=_headers())
    assert res.status_code == 202
    job_id = res.json()["job_id"]
    assert client.get(f"/ingest/jobs/{job_id}", headers=_headers()).status_code == 200

    # The application's own manager never saw the job the fixture's did.
    app_jobs = asyncio.run(api_module.job_manager.list_for_owner("alice"))
    assert [s.job_id for s in app_jobs] == []


def _make_state(*, kind: str = "ingest", resolve: bool = False) -> IngestJobState:
    """Build a standalone :class:`IngestJobState` for calling a runner directly.

    Bypasses :class:`IngestJobManager` entirely — these tests drive
    ``_run_summary_job``/``_run_ingest_job`` as plain functions, not through
    the async worker dispatch. ``kind="ingest"`` gets a real, empty temp
    directory for ``batch_dir`` (the runner's first check is
    ``batch_dir.is_dir()``); ``kind="summary"`` omits it, matching how
    summary jobs are created in practice.

    Args:
        kind (str): ``"ingest"`` or ``"summary"``.
        resolve (bool): Whether the ingest job should attempt entity
            resolution after the pipeline runs. Ingest-only.

    Returns:
        IngestJobState: A job state never registered with any manager.
    """
    batch_dir = Path(tempfile.mkdtemp()) if kind == "ingest" else None
    return IngestJobState(
        job_id="test-job-id",
        owner="alice",
        logical_name="mydocs",
        physical="u0123456789ab__mydocs",
        kind=kind,
        batch_dir=batch_dir,
        resolve=resolve,
    )


def test_run_summary_job_builds_and_reports_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """_run_summary_job scopes the collection, forwards progress, returns non-empty."""
    calls: dict[str, bool] = {}

    def fake_build(progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        assert progress is not None
        progress(1, 2)
        progress(2, 2)
        calls["built"] = True
        return {"response": "sum", "sources": [], "summary_diagnostics": {}}

    monkeypatch.setattr(api_module.rag, "build_tree_summary", fake_build)
    pushed: list[tuple[str, dict[str, Any]]] = []
    state = _make_state(kind="summary")
    result = api_module._run_summary_job(state, lambda ev, p: pushed.append((ev, p)))

    assert calls["built"]
    assert result == {"empty": False, "resolution": None}
    progress_events = [p for ev, p in pushed if ev == "summary_progress"]
    assert progress_events[-1]["mapped"] == 2
    assert progress_events[-1]["total_units"] == 2


def test_ingest_job_runs_summary_stage_after_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """_run_ingest_job calls build_tree_summary when SUMMARY_ON_INGEST is on."""
    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", lambda *a, **k: None)
    built: list[int] = []
    monkeypatch.setattr(api_module.rag, "build_tree_summary", lambda progress=None: built.append(1) or {})
    state = _make_state(kind="ingest", resolve=False)

    api_module._run_ingest_job(state, lambda ev, p: None)

    assert built


def test_ingest_summary_stage_failure_is_warning_not_job_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A summary-stage exception after ingest degrades to a warning, not a job failure."""
    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", lambda *a, **k: None)

    def boom(progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        raise RuntimeError("llm down")

    monkeypatch.setattr(api_module.rag, "build_tree_summary", boom)
    pushed: list[tuple[str, dict[str, Any]]] = []
    state = _make_state(kind="ingest", resolve=False)

    result = api_module._run_ingest_job(state, lambda ev, p: pushed.append((ev, p)))

    assert result["empty"] is False
    assert any(ev == "warning" and "summary" in p["message"].lower() for ev, p in pushed)


def test_summary_on_ingest_false_skips_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """With SUMMARY_ON_INGEST=false the ingest job never calls build_tree_summary."""
    monkeypatch.setenv("SUMMARY_ON_INGEST", "false")
    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", lambda *a, **k: None)
    built: list[int] = []
    monkeypatch.setattr(api_module.rag, "build_tree_summary", lambda progress=None: built.append(1) or {})
    state = _make_state(kind="ingest", resolve=False)

    result = api_module._run_ingest_job(state, lambda ev, p: None)

    assert not built
    assert result == {"empty": False, "resolution": None}


def test_run_job_dispatches_by_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    """_run_job routes a summary-kind state to _run_summary_job and everything else to _run_ingest_job."""
    calls: list[str] = []
    monkeypatch.setattr(
        api_module,
        "_run_summary_job",
        lambda state, push: calls.append("summary") or {"empty": False, "resolution": None},
    )
    monkeypatch.setattr(
        api_module,
        "_run_ingest_job",
        lambda state, push: calls.append("ingest") or {"empty": False, "resolution": None},
    )

    api_module._run_job(_make_state(kind="summary"), lambda ev, p: None)
    api_module._run_job(_make_state(kind="ingest"), lambda ev, p: None)

    assert calls == ["summary", "ingest"]
