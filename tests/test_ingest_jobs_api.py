"""Endpoint tests for the ingest job registry."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docint.core import api as api_module


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Generator[TestClient, None, None]:
    """A TestClient whose ingest jobs run a deterministic stub pipeline."""

    def _runner(state: Any, push: Any) -> dict[str, Any]:
        push("ingestion_progress", {"message": "working"})
        return {"empty": False, "resolution": None}

    monkeypatch.setattr(api_module.job_manager, "_runner", _runner)
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    # job_manager is a process-wide singleton (constructed once at module
    # import), so tests reusing the same (owner, collection) pair would
    # otherwise see jobs left behind by earlier tests. Start every test from
    # an empty registry.
    api_module.job_manager._jobs.clear()
    api_module.job_manager._subscribers.clear()
    # Ingest jobs run as a detached ``asyncio.create_task`` meant to outlive the
    # request that queued them. A bare ``TestClient(app)`` opens a brand-new
    # throwaway event loop for every single call (see starlette's
    # ``_portal_factory``), which orphans that task the instant the queuing
    # request returns — the job then never advances past "queued". Entering
    # the client as a context manager keeps one portal (and its background
    # event-loop thread) alive for the whole test, so the worker actually runs.
    with TestClient(api_module.app) as client:
        yield client


def _headers(user: str = "alice") -> dict[str, str]:
    return {"X-Auth-User": user}


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


def test_finalize_409s_with_the_existing_job_when_already_running(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second finalize for the same collection 409s, carrying the in-flight job id."""
    import threading

    gate = threading.Event()

    def _blocking(state: Any, push: Any) -> dict[str, Any]:
        gate.wait(timeout=5)
        return {"empty": False, "resolution": None}

    monkeypatch.setattr(api_module.job_manager, "_runner", _blocking)
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


def test_delete_409s_while_running(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """`DELETE /ingest/jobs/{job_id}` refuses to dismiss a still-running job."""
    import threading

    gate = threading.Event()

    def _blocking(state: Any, push: Any) -> dict[str, Any]:
        gate.wait(timeout=5)
        return {"empty": False, "resolution": None}

    monkeypatch.setattr(api_module.job_manager, "_runner", _blocking)
    _stage(client, "mydocs")
    job_id = client.post("/ingest/finalize", json={"collection": "mydocs"}, headers=_headers()).json()["job_id"]

    assert client.delete(f"/ingest/jobs/{job_id}", headers=_headers()).status_code == 409
    gate.set()
