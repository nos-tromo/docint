"""API tests for generic client-visible error bodies (full detail stays in logs).

Identity is carried by ``X-Auth-User`` (default header); requests with no header
fall back to ``DOCINT_DEFAULT_IDENTITY`` ("test-operator"), matching the
conventions in ``tests/test_api_collections_ownership.py``.
"""

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from conftest import run_ingest
from fastapi.testclient import TestClient
from loguru import logger

import docint.core.api as api_module

MARKER = "MARKER-SECRET-1234"


@pytest.fixture(autouse=True)
def _default_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a default authenticated identity for every request in this module."""
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "test-operator")


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a TestClient that exercises the real installed handlers.

    Entered as a context manager so a single portal (and its background
    event-loop thread) stays alive for the whole test: ingest jobs run as a
    detached ``asyncio`` task meant to outlive the request that queued them,
    and a bare, non-context-managed ``TestClient`` opens a brand-new
    throwaway event loop per call — orphaning that task the instant the
    queuing request returns.
    """
    with TestClient(api_module.app, raise_server_exceptions=False) as client:
        yield client


def _capture_logs() -> tuple[list[str], int]:
    records: list[str] = []
    sink_id = logger.add(lambda m: records.append(str(m)), level="DEBUG")
    return records, sink_id


def test_unhandled_error_is_generic_and_logged(client: TestClient) -> None:
    """An unhandled exception returns a generic body; the marker only appears in logs."""

    @api_module.app.get("/__test_boom")
    def boom() -> None:
        raise RuntimeError(MARKER)

    records, sink_id = _capture_logs()
    try:
        resp = client.get("/__test_boom")
        assert resp.status_code == 500
        assert resp.json() == {"detail": "Internal server error."}
        assert MARKER not in resp.text
        assert any(MARKER in r for r in records)
    finally:
        logger.remove(sink_id)
        api_module.app.router.routes = [
            route for route in api_module.app.router.routes if getattr(route, "path", None) != "/__test_boom"
        ]


def test_validation_error_returns_generic_body(client: TestClient) -> None:
    """A request-validation failure returns a static, non-echoing detail."""
    # /query requires a "question" field; omit it to trigger RequestValidationError.
    resp = client.post("/query", json={})
    assert resp.status_code == 422
    assert resp.json() == {"detail": "Invalid request."}


def test_validation_error_does_not_log_the_submitted_question(client: TestClient) -> None:
    """A malformed /query body must not put the question in the log.

    Pydantic attaches the offending value to every error it raises, so the
    validation handler was the one path on this API by which a user's
    question could reach a log line. Sending the question as a list makes it
    the rejected ``input`` — exactly the field that used to be logged.
    """
    records, sink_id = _capture_logs()
    try:
        resp = client.post("/query", json={"question": [MARKER]})
        assert resp.status_code == 422
    finally:
        logger.remove(sink_id)

    combined = "\n".join(records)
    assert MARKER not in combined, f"the submitted question reached the log: {combined!r}"
    # The diagnostic itself must survive; only the value is withheld.
    assert any("Validation error on POST /query" in r for r in records)


def test_swept_endpoint_returns_static_detail_and_logs_marker(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """A representative swept endpoint (collections/list) hides exception text from the client."""

    def raiser() -> Any:
        raise RuntimeError(MARKER)

    monkeypatch.setattr(api_module.rag, "ensure_collection_owner_manager", raiser)

    records, sink_id = _capture_logs()
    try:
        resp = client.get("/collections/list")
        assert resp.status_code == 500
        assert resp.json() == {"detail": "Request failed."}
        assert MARKER not in resp.text
        assert any(MARKER in r for r in records)
    finally:
        logger.remove(sink_id)


def test_ingest_finalize_job_error_is_generic_and_logged(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """A raising ingest job fails with a static message; the marker only appears in logs.

    ``/ingest/upload`` no longer runs the pipeline at all — it only stages
    files (see the ``client`` fixture docstring). This now drives the
    failure through ``/ingest/finalize``'s job and asserts on the job's
    snapshot (``GET /ingest/jobs/{job_id}``) instead of an SSE body.

    This replaces both ``test_ingest_upload_stream_error_is_generic_and_logged``
    and ``test_ingest_stream_error_event_carries_code``. The machine-readable
    ``code: "ingestion_failed"`` companion field on the underlying SSE
    ``error`` event is ``docint/core/jobs.py`` behavior
    (``IngestJobManager._worker``'s failure path) and is covered at that
    layer by
    ``tests/test_ingest_jobs.py::test_failed_runner_error_frame_carries_machine_readable_code``,
    not duplicated here. That event is not reachable through an HTTP test
    transport for this assertion: ``/ingest/jobs/events`` never terminates on
    its own, and every in-process ASGI transport available here (sync
    ``TestClient`` and async ``httpx.ASGITransport``) fully drains a response
    before returning anything — see
    ``tests/test_ingest_jobs_api.py::test_events_route_is_not_shadowed_by_the_job_id_route``
    for the same finding. This test asserts only what actually changed: how a
    caller observes a failed run through the new job-polling contract.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture.
    """
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)

    def raising_ingest(
        collection: str,
        path: Path,
        hybrid: bool = True,
        progress_callback: Any = None,
        **kwargs: Any,
    ) -> None:
        _ = (collection, path, hybrid, progress_callback)
        raise RuntimeError(MARKER)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", raising_ingest)

    records, sink_id = _capture_logs()
    try:
        staged = client.post(
            "/ingest/upload",
            data={"collection": "boom-collection"},
            files={"files": ("a.txt", b"hello", "text/plain")},
        )
        assert staged.status_code == 200

        snapshot = run_ingest(client, "boom-collection", {})

        assert snapshot["status"] == "failed"
        assert snapshot["error"] == "Ingestion failed."
        assert MARKER not in json.dumps(snapshot)
        assert any(MARKER in r for r in records)
    finally:
        logger.remove(sink_id)


def test_ingest_save_failure_is_static_with_code_and_filename(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """A file-save failure emits a static message plus code and structured filename."""
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)

    def raising_hash(path: Path) -> str:
        _ = path
        raise RuntimeError(MARKER)

    monkeypatch.setattr(api_module, "compute_file_hash", raising_hash)

    records, sink_id = _capture_logs()
    try:
        resp = client.post(
            "/ingest/upload",
            data={"collection": "boom-collection"},
            files={"files": ("a.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 200
        body = resp.text
        assert "event: error" in body
        assert '"message": "Failed to save file."' in body
        assert '"code": "save_failed"' in body
        assert '"filename": "a.txt"' in body
        assert MARKER not in body
        assert any(MARKER in r for r in records)
    finally:
        logger.remove(sink_id)
