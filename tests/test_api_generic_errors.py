"""API tests for generic client-visible error bodies (full detail stays in logs).

Identity is carried by ``X-Auth-User`` (default header); requests with no header
fall back to ``DOCINT_DEFAULT_IDENTITY`` ("test-operator"), matching the
conventions in ``tests/test_api_collections_ownership.py``.
"""

from pathlib import Path
from typing import Any

import pytest
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
def client() -> TestClient:
    """Create a TestClient that exercises the real installed handlers."""
    return TestClient(api_module.app, raise_server_exceptions=False)


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


def test_ingest_upload_stream_error_is_generic_and_logged(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """A raising ingest pipeline emits a generic SSE ``error`` event; the marker only appears in logs."""
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)

    def raising_ingest(
        collection: str,
        path: Path,
        hybrid: bool = True,
        progress_callback: Any = None,
    ) -> None:
        _ = (collection, path, hybrid, progress_callback)
        raise RuntimeError(MARKER)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", raising_ingest)

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
        assert '"message": "Ingestion failed."' in body
        assert MARKER not in body
        assert any(MARKER in r for r in records)
    finally:
        logger.remove(sink_id)
