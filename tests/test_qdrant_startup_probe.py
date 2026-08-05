"""Tests for the startup Qdrant reachability probe.

Qdrant is contacted lazily, so a mis-wired deployment (backend not on
data-net, data-plane stack down) used to surface only at the first ingest
or query. ``RAG.probe_qdrant`` runs once at application startup and logs a
loud, actionable error instead — without failing the app, since Qdrant may
legitimately come up after the backend on an airgapped boot and the
SQLite-backed endpoints still work without it.
"""

import urllib.error

import pytest
from fastapi.testclient import TestClient

from docint.core import rag as rag_module
from docint.core.rag import RAG


@pytest.fixture()
def rag_instance() -> RAG:
    """A RAG built the way tests/test_rag_unit.py builds one (no live Qdrant)."""
    return RAG(qdrant_collection="test")


class _FakeResponse:
    """Minimal urlopen context-manager response."""

    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_probe_true_when_qdrant_ready(
    monkeypatch: pytest.MonkeyPatch,
    rag_instance: RAG,
) -> None:
    """A 2xx from /readyz reports Qdrant reachable."""

    def _ok(url: str, timeout: float | None = None) -> _FakeResponse:
        return _FakeResponse(200)

    monkeypatch.setattr(rag_module.urllib.request, "urlopen", _ok)

    assert rag_instance.probe_qdrant() is True


def test_probe_false_when_qdrant_unreachable(
    monkeypatch: pytest.MonkeyPatch,
    rag_instance: RAG,
) -> None:
    """A transport failure (e.g. DNS) returns False and never raises."""

    def _boom(url: str, timeout: float | None = None) -> _FakeResponse:
        raise urllib.error.URLError(OSError(-3, "Temporary failure in name resolution"))

    monkeypatch.setattr(rag_module.urllib.request, "urlopen", _boom)

    assert rag_instance.probe_qdrant() is False


def test_probe_targets_readyz_on_configured_host(
    monkeypatch: pytest.MonkeyPatch,
    rag_instance: RAG,
) -> None:
    """The probe must hit the resolved QDRANT_HOST, not some other base."""
    rag_instance.qdrant_host = "http://qdrant:6333"
    seen: list[str] = []

    def _capture(url: str, timeout: float | None = None) -> _FakeResponse:
        seen.append(url)
        assert timeout is not None
        return _FakeResponse(200)

    monkeypatch.setattr(rag_module.urllib.request, "urlopen", _capture)
    rag_instance.probe_qdrant()

    assert seen == ["http://qdrant:6333/readyz"]


def test_lifespan_startup_runs_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Application startup probes Qdrant before serving requests."""
    from docint.core import api as api_module

    calls: list[bool] = []

    def _record(self: RAG) -> bool:
        calls.append(True)
        return True

    monkeypatch.setattr(rag_module.RAG, "probe_qdrant", _record)

    with TestClient(api_module.app):
        assert calls == [True]


def test_lifespan_startup_survives_unreachable_qdrant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreachable Qdrant is logged, not fatal: the app still serves."""
    from docint.core import api as api_module

    def _down(self: RAG) -> bool:
        return False

    monkeypatch.setattr(rag_module.RAG, "probe_qdrant", _down)

    with TestClient(api_module.app) as client:
        response = client.get("/version")
        assert response.status_code == 200


def test_health_reports_ok_when_qdrant_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/health re-runs the probe on demand and reports it healthy."""
    from docint.core import api as api_module

    monkeypatch.setattr(rag_module.RAG, "probe_qdrant", lambda self: True)

    with TestClient(api_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "qdrant": True}


def test_health_reports_degraded_when_qdrant_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead Qdrant degrades /health but keeps it HTTP 200.

    The Docker healthcheck watches /version; /health is a status report,
    so a vector-store outage must not flap the container.
    """
    from docint.core import api as api_module

    monkeypatch.setattr(rag_module.RAG, "probe_qdrant", lambda self: False)

    with TestClient(api_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "qdrant": False}


def test_health_is_unauthenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    """/health needs no principal, like /version — the Makefile curls it."""
    from docint.core import api as api_module

    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)
    monkeypatch.setattr(rag_module.RAG, "probe_qdrant", lambda self: True)

    with TestClient(api_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
