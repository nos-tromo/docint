"""Tests for the GET /metrics Prometheus scrape endpoint."""

from fastapi.testclient import TestClient

import docint.core.api as api_module


def test_metrics_endpoint_returns_prometheus_families() -> None:
    """GET /metrics returns 200 and exposes a known Prometheus metric family."""
    client = TestClient(api_module.app)
    resp = client.get("/metrics")  # unauthenticated, no principal
    assert resp.status_code == 200
    assert "http_requests_total" in resp.text
