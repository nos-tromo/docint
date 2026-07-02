"""Tests for docint.__version__ and the GET /version endpoint."""

from fastapi.testclient import TestClient

import docint
import docint.core.api as api_module


def test_dunder_version_resolves_from_metadata() -> None:
    """docint.__version__ resolves to a non-empty string from metadata."""
    assert isinstance(docint.__version__, str) and docint.__version__


def test_version_endpoint_returns_version() -> None:
    """GET /version returns a non-empty version string."""
    client = TestClient(api_module.app)
    resp = client.get("/version")  # unauthenticated, no principal
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["version"], str) and body["version"]
