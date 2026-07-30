"""Tests for the GET /whoami endpoint."""

import pytest
from fastapi.testclient import TestClient

import docint.core.api as api_module


@pytest.fixture
def client() -> TestClient:
    """Create a TestClient for testing the FastAPI application.

    Returns:
        TestClient: The TestClient instance.
    """
    return TestClient(api_module.app)


def test_whoami_returns_the_trusted_header_identity(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """A present trusted header is echoed back as ``username``."""
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)
    response = client.get("/whoami", headers={"X-Auth-User": "alice"})
    assert response.status_code == 200
    assert response.json() == {"username": "alice"}


def test_whoami_falls_back_to_default_identity(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """With no header, the configured dev default identity is returned."""
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "test-operator")
    response = client.get("/whoami")
    assert response.status_code == 200
    assert response.json() == {"username": "test-operator"}


def test_whoami_fails_closed_without_header_or_default(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """No header and no configured fallback means 401, like every other endpoint."""
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)
    response = client.get("/whoami")
    assert response.status_code == 401
