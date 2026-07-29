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
    """A present trusted header is echoed back as ``username``; no X-Auth-Name means display_name is None."""
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)
    response = client.get("/whoami", headers={"X-Auth-User": "alice"})
    assert response.status_code == 200
    assert response.json() == {"username": "alice", "display_name": None}


def test_whoami_includes_display_name_when_gateway_header_present(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """The decorative X-Auth-Name header, when present, is surfaced as display_name."""
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)
    response = client.get("/whoami", headers={"X-Auth-User": "alice", "X-Auth-Name": "Alice Example"})
    assert response.status_code == 200
    assert response.json() == {"username": "alice", "display_name": "Alice Example"}


def test_whoami_falls_back_to_default_identity(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """With no header, the configured dev default identity is returned."""
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "test-operator")
    response = client.get("/whoami")
    assert response.status_code == 200
    assert response.json() == {"username": "test-operator", "display_name": None}


def test_whoami_fails_closed_without_header_or_default(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """No header and no configured fallback means 401, like every other endpoint."""
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)
    response = client.get("/whoami")
    assert response.status_code == 401


def test_whoami_display_name_is_decorative_not_identity(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """X-Auth-Name never substitutes for the principal — auth stays governed by X-Auth-User."""
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)
    # X-Auth-Name alone, with no trusted identity header, still fails closed.
    response = client.get("/whoami", headers={"X-Auth-Name": "Alice Example"})
    assert response.status_code == 401
