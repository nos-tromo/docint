"""Verifies the SPA static mount behavior."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_static_mount_present_only_when_dist_exists(repo_root: Path) -> None:
    """The /static_index sentinel route always responds; / serves SPA only when built."""
    from docint.core.api import app

    client = TestClient(app)

    dist = repo_root / "frontend" / "dist" / "index.html"
    if dist.is_file():
        res = client.get("/")
        assert res.status_code == 200
        assert "<!doctype html" in res.text.lower() or "<html" in res.text.lower()
    else:
        res = client.get("/")
        assert res.status_code in (404, 405)


def test_spa_fallback_for_client_routes(repo_root: Path) -> None:
    """Hard-refreshing /chat/abc should still return the SPA shell."""
    from docint.core.api import app

    client = TestClient(app)
    dist = repo_root / "frontend" / "dist" / "index.html"
    if not dist.is_file():
        pytest.skip("frontend not built")
    res = client.get("/chat/abc")
    assert res.status_code == 200
    assert "<!doctype html" in res.text.lower() or "<html" in res.text.lower()


def test_api_routes_still_work_with_static_mount(repo_root: Path) -> None:
    """The /collections/list route must not be shadowed by the SPA mount."""
    from docint.core.api import app

    client = TestClient(app)
    # We don't care about the body; we only care that the route isn't 404
    # because of the SPA mount swallowing it.
    res = client.get("/collections/list")
    assert res.status_code != 404
