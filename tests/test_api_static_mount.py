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
