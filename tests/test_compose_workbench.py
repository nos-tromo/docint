"""Assertions for the workbench network seam in compose and the Makefile."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _service(text: str, name: str) -> str:
    """Return one compose service's block.

    The block runs from the ``  <name>:`` line up to the next two-space service
    key, or up to the next top-level key, whichever comes first.

    Args:
        text (str): The compose file's text.
        name (str): The service name.

    Returns:
        str: The service's lines, joined as in the file.
    """
    lines = text.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.rstrip() == f"  {name}:")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if re.match(r"^(  [A-Za-z0-9_-]+:\s*$|[A-Za-z0-9_-]+:)", lines[i]):
            end = i
            break
    return "".join(lines[start:end])


def test_the_backend_joins_workbench_net_and_the_frontend_does_not() -> None:
    """Only the backend reaches workbench; the frontend stays off its network."""
    compose = (REPO_ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")
    backend, frontend = _service(compose, "backend"), _service(compose, "frontend")

    assert "WORKBENCH_API_BASE: ${WORKBENCH_API_BASE:-http://workbench-api:8000}" in backend
    assert "      workbench-net:\n        aliases:\n          - docint-backend" in backend
    assert "workbench-net" not in frontend
    assert "  workbench-net:\n    external: true\n    name: ${WORKBENCH_NET:-workbench-net}" in compose


def test_the_service_slice_stops_at_the_next_service() -> None:
    """The helper keeps the backend block apart from its neighbours."""
    compose = (REPO_ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")
    backend = _service(compose, "backend")

    assert backend.startswith("  backend:")
    assert "  frontend:" not in backend
    assert "image: docint-frontend" not in backend


def test_make_network_creates_workbench_net() -> None:
    """``make network`` creates every external network the backend joins."""
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "NETWORKS := inference-net data-net edge-net workbench-net" in makefile
