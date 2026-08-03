"""Shared pytest configuration and fixtures for the docint test suite."""

import sys
import time
import types
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient


class _MagicModule(types.ModuleType):
    """Magic module for handling file types."""

    class Magic:
        """Magic class for handling file types."""

        def __init__(self, mime: bool = True) -> None:
            """Initialize the Magic class.

            Args:
                mime (bool, optional): Whether to use MIME types. Defaults to True.
            """
            self.mime = mime

        def from_file(self, path: str) -> str:
            """Get the MIME type of a file.

            Args:
                path (str): The path to the file.

            Returns:
                str: The MIME type of the file.
            """
            return "application/octet-stream"


def _install_magic_stub() -> None:
    """Install a stub for the magic module."""
    sys.modules.setdefault("magic", _MagicModule("magic"))


def pytest_configure() -> None:
    """Configure pytest by installing necessary stubs."""
    _install_magic_stub()


@pytest.fixture(autouse=True)
def _hermetic_hybrid_env() -> Iterator[None]:
    """Clear the env vars that drive hybrid-retrieval resolution.

    ``docint.utils.env_cfg`` runs ``load_dotenv()`` at import time, so a
    developer ``.env`` (e.g. ``ENABLE_HYBRID=true`` / ``INFERENCE_PROVIDER=vllm``)
    leaks into ``os.environ`` and makes every ``RAG()`` resolve
    ``enable_hybrid=True`` — at which point ``ingest_docs`` runs the real
    ``probe_sparse_endpoint()`` network call against ``vllm-router``. CI has no
    ``.env``, so the suite must see the same unset baseline locally. Tests that
    exercise hybrid behavior opt in explicitly via ``monkeypatch.setenv`` or by
    setting ``enable_hybrid`` on the instance.

    Uses a private ``MonkeyPatch`` instead of the shared ``monkeypatch``
    fixture: requesting the shared one here reorders it before every module's
    own autouse fixtures, so their teardowns would run against still-patched
    state (this broke ``tests/test_translate_client.py``'s cache-clear
    teardown).

    Yields:
        None.
    """
    mp = pytest.MonkeyPatch()
    for name in ("ENABLE_HYBRID", "SPARSE_API_BASE", "INFERENCE_PROVIDER", "SPARSE_MODEL"):
        mp.delenv(name, raising=False)
    yield
    mp.undo()


def run_ingest(
    client: TestClient,
    collection: str,
    headers: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Queue an ingest job via ``/ingest/finalize`` and wait for it to finish.

    Ingestion is now a server-owned job (``docint/core/jobs.py``): a
    ``POST /ingest/finalize`` only queues the run and returns ``202 {job_id}``;
    the caller polls ``GET /ingest/jobs/{job_id}`` for the terminal snapshot.
    This is the shared two-step helper for every test suite that used to
    consume the old ``/ingest/finalize`` SSE stream directly.

    Args:
        client (TestClient): The API test client. Must be entered as a context
            manager (``with TestClient(app) as client:``) — ingest jobs run as
            a detached ``asyncio`` task that a bare, non-context-managed
            ``TestClient`` would orphan the instant the queuing request
            returns (each such call opens and tears down its own throwaway
            event loop; see starlette's ``TestClient._portal_factory``).
        collection (str): Logical collection name to finalize.
        headers (dict[str, str] | None): Request headers (e.g. ``X-Auth-User``).
        extra (dict[str, Any] | None): Extra ``IngestIn`` fields for the
            finalize payload (e.g. ``{"ner": True}``), merged over
            ``{"collection": collection}``.

    Returns:
        dict[str, Any]: The job's terminal snapshot (``status`` is
        ``"completed"`` or ``"failed"``).

    Raises:
        AssertionError: If finalize does not return 202, or the job does not
            reach a terminal status within the poll budget.
    """
    payload = {"collection": collection, **(extra or {})}
    res = client.post("/ingest/finalize", json=payload, headers=headers)
    assert res.status_code == 202, res.text
    job_id = res.json()["job_id"]
    for _ in range(200):
        snapshot = client.get(f"/ingest/jobs/{job_id}", headers=headers).json()
        if snapshot["status"] in ("completed", "failed"):
            return snapshot
        time.sleep(0.01)
    raise AssertionError("ingest job did not finish")
