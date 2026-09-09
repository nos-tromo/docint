"""Shared pytest configuration and fixtures for the docint test suite."""

import logging
import sys
import time
import types
from collections.abc import Iterator
from concurrent.futures import Future
from typing import Any

import pytest
from _pytest.logging import LogCaptureFixture
from fastapi.testclient import TestClient
from loguru import logger


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


@pytest.fixture(autouse=True, scope="session")
def _hermetic_session_store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Point the default session store at a throwaway SQLite file.

    The app lifespan initializes the session store eagerly at startup, so
    every ``TestClient(app)`` would otherwise create (or migrate!) the real
    ``~/docint/sessions.sqlite3`` on the developer's machine. ``SESSION_STORE``
    is cleared for the same reason a developer ``.env`` clears hybrid vars
    below: it would override this path entirely.

    Yields:
        None.
    """
    mp = pytest.MonkeyPatch()
    db_path = tmp_path_factory.mktemp("session-store") / "sessions.sqlite3"
    mp.setenv("SESSIONS_DB_PATH", str(db_path))
    mp.delenv("SESSION_STORE", raising=False)
    yield
    mp.undo()


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
    setting ``enable_hybrid`` on the instance. ``EMBED_API_BASE`` leaks the
    same way (a developer ``.env`` pointing at the ``embed-only`` shape makes
    the embed backend resolve away from ``openai_api_base``), so it is cleared
    here too. ``OPENAI_API_KEY`` likewise: with it leaked, llama-index's
    ``Settings.llm`` default resolves a real OpenAI client, so a test that
    accidentally reaches it passes locally but fails on keyless CI.

    Uses a private ``MonkeyPatch`` instead of the shared ``monkeypatch``
    fixture: requesting the shared one here reorders it before every module's
    own autouse fixtures, so their teardowns would run against still-patched
    state (this broke ``tests/test_translate_client.py``'s cache-clear
    teardown).

    Yields:
        None.
    """
    mp = pytest.MonkeyPatch()
    for name in (
        "ENABLE_HYBRID",
        "SPARSE_API_BASE",
        "INFERENCE_PROVIDER",
        "SPARSE_MODEL",
        "EMBED_API_BASE",
        "OPENAI_API_KEY",
    ):
        mp.delenv(name, raising=False)
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _hermetic_preprocess() -> Iterator[None]:
    """Keep the preprocessing pool from doing real work behind a test.

    Uploading a ``.jpg`` through the API, or ingesting a directory holding a
    PDF, would otherwise submit a real caption/OCR/layout task to a pool
    worker that then fails against no model endpoint and logs after the
    test's capture has closed. Tests of the hook and the prefetch re-enable
    them explicitly. A private ``MonkeyPatch`` for the same fixture-ordering
    reason as :func:`_hermetic_hybrid_env`.

    Yields:
        None.
    """
    mp = pytest.MonkeyPatch()
    mp.setenv("INGEST_PREPROCESS_ON_UPLOAD", "false")
    mp.setattr("docint.core.rag.prefetch_batch", lambda *args, **kwargs: 0)
    yield
    mp.undo()


@pytest.fixture
def recording_pool() -> Any:
    """A preprocessing pool double that runs every task inline and records the keys.

    Returns:
        A ``PreprocessPool`` subclass instance with ``keys`` (every submitted
        or run key, in order) and ``joins`` (every key waited on).
    """
    from typing_extensions import override

    from docint.core.ingest.preprocess import PreprocessPool

    class RecordingPool(PreprocessPool):
        """Inline, ordered stand-in for the real pool."""

        def __init__(self) -> None:
            """Start with empty logs and no executor."""
            self.keys: list[str] = []
            self.joins: list[str] = []
            self.results: dict[str, Any] = {}

        @override
        def submit(self, key: str, fn: Any) -> Any:
            """Run *fn* now (once per key) and return a settled future holding its value.

            A real :class:`~concurrent.futures.Future`, not a stand-in:
            ``run_all`` passes what this returns to ``as_completed``, which
            only accepts the real thing.
            """
            self.keys.append(key)
            if key not in self.results:
                self.results[key] = fn()
            future: Future[Any] = Future()
            future.set_result(self.results[key])
            return future

        @override
        def run(self, key: str, fn: Any) -> Any:
            """Return the value ``submit`` computed for *key*, running *fn* if it never did."""
            return self.submit(key, fn).result()

        @override
        def join(self, key: str) -> None:
            """Record the key waited on."""
            self.joins.append(key)

        @override
        def shutdown(self) -> None:
            """Nothing to stop."""

    return RecordingPool()


@pytest.fixture
def loguru_caplog(caplog: LogCaptureFixture) -> Iterator[LogCaptureFixture]:
    """Bridge loguru WARNING records into ``caplog`` for the duration of a test.

    Loguru bypasses the stdlib ``logging`` module, so ``caplog`` sees none of
    its records by default. Adding ``caplog.handler`` as a loguru sink is the
    whole bridge; asserting on ``caplog.text`` then works as it would for a
    stdlib logger.

    Args:
        caplog: The standard pytest log-capture fixture.

    Yields:
        The same ``caplog`` fixture, now populated with loguru-sourced
        records at WARNING level and above.
    """
    handler_id = logger.add(caplog.handler, level="WARNING", format="{message}")
    caplog.set_level(logging.WARNING)
    try:
        yield caplog
    finally:
        logger.remove(handler_id)


@pytest.fixture
def loguru_caplog_info(caplog: LogCaptureFixture) -> Iterator[LogCaptureFixture]:
    """Bridge loguru INFO records into ``caplog`` for the duration of a test.

    The INFO-level twin of :func:`loguru_caplog`, for the run-narrative lines
    (banner, per-file progress, run summary) that are emitted at INFO.

    Args:
        caplog: The standard pytest log-capture fixture.

    Yields:
        The same ``caplog`` fixture, now populated with loguru-sourced
        records at INFO level and above.
    """
    handler_id = logger.add(caplog.handler, level="INFO", format="{message}")
    caplog.set_level(logging.INFO)
    try:
        yield caplog
    finally:
        logger.remove(handler_id)


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
