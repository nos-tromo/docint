"""Tests for logging configuration: level plumbing, the stdlib bridge, noise."""

from __future__ import annotations

import http.client
import io
import logging
import socket
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Iterator

import pytest
import uvicorn
from _pytest.logging import LogCaptureFixture
from loguru import logger

from docint.utils.env_cfg import load_logging_env
from docint.utils.logger_cfg import (
    _BRIDGED_LOGGERS,
    _HealthcheckAccessFilter,
    _install_stdlib_bridge,
    _QueryStringAccessFilter,
    init_logger,
)


@pytest.fixture
def restored_stdlib_loggers() -> Iterator[None]:
    """Restore the bridged stdlib loggers after a test mutates them.

    ``_install_stdlib_bridge`` replaces handlers and filters on
    process-wide loggers. Without this, one test's bridge leaks into every
    test that runs after it.

    Yields:
        None.
    """
    saved = [
        (name, logging.getLogger(name).handlers[:], logging.getLogger(name).filters[:], logging.getLogger(name).level)
        for name in _BRIDGED_LOGGERS
    ]
    try:
        yield
    finally:
        for name, handlers, filters, level in saved:
            std_logger = logging.getLogger(name)
            std_logger.handlers = handlers
            std_logger.filters = filters
            std_logger.setLevel(level)


def _access_record(client: str, path: str, status: int) -> logging.LogRecord:
    """Build a record shaped like uvicorn's access log.

    Args:
        client (str): Client address, as uvicorn renders it.
        path (str): Full request path.
        status (int): HTTP status code.

    Returns:
        logging.LogRecord: A record whose ``args`` match uvicorn's
        ``(client_addr, method, full_path, http_version, status)`` tuple.
    """
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg='%s - "%s %s HTTP/%s" %d',
        args=(client, "GET", path, "1.1", status),
        exc_info=None,
    )


# ---------------------------------------------------------------------------
# load_logging_env
# ---------------------------------------------------------------------------


def test_logging_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset env yields INFO and the 30s heartbeat the operator agreed on.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_PROGRESS_INTERVAL_S", raising=False)

    cfg = load_logging_env()

    assert cfg.level == "INFO"
    assert cfg.progress_interval_s == 30.0


def test_logging_env_reads_both_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both knobs are env-driven.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_PROGRESS_INTERVAL_S", "5")

    cfg = load_logging_env()

    assert cfg.level == "DEBUG"
    assert cfg.progress_interval_s == 5.0


def test_logging_env_allows_zero_to_disable_throttling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero is a meaningful value, not a missing one — it is the debug escape hatch.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("LOG_PROGRESS_INTERVAL_S", "0")

    assert load_logging_env().progress_interval_s == 0.0


@pytest.mark.parametrize("raw", ["", "abc", "-1", "nan"])
def test_logging_env_falls_back_on_unusable_interval(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """A typo in an operator's .env must not stop the backend booting.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        raw (str): An unusable ``LOG_PROGRESS_INTERVAL_S`` value.
    """
    monkeypatch.setenv("LOG_PROGRESS_INTERVAL_S", raw)

    assert load_logging_env().progress_interval_s == 30.0


# ---------------------------------------------------------------------------
# healthcheck suppression
# ---------------------------------------------------------------------------


def test_healthcheck_filter_drops_the_container_probe() -> None:
    """The compose healthcheck's own access line is noise, not signal."""
    assert _HealthcheckAccessFilter().filter(_access_record("127.0.0.1:53496", "/version", 200)) is False


@pytest.mark.parametrize(
    ("client", "path", "status", "why"),
    [
        ("127.0.0.1:53496", "/version", 500, "a failing healthcheck is real signal"),
        ("172.19.0.3:60230", "/version", 200, "a browser hitting /version is a real request"),
        ("127.0.0.1:53496", "/collections/list", 200, "only /version is probe noise"),
        ("127.0.0.1:53496", "/versionify", 200, "prefix match must not swallow other routes"),
    ],
)
def test_healthcheck_filter_keeps_everything_else(client: str, path: str, status: int, why: str) -> None:
    """Only a successful loopback probe of /version is dropped.

    Args:
        client (str): Client address on the record.
        path (str): Request path on the record.
        status (int): HTTP status on the record.
        why (str): Why this record must survive.
    """
    assert _HealthcheckAccessFilter().filter(_access_record(client, path, status)) is True, why


@pytest.mark.parametrize(
    "args",
    [
        None,
        (),
        ("127.0.0.1", "GET", "/version"),
        ("127.0.0.1", "GET", "/version", "1.1", "not-a-status"),
        {"client": "127.0.0.1"},
    ],
)
def test_healthcheck_filter_fails_open_on_unexpected_shapes(args: object) -> None:
    """An arg shape we do not recognise is kept, never silently dropped.

    A future uvicorn could change the tuple. Dropping records we failed to
    parse would lose real traffic invisibly; keeping them only costs noise.

    Args:
        args (object): An ``args`` payload the filter must not choke on.
    """
    record = logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname="", lineno=0, msg="x", args=None, exc_info=None
    )
    record.args = args  # pyrefly: ignore  # deliberately malformed

    assert _HealthcheckAccessFilter().filter(record) is True


# ---------------------------------------------------------------------------
# query-string stripping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/collections/ner/sources?entity_text=Example%20Person", "/collections/ner/sources"),
        ("/collections/ner/search?q=example+term&limit=50", "/collections/ner/search"),
        ("/collections/list", "/collections/list"),
    ],
)
def test_query_string_filter_keeps_only_the_path(path: str, expected: str) -> None:
    """The line keeps its route and status and loses what was searched for.

    Args:
        path (str): The ``path?query`` uvicorn put on the record.
        expected (str): The path the line must show.
    """
    record = _access_record("172.19.0.3:60230", path, 200)

    assert _QueryStringAccessFilter().filter(record) is True
    assert record.getMessage() == f'172.19.0.3:60230 - "GET {expected} HTTP/1.1" 200'


@pytest.mark.parametrize(
    "args",
    [
        None,
        (),
        ("127.0.0.1", "GET", "/version"),
        ("127.0.0.1", "GET", 404, "1.1", 200),
        {"client": "127.0.0.1"},
    ],
)
def test_query_string_filter_passes_unexpected_shapes_through(args: object) -> None:
    """A shape it does not recognise is kept as it is, never dropped.

    Args:
        args (object): An ``args`` payload the filter must not choke on.
    """
    record = logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname="", lineno=0, msg="x", args=None, exc_info=None
    )
    record.args = args  # pyrefly: ignore  # deliberately malformed

    assert _QueryStringAccessFilter().filter(record) is True
    assert record.args is args


# ---------------------------------------------------------------------------
# the stdlib -> loguru bridge
# ---------------------------------------------------------------------------


def test_bridge_routes_uvicorn_records_into_loguru(
    loguru_caplog_info: LogCaptureFixture, restored_stdlib_loggers: None
) -> None:
    """A uvicorn record reaches the loguru sink, so one format covers both.

    Args:
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
        restored_stdlib_loggers (None): Restores the mutated stdlib loggers.
    """
    _install_stdlib_bridge("INFO")

    logging.getLogger("uvicorn.error").info("Application startup complete.")

    assert any("Application startup complete." in m for m in loguru_caplog_info.messages)


def test_bridge_applies_the_healthcheck_filter_end_to_end(
    loguru_caplog_info: LogCaptureFixture, restored_stdlib_loggers: None
) -> None:
    """Wiring check: the probe is dropped, a real request survives.

    Args:
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
        restored_stdlib_loggers (None): Restores the mutated stdlib loggers.
    """
    _install_stdlib_bridge("INFO")
    access = logging.getLogger("uvicorn.access")

    access.handle(_access_record("127.0.0.1:53496", "/version", 200))
    access.handle(_access_record("172.19.0.3:60230", "/collections/list", 200))

    combined = "\n".join(loguru_caplog_info.messages)
    assert "/version" not in combined
    assert "/collections/list" in combined


def test_bridge_strips_the_query_string_end_to_end(
    loguru_caplog_info: LogCaptureFixture, restored_stdlib_loggers: None
) -> None:
    """Wiring check: the line that reaches the loguru sink shows the path alone.

    Args:
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
        restored_stdlib_loggers (None): Restores the mutated stdlib loggers.
    """
    _install_stdlib_bridge("INFO")

    logging.getLogger("uvicorn.access").handle(
        _access_record("172.19.0.3:60230", "/collections/ner/sources?entity_text=Example%20Person", 200)
    )

    combined = "\n".join(loguru_caplog_info.messages)
    assert '"GET /collections/ner/sources HTTP/1.1" 200' in combined
    assert "entity_text" not in combined
    assert "Example" not in combined


def test_bridge_decides_the_healthcheck_on_the_path_the_line_would_show(
    loguru_caplog_info: LogCaptureFixture, restored_stdlib_loggers: None
) -> None:
    """The query is cut first, so no kept line reads exactly like a dropped probe.

    Args:
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
        restored_stdlib_loggers (None): Restores the mutated stdlib loggers.
    """
    _install_stdlib_bridge("INFO")

    logging.getLogger("uvicorn.access").handle(_access_record("127.0.0.1:53496", "/version?probe=1", 200))

    assert not any("/version" in m for m in loguru_caplog_info.messages)


async def _no_content_app(scope: object, receive: object, send: Callable[[dict[str, object]], Awaitable[None]]) -> None:
    """Answer every HTTP request with an empty 204, whatever its path.

    Args:
        scope (object): The ASGI connection scope (unused).
        receive (object): The ASGI receive channel (unused).
        send (Callable[[dict[str, object]], Awaitable[None]]): The ASGI send channel.
    """
    await send({"type": "http.response.start", "status": 204, "headers": []})
    await send({"type": "http.response.body", "body": b""})


def test_a_real_uvicorn_access_line_carries_no_query_string(
    loguru_caplog_info: LogCaptureFixture, restored_stdlib_loggers: None
) -> None:
    """Pin the filter to the record uvicorn actually emits, not a hand-built one.

    Every other test here builds its record in uvicorn's argument shape. Were
    a uvicorn release to change that shape, those tests would still pass
    while the filter silently stopped matching and query strings returned to
    the log. ``log_config=None`` keeps uvicorn from replacing the bridge.

    Args:
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
        restored_stdlib_loggers (None): Restores the mutated stdlib loggers.
    """
    _install_stdlib_bridge("INFO")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(_no_content_app, lifespan="off", log_config=None))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "uvicorn did not start"
            time.sleep(0.01)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request("GET", "/collections/ner/sources?entity_text=Example%20Person")
            assert conn.getresponse().status == 204
        finally:
            conn.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()

    access_lines = [m for m in loguru_caplog_info.messages if '"GET ' in m]
    assert len(access_lines) == 1
    assert access_lines[0].endswith('"GET /collections/ner/sources HTTP/1.1" 204')
    combined = "\n".join(loguru_caplog_info.messages)
    assert "entity_text" not in combined
    assert "Example" not in combined


def test_bridge_is_idempotent(restored_stdlib_loggers: None) -> None:
    """init_logger runs once per CLI and again at api import; no doubling.

    Args:
        restored_stdlib_loggers (None): Restores the mutated stdlib loggers.
    """
    _install_stdlib_bridge("INFO")
    _install_stdlib_bridge("INFO")

    access = logging.getLogger("uvicorn.access")
    assert len(access.handlers) == 1
    assert sum(isinstance(f, _QueryStringAccessFilter) for f in access.filters) == 1
    assert sum(isinstance(f, _HealthcheckAccessFilter) for f in access.filters) == 1


def test_bridge_does_not_touch_the_root_logger(restored_stdlib_loggers: None) -> None:
    """Bridging root would pull in httpx/qdrant/llama_index at unknown volume.

    Args:
        restored_stdlib_loggers (None): Restores the mutated stdlib loggers.
    """
    before = logging.getLogger().handlers[:]

    _install_stdlib_bridge("INFO")

    assert logging.getLogger().handlers == before


# ---------------------------------------------------------------------------
# init ordering
# ---------------------------------------------------------------------------


def test_logger_follows_stderr_after_a_redirect_is_closed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    restored_stdlib_loggers: None,
) -> None:
    """Later log records reach stderr after the stream used at setup has closed."""
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    redirected = io.StringIO()
    try:
        with monkeypatch.context() as redirect:
            redirect.setattr(sys, "stderr", redirected)
            init_logger()
            logger.info("during redirection")
            assert "during redirection" in redirected.getvalue()
        redirected.close()

        logger.info("after redirection")

        captured = capsys.readouterr()
        assert "Logging error" not in captured.err
        assert "after redirection" in captured.err
    finally:
        redirected.close()
        logger.remove()


def test_importing_env_cfg_emits_no_log_record() -> None:
    """The offline vars apply at import; announcing them there escapes the sink.

    ``env_cfg`` is imported long before ``init_logger`` installs the
    configured sink, so a log call at its module scope prints through
    loguru's *default* handler — a different format from every other line
    in the run. The application must be silent.
    """
    import docint.utils.env_cfg as env_cfg

    seen: list[str] = []
    sink_id = logger.add(lambda message: seen.append(str(message)), level="DEBUG", format="{message}")
    try:
        assert env_cfg._apply_offline_env() in (True, False)
    finally:
        logger.remove(sink_id)

    assert seen == [], f"applying offline env must not log; got {seen}"


def test_set_offline_env_still_announces_the_mode(loguru_caplog_info: LogCaptureFixture) -> None:
    """The airgap line is useful, so it must survive — just later, and formatted.

    Args:
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
    """
    from docint.utils.env_cfg import set_offline_env

    set_offline_env()

    assert any("offline mode" in m or "online mode" in m for m in loguru_caplog_info.messages)
