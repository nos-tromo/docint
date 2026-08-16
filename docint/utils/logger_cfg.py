"""Loguru sink configuration shared across the backend, UI, and CLIs.

A single stderr sink is configured; the container logging driver owns
log retention and rotation (see ``docker/compose.yaml``).

Uvicorn logs through the standard library, which loguru does not see, so
its records are re-dispatched into the same sink (see
:func:`_install_stdlib_bridge`). Without that, ``docker logs`` carries two
line formats on two different streams — uvicorn's access log on stdout,
everything else on stderr — buffered separately and therefore interleaved
out of order.
"""

from __future__ import annotations

import inspect
import logging
import sys

from loguru import logger
from typing_extensions import override

from docint.utils.env_cfg import load_logging_env

#: Stdlib loggers re-dispatched into loguru. Deliberately *not* the root
#: logger: httpx, qdrant_client, llama_index and transformers all log
#: through it, at a volume that cannot be predicted for an airgapped
#: deployment. Naming the loggers keeps the one-format win without the
#: flood. Add a name here if a specific library ever needs surfacing.
_BRIDGED_LOGGERS: tuple[str, ...] = ("uvicorn", "uvicorn.error", "uvicorn.access")

#: Path whose access-log records are dropped as healthcheck noise.
_HEALTHCHECK_PATH = "/version"

#: Loopback prefixes a container-local healthcheck probe originates from.
_LOOPBACK_PREFIXES: tuple[str, ...] = ("127.0.0.1", "::1", "localhost")


class _InterceptHandler(logging.Handler):
    """Forward standard-library log records into loguru.

    The stock recipe from loguru's documentation: map the stdlib level
    name onto a loguru level, walk back out of the ``logging`` frames so
    the reported source location is the real caller, and re-emit.
    """

    @override
    def emit(self, record: logging.LogRecord) -> None:
        """Re-emit one stdlib record through the loguru logger.

        Args:
            record (logging.LogRecord): The record handed over by the
                standard library.
        """
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class _HealthcheckAccessFilter(logging.Filter):
    """Drop the container healthcheck's own access-log records.

    ``docker/compose.yaml`` probes ``GET /version`` from inside the
    container every 30s (every 3s during ``start_period``), which on a
    measured run was 36 of 75 stdout lines — roughly half the log, and
    the share only grows the longer a container stays up.

    The predicate is deliberately narrow and **fails open**: anything it
    cannot positively identify as a successful loopback probe of
    ``/version`` is kept. A ``/version`` that returns 5xx is a real
    signal and survives, as does a request from any non-loopback client.
    """

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        """Return ``False`` only for a successful loopback ``/version`` probe.

        Args:
            record (logging.LogRecord): A ``uvicorn.access`` record, whose
                ``args`` uvicorn populates as
                ``(client_addr, method, full_path, http_version, status)``.

        Returns:
            bool: ``True`` to keep the record, ``False`` to drop it.
        """
        try:
            args = record.args
            if not isinstance(args, tuple) or len(args) != 5:
                return True
            client_addr, _method, full_path, _http_version, status = args
            if not str(client_addr).startswith(_LOOPBACK_PREFIXES):
                return True
            if str(full_path) != _HEALTHCHECK_PATH:
                return True
            return int(str(status)) >= 400
        except Exception:
            return True


def _install_stdlib_bridge(level: str) -> None:
    """Route the bridged stdlib loggers into the loguru sink.

    Idempotent: existing handlers are cleared before the intercept handler
    is attached, mirroring ``logger.remove()`` above. That matters because
    ``init_logger`` is called once per CLI entry point *and* at
    ``docint.core.api`` import time.

    Args:
        level (str): Minimum level for the bridged stdlib loggers.
    """
    handler = _InterceptHandler()
    healthcheck_filter = _HealthcheckAccessFilter()

    for name in _BRIDGED_LOGGERS:
        std_logger = logging.getLogger(name)
        std_logger.handlers = [handler]
        std_logger.propagate = False
        std_logger.filters = [f for f in std_logger.filters if not isinstance(f, _HealthcheckAccessFilter)]
        if name == "uvicorn.access":
            std_logger.addFilter(healthcheck_filter)
        try:
            std_logger.setLevel(level)
        except ValueError:
            std_logger.setLevel(logging.INFO)


def init_logger(
    backtrace: bool = False,
    diagnose: bool = False,
) -> None:
    """Set up logging for the application.

    Installs a single stderr sink and bridges uvicorn's stdlib loggers
    into it. ``LOG_LEVEL`` selects the minimum level (default ``INFO``).

    ``diagnose`` stays off by default on purpose: it dumps local variables
    into tracebacks, which on this codebase would put user queries and
    document text in the log.

    Args:
        backtrace (bool, optional): Whether to include backtrace information. Defaults to False.
        diagnose (bool, optional): Whether to include diagnostic information. Defaults to False.
    """
    level = load_logging_env().level

    logger.remove()

    logger.add(
        sink=sys.stderr,
        level=level,
        backtrace=backtrace,
        diagnose=diagnose,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name} | {message}",
    )

    _install_stdlib_bridge(level)
