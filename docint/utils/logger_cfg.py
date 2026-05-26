"""Loguru sink configuration shared across the backend, UI, and CLIs.

A single stderr sink is configured; the container logging driver owns
log retention and rotation (see ``docker/compose.yaml``).
"""

from __future__ import annotations

import os
import sys

from loguru import logger


def init_logger(
    backtrace: bool = False,
    diagnose: bool = False,
) -> None:
    """Set up logging for the application.

    Installs a single stderr sink. ``LOG_LEVEL`` selects the minimum
    level (default ``INFO``).

    Args:
        backtrace (bool, optional): Whether to include backtrace information. Defaults to False.
        diagnose (bool, optional): Whether to include diagnostic information. Defaults to False.
    """
    level = os.getenv("LOG_LEVEL", "INFO")

    logger.remove()

    logger.add(
        sink=sys.stderr,
        level=level,
        backtrace=backtrace,
        diagnose=diagnose,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name} | {message}",
    )
