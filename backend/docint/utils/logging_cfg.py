from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

DEFAULT_ROTATION = "5 MB"
DEFAULT_RETENTION = 3


def _resolve_log_path(
    default_log_path: str | os.PathLike[str] | None,
) -> Path:
    """
    Resolve the log file path based on environment variable or default.

    Args:
        default_log_path (str | os.PathLike[str] | None): The default log file path.

    Returns:
        Path: The resolved log file path.
    """

    if default_log_path is None:
        default_log_path = Path(__file__).resolve().parents[2] / ".logs" / "docint.log"

    env_log_path = os.getenv("LOG_PATH")
    log_path = Path(env_log_path) if env_log_path else Path(default_log_path)

    if log_path.is_dir() or not log_path.suffix:
        log_path = log_path / "docint.log"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def setup_logging(
    default_log_path: str | os.PathLike[str] | None = None,
    *,
    rotation: str | int = DEFAULT_ROTATION,
    retention: str | int | None = DEFAULT_RETENTION,
    level: str = "INFO",
    backtrace: bool = False,
    diagnose: bool = False,
) -> Path:
    """
    Set up logging for the application.

    Args:
        default_log_path (str | os.PathLike[str] | None, optional): The default log file path. Defaults to None.
        rotation (str | int, optional): The log file rotation policy. Defaults to DEFAULT_ROTATION.
        retention (str | int | None, optional): The log file retention policy. Defaults to DEFAULT_RETENTION.
        level (str, optional): The log level. Defaults to "INFO".
        backtrace (bool, optional): Whether to include backtrace information. Defaults to False.
        diagnose (bool, optional): Whether to include diagnostic information. Defaults to False.

    Returns:
        Path: The path to the log file.
    """

    log_path = _resolve_log_path(default_log_path)

    logger.remove()

    logger.add(
        sys.stderr,
        level=level,
        backtrace=backtrace,
        diagnose=diagnose,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name} | {message}",
    )

    logger.add(
        log_path,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        level="DEBUG",
        backtrace=backtrace,
        diagnose=diagnose,
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name} | {message}",
    )

    logger.info("DocInt logging initialised")
    return log_path
