from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()
APP_NAME = os.getenv("APP_NAME", "docint")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_ROTATION = os.getenv("LOG_ROTATION", "5 MB")
LOG_RETENTION = os.getenv("LOG_RETENTION", "3")


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
        default_log_path = (
            Path(__file__).resolve().parents[2] / ".logs" / f"{APP_NAME}.log"
        )

    env_log_path = os.getenv("LOG_PATH")
    log_path = Path(env_log_path) if env_log_path else Path(default_log_path)

    if log_path.is_dir() or not log_path.suffix:
        log_path = log_path / f"{APP_NAME}.log"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def setup_logging(
    default_log_path: str | os.PathLike[str] | None = None,
    *,
    encoding="utf-8",
    level: str | None = None,
    rotation: str | int | None = None,
    retention: str | int | None = None,
    backtrace: bool = False,
    diagnose: bool = False,
) -> Path:
    """
    Set up logging for the application.

    Args:
        default_log_path (str | os.PathLike[str] | None, optional): The default log file path. Defaults to None.
        encoding (str, optional): The log file encoding. Defaults to "utf-8".
        level (str, optional): The log level. Defaults to "INFO".
        rotation (str | int, optional): The log file rotation policy. Defaults to LOG_ROTATION.
        retention (str | int | None, optional): The log file retention policy. Defaults to LOG_RETENTION.
        backtrace (bool, optional): Whether to include backtrace information. Defaults to False.
        diagnose (bool, optional): Whether to include diagnostic information. Defaults to False.

    Returns:
        Path: The path to the log file.
    """
    log_path = _resolve_log_path(default_log_path)

    if level is None:
        level = LOG_LEVEL
    if rotation is None:
        rotation = LOG_ROTATION
    if retention is None:
        retention = LOG_RETENTION

    retention = retention if isinstance(retention, int) else int(retention)

    logger.remove()

    logger.add(
        sink=sys.stderr,
        level=level,
        backtrace=backtrace,
        diagnose=diagnose,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name} | {message}",
    )

    logger.add(
        sink=log_path,
        rotation=rotation,
        retention=retention,
        encoding=encoding,
        level="DEBUG",
        backtrace=True,
        diagnose=diagnose,
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {line:<4} | {name} | {message}",
    )

    return log_path
