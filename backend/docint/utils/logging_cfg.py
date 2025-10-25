import logging
import os
from collections.abc import Iterable
from logging import Handler
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _remove_handlers(logger: logging.Logger, handlers: Iterable[Handler]) -> None:
    """Remove specific handlers from a logger."""

    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()


def setup_logging(
    default_log_path: str | os.PathLike[str] | None = None,
    *,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
    force: bool = True,
) -> Path:
    """
    Configure application logging with rotating file and console handlers.

    Args:
        default_log_path: Default path to the log file. Falls back to
            ``<project_root>/.logs/docint.log`` when ``None``.
        max_bytes: Maximum size of each log file before rotation.
        backup_count: Number of rotated backups to keep.
        force: Whether to clear existing handlers before applying the
            configuration. This is helpful when the runtime (e.g. uvicorn)
            installs handlers ahead of time.

    Returns:
        Path to the configured log file.
    """

    if default_log_path is None:
        default_log_path = (
            Path(__file__).resolve().parents[2] / ".logs" / "docint.log"
        )

    env_log_path = os.getenv("LOG_PATH")
    log_path = Path(env_log_path) if env_log_path else Path(default_log_path)

    # Allow callers to pass a directory path instead of a full filename.
    if log_path.is_dir() or not log_path.suffix:
        log_path = log_path / "docint.log"

    log_dir = log_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    formatter = logging.Formatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    root_logger = logging.getLogger()

    if force:
        for existing in list(root_logger.handlers):
            root_logger.removeHandler(existing)
    else:
        # Remove handlers that duplicate the ones we're about to add to avoid
        # stacking them when ``setup_logging`` is called multiple times.
        duplicate_handlers: list[Handler] = []
        for existing in root_logger.handlers:
            if isinstance(existing, RotatingFileHandler):
                if getattr(existing, "baseFilename", None) == str(log_path):
                    duplicate_handlers.append(existing)
            elif isinstance(existing, type(console_handler)):
                duplicate_handlers.append(existing)

        if duplicate_handlers:
            _remove_handlers(root_logger, duplicate_handlers)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger("docint").info("docint logging initialized.")

    return log_path
