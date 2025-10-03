import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    default_log_path: str = str(
        Path(__file__).resolve().parents[2] / ".log" / "docint.log"
    ),
    max_bytes=5_000_000,
    backup_count=3,
) -> None:
    """
    Set up logging with rotating file and console handlers.

    Args:
        default_log_path (str): Path to the log file (can be overridden by LOG_PATH env variable).
        max_bytes (int): Maximum size of each log file before rotation.
        backup_count (int): Number of rotated backups to keep.
    """
    log_path = os.getenv("LOG_PATH", default_log_path)
    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Root logger config
    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])
    logging.getLogger("docint").info("docint logging initialized.")
