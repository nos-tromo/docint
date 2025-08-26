import logging.config
from pathlib import Path

import yaml


def setup_logging(
    log_dir: str = ".log",
    cfg_dir: str = "logging",
    cfg_file: str = "logging.yaml",
) -> None:
    """
    Set up logging configuration for the application.

    Args:
        log_dir (str): Directory where log files will be stored. Defaults to ".log".
        cfg_dir (str): Directory where the logging configuration file is located. Defaults to "logging".
        cfg_file (str): Name of the logging configuration file. Defaults to "logging.yaml".

    Raises:
        FileNotFoundError: If the logging configuration file does not exist.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(__file__).parent / cfg_dir / cfg_file
    if not cfg_path.exists():
        raise FileNotFoundError("Logging configuration file not found: %s", cfg_path)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    logging.config.dictConfig(cfg)

    logging.info("Logging configuration loaded from %s", cfg_path)
    logging.debug("Logging configuration: %s", cfg)
    logging.getLogger("wizard").info("Wizard logging initialized.")
