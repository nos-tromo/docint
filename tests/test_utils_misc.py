import hashlib
import json
from pathlib import Path

import pytest

from docint.utils.clean_text import basic_clean
from docint.utils.env_cfg import load_path_env, load_summary_env
from docint.utils.hashing import compute_file_hash, ensure_file_hash
from docint.utils.logging_cfg import setup_logging
from loguru import logger


def test_basic_clean_normalizes_whitespace() -> None:
    """Test that basic_clean normalizes whitespace and newlines."""
    text = "Line 1\r\n\r\nLine 2\n\n\nLine 3   \n"
    cleaned = basic_clean(text)
    assert cleaned == "Line 1\nLine 2\nLine 3"


def test_compute_file_hash(tmp_path: Path) -> None:
    """Test that compute_file_hash correctly calculates the SHA256 hash of a file.

    Args:
        tmp_path (Path): The temporary path fixture.
    """
    file = tmp_path / "sample.txt"
    file.write_text("content")
    expected = hashlib.sha256(b"content").hexdigest()
    assert compute_file_hash(file) == expected


def test_compute_file_hash_missing(tmp_path: Path) -> None:
    """Test that compute_file_hash raises FileNotFoundError for missing files.

    Args:
        tmp_path (Path): The temporary path fixture.
    """
    with pytest.raises(FileNotFoundError):
        compute_file_hash(tmp_path / "missing.txt")


def test_ensure_file_hash_mutates_metadata(tmp_path: Path) -> None:
    """Test that ensure_file_hash adds the file hash to the metadata dictionary.

    Args:
        tmp_path (Path): The temporary path fixture.
    """
    file = tmp_path / "data.json"
    file.write_text(json.dumps({"key": "value"}))
    metadata: dict[str, object] = {}
    digest = ensure_file_hash(metadata, path=file)
    assert metadata["file_hash"] == digest


def test_ensure_file_hash_requires_inputs() -> None:
    """Test that ensure_file_hash raises ValueError if neither path nor file_hash is provided."""
    with pytest.raises(ValueError):
        ensure_file_hash({}, path=None, file_hash=None)


def test_path_config_artifacts_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """PathConfig.artifacts should default to ``<project_root>/artifacts``.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture to override environment.
    """
    monkeypatch.delenv("PIPELINE_ARTIFACTS_DIR", raising=False)
    cfg = load_path_env()
    assert cfg.artifacts.name == "artifacts"
    assert cfg.artifacts.is_absolute()


def test_path_config_artifacts_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PIPELINE_ARTIFACTS_DIR env var should override the default artifacts path.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture to override environment.
        tmp_path (Path): Temporary path fixture.
    """
    custom = tmp_path / "my-artifacts"
    monkeypatch.setenv("PIPELINE_ARTIFACTS_DIR", str(custom))
    cfg = load_path_env()
    assert cfg.artifacts == custom


def test_setup_logging_respects_env_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """setup_logging should honor LOG_PATH and create the log file.

    Args:
        tmp_path (Path): Temporary directory.
        monkeypatch (pytest.MonkeyPatch): Fixture to override environment.
    """
    log_file = tmp_path / "logs" / "docint.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOG_PATH", str(log_file))

    resolved = setup_logging(rotation="1 MB", retention=1)
    logger.debug("create log entry for file")

    assert resolved == log_file
    assert log_file.exists()


def test_load_summary_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Summary env loader should use documented defaults.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture to clear environment variables.
    """
    monkeypatch.delenv("SUMMARY_COVERAGE_TARGET", raising=False)
    monkeypatch.delenv("SUMMARY_MAX_DOCS", raising=False)
    monkeypatch.delenv("SUMMARY_PER_DOC_TOP_K", raising=False)
    monkeypatch.delenv("SUMMARY_FINAL_SOURCE_CAP", raising=False)

    cfg = load_summary_env()

    assert cfg.coverage_target == 0.70
    assert cfg.max_docs == 30
    assert cfg.per_doc_top_k == 4
    assert cfg.final_source_cap == 24


def test_load_summary_env_clamps_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Summary env loader should parse numeric fields and clamp invalid ranges.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture to set environment variables.
    """
    monkeypatch.setenv("SUMMARY_COVERAGE_TARGET", "1.5")
    monkeypatch.setenv("SUMMARY_MAX_DOCS", "12")
    monkeypatch.setenv("SUMMARY_PER_DOC_TOP_K", "6")
    monkeypatch.setenv("SUMMARY_FINAL_SOURCE_CAP", "10")

    cfg = load_summary_env()

    assert cfg.coverage_target == 1.0
    assert cfg.max_docs == 12
    assert cfg.per_doc_top_k == 6
    assert cfg.final_source_cap == 10
