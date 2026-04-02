"""Tests for miscellaneous utilities (hashing, cleaning, env config)."""

import hashlib
import json
from pathlib import Path

import pytest

from docint.utils.clean_text import basic_clean
from docint.utils.env_cfg import (
    load_frontend_env,
    load_hate_speech_env,
    load_ingestion_env,
    load_model_env,
    load_openai_env,
    load_path_env,
    load_retrieval_env,
    load_runtime_env,
    load_session_env,
    load_summary_env,
)
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
    monkeypatch.delenv("SUMMARY_SOCIAL_CHUNKING_ENABLED", raising=False)
    monkeypatch.delenv("SUMMARY_SOCIAL_CANDIDATE_POOL", raising=False)
    monkeypatch.delenv("SUMMARY_SOCIAL_DIVERSITY_LIMIT", raising=False)

    cfg = load_summary_env()

    assert cfg.coverage_target == 0.70
    assert cfg.max_docs == 30
    assert cfg.per_doc_top_k == 4
    assert cfg.final_source_cap == 24
    assert cfg.social_chunking_enabled is True
    assert cfg.social_candidate_pool == 48
    assert cfg.social_diversity_limit == 2


def test_load_summary_env_clamps_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Summary env loader should parse numeric fields and clamp invalid ranges.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture to set environment variables.
    """
    monkeypatch.setenv("SUMMARY_COVERAGE_TARGET", "1.5")
    monkeypatch.setenv("SUMMARY_MAX_DOCS", "12")
    monkeypatch.setenv("SUMMARY_PER_DOC_TOP_K", "6")
    monkeypatch.setenv("SUMMARY_FINAL_SOURCE_CAP", "10")
    monkeypatch.setenv("SUMMARY_SOCIAL_CHUNKING_ENABLED", "false")
    monkeypatch.setenv("SUMMARY_SOCIAL_CANDIDATE_POOL", "64")
    monkeypatch.setenv("SUMMARY_SOCIAL_DIVERSITY_LIMIT", "3")

    cfg = load_summary_env()

    assert cfg.coverage_target == 1.0
    assert cfg.max_docs == 12
    assert cfg.per_doc_top_k == 6
    assert cfg.final_source_cap == 10
    assert cfg.social_chunking_enabled is False
    assert cfg.social_candidate_pool == 64
    assert cfg.social_diversity_limit == 3


def test_load_frontend_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Frontend env loader should use the documented collection timeout default.

    Args:
        monkeypatch: Fixture to clear environment variables.
    """
    monkeypatch.delenv("FRONTEND_COLLECTION_TIMEOUT", raising=False)
    cfg = load_frontend_env()
    assert cfg.collection_timeout == 120


def test_load_frontend_env_reads_collection_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frontend env loader should parse the collection timeout override.

    Args:
        monkeypatch: Fixture to set environment variables.
    """
    monkeypatch.setenv("FRONTEND_COLLECTION_TIMEOUT", "90")
    cfg = load_frontend_env()
    assert cfg.collection_timeout == 90


def test_load_retrieval_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrieval env loader should use documented defaults.

    Args:
        monkeypatch: Fixture to clear environment variables.
    """
    monkeypatch.delenv("RERANK_USE_FP16", raising=False)
    monkeypatch.delenv("RETRIEVE_TOP_K", raising=False)
    monkeypatch.delenv("CHAT_RESPONSE_MODE", raising=False)
    monkeypatch.delenv("RETRIEVAL_VECTOR_QUERY_MODE", raising=False)
    monkeypatch.delenv("RETRIEVAL_HYBRID_ALPHA", raising=False)
    monkeypatch.delenv("RETRIEVAL_SPARSE_TOP_K", raising=False)
    monkeypatch.delenv("RETRIEVAL_HYBRID_TOP_K", raising=False)
    monkeypatch.delenv("PARENT_CONTEXT_RETRIEVAL_ENABLED", raising=False)

    cfg = load_retrieval_env()

    assert cfg.rerank_use_fp16 is False
    assert cfg.retrieve_top_k == 20
    assert cfg.chat_response_mode == "auto"
    assert cfg.vector_store_query_mode == "auto"
    assert cfg.hybrid_alpha == 0.5
    assert cfg.sparse_top_k == 20
    assert cfg.hybrid_top_k == 20
    assert cfg.parent_context_enabled is True


def test_load_retrieval_env_parses_chat_response_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrieval env loader should clamp unknown chat modes to ``auto``.

    Args:
        monkeypatch: Fixture to set environment variables.
    """
    monkeypatch.setenv("RERANK_USE_FP16", "true")
    monkeypatch.setenv("RETRIEVE_TOP_K", "11")
    monkeypatch.setenv("CHAT_RESPONSE_MODE", "refine")
    monkeypatch.setenv("RETRIEVAL_VECTOR_QUERY_MODE", "hybrid")
    monkeypatch.setenv("RETRIEVAL_HYBRID_ALPHA", "0.25")
    monkeypatch.setenv("RETRIEVAL_SPARSE_TOP_K", "17")
    monkeypatch.setenv("RETRIEVAL_HYBRID_TOP_K", "9")
    monkeypatch.setenv("PARENT_CONTEXT_RETRIEVAL_ENABLED", "false")

    cfg = load_retrieval_env()

    assert cfg.rerank_use_fp16 is True
    assert cfg.retrieve_top_k == 11
    assert cfg.chat_response_mode == "refine"
    assert cfg.vector_store_query_mode == "hybrid"
    assert cfg.hybrid_alpha == 0.25
    assert cfg.sparse_top_k == 17
    assert cfg.hybrid_top_k == 9
    assert cfg.parent_context_enabled is False


def test_load_openai_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI env loader should default thinking to disabled.

    Args:
        monkeypatch: Fixture to clear environment variables.
    """
    monkeypatch.delenv("INFERENCE_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_DIMENSIONS", raising=False)
    monkeypatch.delenv("OPENAI_ENABLE_THINKING", raising=False)
    monkeypatch.delenv("OPENAI_THINKING_EFFORT", raising=False)
    monkeypatch.delenv("OPENAI_CTX_WINDOW", raising=False)
    monkeypatch.delenv("CHAT_MAX_MODEL_LEN", raising=False)

    cfg = load_openai_env()

    assert cfg.dimensions is None
    assert cfg.ctx_window == 4096
    assert cfg.thinking_enabled is False
    assert cfg.thinking_effort == "medium"


def test_load_openai_env_accepts_vllm_and_dimensions_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI env loader should accept vLLM and parse embedding dimensions.

    Args:
        monkeypatch: Fixture to set environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    monkeypatch.setenv("OPENAI_DIMENSIONS", "1024")
    monkeypatch.delenv("OPENAI_CTX_WINDOW", raising=False)
    monkeypatch.delenv("CHAT_MAX_MODEL_LEN", raising=False)

    cfg = load_openai_env()

    assert cfg.inference_provider == "vllm"
    assert cfg.ctx_window == 8192
    assert cfg.dimensions == 1024


def test_load_openai_env_uses_chat_max_model_len_for_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """vLLM context window should follow CHAT_MAX_MODEL_LEN when not overridden.

    Args:
        monkeypatch: Fixture to set environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    monkeypatch.setenv("CHAT_MAX_MODEL_LEN", "16384")
    monkeypatch.delenv("OPENAI_CTX_WINDOW", raising=False)

    cfg = load_openai_env()

    assert cfg.ctx_window == 16384


def test_load_openai_env_prefers_explicit_ctx_window_over_vllm_model_len(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENAI_CTX_WINDOW should override the inferred vLLM model length.

    Args:
        monkeypatch: Fixture to set environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    monkeypatch.setenv("CHAT_MAX_MODEL_LEN", "16384")
    monkeypatch.setenv("OPENAI_CTX_WINDOW", "12288")

    cfg = load_openai_env()

    assert cfg.ctx_window == 12288


def test_load_openai_env_clamps_invalid_thinking_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI env loader should clamp unknown thinking efforts to ``medium``.

    Args:
        monkeypatch: Fixture to set environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_ENABLE_THINKING", "true")
    monkeypatch.setenv("OPENAI_THINKING_EFFORT", "unsupported")

    cfg = load_openai_env()

    assert cfg.thinking_enabled is True
    assert cfg.thinking_effort == "medium"


def test_load_session_env_defaults_to_docint_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session env loader should place the default sqlite file under docint home.

    Args:
        monkeypatch: Fixture to clear environment variables.
    """
    monkeypatch.delenv("SESSION_STORE", raising=False)
    monkeypatch.delenv("SESSIONS_DB_PATH", raising=False)
    cfg = load_session_env()
    assert (
        cfg.session_store == f"sqlite:///{Path.home() / 'docint' / 'sessions.sqlite3'}"
    )


def test_load_session_env_defaults_to_sessions_db_path_when_explicitly_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session env loader should use SESSIONS_DB_PATH as the exact sqlite file.

    Args:
        monkeypatch: Fixture to set environment variables.
    """
    monkeypatch.delenv("SESSION_STORE", raising=False)
    monkeypatch.setenv("SESSIONS_DB_PATH", "/tmp/docint-sessions.sqlite3")

    cfg = load_session_env()

    assert cfg.session_store == "sqlite:////tmp/docint-sessions.sqlite3"


def test_load_session_env_ignores_data_path_without_sessions_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session env loader should not derive sqlite storage from DATA_PATH.

    Args:
        monkeypatch: Fixture to set environment variables.
    """
    monkeypatch.delenv("SESSION_STORE", raising=False)
    monkeypatch.delenv("SESSIONS_DB_PATH", raising=False)
    monkeypatch.setenv("DATA_PATH", "/tmp/docint-data")

    cfg = load_session_env()

    assert (
        cfg.session_store == f"sqlite:///{Path.home() / 'docint' / 'sessions.sqlite3'}"
    )


def test_load_session_env_honors_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """SESSION_STORE env var should override the default sqlite location.

    Args:
        monkeypatch: Fixture to set environment variables.
    """
    monkeypatch.setenv("SESSION_STORE", "sqlite:////tmp/custom-sessions.db")
    cfg = load_session_env()
    assert cfg.session_store == "sqlite:////tmp/custom-sessions.db"


def test_load_runtime_env_defaults_to_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime env loader should default to automatic device selection.

    Args:
        monkeypatch: Fixture to clear environment variables.
    """
    monkeypatch.delenv("USE_DEVICE", raising=False)

    cfg = load_runtime_env()

    assert cfg.use_device == "auto"


def test_load_runtime_env_normalizes_supported_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime env loader should normalize supported device overrides.

    Args:
        monkeypatch: Fixture to set environment variables.
    """
    monkeypatch.setenv("USE_DEVICE", "CUDA:1")

    cfg = load_runtime_env()

    assert cfg.use_device == "cuda:1"


def test_load_model_env_reads_direct_text_and_vision_model_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model env loader should read direct model identifiers from env vars.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture to set environment variables.
    """
    monkeypatch.setenv("TEXT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("VISION_MODEL", "gpt-4.1-mini")

    cfg = load_model_env()

    assert cfg.text_model == "gpt-4o-mini"
    assert cfg.vision_model == "gpt-4.1-mini"


def test_load_model_env_uses_vllm_sparse_and_asr_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """vLLM model defaults should align sparse and ASR models with the profile.

    Args:
        monkeypatch: Fixture to override environment variables.
    """

    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    monkeypatch.setenv("EMBED_MODEL", "BAAI/bge-m3")
    monkeypatch.delenv("SPARSE_MODEL", raising=False)
    monkeypatch.delenv("WHISPER_MODEL", raising=False)

    cfg = load_model_env()

    assert cfg.sparse_model == "BAAI/bge-m3"
    assert cfg.whisper_model == "openai/whisper-large-v3-turbo"


def test_load_hate_speech_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default hate-speech config should be disabled with one worker.

    Args:
        monkeypatch: Fixture to clear environment variables.
    """
    monkeypatch.delenv("ENABLE_HATE_SPEECH_DETECTION", raising=False)
    monkeypatch.delenv("HATE_SPEECH_MAX_CHARS", raising=False)
    monkeypatch.delenv("HATE_SPEECH_MAX_WORKERS", raising=False)

    cfg = load_hate_speech_env()

    assert cfg.enabled is False
    assert cfg.max_chars == 2048
    assert cfg.max_workers == 1


def test_load_hate_speech_env_parses_max_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HATE_SPEECH_MAX_WORKERS env var should configure worker count.

    Args:
        monkeypatch: Fixture to set environment variables.
    """
    monkeypatch.setenv("ENABLE_HATE_SPEECH_DETECTION", "true")
    monkeypatch.setenv("HATE_SPEECH_MAX_CHARS", "512")
    monkeypatch.setenv("HATE_SPEECH_MAX_WORKERS", "4")

    cfg = load_hate_speech_env()

    assert cfg.enabled is True
    assert cfg.max_chars == 512
    assert cfg.max_workers == 4


def test_load_hate_speech_env_clamps_max_workers_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HATE_SPEECH_MAX_WORKERS should be clamped to at least 1.

    Args:
        monkeypatch: Fixture to set environment variables.
    """
    monkeypatch.setenv("HATE_SPEECH_MAX_WORKERS", "0")

    cfg = load_hate_speech_env()

    assert cfg.max_workers == 1


def test_load_ingestion_env_parses_docstore_retry_knobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ingestion env loader should parse and clamp docstore retry settings.

    Args:
        monkeypatch: Fixture to set environment variables.
    """
    monkeypatch.setenv("DOCSTORE_MAX_RETRIES", "7")
    monkeypatch.setenv("DOCSTORE_RETRY_BACKOFF_SECONDS", "0.75")
    monkeypatch.setenv("DOCSTORE_RETRY_BACKOFF_MAX_SECONDS", "5.0")
    monkeypatch.setenv("INGEST_BENCHMARK_ENABLED", "true")

    cfg = load_ingestion_env()

    assert cfg.docstore_max_retries == 7
    assert cfg.docstore_retry_backoff_seconds == 0.75
    assert cfg.docstore_retry_backoff_max_seconds == 5.0
    assert cfg.ingest_benchmark_enabled is True


def test_load_ingestion_env_clamps_negative_docstore_retry_knobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative docstore retry knobs should be clamped to zero.

    Args:
        monkeypatch: Fixture to set environment variables.
    """
    monkeypatch.setenv("DOCSTORE_MAX_RETRIES", "-2")
    monkeypatch.setenv("DOCSTORE_RETRY_BACKOFF_SECONDS", "-1")
    monkeypatch.setenv("DOCSTORE_RETRY_BACKOFF_MAX_SECONDS", "-3")

    cfg = load_ingestion_env()

    assert cfg.docstore_max_retries == 0
    assert cfg.docstore_retry_backoff_seconds == 0.0
    assert cfg.docstore_retry_backoff_max_seconds == 0.0
