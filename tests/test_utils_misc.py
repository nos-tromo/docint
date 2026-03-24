"""Tests for miscellaneous utilities (hashing, cleaning, env config)."""

import hashlib
import json
from pathlib import Path

import pytest

import docint.utils.env_cfg as env_cfg
from docint.utils.clean_text import basic_clean
from docint.utils.env_cfg import (
    bootstrap_config,
    load_frontend_env,
    load_hate_speech_env,
    load_ingestion_env,
    load_config,
    load_model_env,
    load_openai_env,
    load_path_env,
    load_retrieval_env,
    load_session_env,
    load_summary_env,
)
from docint.utils.hashing import compute_file_hash, ensure_file_hash
from docint.utils.logging_cfg import setup_logging
from loguru import logger


def _write_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    active_profile: str = "test",
    profile_body: str = "",
) -> Path:
    """Write a temporary TOML config and point env_cfg at it."""

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        (
            f'active_profile = "{active_profile}"\n\n'
            f"[profiles.{active_profile}.shared]\n\n"
            f"[profiles.{active_profile}.backend]\n\n"
            f"[profiles.{active_profile}.frontend]\n\n"
            f"[profiles.{active_profile}.worker]\n"
        )
        + profile_body,
        encoding="utf-8",
    )
    monkeypatch.setattr(env_cfg, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(env_cfg, "_ACTIVE_PROFILE", None)
    monkeypatch.setattr(env_cfg, "_ACTIVE_ROLE", None)
    monkeypatch.delenv("DOCINT_PROFILE", raising=False)
    return config_path


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


def test_load_config_uses_active_profile_and_backend_role(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The canonical loader should resolve the active profile and backend role."""

    _write_config(tmp_path, monkeypatch)
    cfg = load_config()
    assert cfg.runtime.profile == "test"
    assert cfg.runtime.role == "backend"


def test_load_config_merges_shared_and_role_specific_sections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Role-specific sections should override shared values for that role."""

    _write_config(
        tmp_path,
        monkeypatch,
        profile_body=(
            "\n[profiles.test.shared.inference]\n"
            'provider = "openai"\n'
            'api_base = "https://api.openai.com/v1"\n'
            "\n[profiles.test.frontend.hosts]\n"
            'backend_host = "http://backend-net:8000"\n'
            'backend_public_host = "http://localhost:8000"\n'
            'qdrant_host = "http://qdrant:6333"\n'
            'cors_allowed_origins = "http://localhost:8501"\n'
        ),
    )

    cfg = load_config(role="frontend")

    assert cfg.runtime.role == "frontend"
    assert cfg.inference.inference_provider == "openai"
    assert cfg.hosts.backend_host == "http://backend-net:8000"


def test_path_config_artifacts_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Path config should use the TOML-backed default artifacts path."""

    _write_config(tmp_path, monkeypatch)
    cfg = load_path_env()
    assert cfg.artifacts.name == "artifacts"
    assert cfg.artifacts.is_absolute()


def test_path_config_artifacts_comes_from_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Artifacts paths should come from TOML role config."""

    custom = tmp_path / "my-artifacts"
    _write_config(
        tmp_path,
        monkeypatch,
        profile_body=(
            "\n[profiles.test.backend.paths]\n"
            f'artifacts = "{custom}"\n'
            f'data = "{tmp_path / "data"}"\n'
            f'docint_home_dir = "{tmp_path}"\n'
            f'logs = "{tmp_path / "backend.log"}"\n'
            f'queries = "{tmp_path / "queries.txt"}"\n'
            f'results = "{tmp_path / "results"}"\n'
            f'qdrant_sources = "{tmp_path / "qdrant_sources"}"\n'
            f'hf_hub_cache = "{tmp_path / "hf"}"\n'
        ),
    )
    cfg = load_path_env()
    assert cfg.artifacts == custom


def test_runtime_env_vars_do_not_override_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Legacy runtime env vars should not silently replace TOML config."""

    _write_config(
        tmp_path,
        monkeypatch,
        profile_body=(
            "\n[profiles.test.shared.inference]\n"
            'provider = "ollama"\n'
            'api_base = "http://localhost:11434/v1"\n'
        ),
    )
    monkeypatch.setenv("INFERENCE_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    cfg = load_openai_env()

    assert cfg.inference_provider == "ollama"
    assert cfg.api_base == "http://localhost:11434/v1"


def test_setup_logging_respects_toml_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """setup_logging should honor the log path resolved from TOML."""

    log_file = tmp_path / "logs" / "docint.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    _write_config(
        tmp_path,
        monkeypatch,
        profile_body=(
            "\n[profiles.test.backend.paths]\n"
            f'artifacts = "{tmp_path / "artifacts"}"\n'
            f'data = "{tmp_path / "data"}"\n'
            f'docint_home_dir = "{tmp_path}"\n'
            f'logs = "{log_file}"\n'
            f'queries = "{tmp_path / "queries.txt"}"\n'
            f'results = "{tmp_path / "results"}"\n'
            f'qdrant_sources = "{tmp_path / "qdrant_sources"}"\n'
            f'hf_hub_cache = "{tmp_path / "hf"}"\n'
        ),
    )

    resolved = setup_logging(rotation="1 MB", retention=1)
    logger.debug("create log entry for file")

    assert resolved == log_file
    assert log_file.exists()


def test_load_summary_env_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Summary config should use documented defaults when TOML omits overrides."""

    _write_config(tmp_path, monkeypatch)
    cfg = load_summary_env()

    assert cfg.coverage_target == 0.70
    assert cfg.max_docs == 30
    assert cfg.per_doc_top_k == 4
    assert cfg.final_source_cap == 24
    assert cfg.social_chunking_enabled is True
    assert cfg.social_candidate_pool == 48
    assert cfg.social_diversity_limit == 2


def test_load_summary_env_clamps_and_parses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Summary config should parse and clamp TOML values."""

    _write_config(
        tmp_path,
        monkeypatch,
        profile_body=(
            "\n[profiles.test.shared.summary]\n"
            "coverage_target = 1.5\n"
            "max_docs = 12\n"
            "per_doc_top_k = 6\n"
            "final_source_cap = 10\n"
            "social_chunking_enabled = false\n"
            "social_candidate_pool = 64\n"
            "social_diversity_limit = 3\n"
        ),
    )

    cfg = load_summary_env()

    assert cfg.coverage_target == 1.0
    assert cfg.max_docs == 12
    assert cfg.per_doc_top_k == 6
    assert cfg.final_source_cap == 10
    assert cfg.social_chunking_enabled is False
    assert cfg.social_candidate_pool == 64
    assert cfg.social_diversity_limit == 3


def test_load_frontend_env_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Frontend config should use the documented collection timeout default."""

    _write_config(tmp_path, monkeypatch)
    cfg = load_frontend_env()
    assert cfg.collection_timeout == 120


def test_load_frontend_env_reads_collection_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Frontend config should parse the collection timeout override."""

    _write_config(
        tmp_path,
        monkeypatch,
        profile_body=("\n[profiles.test.shared.frontend]\ncollection_timeout = 90\n"),
    )
    cfg = load_frontend_env(role="frontend")
    assert cfg.collection_timeout == 90


def test_load_retrieval_env_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Retrieval config should use documented defaults."""

    _write_config(tmp_path, monkeypatch)
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
    tmp_path: Path,
) -> None:
    """Retrieval config should read explicit TOML overrides."""

    _write_config(
        tmp_path,
        monkeypatch,
        profile_body=(
            "\n[profiles.test.shared.retrieval]\n"
            "rerank_use_fp16 = true\n"
            "retrieve_top_k = 11\n"
            'chat_response_mode = "refine"\n'
            'vector_store_query_mode = "hybrid"\n'
            "hybrid_alpha = 0.25\n"
            "sparse_top_k = 17\n"
            "hybrid_top_k = 9\n"
            "parent_context_enabled = false\n"
        ),
    )

    cfg = load_retrieval_env()

    assert cfg.rerank_use_fp16 is True
    assert cfg.retrieve_top_k == 11
    assert cfg.chat_response_mode == "refine"
    assert cfg.vector_store_query_mode == "hybrid"
    assert cfg.hybrid_alpha == 0.25
    assert cfg.sparse_top_k == 17
    assert cfg.hybrid_top_k == 9
    assert cfg.parent_context_enabled is False


def test_load_openai_env_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """OpenAI config should default thinking to disabled."""

    _write_config(tmp_path, monkeypatch)

    cfg = load_openai_env()

    assert cfg.dimensions is None
    assert cfg.thinking_enabled is False
    assert cfg.thinking_effort == "medium"


def test_load_openai_env_accepts_vllm_and_dimensions_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """OpenAI config should accept vLLM and parse embedding dimensions."""

    _write_config(
        tmp_path,
        monkeypatch,
        profile_body=(
            "\n[profiles.test.shared.inference]\n"
            'provider = "vllm"\n'
            'api_base = "http://vllm-router:9000/v1"\n'
            "dimensions = 1024\n"
        ),
    )

    cfg = load_openai_env()

    assert cfg.inference_provider == "vllm"
    assert cfg.dimensions == 1024


def test_load_openai_env_clamps_invalid_thinking_effort(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invalid thinking effort values should fail clearly."""

    _write_config(
        tmp_path,
        monkeypatch,
        profile_body=(
            "\n[profiles.test.shared.inference]\n"
            'provider = "openai"\n'
            'api_base = "https://api.openai.com/v1"\n'
            "thinking_enabled = true\n"
            'thinking_effort = "unsupported"\n'
        ),
    )

    with pytest.raises(ValueError, match="thinking_effort"):
        load_openai_env()


def test_load_openai_env_uses_secret_from_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Secrets should still come from env even when runtime config comes from TOML."""

    _write_config(
        tmp_path,
        monkeypatch,
        profile_body=(
            "\n[profiles.test.shared.inference]\n"
            'provider = "openai"\n'
            'api_base = "https://api.openai.com/v1"\n'
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    cfg = load_openai_env()

    assert cfg.api_key == "sk-test-key"


def test_load_session_env_defaults_to_docint_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Session config should default under docint_home_dir for default paths."""

    _write_config(tmp_path, monkeypatch)
    cfg = load_session_env()
    assert cfg.session_store == f"sqlite:///{Path.home() / 'docint' / 'sessions.db'}"


def test_load_session_env_defaults_to_data_path_when_explicitly_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Session config should default inside the configured data path."""

    _write_config(
        tmp_path,
        monkeypatch,
        profile_body=(
            "\n[profiles.test.backend.paths]\n"
            f'artifacts = "{tmp_path / "data" / "artifacts"}"\n'
            f'data = "{tmp_path / "data"}"\n'
            f'docint_home_dir = "{tmp_path / "home"}"\n'
            f'logs = "{tmp_path / "data" / "backend.log"}"\n'
            f'queries = "{tmp_path / "data" / "queries.txt"}"\n'
            f'results = "{tmp_path / "data" / "results"}"\n'
            f'qdrant_sources = "{tmp_path / "data" / "qdrant_sources"}"\n'
            f'hf_hub_cache = "{tmp_path / "data" / "hf"}"\n'
        ),
    )

    cfg = load_session_env()

    assert cfg.session_store == f"sqlite:///{tmp_path / 'data' / 'sessions.db'}"


def test_load_session_env_honors_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit TOML session_store should override the default sqlite location."""

    _write_config(
        tmp_path,
        monkeypatch,
        profile_body=(
            "\n[profiles.test.shared.session]\n"
            'session_store = "sqlite:////tmp/custom-sessions.db"\n'
        ),
    )
    cfg = load_session_env()
    assert cfg.session_store == "sqlite:////tmp/custom-sessions.db"


def test_load_model_env_reads_direct_text_and_vision_model_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Model config should read explicit TOML model identifiers."""

    _write_config(
        tmp_path,
        monkeypatch,
        profile_body=(
            "\n[profiles.test.shared.models]\n"
            'text_model = "gpt-4o-mini"\n'
            'vision_model = "gpt-4.1-mini"\n'
        ),
    )

    cfg = load_model_env()

    assert cfg.text_model == "gpt-4o-mini"
    assert cfg.vision_model == "gpt-4.1-mini"


def test_load_hate_speech_env_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default hate-speech config should be disabled with one worker."""

    _write_config(tmp_path, monkeypatch)
    cfg = load_hate_speech_env()

    assert cfg.enabled is False
    assert cfg.max_chars == 2048
    assert cfg.max_workers == 1


def test_load_hate_speech_env_parses_max_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Hate-speech config should parse TOML values."""

    _write_config(
        tmp_path,
        monkeypatch,
        profile_body=(
            "\n[profiles.test.shared.hate_speech]\n"
            "enabled = true\n"
            "max_chars = 512\n"
            "max_workers = 4\n"
        ),
    )

    cfg = load_hate_speech_env()

    assert cfg.enabled is True
    assert cfg.max_chars == 512
    assert cfg.max_workers == 4


def test_load_hate_speech_env_clamps_max_workers_minimum(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Hate-speech workers should be clamped to at least one."""

    _write_config(
        tmp_path,
        monkeypatch,
        profile_body=("\n[profiles.test.shared.hate_speech]\nmax_workers = 0\n"),
    )

    cfg = load_hate_speech_env()

    assert cfg.max_workers == 1


def test_load_ingestion_env_parses_docstore_retry_knobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ingestion config should parse and clamp docstore retry settings."""

    _write_config(
        tmp_path,
        monkeypatch,
        profile_body=(
            "\n[profiles.test.shared.ingestion]\n"
            "docstore_max_retries = 7\n"
            "docstore_retry_backoff_seconds = 0.75\n"
            "docstore_retry_backoff_max_seconds = 5.0\n"
            "ingest_benchmark_enabled = true\n"
        ),
    )

    cfg = load_ingestion_env()

    assert cfg.docstore_max_retries == 7
    assert cfg.docstore_retry_backoff_seconds == 0.75
    assert cfg.docstore_retry_backoff_max_seconds == 5.0
    assert cfg.ingest_benchmark_enabled is True


def test_load_ingestion_env_clamps_negative_docstore_retry_knobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Negative docstore retry knobs should be clamped to zero."""

    _write_config(
        tmp_path,
        monkeypatch,
        profile_body=(
            "\n[profiles.test.shared.ingestion]\n"
            "docstore_max_retries = -2\n"
            "docstore_retry_backoff_seconds = -1\n"
            "docstore_retry_backoff_max_seconds = -3\n"
        ),
    )

    cfg = load_ingestion_env()

    assert cfg.docstore_max_retries == 0
    assert cfg.docstore_retry_backoff_seconds == 0.0
    assert cfg.docstore_retry_backoff_max_seconds == 0.0


def test_load_config_rejects_unknown_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unknown profile names should fail clearly."""

    _write_config(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="not defined"):
        load_config(profile="missing")


def test_bootstrap_config_applies_runtime_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bootstrap should apply offline runtime flags from TOML."""

    _write_config(
        tmp_path,
        monkeypatch,
        profile_body=("\n[profiles.test.shared.runtime]\ndocint_offline = true\n"),
    )
    monkeypatch.delenv("DOCINT_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    cfg = bootstrap_config(role="frontend")

    assert cfg.runtime.role == "frontend"
    assert cfg.runtime.docint_offline is True
    assert env_cfg._ACTIVE_PROFILE == "test"
    assert env_cfg._ACTIVE_ROLE == "frontend"
    assert env_cfg.os.environ["DOCINT_OFFLINE"] == "1"
    assert env_cfg.os.environ["HF_HUB_OFFLINE"] == "1"
    assert env_cfg.os.environ["TRANSFORMERS_OFFLINE"] == "1"
