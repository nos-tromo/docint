"""Tests for the SPARSE_* remote sparse-encoder client configuration.

Mirrors the RERANK_* contract: each knob falls back to the active
OpenAI client setting unless explicitly overridden, so the full
vllm-service router works with no configuration while the sparse-only
CPU shape needs only SPARSE_API_BASE.
"""

import pytest

from docint.utils.env_cfg import load_sparse_client_env


def test_sparse_client_inherits_openai_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no SPARSE_* set, every field inherits the OpenAI client settings."""
    for name in ("SPARSE_API_BASE", "SPARSE_API_KEY", "SPARSE_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)

    cfg = load_sparse_client_env(
        default_api_base="http://vllm-router:4000/v1",
        default_api_key="sk-master",
        default_timeout=300.0,
    )

    assert cfg.api_base == "http://vllm-router:4000/v1"
    assert cfg.api_key == "sk-master"
    assert cfg.timeout == 300.0


def test_sparse_client_explicit_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """SPARSE_API_BASE points at the sparse-only container; trailing slash is stripped."""
    monkeypatch.setenv("SPARSE_API_BASE", "http://sparse-only:8000/")
    monkeypatch.setenv("SPARSE_TIMEOUT", "45")
    monkeypatch.delenv("SPARSE_API_KEY", raising=False)

    cfg = load_sparse_client_env(
        default_api_base="http://vllm-router:4000/v1",
        default_api_key="sk-master",
        default_timeout=300.0,
    )

    assert cfg.api_base == "http://sparse-only:8000"
    assert cfg.timeout == 45.0


def test_sparse_client_blank_key_disables_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sparse-only shape has no Bearer gate; a blank default means no header."""
    monkeypatch.delenv("SPARSE_API_KEY", raising=False)
    monkeypatch.setenv("SPARSE_API_BASE", "http://sparse-only:8000")

    cfg = load_sparse_client_env(
        default_api_base="http://vllm-router:4000/v1",
        default_api_key="",
        default_timeout=300.0,
    )

    assert cfg.api_key is None
