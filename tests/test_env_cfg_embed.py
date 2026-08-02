"""Tests for the EMBED_* dense-embedding client configuration.

Mirrors the SPARSE_*/RERANK_* contract: each knob falls back to the
active OpenAI client setting unless explicitly overridden, so the full
vllm-service router works with no configuration while a CPU dev host
points dense embeddings at the embed-only container.
"""

import pytest

from docint.utils.env_cfg import load_embed_client_env


def test_embed_client_inherits_openai_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no EMBED_* set, every field inherits the OpenAI client settings."""
    for name in ("EMBED_API_BASE", "EMBED_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    cfg = load_embed_client_env(
        default_api_base="http://vllm-router:4000/v1",
        default_api_key="sk-master",
    )

    assert cfg.api_base == "http://vllm-router:4000/v1"
    assert cfg.api_key == "sk-master"


def test_embed_client_preserves_the_v1_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OpenAI SDK appends /embeddings to this base, so /v1 must survive."""
    monkeypatch.setenv("EMBED_API_BASE", "http://embed-only:8000/v1/")
    monkeypatch.delenv("EMBED_API_KEY", raising=False)

    cfg = load_embed_client_env(
        default_api_base="http://vllm-router:4000/v1",
        default_api_key=None,
    )

    assert cfg.api_base == "http://embed-only:8000/v1"


def test_embed_client_blank_key_disables_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """The embed-only shape has no Bearer gate; a blank default means no header."""
    monkeypatch.delenv("EMBED_API_KEY", raising=False)
    monkeypatch.setenv("EMBED_API_BASE", "http://embed-only:8000/v1")

    cfg = load_embed_client_env(
        default_api_base="http://vllm-router:4000/v1",
        default_api_key="",
    )

    assert cfg.api_key is None


def test_embed_client_explicit_key_wins_over_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicitly set EMBED_API_KEY beats the inherited one."""
    monkeypatch.setenv("EMBED_API_KEY", "sk-embed")

    cfg = load_embed_client_env(
        default_api_base="http://vllm-router:4000/v1",
        default_api_key="sk-master",
    )

    assert cfg.api_key == "sk-embed"
