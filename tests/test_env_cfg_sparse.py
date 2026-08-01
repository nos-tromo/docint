"""Tests for the SPARSE_* remote sparse-encoder client configuration.

Mirrors the RERANK_* contract: each knob falls back to the active
OpenAI client setting unless explicitly overridden, so the full
vllm-service router works with no configuration while the sparse-only
CPU shape needs only SPARSE_API_BASE.
"""

import pytest

from docint.utils.env_cfg import load_model_env, load_sparse_client_env, resolve_enable_hybrid


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


@pytest.fixture(autouse=True)
def _clear_hybrid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every env var that participates in hybrid resolution."""
    for name in ("ENABLE_HYBRID", "SPARSE_API_BASE", "INFERENCE_PROVIDER", "SPARSE_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_hybrid_on_for_vllm_without_explicit_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production: the router already serves /pooling, so hybrid is on by default."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    assert resolve_enable_hybrid() is True


def test_hybrid_on_when_sparse_base_set_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dev: pointing at sparse-only opts in, whatever the provider."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.setenv("SPARSE_API_BASE", "http://sparse-only:8000")
    assert resolve_enable_hybrid() is True


def test_hybrid_off_for_ollama_without_sparse_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """No sparse endpoint means dense-only, not a POST at a route that 404s."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    assert resolve_enable_hybrid() is False


def test_hybrid_off_for_openai_without_sparse_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plain OpenAI has no /pooling route; degrade to dense."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "openai")
    assert resolve_enable_hybrid() is False


def test_explicit_enable_hybrid_overrides_derivation(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit setting always wins over the derived default."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    monkeypatch.setenv("ENABLE_HYBRID", "false")
    assert resolve_enable_hybrid() is False

    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.setenv("ENABLE_HYBRID", "true")
    assert resolve_enable_hybrid() is True


@pytest.mark.parametrize("provider", ["ollama", "vllm", "openai"])
def test_sparse_model_defaults_to_bge_m3_on_every_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    """One sparse model everywhere — the local BM42 default is gone."""
    monkeypatch.setenv("INFERENCE_PROVIDER", provider)
    assert load_model_env().sparse_model == "BAAI/bge-m3"
