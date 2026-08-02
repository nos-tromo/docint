"""Tests that the dense embedding client targets the configured endpoint.

Production (vLLM) sets no EMBED_* vars and must keep inheriting
OPENAI_API_BASE exactly; a CPU dev host overrides EMBED_API_BASE to
reach the embed-only container without moving chat.
"""

import pytest

from docint.core.rag import RAG


@pytest.fixture()
def _clear_embed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every EMBED_* var so each test controls its own state."""
    for name in ("EMBED_API_BASE", "EMBED_API_KEY", "EMBED_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)


def test_embed_config_inherits_openai_base_when_unset(
    monkeypatch: pytest.MonkeyPatch,
    _clear_embed_env: None,
) -> None:
    """Production path: unset EMBED_API_BASE must resolve to OPENAI_API_BASE."""
    monkeypatch.setenv("OPENAI_API_BASE", "http://vllm-router:4000/v1")
    rag = RAG(qdrant_collection="test")
    assert rag.embed_client_config is not None
    assert rag.embed_client_config.api_base == "http://vllm-router:4000/v1"
    assert rag.embed_client_config.api_base == rag.openai_api_base


def test_embed_config_uses_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
    _clear_embed_env: None,
) -> None:
    """Dev path: EMBED_API_BASE moves dense without moving chat."""
    monkeypatch.setenv("OPENAI_API_BASE", "http://ollama:11434/v1")
    monkeypatch.setenv("EMBED_API_BASE", "http://embed-only:8000/v1")
    rag = RAG(qdrant_collection="test")
    assert rag.embed_client_config is not None
    assert rag.embed_client_config.api_base == "http://embed-only:8000/v1"
    assert rag.openai_api_base == "http://ollama:11434/v1"


def test_embed_model_client_targets_the_embed_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    _clear_embed_env: None,
) -> None:
    """The constructed embedding client must use the embed base, not the chat base."""
    monkeypatch.setenv("OPENAI_API_BASE", "http://ollama:11434/v1")
    monkeypatch.setenv("EMBED_API_BASE", "http://embed-only:8000/v1")
    captured: dict[str, object] = {}

    from docint.core import rag as rag_module

    def _capture(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(rag_module, "BudgetedOpenAIEmbedding", _capture)
    rag = RAG(qdrant_collection="test")
    _ = rag.embed_model

    assert captured["api_base"] == "http://embed-only:8000/v1"
