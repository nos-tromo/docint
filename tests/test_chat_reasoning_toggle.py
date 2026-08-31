"""Per-request reasoning override for chat generation.

The chat UI carries a reasoning on/off toggle. It reaches the engine as a
request-scoped override (:meth:`RAG.reasoning_scope`) rather than a parameter
threaded through every ``chat``/``stream_chat``/``build_query_engine``
signature, mirroring :meth:`RAG.collection_scope`: the post-retrieval model
property consults it, and an absent override falls back to the
``OPENAI_ENABLE_THINKING`` process default.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from openai.resources.chat.completions import Completions

from docint.core import rag as rag_module
from docint.core.api import AgentChatIn, QueryIn
from docint.core.rag import RAG
from docint.utils.env_cfg import OpenAIConfig


def _config(*, provider: str, thinking_enabled: bool) -> OpenAIConfig:
    """Build a minimal OpenAI config for the given provider and thinking default.

    Args:
        provider (str): ``INFERENCE_PROVIDER`` value to emulate.
        thinking_enabled (bool): The ``OPENAI_ENABLE_THINKING`` default.

    Returns:
        OpenAIConfig: The config.
    """
    return OpenAIConfig(
        api_base="http://vllm-router:4000/v1",
        api_key="sk-test",
        ctx_window=32768,
        dimensions=1024,
        max_retries=2,
        num_output=256,
        inference_provider=provider,
        reuse_client=False,
        seed=42,
        temperature=0.0,
        thinking_effort="medium",
        thinking_enabled=thinking_enabled,
        timeout=300.0,
        top_p=0.0,
    )


def _rag(*, provider: str, thinking_enabled: bool) -> RAG:
    """Build a RAG whose text-model construction is driven by ``_config``.

    Args:
        provider (str): ``INFERENCE_PROVIDER`` value to emulate.
        thinking_enabled (bool): The ``OPENAI_ENABLE_THINKING`` default.

    Returns:
        RAG: The engine, with the OpenAI knobs copied the way ``__post_init__`` does.
    """
    rag = RAG(qdrant_collection="test")
    rag.text_model_id = "test-model"
    rag.openai_config = _config(provider=provider, thinking_enabled=thinking_enabled)
    rag.openai_api_base = rag.openai_config.api_base
    rag.openai_api_key = rag.openai_config.api_key
    rag.openai_ctx_window = rag.openai_config.ctx_window
    rag.openai_max_retries = rag.openai_config.max_retries
    rag.openai_reuse_client = rag.openai_config.reuse_client
    rag.openai_seed = rag.openai_config.seed
    rag.openai_temperature = rag.openai_config.temperature
    rag.openai_timeout = rag.openai_config.timeout
    rag.openai_top_p = rag.openai_config.top_p
    return rag


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace ``LocalOpenAI`` with a kwargs recorder.

    Args:
        monkeypatch: The monkeypatch fixture.

    Returns:
        list[dict[str, Any]]: One constructor-kwargs dict per instantiation.
    """
    calls: list[dict[str, Any]] = []

    class FakeLocalOpenAI:
        """Record constructor kwargs."""

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            calls.append(kwargs)

    monkeypatch.setattr(rag_module, "LocalOpenAI", FakeLocalOpenAI)
    return calls


def test_request_override_on_beats_env_off(fake_llm: list[dict[str, Any]]) -> None:
    """``reasoning_scope(True)`` selects the reasoning model even when the env default is off."""
    rag = _rag(provider="vllm", thinking_enabled=False)

    assert rag.post_retrieval_text_model is rag.text_model  # env default: off
    with rag.reasoning_scope(True):
        assert rag.post_retrieval_text_model is not rag.text_model
        assert rag.post_retrieval_text_model.kwargs["reasoning_effort"] == "medium"  # type: ignore[attr-defined]
    assert rag.post_retrieval_text_model is rag.text_model  # restored


def test_request_override_off_beats_env_on(fake_llm: list[dict[str, Any]]) -> None:
    """``reasoning_scope(False)`` selects the plain model even when the env default is on."""
    rag = _rag(provider="vllm", thinking_enabled=True)

    assert rag.post_retrieval_text_model is not rag.text_model  # env default: on
    with rag.reasoning_scope(False):
        assert rag.post_retrieval_text_model is rag.text_model
    assert rag.post_retrieval_text_model is not rag.text_model


def test_absent_override_keeps_env_default(fake_llm: list[dict[str, Any]]) -> None:
    """``reasoning_scope(None)`` is a no-op — API clients that omit the field see no change."""
    rag = _rag(provider="vllm", thinking_enabled=False)

    with rag.reasoning_scope(None):
        assert rag.post_retrieval_text_model is rag.text_model


def test_vllm_gets_enable_thinking_template_kwarg(fake_llm: list[dict[str, Any]]) -> None:
    """On vLLM the switch is ``chat_template_kwargs.enable_thinking``, sent explicitly both ways.

    ``reasoning_effort`` alone does not toggle a Qwen3/Gemma-style template;
    and the plain model must send ``false`` so a stack whose ``.env`` defaults
    thinking on is still overridden per request. The switch must ride inside
    ``extra_body``: llama_index merges ``additional_kwargs`` top-level into
    ``Completions.create()``, whose signature rejects unknown kwargs with a
    ``TypeError`` before any request is made, while ``extra_body`` contents are
    merged into the JSON body where vLLM reads them.
    """
    rag = _rag(provider="vllm", thinking_enabled=False)

    rag._create_text_model()
    rag._create_text_model(enable_reasoning=True)

    assert fake_llm[0]["additional_kwargs"] == {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    assert fake_llm[1]["additional_kwargs"] == {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}


def test_other_providers_do_not_get_template_kwargs(fake_llm: list[dict[str, Any]]) -> None:
    """OpenAI proper rejects unknown body fields, so the vLLM-only kwarg stays off elsewhere."""
    rag = _rag(provider="openai", thinking_enabled=False)

    rag._create_text_model(enable_reasoning=True)

    assert fake_llm[0]["additional_kwargs"] == {}


def test_vllm_additional_kwargs_bind_to_openai_sdk() -> None:
    """Every top-level ``additional_kwargs`` key must be a real ``Completions.create`` parameter.

    llama_index forwards ``additional_kwargs`` as top-level kwargs to the
    installed openai SDK, so a key its signature does not name raises
    ``TypeError`` on the first chat call — client-side, before any request
    reaches the endpoint (the v2.2.0 ``chat_template_kwargs`` regression).
    Guarded against the real installed SDK, not a fake, so an SDK upgrade
    that drops a parameter fails here instead of in production.
    """
    accepted = set(inspect.signature(Completions.create).parameters)

    for enable_reasoning in (False, True):
        rag = _rag(provider="vllm", thinking_enabled=False)
        model = rag._create_text_model(enable_reasoning=enable_reasoning)
        unknown = set(model.additional_kwargs) - accepted
        assert not unknown, f"kwargs the openai SDK would reject with TypeError: {unknown}"


def test_request_payloads_carry_optional_reasoning_flag() -> None:
    """Both chat endpoints accept ``reasoning`` and default it to "no override"."""
    assert QueryIn(question="q").reasoning is None
    assert QueryIn(question="q", reasoning=True).reasoning is True
    assert AgentChatIn(message="m").reasoning is None
    assert AgentChatIn(message="m", reasoning=False).reasoning is False
