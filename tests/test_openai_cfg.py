"""Tests for OpenAI-compatible configuration helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from docint.utils.env_cfg import OpenAIConfig
from docint.utils.openai_cfg import (
    LocalOpenAI,
    OpenAIPipeline,
    get_openai_reasoning_effort,
)


def test_get_openai_reasoning_effort_requires_toggle_only() -> None:
    """Reasoning effort should be returned for any enabled provider."""
    base_config = OpenAIConfig(
        api_base="https://api.openai.com/v1",
        api_key="sk-test",
        ctx_window=4096,
        dimensions=1024,
        max_retries=2,
        num_output=256,
        inference_provider="openai",
        reuse_client=False,
        seed=42,
        temperature=0.0,
        thinking_effort="high",
        thinking_enabled=True,
        timeout=300.0,
        top_p=0.0,
    )

    assert get_openai_reasoning_effort(base_config) == "high"
    assert get_openai_reasoning_effort(base_config, enabled=False) is None
    assert (
        get_openai_reasoning_effort(
            OpenAIConfig(
                api_base=base_config.api_base,
                api_key=base_config.api_key,
                ctx_window=base_config.ctx_window,
                dimensions=base_config.dimensions,
                max_retries=base_config.max_retries,
                num_output=base_config.num_output,
                inference_provider=base_config.inference_provider,
                reuse_client=base_config.reuse_client,
                seed=base_config.seed,
                temperature=base_config.temperature,
                thinking_effort=base_config.thinking_effort,
                thinking_enabled=False,
                timeout=base_config.timeout,
                top_p=base_config.top_p,
            )
        )
        is None
    )
    assert (
        get_openai_reasoning_effort(
            OpenAIConfig(
                api_base=base_config.api_base,
                api_key=base_config.api_key,
                ctx_window=base_config.ctx_window,
                dimensions=base_config.dimensions,
                max_retries=base_config.max_retries,
                num_output=base_config.num_output,
                inference_provider="ollama",
                reuse_client=base_config.reuse_client,
                seed=base_config.seed,
                temperature=base_config.temperature,
                thinking_effort=base_config.thinking_effort,
                thinking_enabled=base_config.thinking_enabled,
                timeout=base_config.timeout,
                top_p=base_config.top_p,
            )
        )
        == "high"
    )


def test_openai_pipeline_call_chat_passes_reasoning_effort_for_ollama(
    monkeypatch, tmp_path: Path
) -> None:
    """Reasoning effort should be forwarded for non-OpenAI providers too.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary cache root.
    """
    captured: dict[str, Any] = {}

    class FakeClient:
        """Minimal client with the chat completion API surface."""

        def __init__(self) -> None:
            """Initialize the fake client and set up the chat completions interface."""
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create),
            )

        def _create(self, **kwargs: Any) -> Any:
            """Simulate the chat completion creation method, capturing the input arguments.

            Returns:
                Any: A fake chat completion response.
            """
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    monkeypatch.setattr(
        "docint.utils.openai_cfg.load_model_env",
        lambda: SimpleNamespace(
            text_model="qwen3.5:9b",
            vision_model="qwen3.5-vl:7b",
        ),
    )
    monkeypatch.setattr(
        "docint.utils.openai_cfg.load_openai_env",
        lambda: OpenAIConfig(
            api_base="http://localhost:11434/v1",
            api_key="sk-test",
            ctx_window=200000,
            dimensions=1024,
            max_retries=2,
            num_output=256,
            inference_provider="ollama",
            reuse_client=False,
            seed=42,
            temperature=0.0,
            thinking_effort="high",
            thinking_enabled=True,
            timeout=300.0,
            top_p=0.0,
        ),
    )
    monkeypatch.setattr(
        "docint.utils.openai_cfg.load_path_env",
        lambda: SimpleNamespace(prompts=tmp_path),
    )
    monkeypatch.setattr("docint.utils.openai_cfg.OpenAI", lambda **_: FakeClient())

    pipeline = OpenAIPipeline()
    result = pipeline.call_chat("hello")

    assert result == "ok"
    assert captured["reasoning_effort"] == "high"


def test_openai_pipeline_call_chat_passes_reasoning_effort(
    monkeypatch, tmp_path: Path
) -> None:
    """OpenAI pipeline should forward reasoning effort on chat completions.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary cache root.
    """
    captured: dict[str, Any] = {}

    class FakeClient:
        """Minimal client with the chat completion API surface."""

        def __init__(self) -> None:
            """Initialize the fake client and set up the chat completions interface."""
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create),
            )

        def _create(self, **kwargs: Any) -> Any:
            """Simulate the chat completion creation method, capturing the input arguments.

            Returns:
                Any: A fake chat completion response.
            """
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    monkeypatch.setattr(
        "docint.utils.openai_cfg.load_model_env",
        lambda: SimpleNamespace(
            text_model="gpt-5-mini.gguf",
            vision_model="gpt-4.1-mini.gguf",
        ),
    )
    monkeypatch.setattr(
        "docint.utils.openai_cfg.load_openai_env",
        lambda: OpenAIConfig(
            api_base="https://api.openai.com/v1",
            api_key="sk-test",
            ctx_window=200000,
            dimensions=1024,
            max_retries=2,
            num_output=256,
            inference_provider="openai",
            reuse_client=False,
            seed=42,
            temperature=0.0,
            thinking_effort="high",
            thinking_enabled=True,
            timeout=300.0,
            top_p=0.0,
        ),
    )
    monkeypatch.setattr(
        "docint.utils.openai_cfg.load_path_env",
        lambda: SimpleNamespace(prompts=tmp_path),
    )
    monkeypatch.setattr("docint.utils.openai_cfg.OpenAI", lambda **_: FakeClient())

    pipeline = OpenAIPipeline()
    result = pipeline.call_chat("hello", system_prompt="system")

    assert result == "ok"
    assert captured["model"] == "gpt-5-mini"
    assert captured["reasoning_effort"] == "high"
    assert captured["messages"][0]["role"] == "system"


def test_openai_pipeline_call_chat_omits_reasoning_effort_when_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    """OpenAI pipeline should omit reasoning effort when thinking is disabled.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary cache root.
    """
    captured: dict[str, Any] = {}

    class FakeClient:
        """Minimal client with the chat completion API surface."""

        def __init__(self) -> None:
            """Initialize the fake client and set up the chat completions interface."""
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create),
            )

        def _create(self, **kwargs: Any) -> Any:
            """Simulate the chat completion creation method, capturing the input arguments.

            Returns:
                Any: A fake chat completion response.
            """
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    monkeypatch.setattr(
        "docint.utils.openai_cfg.load_model_env",
        lambda: SimpleNamespace(
            text_model="gpt-5-mini.gguf",
            vision_model="gpt-4.1-mini.gguf",
        ),
    )
    monkeypatch.setattr(
        "docint.utils.openai_cfg.load_openai_env",
        lambda: OpenAIConfig(
            api_base="https://api.openai.com/v1",
            api_key="sk-test",
            ctx_window=200000,
            dimensions=1024,
            max_retries=2,
            num_output=256,
            inference_provider="openai",
            reuse_client=False,
            seed=42,
            temperature=0.0,
            thinking_effort="high",
            thinking_enabled=False,
            timeout=300.0,
            top_p=0.0,
        ),
    )
    monkeypatch.setattr(
        "docint.utils.openai_cfg.load_path_env",
        lambda: SimpleNamespace(prompts=tmp_path),
    )
    monkeypatch.setattr("docint.utils.openai_cfg.OpenAI", lambda **_: FakeClient())

    pipeline = OpenAIPipeline()
    pipeline.call_chat("hello")

    assert "reasoning_effort" not in captured


def _install_vision_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    inference_provider: str,
    thinking_enabled: bool,
    response_content: str,
    captured: dict[str, Any],
) -> "OpenAIPipeline":
    """Build an ``OpenAIPipeline`` whose vision client returns a fixed response.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary cache root.
        inference_provider: Provider string for ``OpenAIConfig``.
        thinking_enabled: Whether reasoning effort is enabled.
        response_content: String the fake vision client returns.
        captured: Dict the fake client fills with the request kwargs.

    Returns:
        OpenAIPipeline: A pipeline wired to the fake client.
    """

    class FakeClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create),
            )

        def _create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=response_content))
                ]
            )

    monkeypatch.setattr(
        "docint.utils.openai_cfg.load_model_env",
        lambda: SimpleNamespace(
            text_model="qwen3.5:9b",
            vision_model="qwen3.5-vl:7b",
        ),
    )
    monkeypatch.setattr(
        "docint.utils.openai_cfg.load_openai_env",
        lambda: OpenAIConfig(
            api_base="http://localhost:11434/v1",
            api_key="sk-test",
            ctx_window=200000,
            dimensions=1024,
            max_retries=2,
            num_output=256,
            inference_provider=inference_provider,
            reuse_client=False,
            seed=42,
            temperature=0.0,
            thinking_effort="high",
            thinking_enabled=thinking_enabled,
            timeout=300.0,
            top_p=0.0,
        ),
    )
    monkeypatch.setattr(
        "docint.utils.openai_cfg.load_path_env",
        lambda: SimpleNamespace(prompts=tmp_path),
    )
    monkeypatch.setattr("docint.utils.openai_cfg.OpenAI", lambda **_: FakeClient())

    return OpenAIPipeline()


def test_call_vision_strips_think_tags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``call_vision`` must strip ``<think>...</think>`` before returning."""
    captured: dict[str, Any] = {}
    pipeline = _install_vision_pipeline(
        monkeypatch,
        tmp_path,
        inference_provider="ollama",
        thinking_enabled=False,
        response_content="<think>scratch work</think>final description",
        captured=captured,
    )

    result = pipeline.call_vision(prompt="describe", img_base64="AA==")

    assert result == "final description"


def test_call_vision_forwards_reasoning_effort_for_ollama(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reasoning effort should be forwarded for the vision path like chat does."""
    captured: dict[str, Any] = {}
    pipeline = _install_vision_pipeline(
        monkeypatch,
        tmp_path,
        inference_provider="ollama",
        thinking_enabled=True,
        response_content="ok",
        captured=captured,
    )

    pipeline.call_vision(prompt="describe", img_base64="AA==")

    assert captured["reasoning_effort"] == "high"


def test_call_vision_omits_reasoning_effort_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reasoning effort must not be sent when the pipeline has thinking disabled."""
    captured: dict[str, Any] = {}
    pipeline = _install_vision_pipeline(
        monkeypatch,
        tmp_path,
        inference_provider="openai",
        thinking_enabled=False,
        response_content="ok",
        captured=captured,
    )

    pipeline.call_vision(prompt="describe", img_base64="AA==")

    assert "reasoning_effort" not in captured


def test_call_vision_strips_reasoning_around_refusal_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reasoning is stripped but refusal detection is left to call sites."""
    captured: dict[str, Any] = {}
    pipeline = _install_vision_pipeline(
        monkeypatch,
        tmp_path,
        inference_provider="ollama",
        thinking_enabled=False,
        response_content=(
            "<think>I wasn't given an image</think>"
            "I don't see any image attached to your message."
        ),
        captured=captured,
    )

    result = pipeline.call_vision(prompt="describe", img_base64="AA==")

    assert result == "I don't see any image attached to your message."


def test_budgeted_embedding_raises_on_oversize(
    monkeypatch,
) -> None:
    """Oversized embedding inputs must raise loudly — no retry, no truncation.

    The pre-embed re-splitter is the only supported path for oversize
    inputs. If an oversize text reaches the embedding call anyway, the
    wrapper must surface the context-limit error by raising
    ``EmbeddingInputTooLongError`` so ingestion aborts and the operator
    can diagnose why the re-splitter missed the input.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    from docint.utils.openai_cfg import (
        BudgetedOpenAIEmbedding,
        EmbeddingInputTooLongError,
    )

    error = RuntimeError(
        "This model's maximum context length is 8192 tokens. However, you requested 0 output tokens "
        "and your prompt contains at least 8193 input tokens, for a total of at least 8193 tokens."
    )
    original_text = "x" * 12000

    def fake_get_text_embeddings(self: Any, texts: list[str]) -> list[list[float]]:
        """Always raise the OpenAI-style oversized-input error.

        Args:
            self: Embedding instance.
            texts: Texts the caller is trying to embed.

        Raises:
            RuntimeError: Always, with the OpenAI phrasing.
        """
        _ = (self, texts)
        raise error

    monkeypatch.setattr(
        "llama_index.embeddings.openai.base.OpenAIEmbedding._get_text_embeddings",
        fake_get_text_embeddings,
    )

    embedding = BudgetedOpenAIEmbedding(
        model_name="BAAI/bge-m3",
        api_key="sk-test",
        api_base="http://localhost:8000/v1",
        reuse_client=False,
        context_window=8192,
    )

    with pytest.raises(EmbeddingInputTooLongError):
        embedding.get_text_embeddings_strict([original_text])


def test_budgeted_embedding_raises_on_vllm_oversize(
    monkeypatch,
) -> None:
    """vLLM-style context overflows must raise loudly — no retry, no truncation.

    Same loud-failure contract as for OpenAI phrasing: the wrapper does
    not silently shrink the input; instead it raises
    ``EmbeddingInputTooLongError`` so the caller knows the pre-embed
    re-splitter failed to bound the request.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    from docint.utils.openai_cfg import (
        BudgetedOpenAIEmbedding,
        EmbeddingInputTooLongError,
    )

    error = RuntimeError(
        "You passed 8193 input tokens and requested 0 output tokens. However, "
        "the model's context length is only 8192 tokens, resulting in a maximum "
        "input length of 8192 tokens. Please reduce the length of the input prompt. "
        "(parameter=input_tokens, value=8193)"
    )
    original_text = "x" * 12000

    def fake_get_text_embeddings(self: Any, texts: list[str]) -> list[list[float]]:
        """Always raise the vLLM-style oversized-input error.

        Args:
            self: Embedding instance.
            texts: Texts to embed.

        Raises:
            RuntimeError: Always, with the vLLM phrasing.
        """
        _ = (self, texts)
        raise error

    monkeypatch.setattr(
        "llama_index.embeddings.openai.base.OpenAIEmbedding._get_text_embeddings",
        fake_get_text_embeddings,
    )

    embedding = BudgetedOpenAIEmbedding(
        model_name="BAAI/bge-m3",
        api_key="sk-test",
        api_base="http://localhost:8000/v1",
        reuse_client=False,
        context_window=8192,
    )

    with pytest.raises(EmbeddingInputTooLongError):
        embedding.get_text_embeddings_strict([original_text])


def test_is_context_limit_error_recognizes_ollama_input_length_phrasing() -> None:
    """Ollama's "input length exceeds context length" wording is a context-limit signal.

    Regression guard: ollama (routing to a generic OpenAI-compatible model such as
    gemma4) reports overflow with the phrase ``"the input length exceeds the context
    length"``. None of the previously recognised phrasings match this wording, so
    ``_is_context_limit_error`` used to return ``False`` and the batch ingest crashed
    instead of falling back to the truncation retry loop.
    """

    exc = RuntimeError(
        "Error code: 400 - {'error': {'message': 'the input length exceeds the "
        "context length', 'type': 'invalid_request_error', 'param': None, "
        "'code': None}}"
    )

    from docint.utils.openai_cfg import BudgetedOpenAIEmbedding

    assert BudgetedOpenAIEmbedding._is_context_limit_error(exc) is True


def test_is_context_limit_error_rejects_unrelated_errors() -> None:
    """Unrelated transport/throttling errors must not be treated as context-limit overflows.

    Regression guard: broadening the context-limit detection to cover the ollama
    phrasing must not accidentally sweep up connection errors or rate limits,
    which should continue to propagate so higher layers can retry or surface them.
    """

    connection_exc = RuntimeError("Connection refused by http://localhost:11434")
    rate_limit_exc = RuntimeError(
        "Error code: 429 - {'error': {'message': 'rate limit exceeded'}}"
    )

    from docint.utils.openai_cfg import BudgetedOpenAIEmbedding

    assert BudgetedOpenAIEmbedding._is_context_limit_error(connection_exc) is False
    assert BudgetedOpenAIEmbedding._is_context_limit_error(rate_limit_exc) is False


def test_budgeted_embedding_raises_on_ollama_phrasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama-phrased context-limit errors must raise loudly — no retry.

    Regression guard preserved from commit ``761ea72``: the ollama 400
    phrasing ``"the input length exceeds the context length"`` is the
    signal that the pre-embed re-splitter failed to bound the request.
    The wrapper must NOT retry with a truncated input; it must raise
    ``EmbeddingInputTooLongError`` so the operator sees the failure
    rather than silently corrupting the vector store with a prefix-only
    embedding.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    from docint.utils.openai_cfg import (
        BudgetedOpenAIEmbedding,
        EmbeddingInputTooLongError,
    )

    error = RuntimeError(
        "Error code: 400 - {'error': {'message': 'the input length exceeds the "
        "context length', 'type': 'invalid_request_error', 'param': None, "
        "'code': None}}"
    )
    original_text = "x" * 12000

    def fake_get_text_embeddings(self: Any, texts: list[str]) -> list[list[float]]:
        """Always raise with the ollama 400 phrasing.

        Args:
            self: Embedding instance.
            texts: Texts to embed.

        Raises:
            RuntimeError: Always, with the ollama phrasing.
        """
        _ = (self, texts)
        raise error

    monkeypatch.setattr(
        "llama_index.embeddings.openai.base.OpenAIEmbedding._get_text_embeddings",
        fake_get_text_embeddings,
    )

    embedding = BudgetedOpenAIEmbedding(
        model_name="BAAI/bge-m3",
        api_key="sk-test",
        api_base="http://localhost:11434/v1",
        reuse_client=False,
        context_window=8192,
    )

    with pytest.raises(EmbeddingInputTooLongError):
        embedding.get_text_embeddings_strict([original_text])


# ---------------------------------------------------------------------------
# LocalOpenAI metadata tests
# ---------------------------------------------------------------------------


def test_local_openai_metadata_overrides_context_window_for_known_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even when llama_index recognises the model, the configured context_window wins.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    from llama_index.core.base.llms.types import LLMMetadata
    from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

    base_meta = LLMMetadata(
        context_window=128000,
        num_output=4096,
        is_chat_model=True,
        is_function_calling_model=True,
        model_name="gpt-4o",
    )
    monkeypatch.setattr(
        LlamaIndexOpenAI,
        "metadata",
        property(lambda self: base_meta),
    )

    model = LocalOpenAI(
        context_window=32768,
        num_output=512,
        model="gpt-4o",
        api_key="sk-test",
        api_base="http://localhost:11434/v1",
    )
    meta = model.metadata
    assert meta.context_window == 32768
    assert meta.num_output == 512
    assert meta.model_name == base_meta.model_name


def test_local_openai_metadata_fallback_for_unknown_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown model names fall back to configured context_window and defaults.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

    def _raise(_self: Any) -> None:
        raise ValueError("Unknown model")

    monkeypatch.setattr(
        LlamaIndexOpenAI,
        "metadata",
        property(_raise),
    )

    model = LocalOpenAI(
        context_window=16384,
        num_output=512,
        model="Qwen/Qwen3.5-2B",
        api_key="sk-test",
        api_base="http://localhost:11434/v1",
    )
    meta = model.metadata
    assert meta.context_window == 16384
    assert meta.num_output == 512
    assert meta.model_name == "Qwen/Qwen3.5-2B"


# ---------------------------------------------------------------------------
# BudgetedOpenAIEmbedding envelope forwarding
# ---------------------------------------------------------------------------


def test_budgeted_embedding_forwards_timeout_max_retries_and_embed_batch_size() -> None:
    """The embed wrapper must forward envelope kwargs to its parent.

    ``timeout``, ``max_retries``, and ``embed_batch_size`` are the three
    fields that control the embed client's request envelope (how long
    it waits, how many times it retries, how many inputs it packs per
    call). They must land on the parent ``OpenAIEmbedding`` model so
    the HTTP client honors them at call time — otherwise the new
    ``EmbeddingConfig`` values would be ignored and the wrapper would
    keep using the llama_index defaults.
    """
    from docint.utils.openai_cfg import BudgetedOpenAIEmbedding

    embedding = BudgetedOpenAIEmbedding(
        model_name="BAAI/bge-m3",
        api_key="sk-test",
        api_base="http://localhost:11434/v1",
        reuse_client=False,
        context_window=8192,
        timeout=1800.0,
        max_retries=1,
        embed_batch_size=16,
    )

    assert embedding.timeout == 1800.0
    assert embedding.max_retries == 1
    assert embedding.embed_batch_size == 16
