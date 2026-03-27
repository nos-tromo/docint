"""Tests for OpenAI-compatible configuration helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from docint.utils.env_cfg import OpenAIConfig
from docint.utils.openai_cfg import (
    OpenAIPipeline,
    TruncatingOpenAIEmbedding,
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


def test_truncating_embedding_retries_oversized_inputs(
    monkeypatch,
) -> None:
    """Oversized embedding inputs should be retried with truncated text.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    error = RuntimeError(
        "This model's maximum context length is 8192 tokens. However, you requested 0 output tokens "
        "and your prompt contains at least 8193 input tokens, for a total of at least 8193 tokens."
    )
    single_inputs: list[str] = []
    warnings: list[str] = []
    original_text = "x" * 12000

    def fake_get_text_embeddings(self: Any, texts: list[str]) -> list[list[float]]:
        """Simulate the batch text embeddings method, raising an error for oversized inputs.

        Args:
            self (Any): The embedding instance.
            texts (list[str]): The list of texts to embed.

        Raises:
            error: If the input text exceeds the model's context length.

        Returns:
            list[list[float]]: A list of embedding vectors.
        """
        raise error

    def fake_get_text_embedding(self: Any, text: str) -> list[float]:
        """Simulate the single text embedding method, capturing the input and raising an error for oversized text.

        Args:
            self (Any): The embedding instance.
            text (str): The text to embed.

        Raises:
            error: If the input text exceeds the model's context length.

        Returns:
            list[float]: A list of embedding vectors.
        """
        single_inputs.append(text)
        if len(text) >= len(original_text):
            raise error
        return [0.25, 0.5]

    monkeypatch.setattr(
        "llama_index.embeddings.openai.base.OpenAIEmbedding._get_text_embeddings",
        fake_get_text_embeddings,
    )
    monkeypatch.setattr(
        "llama_index.embeddings.openai.base.OpenAIEmbedding._get_text_embedding",
        fake_get_text_embedding,
    )

    embedding = TruncatingOpenAIEmbedding(
        model_name="BAAI/bge-m3",
        api_key="sk-test",
        api_base="http://localhost:8000/v1",
        reuse_client=False,
        context_window=8192,
    )
    embedding.set_warning_callback(warnings.append)

    result = embedding._get_text_embeddings([original_text])

    assert result == [[0.25, 0.5]]
    assert single_inputs
    assert len(single_inputs[-1]) < len(original_text)
    assert warnings
    assert "truncated oversized embedding input" in warnings[0].lower()


def test_truncating_embedding_retries_vllm_oversized_inputs(
    monkeypatch,
) -> None:
    """vLLM-style embedding overflows should also trigger truncation retries.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    error = RuntimeError(
        "You passed 8193 input tokens and requested 0 output tokens. However, "
        "the model's context length is only 8192 tokens, resulting in a maximum "
        "input length of 8192 tokens. Please reduce the length of the input prompt. "
        "(parameter=input_tokens, value=8193)"
    )
    single_inputs: list[str] = []
    warnings: list[str] = []
    original_text = "x" * 12000

    def fake_get_text_embeddings(self: Any, texts: list[str]) -> list[list[float]]:
        """Simulate a batch embedding failure for oversized inputs.

        Args:
            self: Embedding instance.
            texts: Texts to embed.

        Raises:
            RuntimeError: Always, to force the fallback path.
        """

        raise error

    def fake_get_text_embedding(self: Any, text: str) -> list[float]:
        """Simulate retry success after truncation.

        Args:
            self: Embedding instance.
            text: Text to embed.

        Raises:
            RuntimeError: When the input is still oversized.

        Returns:
            list[float]: A fake embedding vector.
        """

        single_inputs.append(text)
        if len(text) >= len(original_text):
            raise error
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        "llama_index.embeddings.openai.base.OpenAIEmbedding._get_text_embeddings",
        fake_get_text_embeddings,
    )
    monkeypatch.setattr(
        "llama_index.embeddings.openai.base.OpenAIEmbedding._get_text_embedding",
        fake_get_text_embedding,
    )

    embedding = TruncatingOpenAIEmbedding(
        model_name="BAAI/bge-m3",
        api_key="sk-test",
        api_base="http://localhost:8000/v1",
        reuse_client=False,
        context_window=8192,
    )
    embedding.set_warning_callback(warnings.append)

    result = embedding._get_text_embeddings([original_text])

    assert result == [[0.1, 0.2, 0.3]]
    assert single_inputs
    assert len(single_inputs[-1]) < len(original_text)
    assert warnings
    assert "8193 input tokens" in warnings[0]
