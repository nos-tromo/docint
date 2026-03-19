"""Tests for OpenAI-compatible configuration helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from docint.utils.env_cfg import OpenAIConfig
from docint.utils.openai_cfg import OpenAIPipeline, get_openai_reasoning_effort


def test_get_openai_reasoning_effort_requires_toggle_only() -> None:
    """Reasoning effort should be returned for any enabled provider."""
    base_config = OpenAIConfig(
        api_base="https://api.openai.com/v1",
        api_key="sk-test",
        ctx_window=4096,
        dimensions=1024,
        max_retries=2,
        model_provider="openai",
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
                model_provider=base_config.model_provider,
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
                model_provider="ollama",
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
            model_provider="ollama",
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
            model_provider="openai",
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
            model_provider="openai",
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
