"""Tests for OpenAI-compatible configuration helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from docint.utils.env_cfg import OpenAIConfig
from docint.utils.openai_cfg import OpenAIPipeline, get_openai_reasoning_effort


def test_get_openai_reasoning_effort_requires_toggle_and_provider() -> None:
    """Reasoning effort should only be returned for enabled OpenAI requests."""
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
    assert (
        get_openai_reasoning_effort(
            OpenAIConfig(
                **{
                    **base_config.__dict__,
                    "thinking_enabled": False,
                }
            )
        )
        is None
    )
    assert (
        get_openai_reasoning_effort(
            OpenAIConfig(
                **{
                    **base_config.__dict__,
                    "inference_provider": "ollama",
                }
            )
        )
        is None
    )


def test_openai_pipeline_call_chat_passes_reasoning_effort(
    monkeypatch, tmp_path: Path
) -> None:
    """OpenAI pipeline should forward reasoning effort on chat completions."""
    captured: dict[str, Any] = {}

    class FakeClient:
        """Minimal client with the chat completion API surface."""

        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create),
            )

        def _create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    monkeypatch.setattr(
        "docint.utils.openai_cfg.load_model_env",
        lambda: SimpleNamespace(
            text_model_file="gpt-5-mini.gguf",
            vision_model_file="gpt-4.1-mini.gguf",
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
    """OpenAI pipeline should omit reasoning effort when thinking is disabled."""
    captured: dict[str, Any] = {}

    class FakeClient:
        """Minimal client with the chat completion API surface."""

        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create),
            )

        def _create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    monkeypatch.setattr(
        "docint.utils.openai_cfg.load_model_env",
        lambda: SimpleNamespace(
            text_model_file="gpt-5-mini.gguf",
            vision_model_file="gpt-4.1-mini.gguf",
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
