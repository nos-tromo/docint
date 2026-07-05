"""Tests for the ``ModelConfig.translate_model`` contract.

Pins the default-to-chat-model behavior and env override for the
``translate_model`` field on :class:`ModelConfig`:

- ``TRANSLATE_MODEL`` unset -> defaults to the resolved ``text_model``.
- ``TRANSLATE_MODEL`` set -> operators always win.
"""

from __future__ import annotations

from typing import Any

import pytest

from docint.utils.env_cfg import load_model_env
from docint.utils.openai_cfg import OpenAIPipeline


def test_translate_model_defaults_to_text_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """``translate_model`` must default to the resolved ``text_model``.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.delenv("TRANSLATE_MODEL", raising=False)
    monkeypatch.setenv("TEXT_MODEL", "my-chat-model")

    cfg = load_model_env()

    assert cfg.translate_model == "my-chat-model"


def test_translate_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``TRANSLATE_MODEL`` must override the chat-model default.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("TEXT_MODEL", "my-chat-model")
    monkeypatch.setenv("TRANSLATE_MODEL", "special-mt-model")

    cfg = load_model_env()

    assert cfg.translate_model == "special-mt-model"


class _FakeResponse:
    """Minimal stand-in for an OpenAI ChatCompletion response object."""

    def __init__(self, content: str = "ok") -> None:
        """Build a fake response exposing ``choices[0].message.content``.

        Args:
            content: Text to surface as the completion's message content.
        """
        message = type("_Msg", (), {"content": content})()
        self.choices = [type("_Choice", (), {"message": message})()]


def test_call_chat_uses_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``model`` kwarg must override the pipeline's default text model.

    Args:
        monkeypatch: Fixture used to patch the fake chat-completions client.
    """
    pipe = OpenAIPipeline()
    captured: dict[str, Any] = {}

    def _fake_create(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse("ok")

    monkeypatch.setattr(pipe.client.chat.completions, "create", _fake_create)
    out = pipe.call_chat("hello", system_prompt="sys", model="override-model")

    assert out == "ok"
    assert captured["model"] == "override-model"
    assert captured["messages"][0] == {"role": "system", "content": "sys"}


def test_call_chat_defaults_to_text_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting ``model`` must fall back to the pipeline's configured text model.

    Args:
        monkeypatch: Fixture used to patch the fake chat-completions client.
    """
    pipe = OpenAIPipeline()
    captured: dict[str, Any] = {}

    def _fake_create(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse("x")

    monkeypatch.setattr(pipe.client.chat.completions, "create", _fake_create)
    pipe.call_chat("hi")

    assert captured["model"] == pipe.text_model_id
