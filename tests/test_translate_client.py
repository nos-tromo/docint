"""Tests for the ``ModelConfig.translate_model`` contract.

Pins the default-to-chat-model behavior and env override for the
``translate_model`` field on :class:`ModelConfig`:

- ``TRANSLATE_MODEL`` unset -> defaults to the resolved ``text_model``.
- ``TRANSLATE_MODEL`` set -> operators always win.
"""

from __future__ import annotations

import pytest

from docint.utils.env_cfg import load_model_env


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
