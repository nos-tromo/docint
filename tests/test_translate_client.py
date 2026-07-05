"""Tests for the ``ModelConfig.translate_model`` contract.

Pins the default-to-chat-model behavior and env override for the
``translate_model`` field on :class:`ModelConfig`:

- ``TRANSLATE_MODEL`` unset -> defaults to the resolved ``text_model``.
- ``TRANSLATE_MODEL`` set -> operators always win.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

import docint.utils.translate_client as tc
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


@pytest.fixture(autouse=True)
def _clear_translate_caches() -> Iterator[None]:
    """Clear the ``translate_client`` LRU caches so tests don't leak state.

    Yields:
        None.
    """
    tc._translate_cached.cache_clear()
    tc._pipeline.cache_clear()
    yield
    tc._translate_cached.cache_clear()
    tc._pipeline.cache_clear()


def test_translate_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean translate call returns a stripped translation from the fake pipeline.

    Args:
        monkeypatch: Fixture used to patch the module-level pipeline factory.
    """
    calls: list[tuple[str, str | None]] = []

    class _FakePipe:
        """Fake chat pipeline recording each ``call_chat`` invocation."""

        def call_chat(self, prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
            """Record the call and return a canned, padded translation.

            Args:
                prompt: The user prompt (the source text to translate).
                system_prompt: Optional system prompt (unused by the fake).
                model: Optional model override, recorded for assertions.

            Returns:
                str: A canned translation padded with whitespace.
            """
            calls.append((prompt, model))
            return "  Hallo Welt  "

    monkeypatch.setattr(tc, "_pipeline", lambda: _FakePipe())
    res = tc.translate("Hello world")
    assert res.ok is True
    assert res.translation == "Hallo Welt"  # stripped
    assert res.target_lang  # set from locale
    assert len(calls) == 1


def test_translate_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second call with identical text/target/model is served from cache.

    Args:
        monkeypatch: Fixture used to patch the module-level pipeline factory.
    """
    n: dict[str, int] = {"count": 0}

    class _FakePipe:
        """Fake chat pipeline counting each ``call_chat`` invocation."""

        def call_chat(self, prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
            """Increment the call counter and return a canned translation.

            Args:
                prompt: The user prompt (the source text to translate).
                system_prompt: Optional system prompt (unused by the fake).
                model: Optional model override (unused by the fake).

            Returns:
                str: A canned translation.
            """
            n["count"] += 1
            return "x"

    monkeypatch.setattr(tc, "_pipeline", lambda: _FakePipe())
    tc.translate("same text")
    tc.translate("same text")
    assert n["count"] == 1  # second call served from cache


def test_translate_failsoft(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transport/model error yields a failed result instead of raising.

    Args:
        monkeypatch: Fixture used to patch the module-level pipeline factory.
    """

    class _BoomPipe:
        """Fake chat pipeline that always raises to simulate an outage."""

        def call_chat(self, prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
            """Simulate a downstream transport/model failure.

            Args:
                prompt: The user prompt (the source text to translate).
                system_prompt: Optional system prompt (unused by the fake).
                model: Optional model override (unused by the fake).

            Raises:
                RuntimeError: Always, to simulate the router being unreachable.
            """
            raise RuntimeError("router down")

    monkeypatch.setattr(tc, "_pipeline", lambda: _BoomPipe())
    res = tc.translate("Hello")
    assert res.ok is False
    assert res.translation is None
    assert res.error == "unavailable"


def test_translate_blank_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blank (whitespace-only) input is a no-op success; the pipeline is never called.

    Args:
        monkeypatch: Unused; kept for signature consistency with sibling tests.
    """
    res = tc.translate("   ")
    assert res.ok is True
    assert res.translation == ""
