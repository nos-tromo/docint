"""Tests for the ``ModelConfig.translate_model`` contract.

Pins the default-to-chat-model behavior and env override for the
``translate_model`` field on :class:`ModelConfig`:

- ``TRANSLATE_MODEL`` unset -> defaults to the resolved ``text_model``.
- ``TRANSLATE_MODEL`` set -> operators always win.
"""

from __future__ import annotations

import re
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
    assert res.same_language is False
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
    """Blank/whitespace input short-circuits without touching the pipeline.

    Args:
        monkeypatch: Fixture used to patch the module-level pipeline factory.
    """
    calls: dict[str, int] = {"n": 0}

    class _CountingPipe:
        """Fake chat pipeline counting each ``call_chat`` invocation."""

        def call_chat(self, prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
            """Increment the call counter and return a canned translation.

            Args:
                prompt: The user prompt (the source text to translate).
                system_prompt: Optional system prompt (unused by the fake).
                model: Optional model override (unused by the fake).

            Returns:
                str: A canned translation (never reached for blank input).
            """
            calls["n"] += 1
            return "x"

    monkeypatch.setattr(tc, "_pipeline", lambda: _CountingPipe())
    res = tc.translate("   ")
    assert res.ok is True
    assert res.translation == ""
    assert calls["n"] == 0


def test_translate_target_lang_override_uses_that_locales_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """``target_lang`` must select the destination locale's prompt, not the active one.

    CLIP's text tower is English-only, so image retrieval translates *to* English
    regardless of the operator's ``RESPONSE_LANGUAGE``.

    Args:
        monkeypatch: Fixture used to patch the pipeline and locale.
    """
    monkeypatch.setattr(tc, "load_language_env", lambda: type("_L", (), {"code": "de"})())
    seen: dict[str, str | None] = {}

    class _FakePipe:
        """Fake chat pipeline recording the system prompt it was handed."""

        def call_chat(self, prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
            """Record the system prompt and echo a canned translation.

            Args:
                prompt: The source text.
                system_prompt: The localized translate prompt under test.
                model: Optional model override (unused by the fake).

            Returns:
                str: A canned English translation.
            """
            seen["system_prompt"] = system_prompt
            return "Are there documents about AI?"

    monkeypatch.setattr(tc, "_pipeline", lambda: _FakePipe())

    res = tc.translate("Gibt es Dokumente über KI?", target_lang="en")

    assert res.ok is True
    assert res.target_lang == "en"
    assert "into English" in (seen["system_prompt"] or "")


def test_translate_defaults_to_the_active_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting ``target_lang`` must keep the existing active-locale behavior.

    Args:
        monkeypatch: Fixture used to patch the pipeline and locale.
    """
    monkeypatch.setattr(tc, "load_language_env", lambda: type("_L", (), {"code": "de"})())

    class _FakePipe:
        """Fake chat pipeline returning a canned German translation."""

        def call_chat(self, prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
            """Return a canned translation.

            Args:
                prompt: The source text.
                system_prompt: The localized translate prompt.
                model: Optional model override (unused by the fake).

            Returns:
                str: A canned German translation.
            """
            return "Hallo"

    monkeypatch.setattr(tc, "_pipeline", lambda: _FakePipe())

    res = tc.translate("Hello")

    assert res.target_lang == "de"


class _CannedPipe:
    """Fake chat pipeline answering every call with one canned reply."""

    def __init__(self, reply: str) -> None:
        """Store the reply.

        Args:
            reply: What the model answers.
        """
        self.reply = reply

    def call_chat(self, prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
        """Return the canned reply.

        Args:
            prompt: The source text (unused by the fake).
            system_prompt: The localized translate prompt (unused by the fake).
            model: Optional model override (unused by the fake).

        Returns:
            str: The canned reply.
        """
        return self.reply


@pytest.mark.parametrize(
    "reply",
    [
        "[[SAME_LANGUAGE]]",
        "[SAME_LANGUAGE]",
        "`[[same_language]]`",
        "[[SAME_LANGUAGE]] The text is already in German.",
    ],
)
def test_translate_marker_reply_is_not_a_translation(monkeypatch: pytest.MonkeyPatch, reply: str) -> None:
    """The model's same-language marker, however decorated, yields no translation.

    Passed through, the marker would be shown and frozen into a report as the
    finding's translation.

    Args:
        monkeypatch: Fixture used to patch the module-level pipeline factory.
        reply: The model's raw answer.
    """
    monkeypatch.setattr(tc, "_pipeline", lambda: _CannedPipe(reply))

    res = tc.translate("Das ist bereits ein deutscher Satz.")

    assert res.ok is True
    assert res.translation is None
    assert res.same_language is True


@pytest.mark.parametrize(
    "reply",
    [
        "Das ist bereits ein deutscher Satz.",
        '"Das ist bereits ein deutscher Satz."',
        "„Das ist bereits ein deutscher Satz“",
        "das ist bereits ein   deutscher\nSatz",
    ],
)
def test_translate_echo_is_not_a_translation(monkeypatch: pytest.MonkeyPatch, reply: str) -> None:
    """A reply repeating the source word for word is not a translation.

    A model can still echo despite the marker instruction — verbatim, quoted, or
    with case and spacing changed — and an echo shown as a translation is the
    bug this guards.

    Args:
        monkeypatch: Fixture used to patch the module-level pipeline factory.
        reply: The model's raw answer.
    """
    monkeypatch.setattr(tc, "_pipeline", lambda: _CannedPipe(reply))

    res = tc.translate("Das ist bereits ein deutscher Satz.")

    assert res.ok is True
    assert res.translation is None
    assert res.same_language is True


@pytest.mark.parametrize("locale", ["en", "de"])
def test_translate_recognises_the_marker_its_prompt_asks_for(monkeypatch: pytest.MonkeyPatch, locale: str) -> None:
    """The marker each locale's prompt asks for must be the one ``translate`` detects.

    The fake model obeys the prompt it is sent: it answers the ``[[…]]`` marker
    that prompt names, or a translation when it names none. A prompt edit that
    drops the marker, or a detector renamed away from it, fails here.

    Args:
        monkeypatch: Fixture used to patch the pipeline and locale.
        locale: The active ``RESPONSE_LANGUAGE``.
    """
    monkeypatch.setattr(tc, "load_language_env", lambda: type("_L", (), {"code": locale})())

    class _ObedientPipe:
        """Fake chat pipeline doing what the system prompt asks."""

        def call_chat(self, prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
            """Answer the prompt's marker, or a translation without one.

            Args:
                prompt: The source text (unused by the fake).
                system_prompt: The localized translate prompt under test.
                model: Optional model override (unused by the fake).

            Returns:
                str: The marker the prompt names, else a canned translation.
            """
            marker = re.search(r"\[\[[A-Z_]+\]\]", system_prompt or "")
            return marker.group(0) if marker else "a translation"

    monkeypatch.setattr(tc, "_pipeline", lambda: _ObedientPipe())

    res = tc.translate("Text already in the target language.")

    assert res.same_language is True
    assert res.translation is None
