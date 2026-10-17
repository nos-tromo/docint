"""Tests for the caption-style description a sub-floor image is re-scored against.

A question *about* the imagery ("how is the weather?") scores the right
caption in the same band as an unrelated one, so the image floor re-scores
such images against what a matching picture's caption would say. The helper
writing that description is one chat call: cached on success, fail-soft, and
never carrying the model's reasoning into the rerank query.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from _pytest.logging import LogCaptureFixture

import docint.utils.image_description as image_description
from docint.utils.env_cfg import load_path_env

QUESTION = "Wie ist das Wetter am Hafen?"
DESCRIPTION = "Hafenbecken unter bedecktem Himmel, nasse Pflastersteine, zwei Personen in Regenjacken"


@pytest.fixture
def _hermetic_image_description() -> None:
    """Opt out of the suite-wide stub in ``conftest.py``: these tests drive the real helper."""


@pytest.fixture(autouse=True)
def _clear_description_caches() -> Iterator[None]:
    """Clear the helper's caches so no test sees another's description.

    Yields:
        None.
    """
    image_description._describe_cached.cache_clear()
    image_description._pipeline.cache_clear()
    yield
    image_description._describe_cached.cache_clear()
    image_description._pipeline.cache_clear()


class _FakePipe:
    """Chat pipeline stand-in answering canned replies in order and recording each call."""

    def __init__(self, *replies: str | Exception) -> None:
        """Store the replies; the last one repeats once the others are used up.

        Args:
            *replies: Reply texts, or exceptions to raise instead of replying.
        """
        self.replies = list(replies)
        self.calls: list[dict[str, str | None]] = []

    def call_chat(self, prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
        """Record the call and answer the next canned reply.

        Args:
            prompt: The user message (the question).
            system_prompt: The localized description prompt.
            model: The chat model asked.

        Returns:
            str: The canned reply.

        Raises:
            Exception: The canned exception, when that is the next reply.
        """
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt, "model": model})
        reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply


def _use(monkeypatch: pytest.MonkeyPatch, pipe: _FakePipe) -> _FakePipe:
    """Route the helper's chat calls to ``pipe``.

    Args:
        monkeypatch: Fixture used to patch the module-level pipeline factory.
        pipe: The stand-in pipeline.

    Returns:
        _FakePipe: The same stand-in, for its call log.
    """
    monkeypatch.setattr(image_description, "_pipeline", lambda: pipe)
    return pipe


def test_the_description_is_one_line_without_the_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reply becomes a rerank query: reasoning and line breaks must not reach it.

    Args:
        monkeypatch: Fixture used to patch the pipeline.
    """
    _use(
        monkeypatch,
        _FakePipe("<think>Gefragt ist nach dem Wetter.</think>\nHafenbecken unter bedecktem Himmel,\n   nasse Steine"),
    )

    assert image_description.describe_for_image_match(QUESTION) == "Hafenbecken unter bedecktem Himmel, nasse Steine"


def test_a_description_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same question asks the model once.

    Args:
        monkeypatch: Fixture used to patch the pipeline.
    """
    pipe = _use(monkeypatch, _FakePipe(DESCRIPTION))

    first = image_description.describe_for_image_match(QUESTION)
    second = image_description.describe_for_image_match(QUESTION)

    assert first == second == DESCRIPTION
    assert len(pipe.calls) == 1


def test_a_failure_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """An outage costs one turn its rescue, not every later turn asking the same.

    Args:
        monkeypatch: Fixture used to patch the pipeline.
    """
    pipe = _use(monkeypatch, _FakePipe(RuntimeError("router down"), DESCRIPTION))

    assert image_description.describe_for_image_match(QUESTION) is None
    assert image_description.describe_for_image_match(QUESTION) == DESCRIPTION
    assert len(pipe.calls) == 2


@pytest.mark.parametrize(
    "reply", ["", "  \n ", "<think>Nur Überlegungen.</think>"], ids=["empty", "blank", "reasoning"]
)
def test_an_empty_reply_is_no_description_and_is_not_cached(monkeypatch: pytest.MonkeyPatch, reply: str) -> None:
    """A reply with nothing left to rerank against is a failure, not an answer.

    Args:
        monkeypatch: Fixture used to patch the pipeline.
        reply: The model's raw answer.
    """
    pipe = _use(monkeypatch, _FakePipe(reply, DESCRIPTION))

    assert image_description.describe_for_image_match(QUESTION) is None
    assert image_description.describe_for_image_match(QUESTION) == DESCRIPTION
    assert len(pipe.calls) == 2


def test_a_failure_logs_one_warning_naming_only_the_exception_type(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """The warning says what failed and never repeats the question.

    An upstream error message can quote the request it rejected, so even the
    exception's own text stays out of the log.

    Args:
        monkeypatch: Fixture used to patch the pipeline.
        loguru_caplog: Captures loguru WARNING records.
    """
    _use(monkeypatch, _FakePipe(RuntimeError(f"upstream rejected: {QUESTION}")))

    assert image_description.describe_for_image_match(QUESTION) is None

    warnings = [record.getMessage() for record in loguru_caplog.records]
    assert len(warnings) == 1
    assert "RuntimeError" in warnings[0]
    assert QUESTION not in warnings[0]
    assert "upstream rejected" not in warnings[0]


@pytest.mark.parametrize("code", ["en", "de"])
def test_the_description_is_asked_with_the_response_language_prompt(monkeypatch: pytest.MonkeyPatch, code: str) -> None:
    """Captions are written in the response language, so the description must be too.

    Args:
        monkeypatch: Fixture used to patch the pipeline and locale.
        code: The active ``RESPONSE_LANGUAGE``.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", code)
    monkeypatch.setenv("TEXT_MODEL", "stub-chat-model")
    pipe = _use(monkeypatch, _FakePipe(DESCRIPTION))

    image_description.describe_for_image_match(QUESTION)

    prompt_file = load_path_env().prompts / code / "image_rescore_description.txt"
    assert pipe.calls == [
        {
            "prompt": QUESTION,
            "system_prompt": prompt_file.read_text(encoding="utf-8").strip(),
            "model": "stub-chat-model",
        }
    ]


def test_the_cache_is_keyed_by_language_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A switched locale or chat model asks again rather than reusing another's description.

    Args:
        monkeypatch: Fixture used to patch the pipeline and environment.
    """
    pipe = _use(monkeypatch, _FakePipe(DESCRIPTION))

    for code, model in (("en", "chat-a"), ("en", "chat-a"), ("de", "chat-a"), ("de", "chat-b")):
        monkeypatch.setenv("RESPONSE_LANGUAGE", code)
        monkeypatch.setenv("TEXT_MODEL", model)
        image_description.describe_for_image_match(QUESTION)

    assert [call["model"] for call in pipe.calls] == ["chat-a", "chat-a", "chat-b"]


@pytest.mark.parametrize("query", ["", "   "], ids=["empty", "blank"])
def test_a_blank_question_asks_nothing(monkeypatch: pytest.MonkeyPatch, query: str) -> None:
    """There is no picture to describe for no question.

    Args:
        monkeypatch: Fixture used to patch the pipeline.
        query: The blank question.
    """
    pipe = _use(monkeypatch, _FakePipe(DESCRIPTION))

    assert image_description.describe_for_image_match(query) is None
    assert pipe.calls == []
