"""Turn classification. All questions and answers are invented."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from _pytest.logging import LogCaptureFixture
from assessment_fakes import fragmented_endpoint, join_threads
from typing_extensions import override

from docint.agents.assessment.classify import (
    CLASSIFY_SLOT_POOL,
    CLASSIFY_SLOT_WAIT_S,
    CLASSIFY_SLOTS,
    CLASSIFY_THREAD_NAME,
    DEFAULT_ASSESSMENT_CLASSIFY_PROMPT,
    classify_turn,
)
from docint.agents.types import PriorTurn
from docint.utils.openai_cfg import LocalOpenAI

QUESTION = "zzsecretzz What could be improved in this study?"


@pytest.fixture(autouse=True)
def _classify_capacity_is_whole_again() -> Iterator[None]:
    """Fail any test that leaves one of the process-wide classification slots taken.

    Torn down after every fixture the test requested, so a held call is
    released and joined before the slots are counted.

    Yields:
        None: The test runs.
    """
    yield
    taken = 0
    while taken < CLASSIFY_SLOTS and CLASSIFY_SLOT_POOL.acquire(blocking=False):
        taken += 1
    for _ in range(taken):
        CLASSIFY_SLOT_POOL.release()
    assert taken == CLASSIFY_SLOTS


def _llm(reply: str) -> MagicMock:
    """Build a model double whose completion returns ``reply``.

    Args:
        reply (str): The raw completion text.

    Returns:
        MagicMock: The model double.
    """
    llm = MagicMock()
    llm.complete.return_value = SimpleNamespace(text=reply)
    return llm


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"mode": "assessment"}', "assessment"),
        ('```json\n{"mode": "ASSESSMENT"}\n```', "assessment"),
        ('<think>asks for critique</think>{"mode": "assessment"}', "assessment"),
        ('Sure: {"mode": "lookup"}', "lookup"),
        ('{"mode": "maybe"}', "lookup"),
        ("no json at all", "lookup"),
        ("{not json}", "lookup"),
    ],
)
def test_a_reply_maps_to_a_mode(reply: str, expected: str) -> None:
    """Anything but an explicit assessment verdict is a lookup.

    Args:
        reply (str): The classifier's raw reply.
        expected (str): The mode it must map to.
    """
    assert classify_turn(cast(Any, _llm(reply)), question=QUESTION, prior_turn=None) == expected


def test_no_model_or_a_blank_question_is_a_lookup() -> None:
    """Nothing to classify means today's behaviour, without a model call."""
    llm = _llm('{"mode": "assessment"}')

    assert classify_turn(None, question=QUESTION, prior_turn=None) == "lookup"
    assert classify_turn(cast(Any, llm), question="   ", prior_turn=None) == "lookup"
    llm.complete.assert_not_called()


def test_a_failed_call_is_a_lookup_and_never_logs_the_question(loguru_caplog: LogCaptureFixture) -> None:
    """The warning names the failure's type; the question never reaches the log.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError(QUESTION)

    assert classify_turn(cast(Any, llm), question=QUESTION, prior_turn=None) == "lookup"
    assert "RuntimeError" in loguru_caplog.text
    assert "zzsecretzz" not in loguru_caplog.text


def test_the_prompt_carries_the_question_and_the_prior_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
    """Follow-ups like "and what do you think of that?" need the previous answer.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    llm = _llm('{"mode": "lookup"}')
    prior = PriorTurn(user_text="Summarise it.", assistant_text="It reports one experiment [1].")

    classify_turn(cast(Any, llm), question=QUESTION, prior_turn=prior)

    prompt = llm.complete.call_args.args[0]
    assert QUESTION in prompt
    assert "It reports one experiment [1]." in prompt
    assert '{"mode": "assessment"}' in prompt


def _assert_leans_to_lookup(prompt: str, tie_break: str) -> None:
    """Check that lookup is listed first, wins a tie, and leads the format line.

    Args:
        prompt (str): The rendered classifier prompt.
        tie_break (str): The locale's "when unsure" sentence.
    """
    assert prompt.index('- "lookup"') < prompt.index('- "assessment"')
    assert tie_break in prompt.split('- "assessment"', 1)[1]
    last_line = prompt.strip().splitlines()[-1]
    assert last_line.index('{"mode": "lookup"}') < last_line.index('{"mode": "assessment"}')


@pytest.mark.parametrize(
    ("code", "tie_break"), [("en", "When unsure, answer lookup."), ("de", "Im Zweifel antworte mit lookup.")]
)
def test_the_prompt_leans_to_lookup(monkeypatch: pytest.MonkeyPatch, code: str, tie_break: str) -> None:
    """A model that echoes the format line, or cannot decide, picks the cheap mode.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        code (str): The locale.
        tie_break (str): The locale's "when unsure" sentence.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", code)
    llm = _llm('{"mode": "lookup"}')

    classify_turn(cast(Any, llm), question=QUESTION, prior_turn=None)

    _assert_leans_to_lookup(llm.complete.call_args.args[0], tie_break)


def test_the_bundled_fallback_prompt_leans_to_lookup() -> None:
    """The default used when the prompt file is missing keeps the same order."""
    prompt = DEFAULT_ASSESSMENT_CLASSIFY_PROMPT.format(question=QUESTION, prior_turn="(none)")

    _assert_leans_to_lookup(prompt, "When unsure, answer lookup.")


#: A classifier reply asking for an assessment, as an OpenAI-compatible endpoint sends it.
ASSESSMENT_REPLY: dict[str, Any] = {
    "id": "invented-completion",
    "object": "chat.completion",
    "created": 0,
    "model": "invented-model",
    "choices": [
        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"mode": "assessment"}'}}
    ],
}


def test_a_reply_that_arrives_in_fragments_past_the_deadline_is_a_lookup(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """Codex's reproduction: each fragment lands within the client's timeout, the verdict past the deadline.

    The SDK's timeout bounds each read, not the call, so only the classifier's own deadline stops it;
    the abandoned call is joined here once it ends.

    Args:
        monkeypatch (pytest.MonkeyPatch): Shortens the deadline to a second.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    monkeypatch.setattr("docint.agents.assessment.classify.CLASSIFY_TIMEOUT_S", 1.0)

    with fragmented_endpoint(ASSESSMENT_REPLY) as base:
        llm = LocalOpenAI(api_base=base, api_key="test", model="invented-model", max_retries=0)
        started = time.monotonic()
        mode = classify_turn(llm, question=QUESTION, prior_turn=None)
        elapsed = time.monotonic() - started
        join_threads(CLASSIFY_THREAD_NAME)

    assert elapsed < 2.0
    assert mode == "lookup"
    assert loguru_caplog.messages == ["Turn classification ran past its deadline; answering as a lookup"]


class _NoFreeSlot(threading.BoundedSemaphore):
    """Classification slots that record how long a call would wait and find them all taken."""

    def __init__(self) -> None:
        """Start with one slot and no waits recorded."""
        super().__init__(1)
        self.waits: list[float | None] = []

    @override
    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        """Record the wait and report no free slot.

        Args:
            blocking (bool): Whether the caller would block.
            timeout (float | None): How long it would wait.

        Returns:
            bool: Always ``False``.
        """
        self.waits.append(timeout)
        return False


def test_no_free_slot_is_a_lookup_without_calling_the_model(loguru_caplog: LogCaptureFixture) -> None:
    """With every slot taken, a turn waits a second at most and is answered as a lookup; the warning has no question.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    slots = _NoFreeSlot()
    llm = _llm('{"mode": "assessment"}')

    assert classify_turn(cast(Any, llm), question=QUESTION, prior_turn=None, slots=slots) == "lookup"
    assert slots.waits == [CLASSIFY_SLOT_WAIT_S]
    llm.complete.assert_not_called()
    assert loguru_caplog.messages == ["Turn classification found no free slot; answering as a lookup"]


def test_a_call_past_the_deadline_is_a_lookup_and_keeps_its_slot_until_it_ends(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """The verdict that arrives late is never read, and the abandoned call stays counted until it really finishes.

    Args:
        monkeypatch (pytest.MonkeyPatch): Shortens the deadline.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    monkeypatch.setattr("docint.agents.assessment.classify.CLASSIFY_TIMEOUT_S", 0.2)
    release = threading.Event()
    slots = threading.BoundedSemaphore(1)
    llm = MagicMock()

    def _held(prompt: str, **kwargs: Any) -> SimpleNamespace:
        """Wait for the release, then ask for an assessment.

        Args:
            prompt (str): The classifier prompt, unused.
            **kwargs (Any): The call's options, unused.

        Returns:
            SimpleNamespace: The late completion.
        """
        release.wait(timeout=10)
        return SimpleNamespace(text='{"mode": "assessment"}')

    llm.complete.side_effect = _held
    try:
        mode = classify_turn(cast(Any, llm), question=QUESTION, prior_turn=None, slots=slots)
        held = slots.acquire(blocking=False) is False
    finally:
        release.set()
        join_threads(CLASSIFY_THREAD_NAME)

    assert (mode, held) == ("lookup", True)
    assert llm.complete.call_args.kwargs["timeout"] == 0.2
    assert loguru_caplog.messages == ["Turn classification ran past its deadline; answering as a lookup"]
    assert slots.acquire(blocking=False) is True
    slots.release()


_SCOPE: ContextVar[str | None] = ContextVar("invented_request_scope", default=None)


def test_a_classification_runs_on_its_own_thread_in_the_callers_context() -> None:
    """The call is abandoned at its deadline, so it runs on its own thread, still seeing the request's scope."""
    seen: list[tuple[str, str | None]] = []
    llm = MagicMock()

    def _record(prompt: str, **kwargs: Any) -> SimpleNamespace:
        """Record the thread and the scope the call runs under.

        Args:
            prompt (str): The classifier prompt, unused.
            **kwargs (Any): The call's options, unused.

        Returns:
            SimpleNamespace: A lookup verdict.
        """
        seen.append((threading.current_thread().name, _SCOPE.get()))
        return SimpleNamespace(text='{"mode": "lookup"}')

    llm.complete.side_effect = _record
    token = _SCOPE.set("invented-collection")
    try:
        classify_turn(cast(Any, llm), question=QUESTION, prior_turn=None)
    finally:
        _SCOPE.reset(token)

    assert seen == [(CLASSIFY_THREAD_NAME, "invented-collection")]
