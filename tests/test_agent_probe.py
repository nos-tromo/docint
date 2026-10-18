"""A refusal of tools is remembered only once a probe carrying nothing of any turn is refused too.

A turn's first call can be refused for its own content, so that refusal alone must not switch
agent turns off for every user. Every message, model and endpoint here is invented.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import openai
import pytest
from _pytest.logging import LogCaptureFixture
from agent_fakes import answer_round, call_round, status_error, streaming_tool_llm, timed_out, unreachable
from assessment_fakes import bad_request
from llama_index.core.base.llms.types import MessageRole

from docint.agents.assessment.gate import ToolsRejection
from docint.agents.turn.probe import (
    PROBE_MESSAGE,
    PROBE_SYSTEM_PROMPT,
    PROBE_TIMEOUT_S,
    PROBE_TOOL,
    ProbeOutcome,
    probe_tools,
)

#: How an endpoint without tool support refuses a request offering tools.
TOOLS_REFUSED = "auto tool choice is not enabled on this invented server"
#: How an endpoint says a prompt does not fit its window.
OVERFLOW = "This model's maximum context length is 8192 tokens; your request has 9001 input tokens."


@pytest.mark.parametrize(
    ("reply", "outcome"),
    [
        (answer_round("pong"), "accepted"),
        (call_round(("call_a", "noop", {})), "accepted"),
        ([], "accepted"),
        (bad_request(TOOLS_REFUSED), "refused"),
        (status_error(openai.UnprocessableEntityError, 422, TOOLS_REFUSED), "refused"),
        (bad_request(OVERFLOW), "inconclusive"),
        (status_error(openai.InternalServerError, 503, "invented outage"), "inconclusive"),
        (status_error(openai.RateLimitError, 429, "invented rate limit"), "inconclusive"),
        (timed_out(), "inconclusive"),
        (unreachable(), "inconclusive"),
        (RuntimeError("invented failure"), "inconclusive"),
    ],
    ids=[
        "answered",
        "called-the-tool",
        "empty",
        "400",
        "422",
        "overflow",
        "5xx",
        "429",
        "timeout",
        "unreachable",
        "other",
    ],
)
def test_a_probe_tells_a_refusal_of_tools_from_every_other_outcome(reply: Any, outcome: ProbeOutcome) -> None:
    """Only a 400 or 422 that is no overflow is a refusal; a reply that began is acceptance; the rest tells nothing.

    Args:
        reply (Any): What the endpoint answers the probe: chunks, or the error its first pull raises.
        outcome (ProbeOutcome): What the probe must find.
    """
    llm = streaming_tool_llm([reply])

    assert probe_tools(llm) == outcome
    assert llm.stream_chat.call_count == 1


def test_the_probe_asks_with_nothing_of_any_turn() -> None:
    """The request is fixed: a system line, ``ping``, one tool with no parameters, chosen freely, ten seconds."""
    llm = streaming_tool_llm([answer_round("pong")])

    probe_tools(llm)

    sent = llm.stream_chat.call_args
    assert [(message.role, message.content) for message in sent.args[0]] == [
        (MessageRole.SYSTEM, PROBE_SYSTEM_PROMPT),
        (MessageRole.USER, PROBE_MESSAGE),
    ]
    assert sent.kwargs == {"tools": [PROBE_TOOL], "tool_choice": "auto", "timeout": PROBE_TIMEOUT_S}
    assert PROBE_TOOL["function"]["parameters"] == {"type": "object", "properties": {}}
    assert (PROBE_MESSAGE, PROBE_TIMEOUT_S) == ("ping", 10.0)


def test_the_probe_reads_only_the_reply_s_first_event_and_closes_it() -> None:
    """Acceptance is the reply beginning: the probe stops reading there and closes it."""
    read: list[str] = []
    closed: list[bool] = []

    def _reply(messages: Any, **kwargs: Any) -> Iterator[Any]:
        """Stream two pieces, recording each one read and the close.

        Args:
            messages (Any): The probe's messages, unused.
            **kwargs (Any): The call's options, unused.

        Yields:
            Any: One event per piece.
        """
        try:
            for piece in ("po", "ng"):
                read.append(piece)
                yield SimpleNamespace(delta=piece)
        finally:
            closed.append(True)

    llm = MagicMock()
    llm.stream_chat.side_effect = _reply

    assert probe_tools(llm) == "accepted"
    assert (read, closed) == (["po"], [True])


@pytest.mark.parametrize(
    ("outcome", "cached", "says"),
    [
        ("refused", True, "a probe confirmed it"),
        ("accepted", False, "a probe offering tools was accepted"),
        ("inconclusive", False, "a probe could not tell"),
    ],
)
def test_only_a_refused_probe_keeps_agent_turns_off(
    loguru_caplog: LogCaptureFixture, outcome: ProbeOutcome, cached: bool, says: str
) -> None:
    """Each probed refusal gives one warning saying what the probe found; only a refused probe is remembered.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
        outcome (ProbeOutcome): What the probe found.
        cached (bool): Whether the refusal must be remembered.
        says (str): What the one warning must say.
    """
    rejection = ToolsRejection()

    rejection.confirm(lambda: outcome)

    warnings = [message for message in loguru_caplog.messages if "offering tools" in message]
    assert rejection.cached is cached
    assert len(warnings) == 1
    assert says in warnings[0]


def test_a_probe_that_cannot_run_remembers_nothing_and_frees_the_next(loguru_caplog: LogCaptureFixture) -> None:
    """A probe that raises, say because the model could not be built, is inconclusive, and the next may run.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    rejection = ToolsRejection()

    def _broken() -> ProbeOutcome:
        """Fail before asking anything.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("invented failure building the model")

    rejection.confirm(_broken)
    first = rejection.cached
    rejection.confirm(lambda: "refused")

    assert (first, rejection.cached) == (False, True)
    assert "invented failure" not in loguru_caplog.text


def test_a_refusal_while_a_probe_runs_neither_probes_nor_is_remembered() -> None:
    """At most one probe runs per process: a refusal arriving meanwhile is dropped, whatever it would have found."""
    rejection = ToolsRejection()
    started, release = threading.Event(), threading.Event()
    probes: list[str] = []

    def _slow() -> ProbeOutcome:
        """Probe until released, then report the endpoint accepted tools.

        Returns:
            ProbeOutcome: ``accepted``.
        """
        probes.append("first")
        started.set()
        release.wait(5)
        return "accepted"

    def _second() -> ProbeOutcome:
        """Report a refusal, were this probe ever run.

        Returns:
            ProbeOutcome: ``refused``.
        """
        probes.append("second")
        return "refused"

    first = threading.Thread(target=rejection.confirm, args=(_slow,))
    first.start()
    assert started.wait(5)
    rejection.confirm(_second)
    during = rejection.cached
    release.set()
    first.join(5)

    assert (probes, during, rejection.cached) == (["first"], False, False)
