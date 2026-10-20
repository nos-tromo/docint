"""A probe of whether the chat endpoint accepts tools, carrying nothing of any turn.

An endpoint that refuses an agent turn's first call may refuse tools as such, or only that request:
its messages, a picture, a template the request trips. Only the first is a reason to keep agent
turns off for every user (see :class:`~docint.agents.assessment.gate.ToolsRejection`). So the
question is put again in a fixed request: a system line, ``ping``, and one tool that does nothing,
offered with ``tool_choice="auto"`` and streamed, as a round is. Nothing of any user, collection or
request reaches it, and nothing the endpoint answers is logged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from loguru import logger

from docint.agents.turn.loop import refused_as_sent

if TYPE_CHECKING:
    from llama_index.core.llms import LLM

#: The probe's system line.
PROBE_SYSTEM_PROMPT = "This is a connectivity check. Reply with one word."
#: The probe's whole message.
PROBE_MESSAGE = "ping"
#: The one tool the probe offers: no parameters, and nothing behind it.
PROBE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "noop",
        "description": "Does nothing.",
        "parameters": {"type": "object", "properties": {}},
    },
}
#: Seconds the probe waits on the endpoint.
PROBE_TIMEOUT_S = 10.0

ProbeOutcome = Literal["refused", "accepted", "inconclusive"]
"""What a probe found: tools refused as such, accepted, or neither shown."""


def probe_tools(llm: LLM) -> ProbeOutcome:
    """Ask the endpoint, in a request carrying nothing of any turn, whether it accepts tools.

    The reply's first event means the endpoint accepted the request: the rest is not read, and the
    reply is closed. ``llm`` must make one attempt per call, as the agent model does.

    Args:
        llm (LLM): The agent model's client.

    Returns:
        ProbeOutcome: ``refused`` for an HTTP 400 or 422 that is not a context overflow,
        ``accepted`` once the reply began, ``inconclusive`` for any other failure, such as a
        timeout, a server error or an unreachable endpoint.
    """
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=PROBE_SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=PROBE_MESSAGE),
    ]
    reply: Any = None
    try:
        reply = llm.stream_chat(messages, tools=[PROBE_TOOL], tool_choice="auto", timeout=PROBE_TIMEOUT_S)
        next(iter(reply), None)
    except Exception as exc:
        if refused_as_sent(exc):
            return "refused"
        logger.info("A probe of the chat endpoint's tool support failed ({})", type(exc).__name__)
        return "inconclusive"
    finally:
        close = getattr(reply, "close", None)
        if callable(close):
            close()
    return "accepted"
