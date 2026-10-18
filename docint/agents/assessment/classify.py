"""Decide per turn whether a question asks for a judgement or for information.

The classifier runs before every turn, so its call runs on its own thread, in
slots of its own, and is abandoned at ``CLASSIFY_TIMEOUT_S``; the turn is then
answered as a lookup.
"""

from __future__ import annotations

import functools
import json
import re
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger

from docint.agents.assessment.bounded import NoSlot, run_bounded
from docint.agents.types import AnswerMode, PriorTurn
from docint.utils.llm_sanitize import strip_reasoning
from docint.utils.prompt_loader import load_localized_prompt

if TYPE_CHECKING:
    from llama_index.core.llms import LLM

#: Enough of a prior answer to resolve "that" in a follow-up.
_PRIOR_ANSWER_CHARS = 1500
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
#: Room for the one-line JSON verdict; a reply padded past it reads as a lookup.
CLASSIFY_MAX_TOKENS = 64
#: Seconds the classifier may take; it runs before every turn, so slowness fails soft.
CLASSIFY_TIMEOUT_S = 15.0
#: Name of the thread a classification runs on, so an abandoned one can be found.
CLASSIFY_THREAD_NAME = "assessment-classify"
#: Classifications that may run at once in this process, abandoned ones included.
CLASSIFY_SLOTS = 8
#: Longest a classification waits for a free slot; the wait comes out of ``CLASSIFY_TIMEOUT_S``.
CLASSIFY_SLOT_WAIT_S = 1.0
#: The process-wide slots every classification takes, apart from research's so research never uses them up.
CLASSIFY_SLOT_POOL = threading.BoundedSemaphore(CLASSIFY_SLOTS)

DEFAULT_ASSESSMENT_CLASSIFY_PROMPT = (
    "Decide whether the user's latest message asks for a judgement or for information.\n\n"
    '- "lookup": information the material itself states, even on an evaluative topic.\n'
    '- "assessment": an evaluation, critique, opinion or recommendation about the material.\n\n'
    "When unsure, answer lookup.\n\n"
    "Previous exchange:\n{prior_turn}\n\n"
    "Latest user message:\n{question}\n\n"
    'Return JSON only: {{"mode": "lookup"}} or {{"mode": "assessment"}}\n'
)


def _prior_turn_text(prior_turn: PriorTurn | None) -> str:
    """Render the previous exchange for the classifier.

    Args:
        prior_turn (PriorTurn | None): The previous exchange, if any.

    Returns:
        str: The exchange, or ``(none)``.
    """
    if prior_turn is None:
        return "(none)"
    return f"User: {prior_turn.user_text}\nAssistant: {prior_turn.assistant_text[:_PRIOR_ANSWER_CHARS]}"


def _parse_mode(reply: str) -> AnswerMode:
    """Read the mode out of the classifier's reply.

    Args:
        reply (str): The raw completion.

    Returns:
        AnswerMode: ``"assessment"`` only for an explicit verdict, else ``"lookup"``.
    """
    visible, _ = strip_reasoning(reply)
    match = _JSON_RE.search(visible)
    if match is None:
        return "lookup"
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "lookup"
    mode = str(payload.get("mode", "")).strip().lower() if isinstance(payload, dict) else ""
    return "assessment" if mode == "assessment" else "lookup"


def classify_turn(
    llm: LLM | None,
    *,
    question: str,
    prior_turn: PriorTurn | None,
    slots: threading.Semaphore = CLASSIFY_SLOT_POOL,
    clock: Callable[[], float] = time.monotonic,
) -> AnswerMode:
    """Return whether a turn asks for an assessment.

    Fail-soft by design: no model, no free slot, a call past ``CLASSIFY_TIMEOUT_S``,
    a failed call or an unreadable reply all classify as a lookup, which is what
    the turn would have been without assessment turns. The call runs on its own
    thread and is abandoned at the deadline however its reply arrives; a reply
    after it is never read.

    Args:
        llm (LLM | None): The plain (non-reasoning) chat model. A call is abandoned
            at the deadline; one making a single attempt per call ends an abandoned
            call, and frees its slot, at its own timeout.
        question (str): The user's message, verbatim.
        prior_turn (PriorTurn | None): The previous exchange, for follow-ups.
        slots (threading.Semaphore): The classification slots shared by every turn
            in the process (``CLASSIFY_SLOT_POOL``); injectable for tests.
        clock (Callable[[], float]): Monotonic clock the slot wait is measured with.

    Returns:
        AnswerMode: ``"assessment"`` or ``"lookup"``.
    """
    if llm is None or not question.strip():
        return "lookup"
    template = load_localized_prompt("assessment_classify", default=DEFAULT_ASSESSMENT_CLASSIFY_PROMPT)
    prompt = template.format(question=question.strip(), prior_turn=_prior_turn_text(prior_turn))
    timeout = CLASSIFY_TIMEOUT_S
    try:
        pending = run_bounded(
            functools.partial(llm.complete, prompt, max_tokens=CLASSIFY_MAX_TOKENS, timeout=timeout),
            timeout,
            slots=slots,
            clock=clock,
            name=CLASSIFY_THREAD_NAME,
            slot_wait=CLASSIFY_SLOT_WAIT_S,
        )
        reply = None if pending is None else str(pending.result().text or "")
    except NoSlot:
        logger.warning("Turn classification found no free slot; answering as a lookup")
        return "lookup"
    except Exception as exc:
        logger.warning("Turn classification failed ({}); answering as a lookup", type(exc).__name__)
        return "lookup"
    if reply is None:
        logger.warning("Turn classification ran past its deadline; answering as a lookup")
        return "lookup"
    return _parse_mode(reply)
