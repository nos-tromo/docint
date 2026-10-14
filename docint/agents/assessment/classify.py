"""Decide per turn whether a question asks for a judgement or for information."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from loguru import logger

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


def classify_turn(llm: LLM | None, *, question: str, prior_turn: PriorTurn | None) -> AnswerMode:
    """Return whether a turn asks for an assessment.

    Fail-soft by design: no model, a failed call or an unreadable reply all
    classify as a lookup, which is what the turn would have been without
    assessment turns.

    Args:
        llm (LLM | None): The plain (non-reasoning) chat model.
        question (str): The user's message, verbatim.
        prior_turn (PriorTurn | None): The previous exchange, for follow-ups.

    Returns:
        AnswerMode: ``"assessment"`` or ``"lookup"``.
    """
    if llm is None or not question.strip():
        return "lookup"
    template = load_localized_prompt("assessment_classify", default=DEFAULT_ASSESSMENT_CLASSIFY_PROMPT)
    prompt = template.format(question=question.strip(), prior_turn=_prior_turn_text(prior_turn))
    try:
        reply = str(llm.complete(prompt, max_tokens=CLASSIFY_MAX_TOKENS, timeout=CLASSIFY_TIMEOUT_S).text or "")
    except Exception as exc:
        logger.warning("Turn classification failed ({}); answering as a lookup", type(exc).__name__)
        return "lookup"
    return _parse_mode(reply)
