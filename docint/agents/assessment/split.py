"""The two-part answer protocol of an assessment turn.

The answer model writes what the sources say, then a line holding only
``[[ASSESSMENT]]``, then its own judgement. The backend swaps that sentinel for
a localized label, so the boundary between source-backed text and model
judgement survives into every surface the answer reaches — the chat, persisted
history and report exports — without any of them knowing the protocol.
"""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from docint.utils.ui_strings import ui_string

SENTINEL = "[[ASSESSMENT]]"
#: The rule a stored answer carries between its source-backed part and the label; the same in every language.
ASSESSMENT_RULE = "\n\n---\n\n"

# Tolerates what models do to a marker: case, single brackets, bold, heading marks.
_SENTINEL_RE = re.compile(r"(?:#{1,6}\s*)?(?:\*\*|__)?\[{1,2}\s*assessment\s*\]{1,2}(?:\*\*|__)?", re.IGNORECASE)
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
# str.lower() can change length (U+0130), which would misalign buffer indices.
_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)

AnswerOutcome = Literal["split", "unsplit", "empty"]


@dataclass(frozen=True)
class LabelledAnswer:
    """An assessment answer with the backend-inserted label in place.

    Attributes:
        text (str): The text every surface shows and persists.
        evidence_chars (int | None): Length of the source-backed part at the
            start of ``text``; ``0`` when there is none, ``None`` when the text
            carries no assessment part at all.
        outcome (AnswerOutcome): ``split`` when the model separated the parts,
            ``unsplit`` when it did not (the whole text is labelled), ``empty``
            when there was nothing to label.
    """

    text: str
    evidence_chars: int | None
    outcome: AnswerOutcome


def assessment_label(*, with_rule: bool) -> str:
    """Render the localized label that opens the assessment part.

    Args:
        with_rule (bool): Whether a horizontal rule separates the label from
            source-backed text above it. Omitted when nothing precedes it.

    Returns:
        str: Markdown with blank lines around it, so CommonMark never reads the
        paragraph above as a setext heading.
    """
    label = f"**{ui_string('assessment_heading')}**\n\n*{ui_string('assessment_disclaimer')}*\n\n"
    return f"{ASSESSMENT_RULE}{label}" if with_rule else label


def neutralize_sentinel(text: str) -> str:
    """Defuse sentinel look-alikes inside untrusted document text.

    Args:
        text (str): A passage about to be shown to a model.

    Returns:
        str: The passage with every sentinel variant rewritten so it can no
        longer be mistaken for the model's own boundary.
    """
    return _SENTINEL_RE.sub("(assessment)", text)


def compose_labelled_answer(text: str) -> LabelledAnswer:
    """Split a finished answer at its sentinel and insert the label.

    Args:
        text (str): The model's answer with reasoning already stripped.

    Returns:
        LabelledAnswer: The labelled text and where its evidence part ends.
    """
    body = text.strip()
    if not body:
        return LabelledAnswer(text="", evidence_chars=None, outcome="empty")
    match = _SENTINEL_RE.search(body)
    if match is None:
        return LabelledAnswer(text=f"{assessment_label(with_rule=False)}{body}", evidence_chars=0, outcome="unsplit")
    evidence = body[: match.start()].strip()
    assessment = _SENTINEL_RE.sub("", body[match.end() :]).strip()
    if not evidence and not assessment:
        return LabelledAnswer(text="", evidence_chars=None, outcome="empty")
    if not assessment:
        return LabelledAnswer(text=evidence, evidence_chars=None, outcome="split")
    if not evidence:
        return LabelledAnswer(
            text=f"{assessment_label(with_rule=False)}{assessment}", evidence_chars=0, outcome="split"
        )
    return LabelledAnswer(
        text=f"{evidence}{assessment_label(with_rule=True)}{assessment}",
        evidence_chars=len(evidence),
        outcome="split",
    )


def cited_sources(evidence: str, sources: Sequence[dict[str, Any]], *, limit: int = 6) -> list[dict[str, Any]]:
    """Pick the sources a source-backed part cites, in citation order.

    Args:
        evidence (str): The source-backed part of an answer.
        sources (Sequence[dict[str, Any]]): The turn's sources, each carrying
            its ``citation_index``.
        limit (int): How many sources the validator is shown.

    Returns:
        list[dict[str, Any]]: The cited sources, or the first ``limit`` sources
        when the evidence cites none.
    """
    cited: set[int] = set()
    for match in _CITATION_RE.finditer(evidence):
        cited.update(int(number) for number in re.split(r"\s*,\s*", match.group(1)))
    picked = [
        source
        for source in sources
        if isinstance(source.get("citation_index"), int) and source["citation_index"] in cited
    ]
    picked.sort(key=lambda source: source["citation_index"])
    return (picked or list(sources))[:limit]


def _partial_suffix(text: str, pattern: str) -> int:
    """Return how many trailing characters of ``text`` could still grow into ``pattern``.

    Args:
        text (str): The buffered text, lower-cased.
        pattern (str): The marker being watched for, lower-cased.

    Returns:
        int: Length of the longest proper prefix of ``pattern`` that ``text`` ends with.
    """
    for size in range(min(len(pattern) - 1, len(text)), 0, -1):
        if text.endswith(pattern[:size]):
            return size
    return 0


class SentinelStreamSplitter:
    """Swap the first canonical sentinel in a token stream for the label.

    Holds back only a tail that could still grow into the sentinel or a think
    tag. Variants the stream does not recognise are left to
    :func:`compose_labelled_answer`, whose result replaces the streamed text
    when the turn completes.
    """

    def __init__(self) -> None:
        """Start before any text, outside any reasoning block."""
        self._buffer = ""
        self._done = False
        self._in_think = False
        self._visible = False

    def feed(self, token: str) -> str:
        """Accept one token and return the text that is safe to emit now.

        Args:
            token (str): The next streamed token.

        Returns:
            str: Text to forward; empty while a possible match is pending.
        """
        if self._done:
            return token
        self._buffer += token
        out: list[str] = []
        while self._buffer:
            lower = self._buffer.translate(_ASCII_LOWER)
            if self._in_think:
                end = lower.find(_THINK_CLOSE)
                if end < 0:
                    out.append(self._take(len(self._buffer) - _partial_suffix(lower, _THINK_CLOSE), visible=False))
                    break
                out.append(self._take(end + len(_THINK_CLOSE), visible=False))
                self._in_think = False
                continue
            think = lower.find(_THINK_OPEN)
            sentinel = lower.find(SENTINEL.lower())
            hits = [index for index in (think, sentinel) if index >= 0]
            if not hits:
                keep = max(_partial_suffix(lower, _THINK_OPEN), _partial_suffix(lower, SENTINEL.lower()))
                out.append(self._take(len(self._buffer) - keep, visible=True))
                break
            first = min(hits)
            out.append(self._take(first, visible=True))
            if first == think:
                out.append(self._take(len(_THINK_OPEN), visible=False))
                self._in_think = True
                continue
            self._buffer = self._buffer[len(SENTINEL) :]
            out.append(assessment_label(with_rule=self._visible))
            out.append(self._buffer)
            self._buffer = ""
            self._done = True
        return "".join(out)

    def flush(self) -> str:
        """Release whatever is still held back once the stream has ended.

        Returns:
            str: The pending tail, verbatim.
        """
        rest, self._buffer = self._buffer, ""
        return rest

    def _take(self, size: int, *, visible: bool) -> str:
        """Remove and return the first ``size`` characters of the buffer.

        Args:
            size (int): How many characters to release.
            visible (bool): Whether they are answer text rather than reasoning.

        Returns:
            str: The released characters.
        """
        chunk, self._buffer = self._buffer[:size], self._buffer[size:]
        if visible and chunk.strip():
            self._visible = True
        return chunk
