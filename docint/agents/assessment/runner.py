"""Run one assessment turn: gather evidence, then stream a two-part answer.

Knows nothing about :class:`~docint.core.rag.RAG`: the answer model, the
LLM-view renderer and the optional research step all arrive as arguments.
"""

from __future__ import annotations

import contextlib
import itertools
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import stream_chat_response_to_tokens
from llama_index.core.schema import BaseNode, NodeWithScore
from loguru import logger

from docint.agents.assessment.ledger import (
    OVERFLOW_RETRY_SHARE,
    PRIOR_TURN_TOKENS,
    QUESTION_TOKENS,
    EvidenceLedger,
    Room,
    answer_budget,
    fit_ledger,
    is_context_overflow,
    render_context,
)
from docint.agents.assessment.split import ASSESSMENT_RULE, SentinelStreamSplitter
from docint.agents.types import AgentProgress, PriorTurn
from docint.utils.embed_chunking import estimate_tokens
from docint.utils.prompt_loader import load_localized_prompt
from docint.utils.text_budget import allocate_shortest_first, fit_with_marker

if TYPE_CHECKING:
    from llama_index.core.llms import LLM

SearchFn = Callable[[str], list[NodeWithScore]]
"""Run one search through the turn's engine and return its ranked passages."""

DEFAULT_ASSESSMENT_ANSWER_PROMPT = (
    "You are assessing material from retrieved evidence.\n\n"
    "Conversation continuity:\n{prior_turn}\n\n"
    "Retrieved context snippets:\n---------------------\n{context_str}\n---------------------\n"
    "The user's question: {question}\n\n"
    "Part 1: what the snippets say, every claim cited by its citation_index in square brackets.\n"
    "Then a line containing only [[ASSESSMENT]]\n"
    "Part 2: your own assessment; flag general knowledge, cite snippets you react to.\n\n"
    "Answer:\n"
)

PRIOR_ANSWER_CHARS = 4000
"""Longest earlier assistant answer, in characters, that research and the answer step are shown.

An earlier assessment's two parts share it.
"""

#: How the prior-turn block labels the two parts of an earlier assessment.
PRIOR_SOURCED_LABEL = "Previous assistant answer, source-backed part (its [n] refer to the previous turn's sources)"
PRIOR_ASSESSMENT_LABEL = "Previous assistant answer, the model's own assessment"


class AssessmentDoesNotFit(ValueError):
    """The evidence cannot be shown legibly in one answer, so the turn is answered as a lookup.

    A ``ValueError`` naming the context window, so that if it ever escapes its
    callers the API still reports ``context_overflow``.

    Attributes:
        passages (int): How many passages the evidence held.
    """

    def __init__(self, passages: int) -> None:
        """Record how many passages did not fit.

        Args:
            passages (int): How many passages the evidence held.
        """
        super().__init__(f"{passages} passages cannot be shown legibly in the model's context window.")
        self.passages = passages


@dataclass(frozen=True)
class ResearchOutcome:
    """How a research step ended.

    Attributes:
        steps (int): Model rounds that returned a tool call.
        actions (list[dict[str, Any]]): One ``{action, detail, found}`` per search.
        reason (str | None): Static token for an early stop, or ``None``.
    """

    steps: int
    actions: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None


class Researcher(Protocol):
    """Gathers more evidence into the ledger before the answer is written."""

    def run(
        self, *, question: str, prior_turn: PriorTurn | None, ledger: EvidenceLedger
    ) -> Iterator[AgentProgress | ResearchOutcome]:
        """Search until the evidence suffices.

        Args:
            question (str): The user's message, verbatim.
            prior_turn (PriorTurn | None): The previous exchange, if any.
            ledger (EvidenceLedger): The evidence so far; extended in place.

        Returns:
            Iterator[AgentProgress | ResearchOutcome]: Progress, then one outcome.
        """
        ...


@dataclass(frozen=True)
class AssessmentResult:
    """What an assessment turn produced, for normalization and persistence.

    Attributes:
        raw_text (str): The answer exactly as streamed, sentinel included.
        nodes (list[NodeWithScore]): The evidence the answer was shown, numbered in
            ledger order; research additions left out to fit are not among them.
        steps (int): Research rounds that ran.
        actions (list[dict[str, Any]]): The seed and every search, in order.
        reason (str | None): Why research stopped short, or why the evidence it
            found was cut short (``context_budget``), or ``None``; the first cause is kept.
        shortened (int): Entries cut to fit the answer's context.
    """

    raw_text: str
    nodes: list[NodeWithScore]
    steps: int
    actions: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None
    shortened: int = 0


def _prior_answer(prior_turn: PriorTurn) -> str:
    """Render the earlier answer's lines of the prior-turn block.

    An assessment whose stored boundary is ``0``, or sits on its rule, is shown as
    two labelled parts that share ``PRIOR_ANSWER_CHARS`` shortest-first, so a long
    source-backed part cannot push the judgement out; the assessment part keeps its
    localized heading and disclaimer, and an empty part is left out. Any other
    answer, a boundary out of range or off the rule included, is shown as before.

    Args:
        prior_turn (PriorTurn): The previous exchange.

    Returns:
        str: One labelled line per part shown.
    """
    text, cut = prior_turn.assistant_text, prior_turn.evidence_chars
    aligned = cut is not None and 0 <= cut <= len(text) and (cut == 0 or text.startswith(ASSESSMENT_RULE, cut))
    parts = (text[:cut], text[cut:].removeprefix(ASSESSMENT_RULE)) if aligned else ("", "")
    if not any(parts):
        return f"Previous assistant answer: {fit_with_marker(text, PRIOR_ANSWER_CHARS)[0]}"
    shares = allocate_shortest_first([len(part) for part in parts], PRIOR_ANSWER_CHARS)
    labelled = zip((PRIOR_SOURCED_LABEL, PRIOR_ASSESSMENT_LABEL), parts, shares, strict=True)
    return "\n".join(f"{label}: {fit_with_marker(part, share)[0]}" for label, part, share in labelled if part)


def prior_turn_block(prior_turn: PriorTurn | None) -> str:
    """Render the previous exchange for an assessment prompt.

    Args:
        prior_turn (PriorTurn | None): The previous exchange, if any.

    Returns:
        str: The block, its earlier answer within ``PRIOR_ANSWER_CHARS`` and an earlier
        assessment's two parts each labelled (see :func:`_prior_answer`), or a
        placeholder when there is none.
    """
    if prior_turn is None:
        return "(no prior turn)"
    return f"Previous user question: {prior_turn.user_text}\n{_prior_answer(prior_turn)}"


class AssessmentRunner:
    """Seed the evidence, let research extend it, then stream the answer."""

    def __init__(
        self,
        *,
        answer_llm: LLM,
        llm_view: Callable[[BaseNode], BaseNode],
        ctx_window: int,
        char_token_ratio: float,
        reasoning: bool,
        researcher: Researcher | None = None,
        answer_prompt: str | None = None,
        count_tokens: Callable[[str], int] | None = None,
    ) -> None:
        """Bind the turn's models and limits.

        Args:
            answer_llm (LLM): Writes the answer; the request's reasoning toggle
                has already chosen it.
            llm_view (Callable[[BaseNode], BaseNode]): Hides metadata the
                prompt must not see, as the grounded synthesizer does.
            ctx_window (int): The chat model's context window in tokens.
            char_token_ratio (float): Characters per token for budgeting.
            reasoning (bool): Whether ``answer_llm`` thinks before answering.
            researcher (Researcher | None): Extends the evidence; ``None`` on a
                scoped turn, whose evidence is fixed.
            answer_prompt (str | None): Template override; the localized
                ``assessment_answer`` prompt by default.
            count_tokens (Callable[[str], int] | None): A tokenizer, when one is
                loaded; budgets then follow the measured text instead of the ratio.
        """
        self._answer_llm = answer_llm
        self._llm_view = llm_view
        self._ctx_window = ctx_window
        self._ratio = char_token_ratio
        self._reasoning = reasoning
        self._researcher = researcher
        self._count_tokens = count_tokens
        self._answer_prompt = answer_prompt or load_localized_prompt(
            "assessment_answer", default=DEFAULT_ASSESSMENT_ANSWER_PROMPT
        )

    def _overhead_tokens(self, text: str) -> int:
        """Measure the prior turn and the question, counted when a tokenizer is loaded.

        Args:
            text (str): The prior-turn block followed by the question.

        Returns:
            int: Their tokens; the ratio estimate without a tokenizer, or when it fails.
        """
        if self._count_tokens is not None:
            # A failed count leaves the estimate, as packing does.
            with contextlib.suppress(Exception):
                return self._count_tokens(text)
        return estimate_tokens(text, self._ratio)

    def _fit(
        self, nodes: list[NodeWithScore], *, keep: int, budget_tokens: int, compact: bool = False
    ) -> list[NodeWithScore]:
        """Leave out the latest research additions until every passage is legible in the budget.

        Args:
            nodes (list[NodeWithScore]): The numbered evidence.
            keep (int): The seed count, never left out.
            budget_tokens (int): Tokens the rendered evidence may use.
            compact (bool): Whether headers are compacted.

        Returns:
            list[NodeWithScore]: The evidence to render.
        """
        return fit_ledger(
            nodes,
            keep=keep,
            llm_view=self._llm_view,
            budget_tokens=budget_tokens,
            char_token_ratio=self._ratio,
            count_tokens=self._count_tokens,
            compact=compact,
        )

    def _room(self, nodes: list[NodeWithScore], *, budget_tokens: int, compact: bool = False) -> Room:
        """Measure how the evidence sits in a budget.

        Args:
            nodes (list[NodeWithScore]): The numbered evidence.
            budget_tokens (int): Tokens the rendered evidence may use.
            compact (bool): Whether headers are compacted.

        Returns:
            Room: The room, its density measured on ``nodes``.
        """
        return Room(
            nodes,
            llm_view=self._llm_view,
            budget_tokens=budget_tokens,
            char_token_ratio=self._ratio,
            count_tokens=self._count_tokens,
            compact=compact,
        )

    def _open_answer(
        self, nodes: list[NodeWithScore], *, block: str, question: str, budget_tokens: int, compact: bool = False
    ) -> tuple[Iterator[str], int, bool]:
        """Render the evidence into the answer prompt, send it and pull the first token.

        The endpoint's HTTP error, a context overflow included, surfaces on that
        first pull, before any answer text exists.

        Args:
            nodes (list[NodeWithScore]): The numbered evidence.
            block (str): The rendered prior turn.
            question (str): The user's message, stripped.
            budget_tokens (int): Tokens the rendered evidence may use.
            compact (bool): Whether headers are compacted, as on the overflow retry.

        Returns:
            tuple[Iterator[str], int, bool]: The answer tokens, the first one
            included; how many entries were shortened; and whether the evidence
            sent kept every passage legible (see :class:`RenderedContext`).
        """
        rendered = render_context(
            nodes,
            llm_view=self._llm_view,
            budget_tokens=budget_tokens,
            char_token_ratio=self._ratio,
            count_tokens=self._count_tokens,
            compact=compact,
        )
        prompt = self._answer_prompt.format(prior_turn=block, context_str=rendered.context, question=question)
        tokens = stream_chat_response_to_tokens(
            self._answer_llm.stream_chat([ChatMessage(role=MessageRole.USER, content=prompt)])
        )
        first = next(tokens, None)
        return itertools.chain([] if first is None else [first], tokens), rendered.shortened, rendered.legible

    def run(
        self, *, question: str, prior_turn: PriorTurn | None, seed: Sequence[NodeWithScore]
    ) -> Iterator[AgentProgress | str | AssessmentResult]:
        """Run the turn.

        ``context_budget`` is reported when the room ended research before its step
        cap, when the seed left no room for research to run, when fewer passages
        reach the answer than the ledger holds, or when the evidence sent, after the
        first fit or the overflow retry's, is not legible; a reason already given is kept.

        Args:
            question (str): The user's message, verbatim.
            prior_turn (PriorTurn | None): The previous exchange, in full.
            seed (Sequence[NodeWithScore]): The lookup's passages for the
                question, each section keeping every child retrieved under it,
                or a scoped turn's pinned chunks.

        Yields:
            AgentProgress | str | AssessmentResult: Progress steps, answer text
            with the label already swapped in, then exactly one result.

        Raises:
            AssessmentDoesNotFit: Before anything is yielded, when the seed
                cannot be shown legibly in the answer's context.
            ValueError: Naming the context window, when the answer overflows it
                twice, or when on the retry even compact headers leave no room.
        """
        budget = answer_budget(
            ctx_window=self._ctx_window,
            template=self._answer_prompt,
            reasoning=self._reasoning,
            char_token_ratio=self._ratio,
        )
        block = prior_turn_block(prior_turn)
        # The prior turn and question can outgrow their reserve; the evidence gives way.
        overhead = self._overhead_tokens(block + question.strip())
        extra = max(0, overhead - PRIOR_TURN_TOKENS - QUESTION_TOKENS)
        evidence_budget = max(0, budget.evidence_tokens - extra)
        # Pinned or seeded evidence is never dropped; packing shortens it instead.
        ledger = EvidenceLedger(cap=max(budget.max_entries, len({scored.node.node_id for scored in seed})))
        seeded = ledger.add(seed)
        seed_count = len(ledger)
        # Measured on the seed and held, so research is admitted only while the evidence stays legible.
        room = self._room(ledger.numbered(), budget_tokens=evidence_budget)
        if not room.legible:
            raise AssessmentDoesNotFit(len(ledger))
        ledger.restrict(room)
        actions: list[dict[str, Any]] = [{"action": "seed", "detail": "", "found": len(seeded)}]
        yield AgentProgress(step=0, action="seed", detail="", found=len(seeded))
        steps, reason = 0, None
        if self._researcher is not None and ledger.out_of_room:
            # The seed left no room for one more legible passage, so research never runs.
            reason = "context_budget"
        elif self._researcher is not None and not ledger.full:
            try:
                for item in self._researcher.run(question=question, prior_turn=prior_turn, ledger=ledger):
                    if isinstance(item, ResearchOutcome):
                        steps, reason = item.steps, item.reason
                        actions.extend(item.actions)
                    else:
                        yield item
            except Exception as exc:
                reason = "research_failed"
                logger.warning("Assessment research stopped (research_failed): {}", type(exc).__name__)
        if len(ledger) == 0:
            yield AssessmentResult(raw_text="", nodes=[], steps=steps, actions=actions, reason="no_evidence")
            return
        # Remeasured on everything found, which catches research denser than the seed.
        nodes = self._fit(ledger.numbered(), keep=seed_count, budget_tokens=evidence_budget)
        yield AgentProgress(step=steps + 1, action="answer", detail="", found=len(nodes))
        try:
            tokens, shortened, legible = self._open_answer(
                nodes, block=block, question=question.strip(), budget_tokens=evidence_budget
            )
        except Exception as exc:
            if not is_context_overflow(exc):
                raise
            # Only before the first token: nothing has been shown that a retry would contradict.
            logger.warning(
                "Assessment answer overflowed the context window ({}); retrying with less evidence",
                type(exc).__name__,
            )
            retry_budget = int(evidence_budget * OVERFLOW_RETRY_SHARE)
            nodes = self._fit(nodes, keep=seed_count, budget_tokens=retry_budget, compact=True)
            # Texts below the legible minimum are still sent; headers with no room left are not.
            if not self._room(nodes, budget_tokens=retry_budget, compact=True).headers_fit:
                raise ValueError("The assessment prompt exceeds the model's context window.") from exc
            try:
                tokens, shortened, legible = self._open_answer(
                    nodes, block=block, question=question.strip(), budget_tokens=retry_budget, compact=True
                )
            except Exception as retry_exc:
                if is_context_overflow(retry_exc):
                    raise ValueError("The assessment prompt exceeds the model's context window.") from retry_exc
                raise
        splitter = SentinelStreamSplitter()
        raw: list[str] = []
        for token in tokens:
            raw.append(token)
            visible = splitter.feed(token)
            if visible:
                yield visible
        tail = splitter.flush()
        if tail:
            yield tail
        if reason is None and (len(nodes) < len(ledger) or not legible):
            # Passages were left out, or the evidence sent was cut below its legible minimum.
            reason = "context_budget"
        yield AssessmentResult(
            raw_text="".join(raw), nodes=nodes, steps=steps, actions=actions, reason=reason, shortened=shortened
        )
