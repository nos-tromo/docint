"""Run one assessment turn: gather evidence, then stream a two-part answer.

Knows nothing about :class:`~docint.core.rag.RAG`: the answer model, the
LLM-view renderer and the optional research step all arrive as arguments.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import stream_chat_response_to_tokens
from llama_index.core.schema import BaseNode, NodeWithScore

from docint.agents.assessment.ledger import EvidenceLedger, answer_budget, render_context
from docint.agents.assessment.split import SentinelStreamSplitter
from docint.agents.types import AgentProgress, PriorTurn
from docint.utils.prompt_loader import load_localized_prompt

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
        nodes (list[NodeWithScore]): The evidence, numbered in ledger order.
        steps (int): Research rounds that ran.
        actions (list[dict[str, Any]]): The seed and every search, in order.
        reason (str | None): Why research stopped short, or ``None``.
        shortened (int): Entries cut to fit the answer's context.
    """

    raw_text: str
    nodes: list[NodeWithScore]
    steps: int
    actions: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None
    shortened: int = 0


def prior_turn_block(prior_turn: PriorTurn | None) -> str:
    """Render the previous exchange for an assessment prompt.

    Args:
        prior_turn (PriorTurn | None): The previous exchange, if any.

    Returns:
        str: The block, or a placeholder when there is none.
    """
    if prior_turn is None:
        return "(no prior turn)"
    return f"Previous user question: {prior_turn.user_text}\nPrevious assistant answer: {prior_turn.assistant_text}"


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
        """
        self._answer_llm = answer_llm
        self._llm_view = llm_view
        self._ctx_window = ctx_window
        self._ratio = char_token_ratio
        self._reasoning = reasoning
        self._researcher = researcher
        self._answer_prompt = answer_prompt or load_localized_prompt(
            "assessment_answer", default=DEFAULT_ASSESSMENT_ANSWER_PROMPT
        )

    def run(
        self, *, question: str, prior_turn: PriorTurn | None, seed: Sequence[NodeWithScore]
    ) -> Iterator[AgentProgress | str | AssessmentResult]:
        """Run the turn.

        Args:
            question (str): The user's message, verbatim.
            prior_turn (PriorTurn | None): The previous exchange, in full.
            seed (Sequence[NodeWithScore]): The lookup's own evidence for the
                question, or a scoped turn's pinned chunks.

        Yields:
            AgentProgress | str | AssessmentResult: Progress steps, answer text
            with the label already swapped in, then exactly one result.
        """
        budget = answer_budget(
            ctx_window=self._ctx_window,
            template=self._answer_prompt,
            reasoning=self._reasoning,
            char_token_ratio=self._ratio,
        )
        # Pinned or seeded evidence is never dropped; packing shortens it instead.
        ledger = EvidenceLedger(cap=max(budget.max_entries, len(seed)))
        seeded = ledger.add(seed)
        actions: list[dict[str, Any]] = [{"action": "seed", "detail": "", "found": len(seeded)}]
        yield AgentProgress(step=0, action="seed", detail="", found=len(seeded))
        steps, reason = 0, None
        if self._researcher is not None and not ledger.full:
            for item in self._researcher.run(question=question, prior_turn=prior_turn, ledger=ledger):
                if isinstance(item, ResearchOutcome):
                    steps, reason = item.steps, item.reason
                    actions.extend(item.actions)
                else:
                    yield item
        if len(ledger) == 0:
            yield AssessmentResult(raw_text="", nodes=[], steps=steps, actions=actions, reason=reason or "no_evidence")
            return
        nodes = ledger.numbered()
        context, shortened = render_context(
            nodes, llm_view=self._llm_view, budget_tokens=budget.evidence_tokens, char_token_ratio=self._ratio
        )
        prompt = self._answer_prompt.format(
            prior_turn=prior_turn_block(prior_turn), context_str=context, question=question.strip()
        )
        yield AgentProgress(step=steps + 1, action="answer", detail="", found=len(nodes))
        splitter = SentinelStreamSplitter()
        raw: list[str] = []
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        for token in stream_chat_response_to_tokens(
            self._answer_llm.stream_chat(messages, max_tokens=budget.output_tokens)
        ):
            raw.append(token)
            visible = splitter.feed(token)
            if visible:
                yield visible
        tail = splitter.flush()
        if tail:
            yield tail
        yield AssessmentResult(
            raw_text="".join(raw), nodes=nodes, steps=steps, actions=actions, reason=reason, shortened=shortened
        )
