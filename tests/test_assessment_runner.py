"""The assessment runner: seed, research hook, streamed two-part answer. All text is invented."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, cast

import pytest
from assessment_fakes import llm_view, scored_node, streaming_llm
from llama_index.core.schema import NodeWithScore

from docint.agents.assessment.ledger import EvidenceLedger
from docint.agents.assessment.runner import AssessmentResult, AssessmentRunner, ResearchOutcome, prior_turn_block
from docint.agents.assessment.split import SENTINEL, assessment_label
from docint.agents.types import AgentProgress, PriorTurn
from docint.utils.embed_chunking import estimate_tokens

QUESTION = "What could be improved in this study?"
ANSWER = ["The study evaluates on one dataset [1].", f"\n{SENTINEL}\n", "A second dataset would help."]


@pytest.fixture(autouse=True)
def _english(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the label and prompt language.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")


def _runner(
    tokens: list[str] | None = None,
    *,
    reasoning: bool = False,
    researcher: Any = None,
    count_tokens: Callable[[str], int] | None = None,
) -> tuple[AssessmentRunner, Any]:
    """Build a runner over a streaming double.

    Args:
        tokens (list[str] | None): The answer tokens; ``ANSWER`` by default.
        reasoning (bool): Whether the answer model counts as a reasoning model.
        researcher (Any): An optional research step.
        count_tokens (Callable[[str], int] | None): An optional tokenizer.

    Returns:
        tuple[AssessmentRunner, Any]: The runner and its model double.
    """
    llm = streaming_llm(tokens or ANSWER)
    runner = AssessmentRunner(
        answer_llm=cast(Any, llm),
        llm_view=llm_view,
        ctx_window=8192,
        char_token_ratio=3.5,
        reasoning=reasoning,
        researcher=researcher,
        count_tokens=count_tokens,
    )
    return runner, llm


def _long_seed(count: int) -> list[NodeWithScore]:
    """Return ``count`` passages of about 2,000 characters each, too long to show whole.

    Args:
        count (int): How many passages.

    Returns:
        list[NodeWithScore]: The seed evidence.
    """
    return [
        scored_node(f"n{number}", f"Passage {number}. " + "The result holds. " * 111, page=number)
        for number in range(1, count + 1)
    ]


def _seed() -> list[NodeWithScore]:
    """Return two passages of an invented study.

    Returns:
        list[NodeWithScore]: The seed evidence.
    """
    return [
        scored_node("n1", "We evaluate on one dataset.", filename="study.pdf", page=4),
        scored_node("n2", "Results improve by two points.", filename="study.pdf", page=6),
    ]


def test_a_seeded_turn_streams_progress_then_a_labelled_answer() -> None:
    """The stream carries the seed step, the label-swapped tokens and one result."""
    runner, _ = _runner()

    items = list(runner.run(question=QUESTION, prior_turn=None, seed=_seed()))

    progress = [item for item in items if isinstance(item, AgentProgress)]
    tokens = [item for item in items if isinstance(item, str)]
    result = items[-1]
    assert [step.action for step in progress] == ["seed", "answer"]
    assert progress[0].found == 2
    assert "".join(tokens) == "".join(ANSWER).replace(SENTINEL, assessment_label(with_rule=True))
    assert isinstance(result, AssessmentResult)
    assert result.raw_text == "".join(ANSWER)
    assert [node.node.metadata["citation_index"] for node in result.nodes] == [1, 2]
    assert (result.steps, result.reason) == (0, None)
    assert result.actions == [{"action": "seed", "detail": "", "found": 2}]


def test_the_answer_call_carries_question_evidence_and_prior_turn_but_no_max_tokens() -> None:
    """One user message holds everything; the reply reserve stays inside the budget, never sent as a cap."""
    runner, llm = _runner()
    prior = PriorTurn(user_text="Summarise it.", assistant_text="It has one experiment [1].")

    list(runner.run(question=QUESTION, prior_turn=prior, seed=_seed()))

    call = llm.stream_chat.call_args
    prompt = call.args[0][0].content
    assert QUESTION in prompt
    assert "citation_index: 1" in prompt
    assert "citation_index: 2" in prompt
    assert "It has one experiment [1]." in prompt
    assert "max_tokens" not in call.kwargs


def test_a_reasoning_model_keeps_a_larger_reserve_free_without_sending_it() -> None:
    """A thinking model spends reply tokens on reasoning first, so its evidence is packed tighter."""
    runner, llm = _runner()
    thinking, thinking_llm = _runner(reasoning=True)

    list(runner.run(question=QUESTION, prior_turn=None, seed=_long_seed(10)))
    list(thinking.run(question=QUESTION, prior_turn=None, seed=_long_seed(10)))

    plain_prompt = llm.stream_chat.call_args.args[0][0].content
    thinking_prompt = thinking_llm.stream_chat.call_args.args[0][0].content
    assert len(thinking_prompt) < len(plain_prompt)
    assert "max_tokens" not in thinking_llm.stream_chat.call_args.kwargs


def test_no_evidence_skips_the_answer_call() -> None:
    """With nothing to assess, no model call is made and the turn says why."""
    runner, llm = _runner()

    result = list(runner.run(question=QUESTION, prior_turn=None, seed=[]))[-1]

    assert isinstance(result, AssessmentResult)
    assert (result.reason, result.nodes) == ("no_evidence", [])
    llm.stream_chat.assert_not_called()


def test_a_researcher_extends_the_ledger_and_its_progress_is_forwarded() -> None:
    """Research adds evidence in place; its steps reach the client in order."""

    class _Researcher:
        """Adds one passage, then stops on the time budget."""

        def run(
            self, *, question: str, prior_turn: PriorTurn | None, ledger: EvidenceLedger
        ) -> Iterator[AgentProgress | ResearchOutcome]:
            """Add a limitations passage.

            Args:
                question (str): The user's message.
                prior_turn (PriorTurn | None): The previous exchange.
                ledger (EvidenceLedger): The evidence so far.

            Yields:
                AgentProgress | ResearchOutcome: One search, then the outcome.
            """
            added = ledger.add([scored_node("n3", "Limitations: small sample.", filename="study.pdf", page=9)])
            yield AgentProgress(step=1, action="search", detail="limitations", found=len(added))
            yield ResearchOutcome(
                steps=1, actions=[{"action": "search", "detail": "limitations", "found": 1}], reason="time_budget"
            )

    runner, _ = _runner(researcher=_Researcher())

    items = list(runner.run(question=QUESTION, prior_turn=None, seed=_seed()))

    progress = [item for item in items if isinstance(item, AgentProgress)]
    result = items[-1]
    assert [step.action for step in progress] == ["seed", "search", "answer"]
    assert progress[-1].step == 2
    assert isinstance(result, AssessmentResult)
    assert (len(result.nodes), result.steps, result.reason) == (3, 1, "time_budget")
    assert result.actions[-1] == {"action": "search", "detail": "limitations", "found": 1}


def test_the_runner_never_yields_an_empty_string() -> None:
    """An empty token frame would be noise on the wire."""
    runner, _ = _runner(["Fact [1].", "", f"\n{SENTINEL[:4]}", f"{SENTINEL[4:]}\nView."])

    items = list(runner.run(question=QUESTION, prior_turn=None, seed=_seed()))

    assert "" not in [item for item in items if isinstance(item, str)]


def test_a_seed_larger_than_the_cap_is_kept_whole_and_shortened() -> None:
    """Seeded evidence past the ledger cap is kept whole; packing shortens it instead."""
    seed = [
        scored_node(f"n{number}", f"Passage {number}. " + "The result holds. " * 111, page=number)
        for number in range(1, 31)
    ]
    runner, _ = _runner()

    result = list(runner.run(question=QUESTION, prior_turn=None, seed=seed))[-1]

    assert isinstance(result, AssessmentResult)
    assert len(result.nodes) == 30
    assert result.shortened > 0
    assert sorted(node.node.metadata["citation_index"] for node in result.nodes) == list(range(1, 31))


def test_a_long_prior_answer_still_fits_the_window() -> None:
    """A long earlier answer is capped and the evidence gives way, so the prompt fits the window."""
    seed = [
        scored_node(f"n{number}", f"Passage {number}. " + "The result holds. " * 111, page=number)
        for number in range(1, 24)
    ]
    prior = PriorTurn(user_text="Summarise it.", assistant_text="The study reports a result. " * 143)
    runner, llm = _runner()

    list(runner.run(question=QUESTION, prior_turn=prior, seed=seed))

    prompt = llm.stream_chat.call_args.args[0][0].content
    assert estimate_tokens(prompt, 3.5) + 1024 <= 8192


def test_a_researcher_that_raises_degrades_to_research_failed() -> None:
    """A failing research step ends the search; the answer streams from the ledger so far."""

    class _Researcher:
        """Adds one passage, then fails."""

        def run(
            self, *, question: str, prior_turn: PriorTurn | None, ledger: EvidenceLedger
        ) -> Iterator[AgentProgress | ResearchOutcome]:
            """Add a limitations passage, then raise.

            Args:
                question (str): The user's message.
                prior_turn (PriorTurn | None): The previous exchange.
                ledger (EvidenceLedger): The evidence so far.

            Yields:
                AgentProgress | ResearchOutcome: One search, then a failure.
            """
            ledger.add([scored_node("n3", "Limitations: small sample.", filename="study.pdf", page=9)])
            yield AgentProgress(step=1, action="search", detail="limitations", found=1)
            raise RuntimeError("search backend unavailable")

    runner, llm = _runner(researcher=_Researcher())

    items = list(runner.run(question=QUESTION, prior_turn=None, seed=_seed()))

    result = items[-1]
    assert isinstance(result, AssessmentResult)
    assert result.reason == "research_failed"
    assert len(result.nodes) == 3
    assert result.raw_text == "".join(ANSWER)
    llm.stream_chat.assert_called_once()


def test_an_empty_ledger_always_reports_no_evidence() -> None:
    """Whatever stopped research, an empty ledger is reported as no evidence."""

    class _Researcher:
        """Adds nothing and reports a tool outage."""

        def run(
            self, *, question: str, prior_turn: PriorTurn | None, ledger: EvidenceLedger
        ) -> Iterator[AgentProgress | ResearchOutcome]:
            """Stop at once with a tool outage.

            Args:
                question (str): The user's message.
                prior_turn (PriorTurn | None): The previous exchange.
                ledger (EvidenceLedger): The evidence so far.

            Yields:
                AgentProgress | ResearchOutcome: The outcome alone.
            """
            yield ResearchOutcome(steps=0, actions=[], reason="tools_unavailable")

    runner, llm = _runner(researcher=_Researcher())

    result = list(runner.run(question=QUESTION, prior_turn=None, seed=[]))[-1]

    assert isinstance(result, AssessmentResult)
    assert result.reason == "no_evidence"
    llm.stream_chat.assert_not_called()


def _prompt(runner: AssessmentRunner, llm: Any, seed: list[NodeWithScore], prior: PriorTurn | None = None) -> str:
    """Run a turn and return the prompt the answer call received.

    Args:
        runner (AssessmentRunner): The runner.
        llm (Any): Its model double.
        seed (list[NodeWithScore]): The seed evidence.
        prior (PriorTurn | None): The previous exchange, if any.

    Returns:
        str: The answer prompt.
    """
    list(runner.run(question=QUESTION, prior_turn=prior, seed=seed))
    return str(llm.stream_chat.call_args.args[0][0].content)


def test_a_dense_tokenizer_packs_less_evidence_than_the_configured_ratio() -> None:
    """Text that tokenizes at two characters a token must not be packed at three and a half."""
    plain = _prompt(*_runner(), _long_seed(10))
    dense = _prompt(*_runner(count_tokens=lambda text: len(text) // 2), _long_seed(10))

    assert len(dense) < len(plain)


def test_a_tokenizer_that_fails_falls_back_to_the_configured_ratio() -> None:
    """A broken tokenizer leaves the turn exactly as it is without one."""

    def _broken(text: str) -> int:
        raise RuntimeError("tokenizer unavailable")

    plain = _prompt(*_runner(), _long_seed(10))
    fallback = _prompt(*_runner(count_tokens=_broken), _long_seed(10))

    assert fallback == plain


def test_the_prior_turn_and_question_are_measured_with_the_tokenizer() -> None:
    """Their overhead is counted, not estimated, when a tokenizer is available."""
    measured: list[str] = []

    def _count(text: str) -> int:
        measured.append(text)
        return max(1, len(text) // 4)

    prior = PriorTurn(user_text="Summarise it.", assistant_text="It has one experiment [1].")

    _prompt(*_runner(count_tokens=_count), _seed(), prior)

    assert prior_turn_block(prior) + QUESTION in measured
