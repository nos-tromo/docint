"""The assessment runner: seed, research hook, streamed two-part answer. All text is invented."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest
from assessment_fakes import llm_view, scored_node, streaming_llm
from llama_index.core.schema import NodeWithScore

from docint.agents.assessment.ledger import EvidenceLedger
from docint.agents.assessment.runner import AssessmentResult, AssessmentRunner, ResearchOutcome
from docint.agents.assessment.split import SENTINEL, assessment_label
from docint.agents.types import AgentProgress, PriorTurn

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
    tokens: list[str] | None = None, *, reasoning: bool = False, researcher: Any = None
) -> tuple[AssessmentRunner, Any]:
    """Build a runner over a streaming double.

    Args:
        tokens (list[str] | None): The answer tokens; ``ANSWER`` by default.
        reasoning (bool): Whether the answer model counts as a reasoning model.
        researcher (Any): An optional research step.

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
    )
    return runner, llm


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


def test_the_answer_call_carries_question_evidence_prior_turn_and_reserve() -> None:
    """One user message holds everything; the reserve rides as ``max_tokens``."""
    runner, llm = _runner()
    prior = PriorTurn(user_text="Summarise it.", assistant_text="It has one experiment [1].")

    list(runner.run(question=QUESTION, prior_turn=prior, seed=_seed()))

    call = llm.stream_chat.call_args
    prompt = call.args[0][0].content
    assert QUESTION in prompt
    assert "citation_index: 1" in prompt
    assert "citation_index: 2" in prompt
    assert "It has one experiment [1]." in prompt
    assert call.kwargs["max_tokens"] == 1024


def test_a_reasoning_model_gets_the_larger_reserve() -> None:
    """A thinking model spends reply tokens on reasoning first."""
    runner, llm = _runner(reasoning=True)

    list(runner.run(question=QUESTION, prior_turn=None, seed=_seed()))

    assert llm.stream_chat.call_args.kwargs["max_tokens"] == 3072


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
