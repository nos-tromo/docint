"""The assessment runner: seed, research hook, streamed two-part answer. All text is invented."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any, cast

import openai
import pytest
from assessment_fakes import (
    LONG_LEAF,
    bad_request,
    heavy_post,
    llm_view,
    scored_node,
    search_call,
    social_post,
    streaming_llm,
    tool_llm,
)
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.schema import BaseNode, MetadataMode, NodeWithScore, TextNode

from docint.agents.assessment.ledger import (
    MARKER_RESERVE_CHARS,
    MIN_TEXT_CHARS,
    EvidenceLedger,
    Room,
    answer_budget,
)
from docint.agents.assessment.research import ResearchLoop
from docint.agents.assessment.runner import (
    DEFAULT_ASSESSMENT_ANSWER_PROMPT,
    AssessmentDoesNotFit,
    AssessmentResult,
    AssessmentRunner,
    ResearchOutcome,
    prior_turn_block,
)
from docint.agents.assessment.split import SENTINEL, assessment_label
from docint.agents.types import AgentProgress, PriorTurn
from docint.core.rag import ParentContextPostprocessor
from docint.utils.embed_chunking import estimate_tokens
from docint.utils.prompt_loader import load_localized_prompt
from docint.utils.text_budget import MATCH_SPANS_KEY

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
    failures: list[Exception] | None = None,
    view: Callable[[BaseNode], BaseNode] = llm_view,
    ctx_window: int = 8192,
) -> tuple[AssessmentRunner, Any]:
    """Build a runner over a streaming double.

    Args:
        tokens (list[str] | None): The answer tokens; ``ANSWER`` by default.
        reasoning (bool): Whether the answer model counts as a reasoning model.
        researcher (Any): An optional research step.
        count_tokens (Callable[[str], int] | None): An optional tokenizer.
        failures (list[Exception] | None): Errors the first answer calls raise.
        view (Callable[[BaseNode], BaseNode]): The LLM view; the test double by default.
        ctx_window (int): The context window in tokens.

    Returns:
        tuple[AssessmentRunner, Any]: The runner and its model double.
    """
    llm = streaming_llm(tokens or ANSWER, failures=failures or [])
    runner = AssessmentRunner(
        answer_llm=cast(Any, llm),
        llm_view=view,
        ctx_window=ctx_window,
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
        """Stand in for a tokenizer that fails.

        Args:
            text (str): The text to count, unused.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("tokenizer unavailable")

    plain = _prompt(*_runner(), _long_seed(10))
    fallback = _prompt(*_runner(count_tokens=_broken), _long_seed(10))

    assert fallback == plain


def test_the_prior_turn_and_question_are_measured_with_the_tokenizer() -> None:
    """Their overhead is counted, not estimated, when a tokenizer is available."""
    measured: list[str] = []

    def _count(text: str) -> int:
        """Record what is measured and count it at four characters a token.

        Args:
            text (str): The text to count.

        Returns:
            int: Its tokens.
        """
        measured.append(text)
        return max(1, len(text) // 4)

    prior = PriorTurn(user_text="Summarise it.", assistant_text="It has one experiment [1].")

    _prompt(*_runner(count_tokens=_count), _seed(), prior)

    assert prior_turn_block(prior) + QUESTION in measured


OVERFLOW = "This model's maximum context length is 8192 tokens; your request has 9001 input tokens."


def test_an_overflowing_answer_prompt_is_rebuilt_once_with_less_evidence() -> None:
    """A context error before the first token rebuilds the prompt smaller; the turn then completes."""
    runner, llm = _runner(failures=[bad_request(OVERFLOW)])

    items = list(runner.run(question=QUESTION, prior_turn=None, seed=_long_seed(8)))

    first, second = (call.args[0][0].content for call in llm.stream_chat.call_args_list)
    result = items[-1]
    assert len(second) < len(first)
    assert [item.action for item in items if isinstance(item, AgentProgress)] == ["seed", "answer"]
    assert isinstance(result, AssessmentResult)
    assert result.raw_text == "".join(ANSWER)
    # The first prompt held every passage whole, so the count can only be the retry's.
    assert "characters not shown" not in first
    assert result.shortened == 8


def test_a_second_overflow_ends_the_turn_as_a_context_window_error() -> None:
    """The API reports a ValueError naming the context window as ``context_overflow``."""
    runner, llm = _runner(failures=[bad_request(OVERFLOW), bad_request(OVERFLOW)])

    with pytest.raises(ValueError, match="context window"):
        list(runner.run(question=QUESTION, prior_turn=None, seed=_long_seed(8)))

    assert llm.stream_chat.call_count == 2


@pytest.mark.parametrize(
    "error", [bad_request("The model 'test-model' does not exist."), RuntimeError("upstream unavailable")]
)
def test_a_failure_that_is_not_an_overflow_is_not_retried(error: Exception) -> None:
    """Only a prompt that was too long is worth sending again shorter.

    Args:
        error (Exception): What the answer call raises.
    """
    runner, llm = _runner(failures=[error])

    with pytest.raises(type(error)) as raised:
        list(runner.run(question=QUESTION, prior_turn=None, seed=_seed()))

    assert raised.value is error
    assert llm.stream_chat.call_count == 1


def test_an_answer_that_fails_after_its_first_token_is_never_retried() -> None:
    """Text already streamed cannot be taken back, so a later error propagates unchanged."""
    llm = streaming_llm([])

    def _stream(messages: list[ChatMessage], **kwargs: Any) -> Iterator[ChatResponse]:
        """Stream one token, then fail as an overflowing endpoint would.

        Args:
            messages (list[ChatMessage]): The prompt, unused.
            **kwargs (Any): Call options, unused.

        Yields:
            ChatResponse: The first token.

        Raises:
            openai.BadRequestError: After the first token.
        """
        yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content="Fact"), delta="Fact")
        raise bad_request(OVERFLOW)

    llm.stream_chat.side_effect = _stream
    runner = AssessmentRunner(
        answer_llm=cast(Any, llm), llm_view=llm_view, ctx_window=8192, char_token_ratio=3.5, reasoning=False
    )
    streamed: list[Any] = []

    with pytest.raises(openai.BadRequestError):
        for item in runner.run(question=QUESTION, prior_turn=None, seed=_seed()):
            streamed.append(item)

    assert "Fact" in streamed
    assert llm.stream_chat.call_count == 1


def test_evidence_too_large_to_show_legibly_raises_before_anything_is_yielded() -> None:
    """Fifty pinned posts at 8k leave no room to read them; the caller answers as a lookup instead."""
    runner, llm = _runner(view=ParentContextPostprocessor._emit_with_llm_exclusion)
    stream = runner.run(question=QUESTION, prior_turn=None, seed=[social_post(number) for number in range(1, 51)])

    with pytest.raises(AssessmentDoesNotFit) as raised:
        next(stream)

    assert raised.value.passages == 50
    assert isinstance(raised.value, ValueError)
    assert "context window" in str(raised.value)
    llm.stream_chat.assert_not_called()


class _FinishAtOnce:
    """A research step that searches nothing, as an unscoped turn's may."""

    def run(
        self, *, question: str, prior_turn: PriorTurn | None, ledger: EvidenceLedger
    ) -> Iterator[AgentProgress | ResearchOutcome]:
        """End research at once.

        Args:
            question (str): The user's message.
            prior_turn (PriorTurn | None): The previous exchange.
            ledger (EvidenceLedger): The evidence so far.

        Yields:
            AgentProgress | ResearchOutcome: The outcome alone.
        """
        yield ResearchOutcome(steps=1)


@pytest.mark.parametrize(("window", "reasoning"), [(8192, False), (8192, True), (32768, False)])
def test_an_unscoped_seed_always_fits(window: int, reasoning: bool) -> None:
    """At the default ``RETRIEVE_TOP_K`` a lookup's top-n never exceeds the ledger cap.

    A full cap stays legible even with long social rows, so an unscoped seed always fits.

    Args:
        window (int): The context window in tokens.
        reasoning (bool): Whether the answer model thinks first.
    """
    template = load_localized_prompt("assessment_answer", default=DEFAULT_ASSESSMENT_ANSWER_PROMPT)
    cap = answer_budget(ctx_window=window, template=template, reasoning=reasoning, char_token_ratio=3.5).max_entries
    seed = [social_post(number, text="The result holds. " * 111) for number in range(1, cap + 1)]
    runner, llm = _runner(
        reasoning=reasoning,
        researcher=_FinishAtOnce(),
        view=ParentContextPostprocessor._emit_with_llm_exclusion,
        ctx_window=window,
    )

    items = list(runner.run(question=QUESTION, prior_turn=None, seed=seed))

    assert isinstance(items[0], AgentProgress)
    assert items[0].found == cap
    assert isinstance(items[-1], AssessmentResult)
    llm.stream_chat.assert_called_once()


#: An invented section whose decisive sentence sits deep inside it.
DEEP_FACT = "The pilot excluded every respondent recruited after March."
DEEP_SECTION = ("Background filler sentence. " * 190)[:5214] + DEEP_FACT + " Closing filler sentence." * 40


def test_a_fact_deep_in_a_promoted_section_survives_a_full_ledger() -> None:
    """Codex's reproduction: research fills 23 entries at 8k, and the retrieved fact at offset 5,214 stays.

    The real parent-context postprocessor promotes the retrieved sentence to its whole section; both the
    research digest and the packed answer context must still show that sentence.
    """
    assert DEEP_SECTION.index(DEEP_FACT) == 5214
    section = TextNode(id_="section-7", text=DEEP_SECTION, metadata={"filename": "survey.pdf", "page": 3})
    child = NodeWithScore(
        node=TextNode(
            id_="child-7", text=DEEP_FACT, metadata={"hier.parent_id": "section-7", "docint_hier_type": "fine"}
        ),
        score=0.8,
    )
    postprocessor = ParentContextPostprocessor(
        docstore=SimpleNamespace(
            get_node=lambda node_id, raise_error=False: section if node_id == "section-7" else None
        ),
        usable_tokens=4000,
        per_hit_floor=256,
        char_token_ratio=3.5,
        budget_enforced=True,
    )
    filler = [
        scored_node(
            f"n{number}", f"Passage {number}. " + "Invented filler text. " * 100, filename="notes.pdf", page=number
        )
        for number in range(1, 23)
    ]
    # The section arrives 22nd; one more search fills the 23-entry ledger after the model has read its digest.
    results = {
        "exclusion criteria": lambda: postprocessor._postprocess_nodes([child]),
        "recruitment": lambda: filler[21:],
    }
    research_llm = tool_llm([search_call("exclusion criteria"), search_call("recruitment")])
    researcher = ResearchLoop(
        llm=cast(Any, research_llm),
        search=lambda query: results[query](),
        max_steps=6,
        ctx_window=8192,
        char_token_ratio=3.5,
        prompt="{question}{prior_turn}{digest}",
    )
    runner, llm = _runner(researcher=researcher, view=ParentContextPostprocessor._emit_with_llm_exclusion)

    result = list(runner.run(question=QUESTION, prior_turn=None, seed=filler[:21]))[-1]

    assert isinstance(result, AssessmentResult)
    assert len(result.nodes) == 23
    assert result.shortened == 23
    digest_reply = research_llm.chat_with_tools.call_args_list[1].kwargs["chat_history"][2]
    assert DEEP_FACT in str(digest_reply.content)
    assert DEEP_FACT in llm.stream_chat.call_args.args[0][0].content


class _AddsOneBatch:
    """Research that adds one batch of passages in a single search."""

    def __init__(self, batch: list[NodeWithScore], reason: str | None = None) -> None:
        """Hold the batch.

        Args:
            batch (list[NodeWithScore]): What the search finds.
            reason (str | None): Why research says it stopped.
        """
        self._batch = batch
        self._reason = reason

    def run(
        self, *, question: str, prior_turn: PriorTurn | None, ledger: EvidenceLedger
    ) -> Iterator[AgentProgress | ResearchOutcome]:
        """Offer the whole batch to the ledger at once.

        Args:
            question (str): The user's message.
            prior_turn (PriorTurn | None): The previous exchange.
            ledger (EvidenceLedger): The evidence so far.

        Yields:
            AgentProgress | ResearchOutcome: One search, then the outcome.
        """
        added = ledger.add(self._batch)
        yield AgentProgress(step=1, action="search", detail="invented query", found=len(added))
        yield ResearchOutcome(
            steps=1,
            actions=[{"action": "search", "detail": "invented query", "found": len(added)}],
            reason=self._reason,
        )


def _citation_lines(prompt: str) -> list[str]:
    """Return the citation-number lines of a prompt, in order.

    Args:
        prompt (str): The answer prompt.

    Returns:
        list[str]: Every line that starts with ``citation_index:``.
    """
    return [line for line in prompt.splitlines() if line.startswith("citation_index: ")]


def test_research_admits_metadata_heavy_passages_only_while_the_answer_still_fits() -> None:
    """Codex's reproduction: research passages with ~3 KB headers once grew an 8k answer's context to ~69k characters.

    Admission prices each header, so the turn takes what stays legible and the prompt fits its window.
    """
    researcher = _AddsOneBatch([heavy_post(number) for number in range(1, 23)])
    runner, llm = _runner(researcher=researcher, view=ParentContextPostprocessor._emit_with_llm_exclusion)

    items = list(runner.run(question=QUESTION, prior_turn=None, seed=_seed()))

    result = items[-1]
    assert isinstance(result, AssessmentResult)
    prompt = llm.stream_chat.call_args.args[0][0].content
    assert estimate_tokens(prompt, 3.5) + 1024 <= 8192
    assert 2 < len(result.nodes) < 10
    assert _citation_lines(prompt) == [f"citation_index: {number}" for number in range(1, len(result.nodes) + 1)]
    search = next(item for item in items if isinstance(item, AgentProgress) and item.action == "search")
    assert search.found == len(result.nodes) - 2


def _denser(text: str) -> int:
    """Count tokens like a tokenizer meeting dense research text: each ``§`` costs one token more.

    Args:
        text (str): The text to count.

    Returns:
        int: A token per four characters, plus one per ``§``.
    """
    return len(text) // 4 + text.count("§")


def test_a_ledger_denser_than_its_seed_leaves_out_its_latest_research_passages() -> None:
    """Admission priced research at the seed's density; the final measure trims the newest additions, never the seed."""
    dense = [scored_node(f"d{number}", "§" * 1500, page=100 + number) for number in range(1, 21)]
    runner, llm = _runner(researcher=_AddsOneBatch(dense), count_tokens=_denser)

    items = list(runner.run(question=QUESTION, prior_turn=None, seed=_long_seed(3)))

    result = items[-1]
    assert isinstance(result, AssessmentResult)
    kept = [scored.node.node_id for scored in result.nodes]
    assert 3 < len(kept) < 23
    assert kept == ["n1", "n2", "n3"] + [f"d{number}" for number in range(1, len(kept) - 2)]
    prompt = llm.stream_chat.call_args.args[0][0].content
    assert _citation_lines(prompt) == [f"citation_index: {number}" for number in range(1, len(kept) + 1)]
    answer = [item for item in items if isinstance(item, AgentProgress)][-1]
    assert (answer.action, answer.found) == ("answer", len(kept))
    # What research found and the answer could not show is a reduction the turn reports.
    assert result.reason == "context_budget"


def test_a_reason_research_already_gave_is_kept_when_passages_are_left_out() -> None:
    """The first cause wins: research that ran out of time still says so after the fit leaves passages out."""
    dense = [scored_node(f"d{number}", "§" * 1500, page=100 + number) for number in range(1, 21)]
    runner, _ = _runner(researcher=_AddsOneBatch(dense, reason="time_budget"), count_tokens=_denser)

    result = list(runner.run(question=QUESTION, prior_turn=None, seed=_long_seed(3)))[-1]

    assert isinstance(result, AssessmentResult)
    assert len(result.nodes) < 23
    assert result.reason == "time_budget"


def test_the_overflow_retry_compacts_headers_so_every_passage_keeps_its_text() -> None:
    """At 60% of the budget five full ~3 KB headers would leave no text at all; compact ones leave every post whole."""
    runner, llm = _runner(failures=[bad_request(OVERFLOW)], view=ParentContextPostprocessor._emit_with_llm_exclusion)
    seed = [heavy_post(number) for number in range(1, 6)]

    result = list(runner.run(question=QUESTION, prior_turn=None, seed=seed))[-1]

    first, second = (call.args[0][0].content for call in llm.stream_chat.call_args_list)
    assert len(second) < len(first)
    assert LONG_LEAF in first
    assert LONG_LEAF not in second
    assert _citation_lines(second) == [f"citation_index: {number}" for number in range(1, 6)]
    assert all(f"Harbour post {number}: the stalls open at ten." in second for number in range(1, 6))
    assert isinstance(result, AssessmentResult)
    assert len(result.nodes) == 5


@pytest.mark.parametrize(
    ("posts", "body", "gives_up"),
    [(38, 5, True), (20, 400, False)],
    ids=["compact-headers-exhaust-the-room", "texts-below-the-legible-minimum-still-sent"],
)
def test_the_retry_gives_up_only_when_compact_headers_alone_exhaust_the_room(
    posts: int, body: int, gives_up: bool
) -> None:
    """Short texts are allowed on the retry, as before; only headers with no room left end the turn unsent.

    Args:
        posts (int): How many pinned passages, each with a header of about 420 characters.
        body (int): Characters of text each carries.
        gives_up (bool): Whether the retry ends the turn without a second call.
    """
    seed = [scored_node(f"n{number}", "w" * body, filename="f" * 380, page=number) for number in range(1, posts + 1)]
    runner, llm = _runner(failures=[bad_request(OVERFLOW)])

    if gives_up:
        with pytest.raises(ValueError, match="context window"):
            list(runner.run(question=QUESTION, prior_turn=None, seed=seed))
        assert llm.stream_chat.call_count == 1
    else:
        result = list(runner.run(question=QUESTION, prior_turn=None, seed=seed))[-1]
        assert llm.stream_chat.call_count == 2
        assert isinstance(result, AssessmentResult)
        assert len(_citation_lines(llm.stream_chat.call_args.args[0][0].content)) == posts


SECTIONS = 5


def _decisive(section: int, window: int) -> str:
    """Return the decisive sentence of one window of an invented section.

    Args:
        section (int): The section.
        window (int): The window.

    Returns:
        str: The sentence the window was retrieved for.
    """
    return f"Decisive finding {section}-{window}: the invented cohort {section}{window} left in week {window + 2}."


def _section_window(section: int, window: int) -> NodeWithScore:
    """Build one window of an invented section as parent context emits it, its match in the middle.

    Args:
        section (int): The section; every window of it shares its node id.
        window (int): The window.

    Returns:
        NodeWithScore: The window, saying where its match sits.
    """
    before = (f"Lead-in filler for section {section} window {window}. " * 20)[:600]
    after = (f" Trailing filler for section {section} window {window}." * 20)[:600]
    fact = _decisive(section, window)
    metadata = {
        "filename": f"report-{section}.pdf",
        "page": section,
        MATCH_SPANS_KEY: [[len(before), len(before) + len(fact)]],
    }
    return NodeWithScore(
        node=TextNode(id_=f"section-{section}", text=before + fact + after, metadata=metadata), score=0.9
    )


#: About four thousand characters of an earlier lookup answer.
PRIOR_LOOKUP = PriorTurn(
    user_text="Summarise the report.", assistant_text=("The report covers an invented cohort. " * 110)[:4000]
)


@pytest.mark.parametrize(
    ("reasoning", "max_steps", "held_at_least", "reason"),
    [(True, 6, 30, None), (True, 8, 30, "context_budget"), (False, 6, 35, None)],
    ids=["8k-reasoning-stops-at-its-step-cap", "8k-reasoning-runs-out-of-room", "8k-everything-fits"],
)
def test_every_match_research_keeps_reaches_the_answer(
    reasoning: bool, max_steps: int, held_at_least: int, reason: str | None
) -> None:
    """Codex's reproduction: five seeded sections, each search a new window of every one, and a 4K prior answer.

    Every match the evidence holds reaches the answer prompt. Research the room stopped says so; research that
    ran to its step cap does not, though its last search may have offered more than the room took.

    Args:
        reasoning (bool): Whether the answer model thinks first.
        max_steps (int): The research step cap, and how many searches are scripted.
        held_at_least (int): The fewest matches the evidence must hold, so the check is never vacuous.
        reason (str | None): The reason the turn must report.
    """
    results = {f"q{w}": [_section_window(s, w) for s in range(1, SECTIONS + 1)] for w in range(1, max_steps + 1)}
    research_llm = tool_llm([search_call(f"q{w}") for w in range(1, max_steps + 1)])
    researcher = ResearchLoop(
        llm=cast(Any, research_llm),
        search=lambda query: results[query],
        max_steps=max_steps,
        ctx_window=8192,
        char_token_ratio=3.5,
        prompt="{question}{prior_turn}{digest}",
    )
    runner, llm = _runner(
        researcher=researcher, reasoning=reasoning, view=ParentContextPostprocessor._emit_with_llm_exclusion
    )
    seed = [_section_window(s, 0) for s in range(1, SECTIONS + 1)]

    items = list(runner.run(question=QUESTION, prior_turn=PRIOR_LOOKUP, seed=seed))

    result = items[-1]
    assert isinstance(result, AssessmentResult)
    prompt = llm.stream_chat.call_args.args[0][0].content
    texts = " ".join(scored.node.get_content(metadata_mode=MetadataMode.NONE) for scored in result.nodes)
    held = [_decisive(s, w) for s in range(1, SECTIONS + 1) for w in range(max_steps + 1) if _decisive(s, w) in texts]
    assert len(held) >= held_at_least
    assert [fact for fact in held if fact not in prompt] == []
    assert len(result.nodes) == SECTIONS
    assert result.reason == reason


class _Spy:
    """A research step that records being asked to run."""

    def __init__(self) -> None:
        """Start uncalled."""
        self.calls = 0

    def run(
        self, *, question: str, prior_turn: PriorTurn | None, ledger: EvidenceLedger
    ) -> Iterator[AgentProgress | ResearchOutcome]:
        """Record the call and end at once.

        Args:
            question (str): The user's message.
            prior_turn (PriorTurn | None): The previous exchange.
            ledger (EvidenceLedger): The evidence so far.

        Yields:
            AgentProgress | ResearchOutcome: The outcome alone.
        """
        self.calls += 1
        yield ResearchOutcome(steps=1)


def _seed_leaving(left: int) -> list[NodeWithScore]:
    """Build a legible seed that leaves ``left`` characters of room once each of its passages keeps its minimum.

    Args:
        left (int): The text room to leave, at the default 8k window with the English template.

    Returns:
        list[NodeWithScore]: Two passages, the second with a file name sized to use up the rest.
    """
    template = load_localized_prompt("assessment_answer", default=DEFAULT_ASSESSMENT_ANSWER_PROMPT, lang="en")
    budget = answer_budget(ctx_window=8192, template=template, reasoning=False, char_token_ratio=3.5)
    probe = EvidenceLedger(cap=2)
    probe.add([scored_node("n1", "x" * 1000, page=1), scored_node("n2", "y" * 1000, page=2, filename="")])
    spent = sum(
        len(llm_view(scored.node).get_metadata_str(mode=MetadataMode.LLM)) + 4 + MARKER_RESERVE_CHARS + MIN_TEXT_CHARS
        for scored in probe.numbered()
    )
    name = "f" * (int(budget.evidence_tokens * 3.5) - spent - left)
    seed = [scored_node("n1", "x" * 1000, page=1), scored_node("n2", "y" * 1000, page=2, filename=name)]
    numbered = EvidenceLedger(cap=2)
    numbered.add(seed)
    room = Room(numbered.numbered(), llm_view=llm_view, budget_tokens=budget.evidence_tokens, char_token_ratio=3.5)
    assert (room.legible, room.closed) == (True, True)
    return seed


def test_research_that_never_runs_for_lack_of_room_says_so() -> None:
    """A seed that leaves no room for another legible passage skips research, and the turn reports why."""
    researcher = _Spy()
    runner, _ = _runner(researcher=researcher)

    result = list(runner.run(question=QUESTION, prior_turn=None, seed=_seed_leaving(177)))[-1]

    assert isinstance(result, AssessmentResult)
    assert researcher.calls == 0
    assert (len(result.nodes), result.reason) == (2, "context_budget")


def test_a_scoped_turn_never_researches_so_a_full_room_is_no_reduction() -> None:
    """Without a research step there is nothing the room stopped."""
    runner, _ = _runner()

    result = list(runner.run(question=QUESTION, prior_turn=None, seed=_seed_leaving(177)))[-1]

    assert isinstance(result, AssessmentResult)
    assert result.reason is None


def test_passages_the_overflow_retry_leaves_out_are_reported() -> None:
    """The retry's tighter fit drops research additions the first fit kept, and the turn says so."""
    batch = [scored_node(f"r{number}", "w" * 1500, filename="f" * 220, page=100 + number) for number in range(1, 21)]
    runner, llm = _runner(researcher=_AddsOneBatch(batch), failures=[bad_request(OVERFLOW)])

    items = list(runner.run(question=QUESTION, prior_turn=None, seed=_seed()))

    result = items[-1]
    answer = [item for item in items if isinstance(item, AgentProgress)][-1]
    assert isinstance(result, AssessmentResult)
    assert answer.found == 22
    assert 2 < len(result.nodes) < 22
    assert llm.stream_chat.call_count == 2
    assert result.reason == "context_budget"
