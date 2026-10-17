"""The assessment runner: seed, research hook, streamed two-part answer. All text is invented."""

from __future__ import annotations

import re
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
    evidence_fits,
)
from docint.agents.assessment.research import ResearchLoop
from docint.agents.assessment.runner import (
    DEFAULT_ASSESSMENT_ANSWER_PROMPT,
    PRIOR_ANSWER_CHARS,
    RESOLVED_QUESTION_CHARS,
    RESOLVED_QUESTION_LABEL,
    AssessmentDoesNotFit,
    AssessmentResult,
    AssessmentRunner,
    ResearchOutcome,
    prior_turn_block,
    question_block,
)
from docint.agents.assessment.split import ASSESSMENT_RULE, SENTINEL, assessment_label, compose_labelled_answer
from docint.agents.types import AgentProgress, PriorTurn
from docint.core.rag import ParentContextPostprocessor
from docint.utils.embed_chunking import estimate_tokens
from docint.utils.prompt_loader import load_localized_prompt
from docint.utils.text_budget import MATCH_SPANS_KEY, allocate_shortest_first, fit_with_marker

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
    answer_prompt: str | None = None,
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
        answer_prompt (str | None): An answer template override; the localized one by default.

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
        answer_prompt=answer_prompt,
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
    # Shortened, but each to no less than its minimum: no reduction to report.
    assert result.reason is None


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
    # At 60% the eight passages still keep their minimum, so nothing is reported.
    assert result.reason is None


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


def _unique_filler(label: str, count: int) -> str:
    """Return invented filler whose sentences are numbered, so no stretch of it repeats.

    Args:
        label (str): A word that sets this filler apart from any other.
        count (int): How many sentences.

    Returns:
        str: The filler.
    """
    return "".join(f"{label} filler sentence {index:04d}. " for index in range(count))


#: The decisive fact of an invented section, 50 characters into its second retrieved child.
TOUCH_FACT = "The second cohort was dropped after a protocol breach in week three."
#: An invented section whose two retrieved children touch at 2,100: (100, 2100) and (2100, 4100).
TOUCHING_SECTION = (_unique_filler("Lead", 100)[:2150] + TOUCH_FACT + _unique_filler("Tail", 100))[:4200]


def _child_of_touching_section(start: int, end: int, node_id: str) -> NodeWithScore:
    """Build one retrieved child of :data:`TOUCHING_SECTION`.

    Args:
        start (int): Where the child starts in the section.
        end (int): Where it ends.
        node_id (str): The child's id.

    Returns:
        NodeWithScore: The hit, pointing at its section.
    """
    metadata = {"hier.parent_id": "section-9", "docint_hier_type": "fine"}
    return NodeWithScore(node=TextNode(id_=node_id, text=TOUCHING_SECTION[start:end], metadata=metadata), score=0.8)


def test_a_match_touching_the_seeded_one_keeps_its_own_region_in_a_crowded_8k_answer() -> None:
    """Codex's reproduction: research reads a fact at 2,150 in the child touching the seeded one; the answer keeps it.

    Ten passages crowd an 8k answer, so the section is cut; merged with the seeded match, the second
    match would keep only the opening of the first, and nothing would say so.
    """
    assert TOUCHING_SECTION.index(TOUCH_FACT) == 2150
    section = TextNode(id_="section-9", text=TOUCHING_SECTION, metadata={"filename": "trial.pdf", "page": 4})
    postprocessor = ParentContextPostprocessor(
        docstore=SimpleNamespace(
            get_node=lambda node_id, raise_error=False: section if node_id == "section-9" else None
        ),
        usable_tokens=4000,
        per_hit_floor=256,
        char_token_ratio=3.5,
        budget_enforced=True,
    )
    seeded = postprocessor._postprocess_nodes([_child_of_touching_section(100, 2100, "child-1")])
    found = postprocessor._postprocess_nodes([_child_of_touching_section(2100, 4100, "child-2")])
    assert [scored.node.metadata[MATCH_SPANS_KEY] for scored in (*seeded, *found)] == [[[100, 2100]], [[2100, 4100]]]
    fillers = [
        scored_node(f"n{number}", _unique_filler(f"Passage{number}", 160)[:4000], filename="notes.pdf", page=number)
        for number in range(1, 10)
    ]
    research_llm = tool_llm([search_call("protocol breach")])
    researcher = ResearchLoop(
        llm=cast(Any, research_llm),
        search=lambda query: found,
        max_steps=6,
        ctx_window=8192,
        char_token_ratio=3.5,
        prompt="{question}{prior_turn}{digest}",
    )
    runner, llm = _runner(researcher=researcher, view=ParentContextPostprocessor._emit_with_llm_exclusion)

    result = list(runner.run(question=QUESTION, prior_turn=None, seed=[*seeded, *fillers]))[-1]

    assert isinstance(result, AssessmentResult)
    assert len(result.nodes) == 10
    digest_reply = research_llm.chat_with_tools.call_args_list[1].kwargs["chat_history"][2]
    assert TOUCH_FACT in str(digest_reply.content)
    prompt = llm.stream_chat.call_args.args[0][0].content
    assert TOUCHING_SECTION not in prompt
    assert TOUCH_FACT in prompt
    assert result.reason is None


#: A methodological caveat of 70 characters, 300 characters into the first match of :data:`CAVEAT_SECTION`.
CAVEAT = "Caveat: the invented pilot left out every night-shift worker it found."
#: Where :data:`CAVEAT_SECTION`'s five matches sit: one of 600 characters, then four of 260 after 450-character gaps.
CAVEAT_MATCHES = [(200, 800), (1250, 1510), (1960, 2220), (2670, 2930), (3380, 3640)]
#: An invented section of 4,040 characters: a lead, the five matches with their gaps, and a tail.
CAVEAT_SECTION = (
    _unique_filler("Lead", 20)[:200]
    + _unique_filler("Match", 20)[:300]
    + CAVEAT
    + _unique_filler("Rest", 20)[:230]
    + "".join(
        _unique_filler(f"Gap{index}", 30)[:450] + _unique_filler(f"Hit{index}", 20)[:260] for index in range(2, 6)
    )
    + _unique_filler("Tail", 30)[:400]
)


def _caveat_section(*matches: int) -> NodeWithScore:
    """Return :data:`CAVEAT_SECTION` stamped with some of its matches, in the order given.

    Args:
        *matches (int): Which matches it carries, numbered from 1.

    Returns:
        NodeWithScore: The section, as parent context emits it whole.
    """
    spans = [list(CAVEAT_MATCHES[number - 1]) for number in matches]
    return scored_node("section-s", CAVEAT_SECTION, filename="study.pdf", page=5, **{MATCH_SPANS_KEY: spans})


def _two_match_source(number: int) -> NodeWithScore:
    """Return an invented source of 2,600 characters stamped with two matches.

    Args:
        number (int): The source's number, which also sets its text apart.

    Returns:
        NodeWithScore: The source.
    """
    text = _unique_filler(f"Source{number}", 100)[:2600]
    spans = [[400, 520], [1600, 1720]]
    return scored_node(
        f"source-{number}", text, filename=f"source-{number}.pdf", page=number, **{MATCH_SPANS_KEY: spans}
    )


def _digest_lines(research_llm: Any) -> list[str]:
    """Return every digest line research read: the seed's in its prompt, then each search's in its tool reply.

    Args:
        research_llm (Any): The research model double.

    Returns:
        list[str]: The digest lines, in the order research read them.
    """
    history = research_llm.chat_with_tools.call_args_list[-1].kwargs["chat_history"]
    return [
        line
        for message in history
        if message.role in (MessageRole.USER, MessageRole.TOOL)
        for line in str(message.content).splitlines()
        if re.match(r"\[\d+\] ", line)
    ]


def test_what_research_reads_of_a_crowded_section_reaches_an_8k_reasoning_answer() -> None:
    """Codex's reproduction: research read a caveat in a section's first match that the answer prompt left out.

    At 8K with reasoning on, twelve sources share the answer, and the section's five matches each keep
    under 300 characters, less than the 400-character view research had read around the first. Every
    piece of every digest research reads must reach the prompt, unless the turn reports ``context_budget``.
    """
    assert (len(CAVEAT_SECTION), CAVEAT_SECTION.index(CAVEAT), len(CAVEAT)) == (4040, 500, 70)
    results = {
        "method": [_caveat_section(1, 5), *(_two_match_source(number) for number in range(4, 8))],
        "sampling": [_two_match_source(number) for number in range(8, 12)],
    }
    research_llm = tool_llm([search_call("method"), search_call("sampling")])
    researcher = ResearchLoop(
        llm=cast(Any, research_llm),
        search=lambda query: results[query],
        max_steps=6,
        ctx_window=8192,
        char_token_ratio=3.5,
        prompt="{question}\n{prior_turn}\n{digest}",
    )
    llm = streaming_llm(ANSWER)
    runner = AssessmentRunner(
        answer_llm=cast(Any, llm),
        llm_view=ParentContextPostprocessor._emit_with_llm_exclusion,
        ctx_window=8192,
        char_token_ratio=3.5,
        reasoning=True,
        researcher=researcher,
        answer_prompt=DEFAULT_ASSESSMENT_ANSWER_PROMPT,
    )
    seed = [_caveat_section(1, 2, 3, 4), *(_two_match_source(number) for number in range(1, 4))]

    result = list(runner.run(question=QUESTION, prior_turn=None, seed=seed))[-1]

    assert isinstance(result, AssessmentResult)
    assert len(result.nodes) == 12
    prompt = " ".join(llm.stream_chat.call_args.args[0][0].content.split())
    assert " ".join(CAVEAT_SECTION.split()) not in prompt
    lines = _digest_lines(research_llm)
    assert len(lines) == 13
    pieces = [
        piece for line in lines for piece in re.sub(r"^\[\d+\] \([^()]*\) ", "", line).removeprefix("… ").split(" … ")
    ]
    if result.reason != "context_budget":
        assert [piece for piece in pieces if piece not in prompt] == []


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


def _thin_seed(count: int) -> list[NodeWithScore]:
    """Return short invented passages, each the first window of its section.

    Args:
        count (int): How many passages.

    Returns:
        list[NodeWithScore]: The seed evidence.
    """
    return [scored_node(f"n{number}", f"Seed passage {number}.", page=number) for number in range(1, count + 1)]


def _dense_windows(count: int) -> list[NodeWithScore]:
    """Return a later window of each :func:`_thin_seed` section, in text :func:`_denser` counts as dense.

    Args:
        count (int): How many windows.

    Returns:
        list[NodeWithScore]: The windows, sharing the seed passages' node ids.
    """
    return [scored_node(f"n{number}", "§" * 3000, page=number) for number in range(1, count + 1)]


@pytest.mark.parametrize(("given", "reported"), [(None, "context_budget"), ("time_budget", "time_budget")])
def test_evidence_sent_below_its_legible_minimum_is_reported(given: str | None, reported: str) -> None:
    """Research extends every seed passage with text denser than the seed it was admitted against.

    Nothing can be left out (it is all seed), so the answer is sent with passages cut below their
    minimum; the turn says so, unless research already gave its own reason.

    Args:
        given (str | None): Why research says it stopped.
        reported (str): The reason the turn must report.
    """
    runner, _ = _runner(researcher=_AddsOneBatch(_dense_windows(20), reason=given), count_tokens=_denser)

    result = list(runner.run(question=QUESTION, prior_turn=None, seed=_thin_seed(20)))[-1]

    assert isinstance(result, AssessmentResult)
    assert len(result.nodes) == 20
    assert all("§" in scored.node.get_content(metadata_mode=MetadataMode.NONE) for scored in result.nodes)
    template = load_localized_prompt("assessment_answer", default=DEFAULT_ASSESSMENT_ANSWER_PROMPT, lang="en")
    budget = answer_budget(ctx_window=8192, template=template, reasoning=False, char_token_ratio=3.5)
    assert not evidence_fits(
        result.nodes,
        llm_view=llm_view,
        budget_tokens=budget.evidence_tokens,
        char_token_ratio=3.5,
        count_tokens=_denser,
    )
    assert result.reason == reported


def test_a_retry_that_cuts_the_seed_below_its_minimum_is_reported() -> None:
    """At 60% of the budget forty pinned passages no longer keep their minimum; none can go, so the turn says so."""
    runner, llm = _runner(failures=[bad_request(OVERFLOW)])

    result = list(runner.run(question=QUESTION, prior_turn=None, seed=_long_seed(40)))[-1]

    assert isinstance(result, AssessmentResult)
    assert llm.stream_chat.call_count == 2
    assert len(result.nodes) == 40
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
    [(True, 6, 30, "context_budget"), (True, 8, 30, "context_budget"), (False, 6, 35, None)],
    ids=["8k-reasoning-loses-a-window-at-its-step-cap", "8k-reasoning-runs-out-of-room", "8k-everything-fits"],
)
def test_every_match_research_keeps_reaches_the_answer(
    reasoning: bool, max_steps: int, held_at_least: int, reason: str | None
) -> None:
    """Codex's reproduction: five seeded sections, each search a new window of every one, and a 4K prior answer.

    Every match the evidence holds reaches the answer prompt. A turn whose research the room stopped says so, and
    so does one whose last search, at the step cap, offered a window the room could not take.

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


PRIOR_QUESTION = "Is the harbour survey sound?"
#: An invented source-backed part of 4,439 characters, longer than the prior-answer budget on its own.
SOURCED = ("The survey interviewed invented respondents at three piers [1]. " * 70)[:4438] + "."
RECOMMENDATIONS = (
    "Recommendation one: rerun the survey with a second cohort. Recommendation two: publish the questionnaire."
)
SOURCED_LABEL = "Previous assistant answer, source-backed part (its [n] refer to the previous turn's sources): "
ASSESSMENT_LABEL = "Previous assistant answer, the model's own assessment: "


def _prior_assessment(sourced: str, judgement: str) -> PriorTurn:
    """Store an invented assessment the way a turn persists it and read it back as the prior turn.

    Args:
        sourced (str): The source-backed part.
        judgement (str): The model's own assessment.

    Returns:
        PriorTurn: The exchange, with the boundary between its parts.
    """
    labelled = compose_labelled_answer(f"{sourced}\n{SENTINEL}\n{judgement}")
    return PriorTurn(user_text=PRIOR_QUESTION, assistant_text=labelled.text, evidence_chars=labelled.evidence_chars)


def test_a_follow_up_sees_recommendations_behind_a_long_source_part_in_both_prompts() -> None:
    """Codex's reproduction: past 4,000 characters of sources, the earlier judgement still reaches both prompts."""
    prior = _prior_assessment(SOURCED, RECOMMENDATIONS)
    assert prior.evidence_chars == 4439
    research_llm = tool_llm([])
    researcher = ResearchLoop(
        llm=cast(Any, research_llm),
        search=lambda query: [],
        max_steps=6,
        ctx_window=32768,
        char_token_ratio=3.5,
        prompt="{question}\n{prior_turn}\n{digest}",
    )
    runner, llm = _runner(researcher=researcher, ctx_window=32768)

    list(runner.run(question="Which recommendations should we prioritise?", prior_turn=prior, seed=_seed()))

    research_prompt = str(research_llm.chat_with_tools.call_args_list[0].kwargs["chat_history"][0].content)
    answer_prompt = str(llm.stream_chat.call_args.args[0][0].content)
    judgement = f"{assessment_label(with_rule=False)}{RECOMMENDATIONS}"
    # Shortest first: the judgement is shown whole and the sources get everything it leaves.
    sourced = fit_with_marker(SOURCED, PRIOR_ANSWER_CHARS - len(judgement))[0]
    for prompt in (research_prompt, answer_prompt):
        assert f"{SOURCED_LABEL}{sourced}\n{ASSESSMENT_LABEL}{judgement}" in prompt


def test_a_short_prior_assessment_shows_both_parts_whole_and_labelled() -> None:
    """Each part keeps its own label; the source-backed one says whose sources its numbers cite."""
    prior = _prior_assessment("The survey covered three piers [1].", "Rerun it with a second cohort.")

    assert prior_turn_block(prior) == (
        f"Previous user question: {PRIOR_QUESTION}\n"
        f"{SOURCED_LABEL}The survey covered three piers [1].\n"
        f"{ASSESSMENT_LABEL}{assessment_label(with_rule=False)}Rerun it with a second cohort."
    )


def test_a_prior_assessment_with_no_source_backed_part_shows_only_the_judgement() -> None:
    """A boundary of ``0`` means the whole answer is the model's own assessment."""
    labelled = compose_labelled_answer("Rerun it with a second cohort.")
    assert labelled.evidence_chars == 0
    prior = PriorTurn(user_text=PRIOR_QUESTION, assistant_text=labelled.text, evidence_chars=0)

    assert prior_turn_block(prior) == f"Previous user question: {PRIOR_QUESTION}\n{ASSESSMENT_LABEL}{labelled.text}"


def _today(prior: PriorTurn) -> str:
    """Render a prior turn the way the block rendered every earlier answer before the split.

    Args:
        prior (PriorTurn): The exchange.

    Returns:
        str: The block.
    """
    answer = fit_with_marker(prior.assistant_text, PRIOR_ANSWER_CHARS)[0]
    return f"Previous user question: {prior.user_text}\nPrevious assistant answer: {answer}"


@pytest.mark.parametrize("answer", ["It has one experiment [1].", "The study reports a result. " * 300, ""])
def test_a_prior_lookup_renders_exactly_as_before(answer: str) -> None:
    """With no boundary the whole answer is source-backed and is shown as it always was.

    Args:
        answer (str): The earlier answer.
    """
    prior = PriorTurn(user_text=PRIOR_QUESTION, assistant_text=answer)

    assert prior_turn_block(prior) == _today(prior)


@pytest.mark.parametrize(
    "cut",
    [-5, None, 4439 + 10_000, 10, 4439 - 1, 4439 + 1],
    ids=["negative", "negative-on-the-rule", "past-the-end", "misaligned", "one-early", "one-late"],
)
def test_a_boundary_that_does_not_sit_on_the_rule_renders_exactly_as_before(cut: int | None) -> None:
    """A boundary out of range or off the rule is not trusted to split anything.

    Args:
        cut (int | None): The stored boundary; ``None`` stands for the negative one that lands on the rule.
    """
    stored = _prior_assessment(SOURCED, RECOMMENDATIONS)
    boundary = 4439 - len(stored.assistant_text) if cut is None else cut
    prior = PriorTurn(user_text=PRIOR_QUESTION, assistant_text=stored.assistant_text, evidence_chars=boundary)

    assert prior_turn_block(prior) == _today(prior)


def test_an_assessment_with_nothing_on_either_side_renders_as_before() -> None:
    """Splitting an empty answer would leave no answer line at all."""
    prior = PriorTurn(user_text=PRIOR_QUESTION, assistant_text="", evidence_chars=0)

    assert prior_turn_block(prior) == _today(prior)


def test_both_long_parts_share_the_prior_answer_budget() -> None:
    """The two parts split ``PRIOR_ANSWER_CHARS`` between them shortest-first, so the total is what it was."""
    judgement = ("Recommendation: an invented improvement to the survey design. " * 60)[:3000].strip()
    prior = _prior_assessment(SOURCED, judgement)
    sourced, cut = prior.assistant_text[: prior.evidence_chars], prior.evidence_chars or 0
    part = prior.assistant_text[cut:].removeprefix(ASSESSMENT_RULE)
    assert allocate_shortest_first([len(sourced), len(part)], PRIOR_ANSWER_CHARS) == [2000, 2000]

    block = prior_turn_block(prior)

    assert f"{SOURCED_LABEL}{fit_with_marker(sourced, 2000)[0]}\n" in block
    assert block.endswith(f"{ASSESSMENT_LABEL}{fit_with_marker(part, 2000)[0]}")


def test_a_long_prior_assessment_still_fits_the_window() -> None:
    """Both parts at their share, with their labels and markers, still leave the answer inside its window."""
    judgement = ("Recommendation: an invented improvement to the survey design. " * 60)[:3000].strip()
    prior = _prior_assessment(SOURCED, judgement)
    runner, llm = _runner()

    list(runner.run(question=QUESTION, prior_turn=prior, seed=_long_seed(23)))

    prompt = llm.stream_chat.call_args.args[0][0].content
    assert estimate_tokens(prompt, 3.5) + 1024 <= 8192


FOLLOW_UP = "Should we proceed?"
#: An invented rewrite that recovered a budget set two turns earlier.
RESOLVED = "Should we proceed with the invented pilot deployment within the €10,000 budget?"


@pytest.mark.parametrize(
    "resolved",
    [None, "", " \n\t ", "should we PROCEED", "Should   we\nproceed?!", "<think>the user means it</think>"],
    ids=["none", "empty", "blank", "same-words-other-case", "same-words-other-spacing", "only-reasoning"],
)
def test_a_rewrite_that_adds_no_words_leaves_the_question_alone(resolved: str | None) -> None:
    """Nothing resolved, or the same words up to case, spacing and punctuation, is the user's message, stripped.

    Args:
        resolved (str | None): The search rewrite, if any.
    """
    assert question_block(f"  {FOLLOW_UP}\n", resolved) == FOLLOW_UP


def test_a_rewrite_with_new_words_is_labelled_on_its_own_line() -> None:
    """The user's own words come first, then one labelled line that says the rewrite is no source."""
    assert question_block(FOLLOW_UP, RESOLVED) == f"{FOLLOW_UP}\n{RESOLVED_QUESTION_LABEL}: {RESOLVED}"
    assert RESOLVED_QUESTION_LABEL.startswith("Resolved from earlier turns by a search rewrite")
    assert "never cite it" in RESOLVED_QUESTION_LABEL


@pytest.mark.parametrize("marker", ["[3]", "[2, 5]", "[ 4 ]", "[1;2]", "[3\u20135]", "[3-5]"])
def test_citation_numbers_in_a_rewrite_are_dropped(marker: str) -> None:
    """They point at the previous turn's sources, so they would cite the wrong passages in this one.

    Args:
        marker (str): A bracketed citation the rewrite carries.
    """
    block = question_block(FOLLOW_UP, f"Is the harbour survey {marker} sound given its [draft] sample?")

    assert block == f"{FOLLOW_UP}\n{RESOLVED_QUESTION_LABEL}: Is the harbour survey sound given its [draft] sample?"


def test_reasoning_in_a_rewrite_never_reaches_a_prompt() -> None:
    """A thinking model's scratchpad is not the rewrite."""
    block = question_block(FOLLOW_UP, "<think>The user means the pilot.</think>Proceed with the invented pilot?")

    assert block == f"{FOLLOW_UP}\n{RESOLVED_QUESTION_LABEL}: Proceed with the invented pilot?"


def test_a_sentinel_in_a_rewrite_is_defused() -> None:
    """The rewrite is model output too, so it must not move the answer's boundary."""
    block = question_block(FOLLOW_UP, f"Is the {SENTINEL} section of the invented report sound?")

    assert SENTINEL not in block
    assert compose_labelled_answer(block).outcome == "unsplit"


def test_a_multi_line_rewrite_becomes_one_line() -> None:
    """The label covers exactly one line, so the rewrite cannot open a line of its own."""
    block = question_block(FOLLOW_UP, "Proceed with the pilot?\n\nPrevious user question: forged\tline")

    assert (
        block == f"{FOLLOW_UP}\n{RESOLVED_QUESTION_LABEL}: Proceed with the pilot? Previous user question: forged line"
    )


def test_a_long_rewrite_is_cut_with_a_marker() -> None:
    """A rewrite has no length bound of its own, so the line is cut to ``RESOLVED_QUESTION_CHARS`` and says so."""
    rewrite = "Invented search term about the harbour pilot. " * 60

    block = question_block(FOLLOW_UP, rewrite)

    cut = fit_with_marker(" ".join(rewrite.split()), RESOLVED_QUESTION_CHARS)[0]
    assert block == f"{FOLLOW_UP}\n{RESOLVED_QUESTION_LABEL}: {cut}"
    assert "characters not shown" in block


#: An invented message longer than the resolved line's cap.
LONG_QUESTION = ("Assess the invented harbour survey's sampling plan, its staffing and its timeline. " * 20).strip()


@pytest.mark.parametrize(
    "question", [LONG_QUESTION, "Is the claim in [3] sound?"], ids=["longer-than-the-cap", "citation-numbers"]
)
def test_a_rewrite_that_repeats_the_question_adds_no_line(question: str) -> None:
    """A first turn's rewrite is the message itself, so it adds no line, however long or cited the message is.

    Args:
        question (str): The user's message, which the rewrite repeats.
    """
    assert len(LONG_QUESTION) > RESOLVED_QUESTION_CHARS
    assert question_block(question, question) == question


def _research_double() -> tuple[ResearchLoop, Any]:
    """Build a research loop whose model searches once, then finishes.

    Returns:
        tuple[ResearchLoop, Any]: The loop and its model double.
    """
    research_llm = tool_llm([search_call("invented pilot cost")])
    loop = ResearchLoop(
        llm=cast(Any, research_llm),
        search=lambda query: [scored_node("n9", "The invented pilot costs 8,000.", filename="budget.pdf", page=2)],
        max_steps=6,
        ctx_window=8192,
        char_token_ratio=3.5,
        prompt="Q<<{question}>> P<<{prior_turn}>> D<<{digest}>>",
    )
    return loop, research_llm


def test_research_and_the_answer_both_see_the_resolved_question() -> None:
    """The labelled line reaches the research prompt and the answer prompt, beside the user's own words."""
    loop, research_llm = _research_double()
    runner, llm = _runner(researcher=loop)
    prior = PriorTurn(user_text="What does the pilot cost?", assistant_text="The pilot costs 8,000 [1].")

    result = list(runner.run(question=FOLLOW_UP, prior_turn=prior, seed=_seed(), resolved_question=RESOLVED))[-1]

    asked = f"{FOLLOW_UP}\n{RESOLVED_QUESTION_LABEL}: {RESOLVED}"
    research_prompt = str(research_llm.chat_with_tools.call_args_list[0].kwargs["chat_history"][0].content)
    assert f"Q<<{asked}>>" in research_prompt
    assert f"The user's question: {asked}\n" in str(llm.stream_chat.call_args.args[0][0].content)
    assert isinstance(result, AssessmentResult)
    assert result.reason is None


def test_the_overflow_retry_still_shows_the_resolved_question() -> None:
    """The rebuilt prompt keeps the labelled line, as the first one did."""
    runner, llm = _runner(failures=[bad_request(OVERFLOW)])

    list(runner.run(question=FOLLOW_UP, prior_turn=None, seed=_long_seed(8), resolved_question=RESOLVED))

    prompts = [str(call.args[0][0].content) for call in llm.stream_chat.call_args_list]
    assert len(prompts) == 2
    assert all(f"The user's question: {FOLLOW_UP}\n{RESOLVED_QUESTION_LABEL}: {RESOLVED}\n" in p for p in prompts)


def test_the_resolved_question_is_measured_with_the_prior_turn() -> None:
    """The labelled line takes room in the answer prompt, so the overhead counts it."""
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

    prior = PriorTurn(user_text="What does the pilot cost?", assistant_text="The pilot costs 8,000 [1].")
    runner, _ = _runner(count_tokens=_count)

    list(runner.run(question=FOLLOW_UP, prior_turn=prior, seed=_seed(), resolved_question=RESOLVED))

    assert prior_turn_block(prior) + question_block(FOLLOW_UP, RESOLVED) in measured


@pytest.mark.parametrize("resolved", [None, "", "should we proceed"], ids=["none", "empty", "same-words"])
def test_without_a_new_resolved_question_both_prompts_are_unchanged(resolved: str | None) -> None:
    """A first turn, a failed rewrite or one that adds nothing sends exactly the prompts a turn sent before.

    Args:
        resolved (str | None): The search rewrite, if any.
    """
    loop, research_llm = _research_double()
    runner, llm = _runner(researcher=loop)
    plain_loop, plain_research_llm = _research_double()
    plain_runner, plain_llm = _runner(researcher=plain_loop)

    list(runner.run(question=f" {FOLLOW_UP} ", prior_turn=None, seed=_seed(), resolved_question=resolved))
    list(plain_runner.run(question=f" {FOLLOW_UP} ", prior_turn=None, seed=_seed()))

    research_prompt = research_llm.chat_with_tools.call_args_list[0].kwargs["chat_history"][0].content
    plain_research_prompt = plain_research_llm.chat_with_tools.call_args_list[0].kwargs["chat_history"][0].content
    assert research_prompt == plain_research_prompt
    assert f"Q<<{FOLLOW_UP}>>" in str(research_prompt)
    answer_prompt = str(llm.stream_chat.call_args.args[0][0].content)
    assert answer_prompt == str(plain_llm.stream_chat.call_args.args[0][0].content)
    assert f"The user's question: {FOLLOW_UP}\n" in answer_prompt
    assert RESOLVED_QUESTION_LABEL not in answer_prompt


class _QuestionRecorder:
    """A research step written before the resolved question existed: its ``run`` knows nothing of it."""

    def __init__(self) -> None:
        """Start with nothing asked."""
        self.asked: list[str] = []

    def run(
        self, *, question: str, prior_turn: PriorTurn | None, ledger: EvidenceLedger
    ) -> Iterator[AgentProgress | ResearchOutcome]:
        """Record the question it is shown and end at once.

        Args:
            question (str): The question as the prompts show it.
            prior_turn (PriorTurn | None): The previous exchange.
            ledger (EvidenceLedger): The evidence so far.

        Yields:
            AgentProgress | ResearchOutcome: The outcome alone.
        """
        self.asked.append(question)
        yield ResearchOutcome(steps=1)


def test_a_research_step_without_the_keyword_receives_the_block_as_its_question() -> None:
    """The researcher protocol is unchanged: the resolved line rides in ``question``, so no double fails on it."""
    recorder = _QuestionRecorder()
    runner, _ = _runner(researcher=recorder)

    result = list(runner.run(question=FOLLOW_UP, prior_turn=None, seed=_seed(), resolved_question=RESOLVED))[-1]

    assert recorder.asked == [f"{FOLLOW_UP}\n{RESOLVED_QUESTION_LABEL}: {RESOLVED}"]
    assert isinstance(result, AssessmentResult)
    assert result.reason is None


class _CheckingResearcher:
    """Records keyword checks as research does, adding no passage."""

    def __init__(self, actions: list[dict[str, Any]]) -> None:
        """Keep the actions to report.

        Args:
            actions (list[dict[str, Any]]): The research actions.
        """
        self.actions = actions

    def run(
        self, *, question: str, prior_turn: PriorTurn | None, ledger: EvidenceLedger
    ) -> Iterator[AgentProgress | ResearchOutcome]:
        """End at once with the recorded actions.

        Args:
            question (str): The question as the prompts show it.
            prior_turn (PriorTurn | None): The previous exchange.
            ledger (EvidenceLedger): The evidence so far.

        Yields:
            AgentProgress | ResearchOutcome: The outcome alone.
        """
        yield ResearchOutcome(steps=len(self.actions), actions=list(self.actions))


CHECKS = [
    {"action": "keyword", "detail": "Kontrollgruppe", "field": "text", "found": 0, "matches": 0},
    {"action": "keyword", "detail": "Krieger", "field": "author", "found": 0, "matches": None},
]


def test_keyword_checks_reach_the_answer_as_counts_beside_the_snippets() -> None:
    """The answer reads every check with a known count; a check whose count failed says nothing."""
    runner, llm = _runner(researcher=_CheckingResearcher(CHECKS), answer_prompt="{context_str}<<{keyword_checks}>>")

    prompt = _prompt(runner, llm, _seed())

    assert prompt.endswith('<<- "Kontrollgruppe" (text): 0>>')


def test_an_answer_without_checks_says_so() -> None:
    """The block is always there, so a turn with no check states it rather than leaving a gap."""
    runner, llm = _runner(answer_prompt="{context_str}<<{keyword_checks}>>")

    assert _prompt(runner, llm, _seed()).endswith("<<(none)>>")


def test_both_localized_answer_prompts_carry_the_checks() -> None:
    """Either locale places the checks, so a check made in German reaches a German answer."""
    for code in ("en", "de"):
        template = load_localized_prompt("assessment_answer", default=DEFAULT_ASSESSMENT_ANSWER_PROMPT, lang=code)
        assert "{keyword_checks}" in template


def test_the_checks_come_out_of_the_evidence_room_once_the_reserve_is_spent() -> None:
    """A long prior turn spends the reserve, so the checks cost evidence; no check costs nothing."""
    template = "{prior_turn}{question}<<{context_str}>>{keyword_checks}"
    prior = PriorTurn(user_text="Earlier question?", assistant_text="An earlier answer. " * 300)
    checks = [
        {"action": "keyword", "detail": f"term{i} word{i}", "field": "text", "found": 0, "matches": i} for i in range(6)
    ]

    def _evidence(actions: list[dict[str, Any]]) -> str:
        """Return the evidence an answer is shown after research reported ``actions``.

        Args:
            actions (list[dict[str, Any]]): The research actions.

        Returns:
            str: The rendered context.
        """
        runner, llm = _runner(researcher=_CheckingResearcher(actions), answer_prompt=template)
        return _prompt(runner, llm, _long_seed(10), prior).split("<<", 1)[1].split(">>", 1)[0]

    assert len(_evidence(checks)) < len(_evidence([]))
