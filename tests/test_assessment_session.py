"""Assessment turns through the session manager. All text is invented."""

from __future__ import annotations

from collections.abc import Callable, Generator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from _pytest.logging import LogCaptureFixture
from assessment_fakes import llm_view, scored_node, social_post, streaming_llm
from llama_index.core import PromptTemplate
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import Refine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing_extensions import override

from docint.agents.assessment.runner import AssessmentRunner
from docint.agents.assessment.split import SENTINEL
from docint.agents.history import build_prior_turn
from docint.agents.types import AgentProgress, PriorTurn
from docint.core.rag import RAG, ParentContextPostprocessor
from docint.core.state.base import Base
from docint.core.state.session_manager import SessionManager
from docint.core.state.turn import Turn

QUESTION = "What could be improved in this study?"
EVIDENCE = "The study evaluates on one dataset [1]."
ANSWER = [EVIDENCE, f"\n{SENTINEL}\n", "A second dataset would help."]


def _normalize(query: str, result: Any, **kwargs: Any) -> dict[str, Any]:
    """Mirror the normalizer's shape: text, numbered sources and routing fields.

    Args:
        query (str): The user's message.
        result (Any): The response object built from the runner's result.
        **kwargs (Any): The normalizer's keyword arguments.

    Returns:
        dict[str, Any]: A normalized payload; an empty answer becomes ``No answer.``.
    """
    return {
        "query": query,
        "response": result.response or "No answer.",
        "sources": [
            {"id": scored.node.node_id, "citation_index": scored.node.metadata.get("citation_index")}
            for scored in result.source_nodes
        ],
        "reasoning": None,
        "retrieval_query": kwargs["retrieval_query"],
        "coverage_unit": kwargs["coverage_unit"],
        "retrieval_mode": kwargs["retrieval_mode"],
        "rerank": None,
        "visual": None,
    }


@pytest.fixture(autouse=True)
def _english(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the label language.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")


@pytest.fixture
def session_manager() -> Generator[SessionManager, None, None]:
    """SessionManager on an in-memory DB with a RAG double wired for assessments.

    Returns:
        Generator[SessionManager, None, None]: The session manager.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    rag_mock = MagicMock()
    rag_mock.index = object()
    rag_mock.qdrant_collection = "test_collection"
    rag_mock.rewrite_retrieval_query.return_value = "rewritten question"
    rag_mock.expand_query_with_graph_with_debug.return_value = ("expanded question", {"applied": False})
    rag_mock.graph_debug_skipped.return_value = {"applied": False, "reason": "scoped"}
    rag_mock._infer_collection_profile.return_value = {"coverage_unit": "documents"}
    mode = MagicMock()
    mode.value = "compact"
    rag_mock._resolve_chat_response_mode.return_value = mode
    rag_mock._normalize_response_data.side_effect = _normalize
    cast(Any, rag_mock.get_source_by_node_id).return_value = None
    sm = SessionManager(rag=rag_mock)
    sm._SessionMaker = sessionmaker(bind=engine)
    yield sm
    engine.dispose()


def _seed() -> list[NodeWithScore]:
    """Return two passages of an invented study, each with its own file hash.

    Returns:
        list[NodeWithScore]: The seed evidence.
    """
    return [
        scored_node("n1", "We evaluate on one dataset.", filename="study.pdf", page=4, file_hash="h1"),
        scored_node("n2", "Results improve by two points.", filename="study.pdf", page=6, file_hash="h2"),
    ]


def _arm(
    sm: SessionManager,
    *,
    nodes: list[NodeWithScore],
    tokens: list[str] | None = None,
    view: Callable[[BaseNode], BaseNode] = llm_view,
) -> MagicMock:
    """Route the turn's engine and runner to doubles.

    Args:
        sm (SessionManager): The session manager under test.
        nodes (list[NodeWithScore]): What the engine's ``retrieve`` returns.
        tokens (list[str] | None): The answer tokens; ``ANSWER`` by default.
        view (Callable[[BaseNode], BaseNode]): The runner's LLM view; the test double by default.

    Returns:
        MagicMock: The engine double.
    """
    engine = MagicMock()
    engine.retrieve.return_value = nodes
    rag = cast(Any, sm.rag)
    rag.build_query_engine.return_value = engine
    rag.query_engine = engine
    rag.assessment_runner.side_effect = lambda engine, scoped: AssessmentRunner(
        answer_llm=cast(Any, streaming_llm(tokens or ANSWER)),
        llm_view=view,
        ctx_window=8192,
        char_token_ratio=3.5,
        reasoning=False,
    )
    return engine


def test_an_assessment_turn_streams_progress_text_then_a_final_frame(session_manager: SessionManager) -> None:
    """Progress and labelled text stream first; the final frame names the mode and trail.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    engine = _arm(session_manager, nodes=_seed())

    items = list(session_manager.stream_chat(QUESTION, session_id="s1", answer_mode="assessment"))

    progress = [item for item in items if isinstance(item, AgentProgress)]
    text = "".join(item for item in items if isinstance(item, str))
    final = items[-1]
    assert [step.action for step in progress] == ["seed", "answer"]
    assert "**Model assessment**" in text
    assert isinstance(final, dict)
    assert final["answer_mode"] == "assessment"
    assert final["retrieval_mode"] == "agent"
    assert final["agent"]["answer"] == "split"
    assert final["agent"]["evidence_chars"] == len(EVIDENCE)
    assert final["response"].startswith(EVIDENCE)
    assert final["turn_idx"] == 0
    assert engine.retrieve.call_args.args[0].query_str == "expanded question"


def test_assessment_citations_persist_in_ledger_order(session_manager: SessionManager) -> None:
    """Replay numbers citations by position, so they must be stored in ledger order.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(session_manager, nodes=_seed())

    list(session_manager.stream_chat(QUESTION, session_id="s1", answer_mode="assessment"))

    assistant = session_manager.get_session_history("s1", owner=None)[1]
    assert [(src["file_hash"], src["citation_index"]) for src in assistant["sources"]] == [("h1", 1), ("h2", 2)]
    assert assistant["answer_mode"] == "assessment"


def test_a_scoped_assessment_keeps_its_scope_and_never_researches(session_manager: SessionManager) -> None:
    """A pinned scope is never widened, and the turn still reports itself as scoped.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(session_manager, nodes=_seed())

    final = list(
        session_manager.stream_chat(QUESTION, session_id="s1", scoped_node_ids=["n1", "n2"], answer_mode="assessment")
    )[-1]

    assert isinstance(final, dict)
    assert final["retrieval_mode"] == "scoped"
    assert final["scoped_chunk_count"] == 2
    assert cast(Any, session_manager.rag).assessment_runner.call_args.kwargs["scoped"] is True


def test_a_lookup_turn_never_builds_a_runner(session_manager: SessionManager) -> None:
    """The default mode is today's pipeline, byte for byte.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    streaming_engine = MagicMock()
    response = MagicMock()
    response.response_gen = iter(["Alice [1]."])
    response.source_nodes = cast(list[Any], [])
    streaming_engine.query.return_value = response
    cast(Any, session_manager.rag).build_query_engine.return_value = streaming_engine

    final = list(session_manager.stream_chat("Who wrote it?", session_id="s1"))[-1]

    assert isinstance(final, dict)
    assert "answer_mode" not in final
    cast(Any, session_manager.rag).assessment_runner.assert_not_called()


def test_chat_drains_the_assessment_to_its_final_payload(session_manager: SessionManager) -> None:
    """The non-streaming path returns the same final payload.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(session_manager, nodes=_seed())

    response = session_manager.chat(QUESTION, session_id="s1", answer_mode="assessment")

    assert response["answer_mode"] == "assessment"
    assert response["agent"]["answer"] == "split"


def test_an_unsplit_answer_is_labelled_whole(session_manager: SessionManager) -> None:
    """Without a sentinel nothing may pass as source-backed.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(session_manager, nodes=_seed(), tokens=["No separation here [1]."])

    final = list(session_manager.stream_chat(QUESTION, session_id="s1", answer_mode="assessment"))[-1]

    assert isinstance(final, dict)
    assert final["agent"]["answer"] == "unsplit"
    assert final["agent"]["evidence_chars"] == 0
    assert final["response"].startswith("**Model assessment**")


def test_no_evidence_answers_with_the_unlabelled_fallback(session_manager: SessionManager) -> None:
    """There is nothing to judge, so the normalizer's fallback stands unlabelled.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(session_manager, nodes=[])

    final = list(session_manager.stream_chat(QUESTION, session_id="s1", answer_mode="assessment"))[-1]

    assert isinstance(final, dict)
    assert (final["agent"]["reason"], final["agent"]["answer"]) == ("no_evidence", "empty")
    assert final["response"] == "No answer."


def test_a_sentinel_only_reply_never_reaches_the_user_as_text(session_manager: SessionManager) -> None:
    """A reply holding nothing but the sentinel is empty, so the fallback stands, not the token.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(session_manager, nodes=_seed(), tokens=[SENTINEL])

    final = list(session_manager.stream_chat(QUESTION, session_id="s1", answer_mode="assessment"))[-1]

    assert isinstance(final, dict)
    assert (final["agent"]["answer"], final["agent"]["evidence_chars"]) == ("empty", None)
    assert final["response"] == "No answer."
    assert session_manager.get_session_history("s1", owner=None)[1]["content"] == "No answer."


def test_a_reply_of_only_reasoning_still_reaches_the_normalizer(session_manager: SessionManager) -> None:
    """The normalizer lifts reasoning off the raw reply, so a reply with no answer must not be blanked.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    reply = "<think>weighing the passages</think>"
    _arm(session_manager, nodes=_seed(), tokens=[reply])

    final = list(session_manager.stream_chat(QUESTION, session_id="s1", answer_mode="assessment"))[-1]

    assert isinstance(final, dict)
    assert final["agent"]["answer"] == "empty"
    assert cast(Any, session_manager.rag)._normalize_response_data.call_args.args[1].response == reply


def test_a_stateless_assessment_persists_nothing(session_manager: SessionManager) -> None:
    """``persist=False`` serves stateless requests: no turn row, no turn index.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    engine = _arm(session_manager, nodes=_seed())

    final = list(
        session_manager.stream_assessment(
            QUESTION,
            engine=engine,
            expanded_query=QUESTION,
            retrieval_query=QUESTION,
            prior_turn=None,
            scoped_node_ids=None,
            graph_debug={},
            coverage_unit="documents",
            metadata_filters_active=False,
            metadata_filter_rules=None,
            retrieval_target="all",
            session_id=None,
            owner=None,
            replace_turn_idx=None,
            turn_started_at=0.0,
            persist=False,
        )
    )[-1]

    assert isinstance(final, dict)
    assert final["turn_idx"] is None
    with session_manager._session_scope() as s:
        assert s.query(Turn).count() == 0


def test_a_stateless_assessment_runs_without_a_session() -> None:
    """The stateless entry point builds the turn's engine and never persists."""
    rag = MagicMock()
    rag._infer_collection_profile.return_value = {"coverage_unit": "documents"}
    manager = rag.ensure_session_manager.return_value
    manager.stream_assessment.return_value = iter(["text", {"answer_mode": "assessment"}])

    items = list(RAG.stream_stateless_assessment(rag, QUESTION, expanded_query="expanded", graph_debug={}))

    kwargs = manager.stream_assessment.call_args.kwargs
    assert items[-1] == {"answer_mode": "assessment"}
    assert (kwargs["persist"], kwargs["session_id"], kwargs["prior_turn"]) == (False, None, None)
    assert kwargs["engine"] is rag.build_query_engine.return_value
    assert kwargs["expanded_query"] == "expanded"
    assert rag.build_query_engine.call_args.kwargs["streaming"] is True


def test_a_stateless_assessment_streams_through_the_real_manager(session_manager: SessionManager) -> None:
    """The entry point's keywords fit ``stream_assessment``, filters count as active, nothing is written.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    rag = cast(Any, session_manager.rag)
    rag.ensure_session_manager.return_value = session_manager
    _arm(session_manager, nodes=_seed())

    items = list(
        RAG.stream_stateless_assessment(
            rag, QUESTION, expanded_query="expanded", graph_debug={}, vector_store_kwargs={"qdrant_filters": object()}
        )
    )

    final = items[-1]
    assert isinstance(final, dict)
    assert [item.action for item in items if isinstance(item, AgentProgress)] == ["seed", "answer"]
    assert (final["answer_mode"], final["session_id"], final["turn_idx"]) == ("assessment", None, None)
    assert rag.build_query_engine.call_args.kwargs["metadata_filters_active"] is True
    assert rag._normalize_response_data.call_args.kwargs["metadata_filters_active"] is True
    with session_manager._session_scope() as s:
        assert s.query(Turn).count() == 0


@pytest.mark.parametrize(("scope", "expected_reason"), [(["n1", "n2"], "scoped"), (None, "unavailable")])
def test_a_stateless_assessment_reports_why_graph_expansion_did_not_run(
    session_manager: SessionManager, scope: list[str] | None, expected_reason: str
) -> None:
    """A missing expansion payload is owed the skipped shape the session path reports, not an empty one.

    Args:
        session_manager (SessionManager): The session manager fixture.
        scope (list[str] | None): The pinned chunk ids, if any.
        expected_reason (str): The reason the panel must name.
    """
    rag = cast(Any, session_manager.rag)
    rag.ensure_session_manager.return_value = session_manager
    rag.graphrag_enabled = True
    rag._graph_debug_base.side_effect = lambda query: RAG._graph_debug_base(rag, query)
    rag.graph_debug_skipped.side_effect = lambda query, reason: RAG.graph_debug_skipped(rag, query, reason)
    _arm(session_manager, nodes=_seed())

    final = list(
        RAG.stream_stateless_assessment(
            rag, QUESTION, expanded_query="expanded", graph_debug=None, scoped_node_ids=scope
        )
    )[-1]

    assert isinstance(final, dict)
    graph_debug = final["graph_debug"]
    assert (graph_debug["applied"], graph_debug["reason"]) == (False, expected_reason)
    assert graph_debug["original_query"] == "expanded"


def test_a_stateless_assessment_starts_its_clock_before_the_engine_is_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """The logged duration covers the engine build, as it does on the session path.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    now = [100.0]
    monkeypatch.setattr("docint.core.rag.time", SimpleNamespace(monotonic=lambda: now[0]))
    rag = MagicMock()
    rag._infer_collection_profile.return_value = {"coverage_unit": "documents"}
    manager = rag.ensure_session_manager.return_value
    # Like the real stream, it ends with the final payload.
    manager.stream_assessment.return_value = iter([{"answer_mode": "assessment"}])

    def _build_engine(**_kwargs: Any) -> MagicMock:
        now[0] = 105.0
        return MagicMock()

    rag.build_query_engine.side_effect = _build_engine

    list(RAG.stream_stateless_assessment(rag, QUESTION, expanded_query="expanded", graph_debug={}))

    assert manager.stream_assessment.call_args.kwargs["turn_started_at"] == 100.0


def test_an_empty_assessment_stream_ends_a_streaming_turn_quietly(
    session_manager: SessionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pulling the first item of nothing must end the turn, not raise from inside the generator.

    Args:
        session_manager (SessionManager): The session manager fixture.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    _arm(session_manager, nodes=_seed())
    monkeypatch.setattr(SessionManager, "stream_assessment", lambda _self, *_args, **_kwargs: iter([]))

    assert list(session_manager.stream_chat(QUESTION, session_id="s1", answer_mode="assessment")) == []


def test_an_empty_stateless_assessment_stream_ends_quietly() -> None:
    """The stateless entry point ends the same way when the stream yields nothing."""
    rag = MagicMock()
    rag._infer_collection_profile.return_value = {"coverage_unit": "documents"}
    rag.ensure_session_manager.return_value.stream_assessment.return_value = iter([])

    assert list(RAG.stream_stateless_assessment(rag, QUESTION, expanded_query="expanded", graph_debug={})) == []


#: Fifty pinned posts: too many to show legibly in one 8k answer.
PINNED = [f"post-{number}" for number in range(1, 51)]
LOOKUP_ANSWER = "The festival moved to Sunday [1]."


def _arm_oversized_scope(sm: SessionManager, monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, list[Any]]:
    """Pin fifty social posts and give the engine a lookup answer to fall back to.

    Args:
        sm (SessionManager): The session manager under test.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.

    Returns:
        tuple[MagicMock, list[Any]]: The engine double and the prior turns the lookup was bound to.
    """
    engine = _arm(
        sm,
        nodes=[social_post(number) for number in range(1, 51)],
        view=ParentContextPostprocessor._emit_with_llm_exclusion,
    )
    engine.query.return_value = SimpleNamespace(
        response=LOOKUP_ANSWER, response_gen=iter([LOOKUP_ANSWER]), source_nodes=[]
    )
    bound: list[Any] = []
    monkeypatch.setattr(
        SessionManager, "_bind_prior_turn_context", lambda _self, _engine, prior_turn: bound.append(prior_turn)
    )
    return engine, bound


@pytest.mark.parametrize(
    ("entry", "bound_prior"),
    [
        # A streaming turn builds its own engine, so the source-backed part may ride along.
        ("stream_chat", PriorTurn(user_text="Is the schedule reliable?", assistant_text="It changed twice [1].")),
        # /query may run on the shared cached engine, and its lookups bind no prior turn.
        ("chat", None),
    ],
)
def test_a_scope_too_large_to_show_legibly_is_answered_as_a_lookup(
    session_manager: SessionManager,
    monkeypatch: pytest.MonkeyPatch,
    loguru_caplog: LogCaptureFixture,
    entry: str,
    bound_prior: PriorTurn | None,
) -> None:
    """Lookup is the floor: no research steps, no assessment keys, the engine answers, the judgement stays out.

    Args:
        session_manager (SessionManager): The session manager fixture.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
        entry (str): The session manager method under test.
        bound_prior (PriorTurn | None): The prior turn the lookup may bind.
    """
    engine, bound = _arm_oversized_scope(session_manager, monkeypatch)
    prior = PriorTurn(
        user_text="Is the schedule reliable?",
        assistant_text="It changed twice [1].\n\n---\n\n**Model assessment**\n\nIt seems unreliable.",
        evidence_chars=21,
    )
    kwargs: dict[str, Any] = {
        "session_id": "s1",
        "scoped_node_ids": PINNED,
        "prior_turn": prior,
        "answer_mode": "assessment",
    }

    if entry == "stream_chat":
        items = list(session_manager.stream_chat(QUESTION, **kwargs))
        assert not any(isinstance(item, AgentProgress) for item in items)
        final = items[-1]
    else:
        final = session_manager.chat(QUESTION, **kwargs)

    assert isinstance(final, dict)
    assert "answer_mode" not in final
    assert final["response"] == LOOKUP_ANSWER
    engine.query.assert_called_once()
    assert bound == [bound_prior]
    assert "does not fit the context window (50 passages); answering as a lookup" in loguru_caplog.text
    assert "Harbour festival" not in loguru_caplog.text


def test_a_stateless_scope_too_large_to_show_legibly_falls_back_to_run_query(
    session_manager: SessionManager, loguru_caplog: LogCaptureFixture
) -> None:
    """Stateless has the same floor: the request's own scope, target and filters answer as a lookup.

    Args:
        session_manager (SessionManager): The session manager fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    rag = cast(Any, session_manager.rag)
    rag.ensure_session_manager.return_value = session_manager
    _arm(
        session_manager,
        nodes=[social_post(number) for number in range(1, 51)],
        view=ParentContextPostprocessor._emit_with_llm_exclusion,
    )
    data = {"response": LOOKUP_ANSWER, "sources": [{"id": "post-1"}]}
    rag.run_query.return_value = data
    filters, store_kwargs = object(), {"qdrant_filters": object()}

    items = list(
        RAG.stream_stateless_assessment(
            rag,
            QUESTION,
            expanded_query="expanded question",
            graph_debug=None,
            metadata_filters=cast(Any, filters),
            metadata_filter_rules=["rule"],
            vector_store_kwargs=store_kwargs,
            scoped_node_ids=PINNED,
            retrieval_target="documents",
        )
    )

    assert items == [LOOKUP_ANSWER, data]
    rag.run_query.assert_called_once_with(
        "expanded question",
        metadata_filters=filters,
        metadata_filter_rules=["rule"],
        vector_store_kwargs=store_kwargs,
        scoped_node_ids=PINNED,
        retrieval_target="documents",
    )
    assert "(50 passages); answering as a lookup" in loguru_caplog.text


#: A grounded template whose rendering shows which prior turn, if any, was bound.
ECHO_TEMPLATE = "PRIOR<<{prior_turn_context}>> CTX<<{context_str}>> Q<<{query_str}>>"
PRIOR_QUESTION = "Which invented survey wave lost respondents?"


class _SeedRetriever(BaseRetriever):
    """Return the invented study passages for any query."""

    @override
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Return the seed passages.

        Args:
            query_bundle (QueryBundle): The query, unused.

        Returns:
            list[NodeWithScore]: The seed evidence.
        """
        return _seed()


def _echo_template(social_table: bool) -> PromptTemplate:
    """Build the grounded template with its default "no prior turn" binding.

    Args:
        social_table (bool): Whether the collection is a social table, unused.

    Returns:
        PromptTemplate: The partially formatted template.
    """
    return PromptTemplate(ECHO_TEMPLATE).partial_format(prior_turn_context="(no prior turn)")


def test_a_query_fallback_never_binds_a_prior_turn_onto_the_shared_engine(session_manager: SessionManager) -> None:
    """The cached engine serves every session, so a fallback's prior turn must not stay bound on it.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    rag = cast(Any, session_manager.rag)
    rag._build_grounded_text_qa_template.side_effect = _echo_template
    rag._build_grounded_refine_template.side_effect = _echo_template
    # MockLLM echoes its prompt, so each answer shows the template the synthesizer used.
    shared = RetrieverQueryEngine(
        retriever=_SeedRetriever(),
        response_synthesizer=Refine(llm=MockLLM(), text_qa_template=_echo_template(False)),
    )
    rag.query_engine = shared
    rag.assessment_runner.side_effect = lambda engine, scoped: AssessmentRunner(
        answer_llm=cast(Any, streaming_llm(["unused"])),
        llm_view=llm_view,
        ctx_window=8192,
        char_token_ratio=3.5,
        reasoning=False,
    )
    prior = PriorTurn(user_text=PRIOR_QUESTION, assistant_text="The second wave [2].")

    # Session A: an assessment question too long to leave any room for evidence.
    first = session_manager.chat(
        "Assess the survey. " + "invented filler " * 2500, session_id="a", prior_turn=prior, answer_mode="assessment"
    )
    # Session B: an ordinary lookup on the same cached engine.
    second = session_manager.chat("Where was the survey run?", session_id="b")

    assert "answer_mode" not in first
    assert rag.query_engine is shared
    assert PRIOR_QUESTION not in second["response"]
    assert PRIOR_QUESTION not in first["response"]


#: What an assessment turn said, as one answer: the sourced part, the label, then the judgement.
FACT = "The second wave lost respondents [2]."
JUDGEMENT = "The attrition probably biases the findings."
ASSESSMENT_ANSWER = f"{FACT}\n\n---\n\n**Model assessment**\n\n{JUDGEMENT}"


def _bound_block(sm: SessionManager, prior: PriorTurn | None) -> str | None:
    """Bind a prior turn onto an engine double and read back what the generation template carries.

    Args:
        sm (SessionManager): The session manager under test.
        prior (PriorTurn | None): The prior turn to bind.

    Returns:
        str | None: The bound prior-turn block, or ``None`` when nothing was bound.
    """
    rag = cast(Any, sm.rag)
    rag._build_grounded_text_qa_template.side_effect = _echo_template
    rag._build_grounded_refine_template.side_effect = _echo_template
    engine = MagicMock()

    sm._bind_prior_turn_context(engine, prior)

    if not engine.update_prompts.called:
        return None
    template = engine.update_prompts.call_args.args[0]["response_synthesizer:text_qa_template"]
    return template.format(context_str="", query_str="")


def test_a_lookup_binds_only_the_source_backed_part_of_an_assessment(session_manager: SessionManager) -> None:
    """The model's judgement must never reach a grounded prompt as if a source had said it.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    prior = PriorTurn(user_text=PRIOR_QUESTION, assistant_text=ASSESSMENT_ANSWER, evidence_chars=len(FACT))

    block = _bound_block(session_manager, prior)

    assert block is not None
    assert PRIOR_QUESTION in block
    assert FACT in block
    assert JUDGEMENT not in block
    assert "Model assessment" not in block


def test_a_lookup_binds_nothing_when_no_part_of_the_assessment_was_source_backed(
    session_manager: SessionManager,
) -> None:
    """An unsplit assessment is judgement throughout, so there is no prior turn to show a lookup.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    prior = PriorTurn(user_text=PRIOR_QUESTION, assistant_text=f"**Model assessment**\n\n{JUDGEMENT}", evidence_chars=0)

    assert _bound_block(session_manager, prior) is None


def test_a_lookup_answer_is_bound_whole(session_manager: SessionManager) -> None:
    """A lookup is source-backed throughout, so the boundary leaves it as it was.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    block = _bound_block(session_manager, PriorTurn(user_text=PRIOR_QUESTION, assistant_text=FACT))

    assert (
        block == f"PRIOR<<Previous user question: {PRIOR_QUESTION}\nPrevious assistant answer: {FACT}>> CTX<<>> Q<<>>"
    )


@pytest.mark.parametrize("entry", ["chat", "stream_chat"])
def test_a_legacy_agent_follow_up_never_sees_the_judgement_of_a_stored_assessment(
    session_manager: SessionManager, entry: str
) -> None:
    """``/agent/*`` builds its prior turn from stored history, whose assessment turns carry ``agent.evidence_chars``.

    The lookup that follows is answered by the real synthesizer; the echo template shows
    which prior turn reached it.

    Args:
        session_manager (SessionManager): The session manager fixture.
        entry (str): The session manager method under test.
    """
    rag = cast(Any, session_manager.rag)
    rag._build_grounded_text_qa_template.side_effect = _echo_template
    rag._build_grounded_refine_template.side_effect = _echo_template
    engine = RetrieverQueryEngine(
        retriever=_SeedRetriever(),
        response_synthesizer=Refine(
            llm=MockLLM(), text_qa_template=_echo_template(False), streaming=entry == "stream_chat"
        ),
    )
    rag.query_engine = engine
    rag.build_query_engine.return_value = engine
    history = [
        {"role": "user", "content": PRIOR_QUESTION},
        {
            "role": "assistant",
            "content": ASSESSMENT_ANSWER,
            "agent": {"answer": "split", "evidence_chars": len(FACT)},
        },
    ]
    prior = build_prior_turn(history)

    if entry == "chat":
        answer = session_manager.chat("Where was the survey run?", session_id="s1", prior_turn=prior)["response"]
    else:
        answer = "".join(
            item
            for item in session_manager.stream_chat("Where was the survey run?", session_id="s1", prior_turn=prior)
            if isinstance(item, str)
        )

    assert FACT in answer
    assert JUDGEMENT not in answer
