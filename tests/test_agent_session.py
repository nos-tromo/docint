"""Agent turns through the session manager: the window, the fallbacks, the final frame and what is stored.

The loop is real and its model scripted (``agent_fakes``). Every message, passage, file and
collection here is invented.
"""

from __future__ import annotations

import threading
from collections.abc import Generator, Iterator, Sequence
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, call

import pytest
from _pytest.logging import LogCaptureFixture
from agent_fakes import (
    GrantedSlots,
    ScriptedSearch,
    ThumbnailStore,
    answer_round,
    call_round,
    chunk,
    frame_passage,
    programme_passage,
    scripted_loop,
    streaming_tool_llm,
    turn_budget,
)
from assessment_fakes import bad_request, scored_node, social_post
from llama_index.core import PromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import Refine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing_extensions import override

from docint.agents.assessment.bounded import NoSlot
from docint.agents.assessment.ledger import EvidenceLedger
from docint.agents.assessment.split import SENTINEL
from docint.agents.history import build_prior_turn
from docint.agents.turn.history import CarriedPassage, ConversationWindow, question_block
from docint.agents.turn.loop import AgentLoop
from docint.agents.types import AgentProgress, PriorTurn
from docint.core.rag import AGENT_LEDGER_CAP, RAG, AgentTurn, agent_llm_view
from docint.core.state import session_manager as session_module
from docint.core.state.base import Base
from docint.core.state.session_manager import SessionManager
from docint.core.state.turn import Turn

QUESTION = "Wann öffnen die Stände beim Hafenfest?"
#: A lookup the agent writes.
LOOKUP = "Die Stände öffnen am Sonntag um zehn Uhr [1]."
#: An assessment the agent writes: what the sources say, the sentinel, then its own judgement.
EVIDENCE = "Das Hafenfest wurde wegen der Sturmwarnung auf Sonntag verschoben [1]."
JUDGEMENT = "Vermutlich kommen dadurch weniger Besucher."
ASSESSMENT = f"{EVIDENCE}\n{SENTINEL}\n{JUDGEMENT}"
#: What the plain path's engine answers, so a fallback is told apart from the agent's answer.
PLAIN_ANSWER = "Die Musik beginnt am Mittag [1]."
#: How an endpoint says a prompt does not fit its window.
OVERFLOW = "This model's maximum context length is 8192 tokens; your request has 9001 input tokens."
#: How an endpoint refuses a request offering tools; it is never logged.
TOOLS_REFUSED = "auto tool choice is not enabled on this invented server"
#: The physical name of the request's collection.
PHYSICAL = "u0123456789ab__hafen"
#: A marker that must never reach a log line.
SECRET = "zzsecretzz"


def _normalize(query: str, result: Any, reason: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Mirror the normalizer's shape: text, numbered sources, reasoning and routing fields.

    Args:
        query (str): The user's message.
        result (Any): The response object built from the turn's result.
        reason (str | None): The reasoning the caller hands over.
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
        "reasoning": reason,
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
    """SessionManager on an in-memory DB with a RAG double whose agent turns are built per test.

    Returns:
        Generator[SessionManager, None, None]: The session manager.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    rag_mock = MagicMock()
    rag_mock.index = object()
    rag_mock.qdrant_collection = PHYSICAL
    rag_mock.rewrite_retrieval_query.return_value = "rewritten question"
    rag_mock.expand_query_with_graph_with_debug.return_value = ("expanded question", {"applied": False})
    rag_mock.graph_debug_skipped.side_effect = lambda query, reason: {
        "applied": False,
        "reason": reason,
        "original_query": query,
    }
    rag_mock._infer_collection_profile.return_value = {"coverage_unit": "documents"}
    mode = MagicMock()
    mode.value = "compact"
    rag_mock._resolve_chat_response_mode.return_value = mode
    rag_mock._normalize_response_data.side_effect = _normalize
    cast(Any, rag_mock.get_source_by_node_id).return_value = None
    rag_mock.reasoning_requested = False
    rag_mock.agent_char_token_ratio = 3.5
    rag_mock.graphrag_enabled = False
    rag_mock.agent_carried_evidence.return_value = cast(list[NodeWithScore], [])
    sm = SessionManager(rag=rag_mock)
    sm._SessionMaker = sessionmaker(bind=engine)
    yield sm
    engine.dispose()


class _Recorder:
    """Stand in for a turn's loop: record what each run was handed, then run the real loop."""

    def __init__(self, loop: AgentLoop) -> None:
        """Wrap a loop.

        Args:
            loop (AgentLoop): The real loop.
        """
        self._loop = loop
        self.runs: list[dict[str, Any]] = []
        self.outputs: list[Iterator[Any]] = []

    def run(self, **kwargs: Any) -> Iterator[Any]:
        """Record the run's arguments and run the loop, keeping its output referenced.

        Args:
            **kwargs (Any): The run's keyword arguments.

        Returns:
            Iterator[Any]: The loop's output.
        """
        self.runs.append(kwargs)
        output = self._loop.run(**kwargs)
        self.outputs.append(output)
        return output


def _arm(
    sm: SessionManager,
    script: Sequence[Any],
    *,
    retrieved: Sequence[NodeWithScore] = (),
    **loop_kwargs: Any,
) -> SimpleNamespace:
    """Route the turn's engine and its agent turn to doubles; the loop is real, its model scripted.

    The engine's ``retrieve`` returns ``retrieved``, as a scoped engine returns its pinned
    passages, and its ``synthesize`` answers the plain path with ``PLAIN_ANSWER``, streaming or not.

    Args:
        sm (SessionManager): The session manager under test.
        script (Sequence[Any]): The model's rounds (see ``streaming_tool_llm``).
        retrieved (Sequence[NodeWithScore]): What the engine's ``retrieve`` returns.
        **loop_kwargs (Any): Passed to ``scripted_loop``: searches, view, slots, limits.

    Returns:
        SimpleNamespace: ``engine``, the model double ``llm``, and ``loops``, one recorder per turn built.
    """
    engine = MagicMock()
    engine.retrieve.return_value = list(retrieved)
    engine.synthesize.side_effect = lambda _query, _nodes: SimpleNamespace(
        response=PLAIN_ANSWER, response_gen=iter([PLAIN_ANSWER]), source_nodes=[]
    )
    rag = cast(Any, sm.rag)
    rag.build_query_engine.return_value = engine
    rag.query_engine = engine
    llm = streaming_tool_llm(script)
    loops: list[_Recorder] = []

    def _turn(_engine: Any, **kwargs: Any) -> AgentTurn:
        """Build the turn the factory would: a fresh loop and a ledger that holds every pinned passage.

        Args:
            _engine (Any): The turn's engine, unused: the searches are scripted.
            **kwargs (Any): The factory's keyword arguments.

        Returns:
            AgentTurn: The turn.
        """
        recorder = _Recorder(scripted_loop(llm, target=kwargs["retrieval_target"], **loop_kwargs))
        loops.append(recorder)
        return AgentTurn(
            loop=cast(Any, recorder),
            ledger=EvidenceLedger(cap=max(AGENT_LEDGER_CAP, kwargs["pinned_count"])),
            budget=turn_budget(),
            reasoning=False,
        )

    rag.agent_turn.side_effect = _turn
    return SimpleNamespace(engine=engine, llm=llm, loops=loops)


def _final(items: Sequence[Any]) -> dict[str, Any]:
    """Return a streamed turn's final payload, checking it came last.

    Args:
        items (Sequence[Any]): What the turn yielded.

    Returns:
        dict[str, Any]: The final payload.
    """
    final = items[-1]
    assert isinstance(final, dict)
    return final


def _answer(sm: SessionManager, entry: str, message: str = QUESTION, **kwargs: Any) -> dict[str, Any]:
    """Run one turn of session ``s1`` through ``chat`` or ``stream_chat`` and return its final payload.

    Args:
        sm (SessionManager): The session manager under test.
        entry (str): ``chat`` or ``stream_chat``.
        message (str): The user's message.
        **kwargs (Any): Further keyword arguments for the entry point.

    Returns:
        dict[str, Any]: The final payload.
    """
    if entry == "chat":
        return sm.chat(message, session_id="s1", **kwargs)
    return _final(list(sm.stream_chat(message, session_id="s1", **kwargs)))


def _rows(sm: SessionManager) -> list[tuple[str | None, str | None]]:
    """Read every stored turn's answer mode and rewritten query, oldest first.

    Args:
        sm (SessionManager): The session manager under test.

    Returns:
        list[tuple[str | None, str | None]]: ``(answer_mode, rewritten_query)`` per turn.
    """
    with sm._session_scope() as s:
        return [(row.answer_mode, row.rewritten_query) for row in s.query(Turn).order_by(Turn.idx)]


def _sent(llm: MagicMock, call_index: int = 0) -> list[ChatMessage]:
    """Return the messages one model call was sent.

    Args:
        llm (MagicMock): The model double.
        call_index (int): The call's position, from zero.

    Returns:
        list[ChatMessage]: The messages.
    """
    return list(llm.stream_chat.call_args_list[call_index].args[0])


def _plain_asked(engine: MagicMock) -> tuple[str, str]:
    """Read what the plain path asked the engine: the query it retrieved with, then the question it answered.

    Args:
        engine (MagicMock): The engine double; the last ``retrieve`` is the plain path's.

    Returns:
        tuple[str, str]: The retrieval query and the question the synthesizer was handed.
    """
    engine.synthesize.assert_called_once()
    return engine.retrieve.call_args.args[0].query_str, engine.synthesize.call_args.args[0].query_str


def _search(call_id: str, query: str) -> tuple[str, str, dict[str, Any]]:
    """Name one ``search`` call.

    Args:
        call_id (str): The call's id.
        query (str): What it searches for.

    Returns:
        tuple[str, str, dict[str, Any]]: The call, as ``call_round`` takes it.
    """
    return (call_id, "search", {"query": query})


def _until_raised(items: Iterator[Any]) -> tuple[list[Any], BaseException]:
    """Collect what a turn yields until it raises.

    Args:
        items (Iterator[Any]): The turn's output.

    Returns:
        tuple[list[Any], BaseException]: What it yielded first, and what it raised.
    """
    collected: list[Any] = []
    try:
        for item in items:
            collected.append(item)
    except BaseException as exc:
        return collected, exc
    raise AssertionError("The turn ended without raising.")


def test_an_agent_turn_streams_steps_tokens_then_a_final_frame(session_manager: SessionManager) -> None:
    """The search step streams first, then the answer's text, then one final frame naming the mode and trail.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(
        session_manager,
        [call_round(_search("call_a", "Hafenfest Stände")), answer_round(LOOKUP)],
        search=ScriptedSearch([programme_passage()]),
    )

    items = list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True))

    steps = [item for item in items if isinstance(item, AgentProgress)]
    texts = [index for index, item in enumerate(items) if isinstance(item, str)]
    final = _final(items)
    assert [(step.action, step.detail, step.found) for step in steps] == [("search", "Hafenfest Stände", 1)]
    assert items.index(steps[0]) < texts[0]
    assert "".join(item for item in items if isinstance(item, str)) == LOOKUP
    assert (final["retrieval_mode"], final["answer_mode"], final["response"]) == ("agent", "lookup", LOOKUP)
    assert final["agent"] == {
        "steps": 1,
        "actions": [{"action": "search", "detail": "Hafenfest Stände", "found": 1}],
        "reason": None,
        "answer": "plain",
    }
    assert [source["id"] for source in final["sources"]] == ["programme"]
    assert (final["retrieval_target"], final["session_id"], final["turn_idx"]) == ("all", "s1", 0)


def test_a_lookup_carries_answer_mode_lookup_and_a_trail(session_manager: SessionManager) -> None:
    """A reply without the sentinel is a lookup: stored with its plain trail, which holds no boundary.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(session_manager, [answer_round(LOOKUP)])

    final = _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True)))

    assistant = session_manager.get_session_history("s1", owner=None)[1]
    turns, _ = session_manager.get_conversation("s1", owner=None)
    trail: dict[str, Any] = {"steps": 0, "actions": [], "reason": None, "answer": "plain"}
    assert (final["answer_mode"], final["agent"], final["response"]) == ("lookup", trail, LOOKUP)
    assert (assistant["answer_mode"], assistant["agent"], assistant["content"]) == ("lookup", trail, LOOKUP)
    assert [mode for mode, _ in _rows(session_manager)] == ["lookup"]
    assert turns[0].evidence_chars is None


def test_a_sentinel_answer_is_stored_as_an_assessment(session_manager: SessionManager) -> None:
    """The sentinel makes the turn an assessment: labelled, split, and stored with its boundary.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(session_manager, [answer_round(ASSESSMENT)])

    final = _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True)))

    assistant = session_manager.get_session_history("s1", owner=None)[1]
    turns, _ = session_manager.get_conversation("s1", owner=None)
    assert final["answer_mode"] == "assessment"
    assert (final["agent"]["answer"], final["agent"]["evidence_chars"]) == ("split", len(EVIDENCE))
    assert final["response"].startswith(EVIDENCE)
    assert "**Model assessment**" in final["response"]
    assert SENTINEL not in final["response"]
    assert (assistant["answer_mode"], assistant["content"]) == ("assessment", final["response"])
    assert [mode for mode, _ in _rows(session_manager)] == ["assessment"]
    assert turns[0].evidence_chars == len(EVIDENCE)


@pytest.mark.parametrize("entry", ["chat", "stream_chat"])
def test_tools_unavailable_falls_back_to_the_plain_lookup_with_the_reason(
    session_manager: SessionManager, loguru_caplog_info: LogCaptureFixture, entry: str
) -> None:
    """An endpoint refusing tools: the plain path answers, reports the reason, keeps its own mode, and says why.

    The refusal is noted at INFO with its status: the API, which keeps agent turns off after it,
    gives the one warning naming what the endpoint must accept.

    Args:
        session_manager (SessionManager): The session manager fixture.
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
        entry (str): The entry point under test.
    """
    armed = _arm(session_manager, [bad_request(TOOLS_REFUSED)])

    final = _answer(session_manager, entry, agent=True)

    trail: dict[str, Any] = {"steps": 0, "actions": [], "reason": "tools_unavailable", "answer": "plain"}
    assistant = session_manager.get_session_history("s1", owner=None)[1]
    assert (final["agent"], final["retrieval_mode"], final["response"]) == (trail, "rewrite_compact", PLAIN_ANSWER)
    assert "answer_mode" not in final
    cast(Any, session_manager.rag).rewrite_retrieval_query.assert_called_once()
    assert _plain_asked(armed.engine) == ("expanded question", question_block(QUESTION, "rewritten question"))
    assert (assistant["answer_mode"], assistant["agent"]) == ("lookup", trail)
    refusals = [record for record in loguru_caplog_info.records if "offering tools" in record.getMessage()]
    assert [(record.levelname, record.getMessage()) for record in refusals] == [
        ("INFO", "The chat endpoint refused a request offering tools (HTTP 400); answering as a lookup")
    ]
    assert "reason=tools_unavailable" in next(m for m in loguru_caplog_info.messages if "Turn complete" in m)
    assert TOOLS_REFUSED not in loguru_caplog_info.text


#: What a corrective retry retrieves with: the reformulation the API wrote.
RETRY_QUERY = "Hafenfest Öffnungszeiten der Stände am Sonntag"


def test_a_retry_pass_retrieves_with_its_query_and_answers_the_users_words(session_manager: SessionManager) -> None:
    """The corrective retry searches with its reformulation, unrewritten, and asks the answer the user's own message.

    The retry is graded against the user's question, so the answer must be asked that question;
    the reformulation rides beside it as the resolved question, as a rewrite does.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    armed = _arm(session_manager, [])
    rag = cast(Any, session_manager.rag)

    final = _final(list(session_manager.stream_chat(QUESTION, session_id="s1", retrieval_query=RETRY_QUERY)))

    rag.rewrite_retrieval_query.assert_not_called()
    rag.expand_query_with_graph_with_debug.assert_called_once_with(RETRY_QUERY)
    assert _plain_asked(armed.engine) == ("expanded question", question_block(QUESTION, RETRY_QUERY))
    assert (final["response"], final["retrieval_query"]) == (PLAIN_ANSWER, RETRY_QUERY)
    assert session_manager.get_session_history("s1", owner=None)[0]["content"] == QUESTION
    assert _rows(session_manager) == [("lookup", RETRY_QUERY)]


def test_a_retry_of_a_fallback_keeps_its_trail(session_manager: SessionManager) -> None:
    """A retry of a turn whose endpoint refused tools writes the fallback's trail over the row: a reload says why.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(session_manager, [bad_request(TOOLS_REFUSED)])
    first = _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True)))

    retried = _final(
        list(
            session_manager.stream_chat(
                QUESTION,
                session_id="s1",
                retrieval_query=RETRY_QUERY,
                replace_turn_idx=first["turn_idx"],
                fallback_trail=first["agent"],
            )
        )
    )

    trail: dict[str, Any] = {"steps": 0, "actions": [], "reason": "tools_unavailable", "answer": "plain"}
    history = session_manager.get_session_history("s1", owner=None)
    assert (first["agent"], retried["agent"], retried["turn_idx"]) == (trail, trail, first["turn_idx"])
    assert [(message["role"], message.get("agent")) for message in history] == [("user", None), ("assistant", trail)]
    assert _rows(session_manager) == [("lookup", RETRY_QUERY)]


def test_the_rag_proxy_hands_a_retry_pass_on() -> None:
    """``RAG.stream_chat`` passes the retry's query and the fallback's trail on to the session manager."""
    rag = MagicMock()
    trail: dict[str, Any] = {"steps": 0, "actions": [], "reason": "tools_unavailable", "answer": "plain"}

    RAG.stream_chat(rag, QUESTION, session_id="s1", retrieval_query=RETRY_QUERY, fallback_trail=trail)

    passed = rag.ensure_session_manager.return_value.stream_chat.call_args
    assert passed.args == (QUESTION,)
    assert (passed.kwargs["retrieval_query"], passed.kwargs["fallback_trail"]) == (RETRY_QUERY, trail)


#: Fifty pinned posts: too many to show legibly in one 8k round under the agent's own view.
PINNED = [f"post-{number}" for number in range(1, 51)]


@pytest.mark.parametrize(
    ("entry", "bound_prior"),
    [
        # A streaming turn builds its own engine, so the source-backed part may ride along.
        ("stream_chat", PriorTurn(user_text="Steht das Programm fest?", assistant_text="Es änderte sich zweimal [1].")),
        # /query may run on the shared cached engine, and its lookups bind no prior turn.
        ("chat", None),
    ],
)
def test_a_scope_too_large_for_the_round_falls_back_to_the_scoped_lookup(
    session_manager: SessionManager,
    monkeypatch: pytest.MonkeyPatch,
    loguru_caplog: LogCaptureFixture,
    entry: str,
    bound_prior: PriorTurn | None,
) -> None:
    """The existing scoped path answers: no steps, no agent keys, the scope kept, and the judgement left out.

    Args:
        session_manager (SessionManager): The session manager fixture.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
        entry (str): The entry point under test.
        bound_prior (PriorTurn | None): The prior turn the lookup may bind.
    """
    armed = _arm(session_manager, [], retrieved=[social_post(number) for number in range(1, 51)], view=agent_llm_view)
    bound: list[Any] = []
    monkeypatch.setattr(
        SessionManager, "_bind_prior_turn_context", lambda _self, _engine, prior_turn: bound.append(prior_turn)
    )
    prior = PriorTurn(
        user_text="Steht das Programm fest?",
        assistant_text="Es änderte sich zweimal [1].\n\n---\n\n**Model assessment**\n\nEs wirkt unzuverlässig.",
        evidence_chars=len("Es änderte sich zweimal [1]."),
    )
    kwargs: dict[str, Any] = {"scoped_node_ids": PINNED, "prior_turn": prior, "agent": True}

    if entry == "stream_chat":
        items = list(session_manager.stream_chat(QUESTION, session_id="s1", **kwargs))
        assert not any(isinstance(item, AgentProgress) for item in items)
        final = _final(items)
    else:
        final = session_manager.chat(QUESTION, session_id="s1", **kwargs)

    assert "answer_mode" not in final
    assert "agent" not in final
    assert (final["retrieval_mode"], final["scoped_chunk_count"], final["response"]) == ("scoped", 50, PLAIN_ANSWER)
    # A prior turn handed in without skip_query_rewrite stands for the rewrite, which is then skipped.
    resolved = QUESTION if bound_prior is not None else "rewritten question"
    assert _plain_asked(armed.engine) == (resolved, question_block(QUESTION, resolved))
    armed.llm.stream_chat.assert_not_called()
    assert bound == [bound_prior]
    assert "(50 passages)" in loguru_caplog.text
    assert "Harbour festival" not in loguru_caplog.text


def test_a_stateless_agent_turn_persists_nothing(session_manager: SessionManager) -> None:
    """No session: an empty window, no row, no turn index, and the session's history is never read.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    rag = cast(Any, session_manager.rag)
    rag.ensure_session_manager.return_value = session_manager
    armed = _arm(session_manager, [answer_round(LOOKUP)])

    final = _final(list(RAG.stream_stateless_agent_turn(rag, QUESTION, expanded_query="expanded", graph_debug=None)))

    assert (final["answer_mode"], final["session_id"], final["turn_idx"]) == ("lookup", None, None)
    assert armed.loops[0].runs[0]["window"] == ConversationWindow()
    assert [message.role for message in _sent(armed.llm)] == [MessageRole.SYSTEM, MessageRole.USER]
    with session_manager._session_scope() as s:
        assert s.query(Turn).count() == 0


def test_sources_persist_contiguously_in_number_order(session_manager: SessionManager) -> None:
    """Replay numbers citations by position, so they are stored in the ledger's order, a re-found passage once.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    stalls = scored_node("n1", "Die Stände öffnen um zehn Uhr.", filename="programm.pdf", page=1, file_hash="h1")
    music = scored_node("n2", "Musik gibt es ab Mittag.", filename="programm.pdf", page=2, file_hash="h2")
    weather = scored_node("n3", "Für Sonntag ist bedeckter Himmel gemeldet.", filename="wetter.pdf", file_hash="h3")
    _arm(
        session_manager,
        [call_round(_search("call_a", "Stände")), call_round(_search("call_b", "Wetter")), answer_round(LOOKUP)],
        search=ScriptedSearch([stalls, music], [music, weather]),
    )

    final = _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True)))

    assistant = session_manager.get_session_history("s1", owner=None)[1]
    assert [source["citation_index"] for source in final["sources"]] == [1, 2, 3]
    assert [(source["file_hash"], source["citation_index"]) for source in assistant["sources"]] == [
        ("h1", 1),
        ("h2", 2),
        ("h3", 3),
    ]


@pytest.mark.parametrize("entry", ["chat", "stream_chat"])
def test_an_agent_turn_makes_no_rewrite_call_and_records_why_graph_expansion_did_not_run(
    session_manager: SessionManager, entry: str
) -> None:
    """The model reads the user's own words and writes its own searches: no rewrite, no turn-level expansion.

    Args:
        session_manager (SessionManager): The session manager fixture.
        entry (str): The entry point under test.
    """
    armed = _arm(session_manager, [answer_round(LOOKUP)])

    final = _answer(session_manager, entry, agent=True)

    rag = cast(Any, session_manager.rag)
    rag.rewrite_retrieval_query.assert_not_called()
    rag.expand_query_with_graph_with_debug.assert_not_called()
    assert final["graph_debug"] == {"applied": False, "reason": "agent", "original_query": QUESTION}
    assert final["retrieval_query"] is None
    assert _rows(session_manager) == [("lookup", None)]
    assert _sent(armed.llm)[-1].content == QUESTION


@pytest.mark.parametrize("cause", ["no-room", "overflow"])
@pytest.mark.parametrize("entry", ["chat", "stream_chat"])
def test_a_fallback_runs_the_rewrite_and_the_expansion(
    session_manager: SessionManager, loguru_caplog: LogCaptureFixture, entry: str, cause: str
) -> None:
    """A turn that cannot start is answered the plain way, which pays for the rewrite and the expansion.

    Args:
        session_manager (SessionManager): The session manager fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
        entry (str): The entry point under test.
        cause (str): No room for tool replies, or a first round that overflows the window.
    """
    armed = _arm(session_manager, [bad_request(OVERFLOW)])
    rag = cast(Any, session_manager.rag)
    if cause == "no-room":
        rag.agent_turn.side_effect = None
        rag.agent_turn.return_value = None

    final = _answer(session_manager, entry, agent=True)

    rag.rewrite_retrieval_query.assert_called_once()
    assert rag.rewrite_retrieval_query.call_args.kwargs["user_msg"] == QUESTION
    rag.expand_query_with_graph_with_debug.assert_called_once_with("rewritten question")
    assert _plain_asked(armed.engine) == ("expanded question", question_block(QUESTION, "rewritten question"))
    assert "agent" not in final
    assert "answer_mode" not in final
    assert (final["retrieval_mode"], final["response"]) == ("rewrite_compact", PLAIN_ANSWER)
    assert _rows(session_manager) == [("lookup", "rewritten question")]
    if cause == "overflow":
        assert "context window" in loguru_caplog.text


def _search_outage(_query: str) -> list[NodeWithScore]:
    """Fail as a search outage would.

    Args:
        _query (str): The query, unused.

    Raises:
        RuntimeError: Always.
    """
    raise RuntimeError("invented search outage")


@pytest.mark.parametrize(
    ("script", "slots", "model_calls"),
    [([], 0, 0), ([call_round(_search("call_a", "Stände"))], 1, 1)],
    ids=["first-round", "answer-after-unanswered-calls"],
)
@pytest.mark.parametrize("entry", ["chat", "stream_chat"])
def test_no_slot_before_any_yield_falls_back(
    session_manager: SessionManager,
    loguru_caplog: LogCaptureFixture,
    entry: str,
    script: list[Any],
    slots: int,
    model_calls: int,
) -> None:
    """A turn that has shown nothing falls back when no model slot comes free, its answer round's included.

    A call whose search failed shows no step, so a turn whose calls all went unanswered has shown
    nothing when its answer round finds no slot: the plain path answers, and nothing of the agent
    turn is stored.

    Args:
        session_manager (SessionManager): The session manager fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
        entry (str): The entry point under test.
        script (list[Any]): The model's rounds.
        slots (int): Model slots granted: none, or the first round's alone.
        model_calls (int): Model calls the turn makes before it falls back.
    """
    armed = _arm(session_manager, script, search=_search_outage, model_slots=GrantedSlots(slots), max_rounds=1)

    if entry == "chat":
        final = session_manager.chat(QUESTION, session_id="s1", agent=True)
    else:
        items = list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True))
        assert not any(isinstance(item, AgentProgress) for item in items)
        final = _final(items)

    assert (final["response"], final["retrieval_mode"]) == (PLAIN_ANSWER, "rewrite_compact")
    assert "agent" not in final
    assert _plain_asked(armed.engine) == ("expanded question", question_block(QUESTION, "rewritten question"))
    assert armed.llm.stream_chat.call_count == model_calls
    assert _rows(session_manager) == [("lookup", "rewritten question")]
    assert "agent" not in session_manager.get_session_history("s1", owner=None)[1]
    assert "No model slot" in loguru_caplog.text


def _broken_answer() -> list[Any]:
    """Script a turn whose search answers, then whose model fails twice: the tool round, then the answer.

    Returns:
        list[Any]: The model's rounds.
    """
    return [call_round(_search("call_a", "Stände")), RuntimeError("invented outage"), RuntimeError("invented outage")]


@pytest.mark.parametrize(
    ("script", "slots", "raised"),
    [
        (_broken_answer(), 8, RuntimeError),
        ([call_round(_search("call_a", "Stände"))], 1, NoSlot),
        ([call_round(_search("call_a", "Stände")), bad_request(OVERFLOW), bad_request(OVERFLOW)], 8, ValueError),
    ],
    ids=["endpoint-error", "no-slot", "overflow"],
)
@pytest.mark.parametrize("entry", ["chat", "stream_chat"])
def test_an_error_after_the_first_yield_propagates_and_persists_nothing(
    session_manager: SessionManager, entry: str, script: list[Any], slots: int, raised: type[Exception]
) -> None:
    """Once a step was shown, a failure is the turn's: no plain answer, no row, the error as it came.

    Args:
        session_manager (SessionManager): The session manager fixture.
        entry (str): The entry point under test.
        script (list[Any]): The model's rounds.
        slots (int): Model slots granted, the first round's included.
        raised (type[Exception]): What the turn must raise.
    """
    armed = _arm(session_manager, script, search=ScriptedSearch([programme_passage()]), model_slots=GrantedSlots(slots))

    if entry == "chat":
        with pytest.raises(raised):
            session_manager.chat(QUESTION, session_id="s1", agent=True)
    else:
        shown, exc = _until_raised(session_manager.stream_chat(QUESTION, session_id="s1", agent=True))
        assert isinstance(exc, raised)
        assert [item.action for item in shown if isinstance(item, AgentProgress)] == ["search"]

    armed.engine.synthesize.assert_not_called()
    cast(Any, session_manager.rag).rewrite_retrieval_query.assert_not_called()
    assert _rows(session_manager) == []


#: An answer past the round's release hold, so its first text goes out while the round holds its model slot.
LONG_ANSWER = (
    "Die Stände öffnen am Sonntag um zehn Uhr, die Musik beginnt am Mittag, "
    "und der Kai bleibt bis zum Abend für Besucher geöffnet [1]. Bei Regen ziehen die Stände in die Halle um [1]."
)


@pytest.mark.parametrize("entry", ["stream_chat", "stateless"])
def test_a_turn_closed_after_its_first_text_frees_its_model_slot_while_still_referenced(
    session_manager: SessionManager, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    """A client that stops reading frees the model slot at once, whoever still holds the turn's generators.

    The test keeps the outer generator, the agent turn's and its loop's referenced, so reference
    counting alone would leave the round suspended with its slot taken. The first round, held
    whole, searches, so the text comes from a round that streams it.

    Args:
        session_manager (SessionManager): The session manager fixture.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        entry (str): The entry point under test: a session turn, or a stateless one.
    """
    slots = threading.BoundedSemaphore(1)
    armed = _arm(
        session_manager,
        [call_round(_search("call_a", "Stände")), answer_round(LONG_ANSWER)],
        model_slots=slots,
        search=ScriptedSearch([programme_passage()]),
    )
    rag = cast(Any, session_manager.rag)
    rag.ensure_session_manager.return_value = session_manager
    kept: list[Iterator[Any]] = []
    stream_agent_turn = SessionManager.stream_agent_turn

    def _kept(self: SessionManager, *args: Any, **kwargs: Any) -> Iterator[Any]:
        """Run the agent turn, keeping its generator referenced.

        Args:
            self (SessionManager): The session manager.
            *args (Any): Positional arguments, passed on.
            **kwargs (Any): Keyword arguments, passed on.

        Returns:
            Iterator[Any]: The turn's output.
        """
        stream = stream_agent_turn(self, *args, **kwargs)
        kept.append(stream)
        return stream

    monkeypatch.setattr(SessionManager, "stream_agent_turn", _kept)
    outer = cast(
        "Generator[Any, None, None]",
        session_manager.stream_chat(QUESTION, session_id="s1", agent=True)
        if entry == "stream_chat"
        else RAG.stream_stateless_agent_turn(rag, QUESTION, expanded_query="expanded", graph_debug=None),
    )

    first = next(item for item in outer if isinstance(item, str))
    held = not slots.acquire(blocking=False)
    outer.close()
    freed = slots.acquire(blocking=False)

    assert first == LONG_ANSWER[:120]
    assert (held, freed) == (True, True)
    assert (len(kept), len(armed.loops[0].outputs)) == (1, 1)
    assert _rows(session_manager) == []


@pytest.mark.parametrize(
    "error", [RuntimeError("invented outage"), ValueError("invented malformed reply")], ids=["endpoint", "value"]
)
@pytest.mark.parametrize("entry", ["chat", "stream_chat"])
def test_a_first_round_failure_outside_the_fallbacks_propagates(
    session_manager: SessionManager, entry: str, error: Exception
) -> None:
    """Only the ruled failures fall back: the endpoint's own error is the turn's even before anything showed.

    Args:
        session_manager (SessionManager): The session manager fixture.
        entry (str): The entry point under test.
        error (Exception): What the first round raises; it names no context window.
    """
    armed = _arm(session_manager, [error])

    with pytest.raises(type(error)):
        _answer(session_manager, entry, agent=True)

    armed.engine.synthesize.assert_not_called()
    cast(Any, session_manager.rag).rewrite_retrieval_query.assert_not_called()
    assert _rows(session_manager) == []


@pytest.mark.parametrize(
    ("earlier", "replace", "folded"),
    [(4, None, True), (5, 4, False)],
    ids=["fifth-turn", "written-over"],
)
def test_an_agent_turn_folds_the_rolling_summary_as_any_turn_does(
    session_manager: SessionManager, earlier: int, replace: int | None, folded: bool
) -> None:
    """The fifth turn folds the last five into the summary; a turn written over an earlier one folds nothing.

    Args:
        session_manager (SessionManager): The session manager fixture.
        earlier (int): Turns stored before this one.
        replace (int | None): The turn this one writes over, if any.
        folded (bool): Whether the summary must be folded.
    """
    rag = cast(Any, session_manager.rag)
    rag.conversation_summary_prompt = "Fasse das Gespräch zusammen:\n"
    rag.text_model.complete.return_value = SimpleNamespace(text="Kurzfassung: Hafenfest am Sonntag.")
    resp = SimpleNamespace(source_nodes=[], metadata={})
    for number in range(earlier):
        session_manager._persist_turn("s1", f"Frage {number}?", resp, {"response": f"Antwort {number}."})
    _arm(session_manager, [answer_round(LOOKUP)])

    _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True, replace_turn_idx=replace)))

    _, summary = session_manager.get_conversation("s1", owner=None)
    assert (summary == "Kurzfassung: Hafenfest am Sonntag.") is folded


@pytest.mark.parametrize(
    ("target", "scope", "script", "visual"),
    [
        ("visual", None, [answer_round("Auf keinem Bild ist ein Schirm zu sehen.")], {"images_attached": 0}),
        (
            "visual",
            None,
            [
                call_round(("call_a", "search_images", {"description": "Person mit Schirm vor dem Hafenamt"})),
                answer_round("Person A hält einen Schirm vor dem Hafenamt [1]."),
            ],
            {"images_attached": 1},
        ),
        ("visual", ["frame-1"], [answer_round("Person A hält einen Schirm [1].")], None),
        ("documents", None, [answer_round(LOOKUP)], None),
    ],
    ids=["visual-no-picture", "visual-one-picture", "visual-scoped", "documents"],
)
def test_a_visual_turn_reports_the_pictures_its_model_saw_even_none(
    session_manager: SessionManager, target: str, scope: list[str] | None, script: list[Any], visual: Any
) -> None:
    """A visual answer that saw no pixels says so with a zero; a scope outranks the target, as on the plain path.

    Args:
        session_manager (SessionManager): The session manager fixture.
        target (str): The turn's retrieval target.
        scope (list[str] | None): The pinned chunk ids, if any.
        script (list[Any]): The model's rounds.
        visual (Any): The ``visual`` the final frame must carry.
    """
    _arm(
        session_manager,
        script,
        retrieved=[frame_passage(1)] if scope else (),
        search_images=ScriptedSearch([frame_passage(1)]),
        fetch_thumbnails=ThumbnailStore(1),
        max_images=4,
    )

    final = _final(
        list(
            session_manager.stream_chat(
                QUESTION, session_id="s1", agent=True, retrieval_target=cast(Any, target), scoped_node_ids=scope
            )
        )
    )

    assert final["visual"] == visual
    assert final["retrieval_target"] == target


@pytest.mark.parametrize(
    ("scope", "retrieved", "pinned_ids", "pinned_count"),
    [
        (None, [], None, 0),
        (["n1", "n2"], ["n1", "n2"], ["n1", "n2"], 2),
        (["n1", "n2", "n1"], ["n1", "n2", "n1"], ["n1", "n2", "n1"], 2),
        (["gone"], [], [], 0),
    ],
    ids=["unscoped", "scoped", "repeated-ids", "nothing-resolved"],
)
def test_a_scoped_turn_passes_its_pinned_nodes_and_an_unscoped_turn_none(
    session_manager: SessionManager,
    scope: list[str] | None,
    retrieved: list[str],
    pinned_ids: list[str] | None,
    pinned_count: int,
) -> None:
    """The pinned passages are what the scoped engine retrieves; an empty one is still a scope, never widened.

    Args:
        session_manager (SessionManager): The session manager fixture.
        scope (list[str] | None): The pinned chunk ids, if any.
        retrieved (list[str]): The ids of what the scoped engine resolves them to.
        pinned_ids (list[str] | None): The ids the loop must be handed; ``None`` for no scope.
        pinned_count (int): The distinct passages the factory must size the ledger for.
    """
    nodes = [
        scored_node(node_id, f"Passage {node_id} zum Hafenfest.", filename="programm.pdf") for node_id in retrieved
    ]
    armed = _arm(session_manager, [answer_round(LOOKUP)], retrieved=nodes)

    final = _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True, scoped_node_ids=scope)))

    pinned = armed.loops[0].runs[0]["pinned"]
    factory = cast(Any, session_manager.rag).agent_turn.call_args.kwargs
    assert (factory["scoped"], factory["pinned_count"]) == (scope is not None, pinned_count)
    if pinned_ids is None:
        assert pinned is None
        armed.engine.retrieve.assert_not_called()
        assert final["retrieval_mode"] == "agent"
    else:
        assert [scored.node.node_id for scored in pinned] == pinned_ids
        assert isinstance(armed.engine.retrieve.call_args.args[0], QueryBundle)
        assert (final["retrieval_mode"], final["scoped_chunk_count"]) == ("scoped", len(cast(list[str], scope)))


@pytest.mark.parametrize(
    ("target", "scope", "graphrag", "warmed"),
    [
        ("all", None, True, True),
        ("documents", None, True, True),
        ("visual", None, True, False),
        ("all", ["n1"], True, False),
        ("all", None, False, False),
    ],
    ids=["all", "documents", "visual", "scoped", "graphrag-off"],
)
def test_the_entity_graph_is_warmed_once_before_an_agent_turns_searches(
    session_manager: SessionManager, target: str, scope: list[str] | None, graphrag: bool, warmed: bool
) -> None:
    """Each search expands its own query, so a cold aggregate is built once, before the first round, not per search.

    Args:
        session_manager (SessionManager): The session manager fixture.
        target (str): The turn's retrieval target.
        scope (list[str] | None): The pinned chunk ids, if any; a scoped turn runs no search.
        graphrag (bool): Whether GraphRAG is enabled.
        warmed (bool): Whether the aggregate must be warmed.
    """
    rag = cast(Any, session_manager.rag)
    rag.graphrag_enabled = graphrag
    armed = _arm(session_manager, [answer_round(LOOKUP)], retrieved=[programme_passage()] if scope else ())
    order: list[str] = []
    rag._get_collection_ner_aggregate.side_effect = lambda **_kwargs: order.append("warm")
    stream = armed.llm.stream_chat.side_effect
    armed.llm.stream_chat.side_effect = lambda *args, **kwargs: (order.append("model"), stream(*args, **kwargs))[1]

    _final(
        list(
            session_manager.stream_chat(
                QUESTION, session_id="s1", agent=True, retrieval_target=cast(Any, target), scoped_node_ids=scope
            )
        )
    )

    if warmed:
        assert rag._get_collection_ner_aggregate.call_args_list == [call(refresh=False)]
        assert order == ["warm", "model"]
    else:
        rag._get_collection_ner_aggregate.assert_not_called()


def test_a_failed_warm_up_never_fails_the_turn(
    session_manager: SessionManager, loguru_caplog: LogCaptureFixture
) -> None:
    """Expansion is best effort: the turn answers, and the log names the failure's type, never its text.

    Args:
        session_manager (SessionManager): The session manager fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    rag = cast(Any, session_manager.rag)
    rag.graphrag_enabled = True
    rag._get_collection_ner_aggregate.side_effect = RuntimeError(f"{SECRET} invented scroll failure")
    _arm(session_manager, [answer_round(LOOKUP)])

    final = _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True)))

    assert (final["answer_mode"], final["response"]) == ("lookup", LOOKUP)
    assert "RuntimeError" in loguru_caplog.text
    assert SECRET not in loguru_caplog.text


def _earlier_turns(sm: SessionManager) -> None:
    """Store two answered lookups in session ``s1``.

    Args:
        sm (SessionManager): The session manager under test.
    """
    resp = SimpleNamespace(source_nodes=[], metadata={})
    sm._persist_turn("s1", "Wann beginnt das Hafenfest?", resp, {"response": "Am Sonntag [1]."}, answer_mode="lookup")
    sm._persist_turn("s1", "Und das Wetter?", resp, {"response": "Bedeckter Himmel [2]."}, answer_mode="lookup")


def test_the_window_holds_the_earlier_turns_but_never_the_turn_itself(session_manager: SessionManager) -> None:
    """The window is read before the turn is stored: the earlier exchanges as messages, the message once, last.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _earlier_turns(session_manager)
    armed = _arm(session_manager, [answer_round(LOOKUP)])

    _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True)))

    sent = _sent(armed.llm)
    run = armed.loops[0].runs[0]
    assert [message.role for message in sent] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
    ]
    assert [message.content for message in sent[1:]] == [
        "Wann beginnt das Hafenfest?",
        "Am Sonntag.",
        "Und das Wetter?",
        "Bedeckter Himmel.",
        QUESTION,
    ]
    assert cast(Any, session_manager.rag).agent_turn.call_args.kwargs["window"] is run["window"]
    assert (run["message"], len(run["window"].recent)) == (QUESTION, 2)


def test_a_replaced_turn_is_not_in_its_own_window(session_manager: SessionManager) -> None:
    """A turn written over an earlier row answers as that row did: without it in its window.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _earlier_turns(session_manager)
    armed = _arm(session_manager, [answer_round(LOOKUP)])

    _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True, replace_turn_idx=1)))

    assert [message.content for message in _sent(armed.llm)[1:]] == [
        "Wann beginnt das Hafenfest?",
        "Am Sonntag.",
        QUESTION,
    ]
    assert len(_rows(session_manager)) == 2


@pytest.mark.parametrize(("reasoning", "allowance"), [(False, 1300), (True, 750)])
def test_the_window_is_built_at_the_allowance_of_the_turns_reasoning(
    session_manager: SessionManager, monkeypatch: pytest.MonkeyPatch, reasoning: bool, allowance: int
) -> None:
    """The factory measures the turn for the request's reasoning, so the window must be built for it too.

    Args:
        session_manager (SessionManager): The session manager fixture.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        reasoning (bool): Whether the request asks the model to think.
        allowance (int): The history allowance the window must be built at.
    """
    rag = cast(Any, session_manager.rag)
    rag.reasoning_requested = reasoning
    rag.agent_char_token_ratio = 3.0
    built: list[tuple[int, float]] = []
    real = session_module.build_window

    def _build(
        turns: Any, summary: str, *, allowance_tokens: int, char_token_ratio: float, carried: Any
    ) -> ConversationWindow:
        """Record the window's allowance and ratio, then build it.

        Args:
            turns (Any): The stored turns.
            summary (str): The rolling summary.
            allowance_tokens (int): The allowance.
            char_token_ratio (float): The ratio.
            carried (Any): The passages to list.

        Returns:
            ConversationWindow: The window.
        """
        built.append((allowance_tokens, char_token_ratio))
        return real(
            turns, summary, allowance_tokens=allowance_tokens, char_token_ratio=char_token_ratio, carried=carried
        )

    monkeypatch.setattr(session_module, "build_window", _build)
    _earlier_turns(session_manager)
    _arm(session_manager, [answer_round(LOOKUP)])

    _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True)))

    assert built == [(allowance, 3.0)]


def _sources() -> tuple[NodeWithScore, NodeWithScore, NodeWithScore]:
    """Build the three invented passages an earlier answer was given, ``n1`` to ``n3``.

    Returns:
        tuple[NodeWithScore, NodeWithScore, NodeWithScore]: The stalls, the music and the weather.
    """
    return (
        scored_node("n1", "Die Stände öffnen um zehn Uhr.", filename="programm.pdf", page=1),
        scored_node("n2", "Musik gibt es ab Mittag.", filename="programm.pdf", page=2),
        scored_node("n3", "Für Sonntag ist bedeckter Himmel gemeldet.", filename="wetter.pdf"),
    )


def _cited_turn(sm: SessionManager, question: str, answer: str, *sources: NodeWithScore) -> None:
    """Store one answered lookup in session ``s1``, its citations in number order.

    Args:
        sm (SessionManager): The session manager under test.
        question (str): The user's message.
        answer (str): The answer, its markers numbering ``sources``.
        *sources (NodeWithScore): The passages the answer was given, ``[1]`` first.
    """
    resp = SimpleNamespace(source_nodes=list(sources), metadata={})
    sm._persist_turn("s1", question, resp, {"response": answer}, answer_mode="lookup")


def _stored(sm: SessionManager, *nodes: NodeWithScore) -> None:
    """Let the RAG double load these passages as carried evidence, each labelled by its file name.

    Args:
        sm (SessionManager): The session manager under test.
        *nodes (NodeWithScore): The passages the store holds.
    """
    rag = cast(Any, sm.rag)
    held = {scored.node.node_id: scored for scored in nodes}
    rag.agent_carried_evidence.side_effect = lambda node_ids, **_kwargs: [held[i] for i in node_ids if i in held]
    rag.agent_passage_label.side_effect = lambda scored: f"filename: {scored.node.metadata['filename']}"


def test_a_follow_up_carries_what_its_last_answer_cites(session_manager: SessionManager) -> None:
    """The passage an answer cites as [2] is loaded under the turn's target and filters and opens the turn as [1].

    The window renumbers the answer's marker to match, the loop is handed the passage, and the new
    answer stores it as its own first source, so a further follow-up can carry it again.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    stalls, music, weather = _sources()
    _cited_turn(session_manager, "Wann beginnt die Musik?", "Die Musik beginnt am Mittag [2].", stalls, music, weather)
    _stored(session_manager, stalls, music, weather)
    armed = _arm(session_manager, [answer_round("Ab Mittag [1].")])
    rules = [{"field": "mimetype", "operator": "eq", "value": "application/pdf"}]

    final = _final(
        list(
            session_manager.stream_chat(
                QUESTION,
                session_id="s1",
                agent=True,
                retrieval_target="documents",
                metadata_filters_active=True,
                metadata_filter_rules=rules,
            )
        )
    )

    rag = cast(Any, session_manager.rag)
    run = armed.loops[0].runs[0]
    turns, _ = session_manager.get_conversation("s1", owner=None, sources_for=1)
    assert rag.agent_carried_evidence.call_args == call(
        ("n2",), retrieval_target="documents", metadata_filter_rules=rules, metadata_filters_active=True
    )
    assert [scored.node.node_id for scored in run["carried"]] == ["n2"]
    assert run["window"].carried == (CarriedPassage("n2", "filename: programm.pdf"),)
    assert _sent(armed.llm)[2].content == "Die Musik beginnt am Mittag [1]."
    assert final["agent"]["actions"][0] == {"action": "carried", "detail": "", "found": 1}
    assert [(source["id"], source["citation_index"]) for source in final["sources"]] == [("n2", 1)]
    assert turns[-1].sources == ("n2",)


@pytest.mark.parametrize(("reasoning", "carried"), [(False, ("n1", "n3")), (True, ("n3",))], ids=["plain", "thinking"])
def test_a_follow_up_carries_from_the_exchanges_its_window_sends(
    session_manager: SessionManager, reasoning: bool, carried: tuple[str, ...]
) -> None:
    """A thinking turn's window sends the last exchange alone, so only that answer's passages are carried.

    Args:
        session_manager (SessionManager): The session manager fixture.
        reasoning (bool): Whether the request asks the model to think.
        carried (tuple[str, ...]): The ids the turn must ask to carry.
    """
    rag = cast(Any, session_manager.rag)
    rag.reasoning_requested = reasoning
    stalls, music, weather = _sources()
    _cited_turn(session_manager, "Wann öffnen die Stände?", "Um zehn Uhr [1].", stalls)
    _cited_turn(session_manager, "Und das Wetter?", "Bedeckter Himmel [1].", weather)
    _stored(session_manager, stalls, music, weather)
    _arm(session_manager, [answer_round(LOOKUP)])

    _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True)))

    assert rag.agent_carried_evidence.call_args.args == (carried,)


def test_a_follow_up_carries_exactly_the_passages_its_window_lists(session_manager: SessionManager) -> None:
    """When the list cannot hold every cited passage, the newest are listed, and the loop is handed those, by id.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    reports = [
        scored_node(f"n{number}", f"Bericht {number}.", filename=f"hafenbericht_{number}_{'x' * 100}.pdf")
        for number in range(1, 9)
    ]
    answer = "Es gibt acht Berichte " + " ".join(f"[{number}]" for number in range(1, 9)) + "."
    _cited_turn(session_manager, "Welche Berichte gibt es?", answer, *reports)
    _stored(session_manager, *reports)
    armed = _arm(session_manager, [answer_round(LOOKUP)])

    _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True)))

    run = armed.loops[0].runs[0]
    listed = [passage.node_id for passage in run["window"].carried]
    assert 0 < len(listed) < len(reports)
    assert listed == [f"n{number}" for number in range(len(reports) - len(listed) + 1, len(reports) + 1)]
    assert [scored.node.node_id for scored in run["carried"]] == listed


def test_a_scoped_turn_carries_nothing(session_manager: SessionManager) -> None:
    """A pinned scope is the whole evidence: nothing is loaded, the loop is handed nothing, and the old marker goes.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    stalls, music, weather = _sources()
    _cited_turn(session_manager, "Wann beginnt die Musik?", "Die Musik beginnt am Mittag [2].", stalls, music, weather)
    _stored(session_manager, stalls, music, weather)
    armed = _arm(session_manager, [answer_round("Ab Mittag [1].")], retrieved=[music])

    _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True, scoped_node_ids=["n2"])))

    run = armed.loops[0].runs[0]
    cast(Any, session_manager.rag).agent_carried_evidence.assert_not_called()
    assert (list(run["carried"]), run["window"].carried) == ([], ())
    assert _sent(armed.llm)[2].content == "Die Musik beginnt am Mittag."


def test_a_stateless_turn_carries_nothing(session_manager: SessionManager) -> None:
    """Without a session there is no earlier answer: nothing is loaded and the loop is handed nothing.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    rag = cast(Any, session_manager.rag)
    rag.ensure_session_manager.return_value = session_manager
    armed = _arm(session_manager, [answer_round(LOOKUP)])

    _final(list(RAG.stream_stateless_agent_turn(rag, QUESTION, expanded_query="expanded", graph_debug=None)))

    rag.agent_carried_evidence.assert_not_called()
    assert list(armed.loops[0].runs[0]["carried"]) == []


def test_a_replaced_turn_carries_from_the_turns_before_it(session_manager: SessionManager) -> None:
    """A turn written over reads the conversation before it, so it carries what those answers cite, never its own.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    stalls, music, weather = _sources()
    _cited_turn(session_manager, "Wann öffnen die Stände?", "Um zehn Uhr [1].", stalls)
    _cited_turn(session_manager, "Und das Wetter?", "Bedeckter Himmel [1].", weather)
    _stored(session_manager, stalls, music, weather)
    armed = _arm(session_manager, [answer_round(LOOKUP)])

    _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True, replace_turn_idx=1)))

    assert cast(Any, session_manager.rag).agent_carried_evidence.call_args.args == (("n1",),)
    assert [scored.node.node_id for scored in armed.loops[0].runs[0]["carried"]] == ["n1"]
    assert [message.content for message in _sent(armed.llm)[1:]] == [
        "Wann öffnen die Stände?",
        "Um zehn Uhr [1].",
        QUESTION,
    ]


def test_the_turn_is_built_from_the_requests_engine_target_and_filters(session_manager: SessionManager) -> None:
    """The factory gets the turn's own engine, target, filter inputs and the message verbatim.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    armed = _arm(session_manager, [answer_round(LOOKUP)])
    rules = [{"field": "mimetype", "operator": "eq", "value": "application/pdf"}]
    store_kwargs = {"qdrant_filters": object()}

    _final(
        list(
            session_manager.stream_chat(
                QUESTION,
                session_id="s1",
                agent=True,
                retrieval_target="documents",
                metadata_filters=cast(Any, object()),
                metadata_filters_active=True,
                metadata_filter_rules=rules,
                vector_store_kwargs=store_kwargs,
            )
        )
    )

    rag = cast(Any, session_manager.rag)
    args, kwargs = rag.agent_turn.call_args
    assert args == (armed.engine,)
    assert (kwargs["retrieval_target"], kwargs["message"], kwargs["scoped"]) == ("documents", QUESTION, False)
    assert kwargs["metadata_filter_rules"] is rules
    assert kwargs["metadata_filters_active"] is True
    assert kwargs["vector_store_kwargs"] is store_kwargs
    assert rag.build_query_engine.call_args.kwargs["streaming"] is True


def test_an_agent_turn_carries_the_models_reasoning(session_manager: SessionManager) -> None:
    """The reasoning the rounds wrote reaches the final frame and the stored turn, never the answer.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    thought = "Die Frage betrifft das Programm des Hafenfests."
    _arm(session_manager, [answer_round(LOOKUP, reasoning=thought)])

    final = _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True)))

    assert (final["reasoning"], final["response"]) == (thought, LOOKUP)
    assert session_manager.get_session_history("s1", owner=None)[1]["reasoning"] == thought


def test_a_sentinel_only_reply_never_reaches_the_user_as_text(session_manager: SessionManager) -> None:
    """A reply of nothing but the sentinel is an empty assessment; the fallback stands, not the token.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(session_manager, [answer_round(SENTINEL)])

    final = _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True)))

    assert final["answer_mode"] == "assessment"
    assert (final["agent"]["answer"], final["agent"]["evidence_chars"]) == ("empty", None)
    assert final["response"] == "No answer."
    assert cast(Any, session_manager.rag)._normalize_response_data.call_args.args[1].response == ""
    assert session_manager.get_session_history("s1", owner=None)[1]["content"] == "No answer."


def test_an_empty_answer_is_a_lookup_and_the_normalizers_fallback_stands(session_manager: SessionManager) -> None:
    """An answer round that wrote nothing carries no sentinel, so it is a lookup with the fallback text.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(session_manager, [answer_round("")])

    final = _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True)))

    assert (final["answer_mode"], final["agent"]["answer"], final["response"]) == ("lookup", "plain", "No answer.")


def test_a_failed_search_is_a_reason_notice_on_a_complete_answer(session_manager: SessionManager) -> None:
    """``research_failed`` says evidence may be missing; the answer is still the turn's, and is stored.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(session_manager, [call_round(_search("call_a", "Stände")), answer_round(LOOKUP)], search=_search_outage)

    final = _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True)))

    assert (final["agent"]["reason"], final["response"]) == ("research_failed", LOOKUP)
    assert session_manager.get_session_history("s1", owner=None)[1]["agent"]["reason"] == "research_failed"


def test_an_answer_cut_at_the_length_limit_is_reported_stored_and_logged_as_cut(
    session_manager: SessionManager, loguru_caplog_info: LogCaptureFixture
) -> None:
    """``answer_cut`` reaches the final frame, the stored trail and the turn's log line; the cut text is kept.

    Args:
        session_manager (SessionManager): The session manager fixture.
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
    """
    cut = [*answer_round(LOOKUP)[:-1], chunk(finish_reason="length")]
    _arm(session_manager, [call_round(_search("call_a", "Stände")), cut], search=ScriptedSearch([programme_passage()]))

    final = _final(list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True)))

    assert (final["agent"]["reason"], final["response"]) == ("answer_cut", LOOKUP)
    assert session_manager.get_session_history("s1", owner=None)[1]["agent"]["reason"] == "answer_cut"
    line = next(message for message in loguru_caplog_info.messages if "Turn complete" in message)
    assert "reason=answer_cut" in line


def test_an_agent_turn_logs_its_mode_answer_steps_and_reason_but_never_its_text(
    session_manager: SessionManager, loguru_caplog_info: LogCaptureFixture
) -> None:
    """One turn line with the agent's counts and tokens; the message, the searches and the answer stay out.

    Args:
        session_manager (SessionManager): The session manager fixture.
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
    """
    _arm(
        session_manager,
        [call_round(_search("call_a", f"{SECRET} Stände")), answer_round(f"{SECRET} {LOOKUP}")],
        search=ScriptedSearch([programme_passage()]),
    )

    _final(list(session_manager.stream_chat(f"{SECRET} {QUESTION}", session_id="s1", agent=True)))

    line = next(message for message in loguru_caplog_info.messages if "Turn complete" in message)
    for field in ("mode=agent", "answer=lookup", "steps=1", "outcome=plain", "reason=none"):
        assert field in line
    assert SECRET not in "\n".join(loguru_caplog_info.messages)


def test_chat_drains_an_agent_turn_to_its_final_payload(session_manager: SessionManager) -> None:
    """The non-streaming path returns the same final payload.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _arm(
        session_manager,
        [call_round(_search("call_a", "Hafenfest")), answer_round(ASSESSMENT)],
        search=ScriptedSearch([programme_passage()]),
    )

    response = session_manager.chat(QUESTION, session_id="s1", agent=True)

    assert (response["answer_mode"], response["agent"]["answer"], response["turn_idx"]) == ("assessment", "split", 0)
    assert response["agent"]["steps"] == 1


@pytest.mark.parametrize("entry", ["chat", "stream_chat"])
def test_a_plain_turn_never_builds_an_agent_turn(session_manager: SessionManager, entry: str) -> None:
    """Without the flag the turn is today's pipeline: no agent turn, no agent keys.

    Args:
        session_manager (SessionManager): The session manager fixture.
        entry (str): The entry point under test.
    """
    _arm(session_manager, [])

    final = _answer(session_manager, entry)

    cast(Any, session_manager.rag).agent_turn.assert_not_called()
    assert "answer_mode" not in final
    assert "agent" not in final
    assert final["response"] == PLAIN_ANSWER


def test_an_empty_agent_stream_ends_a_streaming_turn_quietly(
    session_manager: SessionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pulling the first item of nothing must end the turn, not raise from inside the generator.

    Args:
        session_manager (SessionManager): The session manager fixture.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    _arm(session_manager, [])
    monkeypatch.setattr(SessionManager, "stream_agent_turn", lambda _self, *_args, **_kwargs: iter([]))

    assert list(session_manager.stream_chat(QUESTION, session_id="s1", agent=True)) == []


def test_a_stateless_agent_turn_runs_without_a_session() -> None:
    """The stateless entry point builds the turn's streaming engine and never persists."""
    rag = MagicMock()
    rag._infer_collection_profile.return_value = {"coverage_unit": "documents"}
    manager = rag.ensure_session_manager.return_value
    manager.stream_agent_turn.return_value = iter(["text", {"answer_mode": "lookup"}])

    items = list(RAG.stream_stateless_agent_turn(rag, QUESTION, expanded_query="expanded", graph_debug={}))

    kwargs = manager.stream_agent_turn.call_args.kwargs
    assert items == ["text", {"answer_mode": "lookup"}]
    assert manager.stream_agent_turn.call_args.args == (QUESTION,)
    assert (kwargs["persist"], kwargs["session_id"], kwargs["owner"], kwargs["replace_turn_idx"]) == (
        False,
        None,
        None,
        None,
    )
    assert kwargs["engine"] is rag.build_query_engine.return_value
    assert kwargs["coverage_unit"] == "documents"
    assert rag.build_query_engine.call_args.kwargs["streaming"] is True


def test_a_stateless_agent_turn_streams_through_the_real_manager(session_manager: SessionManager) -> None:
    """The entry point's keywords fit ``stream_agent_turn``: filters count as active, nothing is written.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    rag = cast(Any, session_manager.rag)
    rag.ensure_session_manager.return_value = session_manager
    _arm(
        session_manager,
        [call_round(_search("call_a", "Stände")), answer_round(LOOKUP)],
        search=ScriptedSearch([programme_passage()]),
    )
    store_kwargs = {"qdrant_filters": object()}

    items = list(
        RAG.stream_stateless_agent_turn(
            rag, QUESTION, expanded_query="expanded", graph_debug=None, vector_store_kwargs=store_kwargs
        )
    )

    final = _final(items)
    assert [item.action for item in items if isinstance(item, AgentProgress)] == ["search"]
    assert (final["answer_mode"], final["session_id"], final["turn_idx"]) == ("lookup", None, None)
    assert final["graph_debug"] == {"applied": False, "reason": "agent", "original_query": QUESTION}
    assert rag.build_query_engine.call_args.kwargs["metadata_filters_active"] is True
    assert rag.agent_turn.call_args.kwargs["metadata_filters_active"] is True
    assert rag.agent_turn.call_args.kwargs["vector_store_kwargs"] is store_kwargs
    with session_manager._session_scope() as s:
        assert s.query(Turn).count() == 0


@pytest.mark.parametrize(
    ("cause", "trail"),
    [
        ("tools-unavailable", {"steps": 0, "actions": [], "reason": "tools_unavailable", "answer": "plain"}),
        ("scope-too-large", None),
    ],
)
def test_a_stateless_agent_turn_falls_back_to_run_query(
    session_manager: SessionManager, cause: str, trail: dict[str, Any] | None
) -> None:
    """Stateless has the same floor: the request's own query, scope, target and filters answer through run_query.

    Args:
        session_manager (SessionManager): The session manager fixture.
        cause (str): Why the agent turn falls back.
        trail (dict[str, Any] | None): The trail the plain answer must report.
    """
    rag = cast(Any, session_manager.rag)
    rag.ensure_session_manager.return_value = session_manager
    scoped = cause == "scope-too-large"
    _arm(
        session_manager,
        [] if scoped else [bad_request(TOOLS_REFUSED)],
        retrieved=[social_post(number) for number in range(1, 51)] if scoped else (),
        view=agent_llm_view,
    )
    rag.run_query.return_value = {"response": PLAIN_ANSWER, "sources": [{"id": "post-1"}]}
    filters, store_kwargs = object(), {"qdrant_filters": object()}
    expansion = {"applied": True, "reason": None}
    scope = PINNED if scoped else None

    items = list(
        RAG.stream_stateless_agent_turn(
            rag,
            QUESTION,
            expanded_query="expanded question",
            graph_debug=expansion,
            metadata_filters=cast(Any, filters),
            metadata_filter_rules=["rule"],
            vector_store_kwargs=store_kwargs,
            scoped_node_ids=scope,
            retrieval_target="documents",
        )
    )

    data = items[-1]
    assert items[:-1] == [PLAIN_ANSWER]
    rag.run_query.assert_called_once_with(
        QUESTION,
        expanded_query="expanded question",
        metadata_filters=filters,
        metadata_filter_rules=["rule"],
        vector_store_kwargs=store_kwargs,
        scoped_node_ids=scope,
        retrieval_target="documents",
    )
    assert data.get("agent") == trail
    assert data["graph_debug"] is expansion


def test_a_stateless_agent_turn_starts_its_clock_before_the_engine_is_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """The logged duration covers the engine build, as it does on the session path.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    now = [100.0]
    monkeypatch.setattr("docint.core.rag.time", SimpleNamespace(monotonic=lambda: now[0]))
    rag = MagicMock()
    rag._infer_collection_profile.return_value = {"coverage_unit": "documents"}
    manager = rag.ensure_session_manager.return_value
    manager.stream_agent_turn.return_value = iter([{"answer_mode": "lookup"}])

    def _build_engine(**_kwargs: Any) -> MagicMock:
        """Let five seconds pass while the engine is built.

        Args:
            **_kwargs (Any): The engine's settings, unused.

        Returns:
            MagicMock: The engine.
        """
        now[0] = 105.0
        return MagicMock()

    rag.build_query_engine.side_effect = _build_engine

    list(RAG.stream_stateless_agent_turn(rag, QUESTION, expanded_query="expanded", graph_debug={}))

    assert manager.stream_agent_turn.call_args.kwargs["turn_started_at"] == 100.0


def test_an_empty_stateless_agent_stream_ends_quietly() -> None:
    """The stateless entry point ends the same way when the stream yields nothing."""
    rag = MagicMock()
    rag._infer_collection_profile.return_value = {"coverage_unit": "documents"}
    rag.ensure_session_manager.return_value.stream_agent_turn.return_value = iter([])

    assert list(RAG.stream_stateless_agent_turn(rag, QUESTION, expanded_query="expanded", graph_debug={})) == []


#: A grounded template whose rendering shows which prior turn, if any, was bound.
ECHO_TEMPLATE = "PRIOR<<{prior_turn_context}>> CTX<<{context_str}>> Q<<{query_str}>>"
PRIOR_QUESTION = "Which invented survey wave lost respondents?"


class _SeedRetriever(BaseRetriever):
    """Return two invented study passages for any query."""

    @override
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Return the passages.

        Args:
            query_bundle (QueryBundle): The query, unused.

        Returns:
            list[NodeWithScore]: The passages.
        """
        return [
            scored_node("n1", "We evaluate on one dataset.", filename="study.pdf", page=4, file_hash="h1"),
            scored_node("n2", "Results improve by two points.", filename="study.pdf", page=6, file_hash="h2"),
        ]


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
    # No room for an agent turn, so it falls back to the plain path on the shared engine.
    rag.agent_turn.return_value = None
    prior = PriorTurn(user_text=PRIOR_QUESTION, assistant_text="The second wave [2].")

    first = session_manager.chat("Assess the survey.", session_id="a", prior_turn=prior, agent=True)
    second = session_manager.chat("Where was the survey run?", session_id="b")

    assert "answer_mode" not in first
    assert rag.query_engine is shared
    assert PRIOR_QUESTION not in second["response"]
    assert PRIOR_QUESTION not in first["response"]


#: What an assessment turn said, as one answer: the sourced part, the label, then the judgement.
FACT = "The second wave lost respondents [2]."
VIEW = "The attrition probably biases the findings."
ASSESSMENT_ANSWER = f"{FACT}\n\n---\n\n**Model assessment**\n\n{VIEW}"


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
    assert VIEW not in block
    assert "Model assessment" not in block


def test_a_lookup_binds_nothing_when_no_part_of_the_assessment_was_source_backed(
    session_manager: SessionManager,
) -> None:
    """An unsplit assessment is judgement throughout, so there is no prior turn to show a lookup.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    prior = PriorTurn(user_text=PRIOR_QUESTION, assistant_text=f"**Model assessment**\n\n{VIEW}", evidence_chars=0)

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
    assert VIEW not in answer
