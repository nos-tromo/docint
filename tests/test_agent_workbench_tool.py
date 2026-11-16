"""The run_python tool of an agent turn: the turn's evidence handed to workbench inline, its tables and charts kept.

Every passage, name, number, code and output here is invented.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, get_args

import pytest
from _pytest.logging import LogCaptureFixture
from agent_fakes import (
    COUNTS_CSV,
    WORKBENCH_JOB,
    FakeWorkbench,
    run_result,
    tiny_png,
    tool_call,
    tool_context,
    two_phase_round,
)
from assessment_fakes import scored_node
from llama_index.core.schema import NodeWithScore

import docint.agents.turn.workbench_tool as wt
from docint.agents.turn.keyword import check_key
from docint.agents.turn.tools import ToolContext, ToolRegistry, ToolReply
from docint.agents.turn.workbench_tool import (
    ANALYSIS_NOTE,
    CELL_CHARS,
    MAX_CHARTS,
    MAX_CODE_CHARS,
    MAX_COLUMNS,
    MAX_ROWS,
    MAX_TABLES,
    NAME,
    PASSAGE_TEXT_CHARS,
    PythonTurn,
    WorkbenchPythonTool,
    parse_table,
)
from docint.core.rag import agent_llm_view
from docint.utils.audit_disclose import DisclosureLedger, DisclosureRecord, binding
from docint.utils.workbench_client import Discovery, FailureKind, OutputFile, RunFailure

TURN = PythonTurn("user-a", "session-1", deadline=1_000.0)
DISCOVERY = Discovery(60, "0.2.0")
JOB = WORKBENCH_JOB
CODE = "import json\nprint(len(json.load(open('passages.json'))))"
CSV = COUNTS_CSV
TEXT = "The harbour office counted twelve berths in May and seven in June."
HASH = "a" * 64
TITLE = "Posts per author"
KEYWORD_FACT: dict[str, Any] = {
    "action": "keyword",
    "detail": "Hafenfest",
    "field": "author",
    "matches": 3,
    "unchecked": 2,
}
TOPICS_FACT: dict[str, Any] = {
    "action": "topics",
    "detail": "",
    "total": 400,
    "filtered": False,
    "topics": [[4, "Hafenfest und Stände", 120]],
}


def _tool(
    workbench: FakeWorkbench, *, now: float = 880.0, slots: threading.Semaphore | None = None
) -> WorkbenchPythonTool:
    """Build the tool over a workbench double, its clock standing still.

    Args:
        workbench (FakeWorkbench): The double the calls reach.
        now (float): What the clock reads; the default leaves the turn 120 s.
        slots (threading.Semaphore | None): The process's workbench slots; two free ones by default.

    Returns:
        WorkbenchPythonTool: The tool.
    """
    pool = slots if slots is not None else threading.BoundedSemaphore(2)
    return WorkbenchPythonTool(workbench, DISCOVERY, TURN, clock=lambda: now, slots=pool)


def _context() -> ToolContext:
    """Build an empty turn context whose LLM view is the production one.

    Returns:
        ToolContext: The context.
    """
    return tool_context(view=agent_llm_view)


def _passage(number: int, text: str = TEXT, **metadata: Any) -> NodeWithScore:
    """Build an invented passage, its point id ``p<number>``.

    Args:
        number (int): Its place, which names it.
        text (str): Its text.
        **metadata (Any): Its metadata.

    Returns:
        NodeWithScore: The passage.
    """
    return scored_node(f"p{number}", text, **metadata)


def _held(ctx: ToolContext, *passages: NodeWithScore) -> None:
    """Admit invented passages into the turn's evidence, numbered in the order given.

    Args:
        ctx (ToolContext): The turn's context.
        *passages (NodeWithScore): The passages.
    """
    ctx.ledger.add(passages)


def _three(ctx: ToolContext) -> None:
    """Admit three invented passages.

    Args:
        ctx (ToolContext): The turn's context.
    """
    _held(ctx, _passage(1), _passage(2), _passage(3))


def _round(
    registry: ToolRegistry, ctx: ToolContext, *arguments: dict[str, Any], ledger: DisclosureLedger | None = None
) -> list[ToolReply]:
    """Answer one round of run_python calls, the fetches inside a bound ledger.

    Args:
        registry (ToolRegistry): The turn's tools.
        ctx (ToolContext): The turn's context.
        *arguments (dict[str, Any]): Per call, its arguments.
        ledger (DisclosureLedger | None): The turn's ledger; a fresh one by default.

    Returns:
        list[ToolReply]: One reply per call.
    """
    registry.begin_round()
    with binding(ledger if ledger is not None else DisclosureLedger()):
        return two_phase_round(registry, [tool_call(NAME, **call) for call in arguments], ctx)


def _run(
    tool: WorkbenchPythonTool, ctx: ToolContext, *, ledger: DisclosureLedger | None = None, **arguments: Any
) -> ToolReply:
    """Answer one run_python call in a round of its own.

    Args:
        tool (WorkbenchPythonTool): The tool.
        ctx (ToolContext): The turn's context.
        ledger (DisclosureLedger | None): The turn's ledger; a fresh one by default.
        **arguments (Any): The call's arguments.

    Returns:
        ToolReply: The reply.
    """
    [reply] = _round(ToolRegistry([tool]), ctx, arguments, ledger=ledger)
    return reply


def _sent(workbench: FakeWorkbench) -> dict[str, str]:
    """Return the files the last call handed over, by name, in the order sent.

    Args:
        workbench (FakeWorkbench): The double.

    Returns:
        dict[str, str]: Each file's content.
    """
    return {item.name: item.content for item in workbench.calls[-1]["data"]}


def _entry(number: int, text: str = TEXT) -> dict[str, Any]:
    """Return the passages.json entry of a passage that carries no metadata.

    Args:
        number (int): Its number.
        text (str): Its text.

    Returns:
        dict[str, Any]: The entry.
    """
    keys = ("file", "page", "time", "author", "network", "kind")
    return {"n": number, "text": text, **dict.fromkeys(keys)}


def _size(entries: list[Any]) -> int:
    """Return the UTF-8 bytes of a file holding these entries, as the tool writes it.

    Args:
        entries (list[Any]): The entries.

    Returns:
        int: The size.
    """
    return len(json.dumps(entries, ensure_ascii=False).encode("utf-8"))


# --- limits ------------------------------------------------------------------


def test_a_parallel_pair_in_one_round_runs_one_and_refuses_the_other() -> None:
    """Two calls in one round: the first runs, the second is told to call again next round, which then runs."""
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench), _context()
    _three(ctx)
    registry = ToolRegistry([tool])

    first, second = _round(registry, ctx, {"code": CODE}, {"code": CODE})

    assert not first.error
    assert (second.error, second.kind) == (True, "call_limit") and "call it again next round" in second.text
    [third] = _round(registry, ctx, {"code": CODE})
    assert not third.error and len(workbench.calls) == 2


def test_refusals_leave_the_turns_two_intact() -> None:
    """A call refused for its arguments or the round's limit spends none of the turn's two runs."""
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench), _context()
    _three(ctx)
    registry = ToolRegistry([tool])

    [unheld] = _round(registry, ctx, {"code": CODE, "passages": [99]})
    [ran] = _round(registry, ctx, {"code": CODE})
    second_run, past_round = _round(registry, ctx, {"code": CODE}, {"code": CODE})
    [past_turn] = _round(registry, ctx, {"code": CODE})

    assert unheld.kind == "no_such_passage"
    assert not ran.error and not second_run.error
    assert past_round.kind == "call_limit" and "next round" in past_round.text
    assert past_turn.kind == "call_limit" and "no more runs in this turn" in past_turn.text
    assert len(workbench.calls) == 2


def test_a_run_never_sent_while_workbench_is_busy_is_no_run() -> None:
    """A call refused for want of a workbench slot sends nothing and leaves the turn its two runs."""
    workbench = FakeWorkbench()
    slots = threading.BoundedSemaphore(2)
    assert slots.acquire(blocking=False) and slots.acquire(blocking=False)
    tool, ctx = _tool(workbench, slots=slots), _context()
    _three(ctx)
    registry = ToolRegistry([tool])

    [busy] = _round(registry, ctx, {"code": CODE})
    slots.release()
    slots.release()
    [first] = _round(registry, ctx, {"code": CODE})
    [second] = _round(registry, ctx, {"code": CODE})
    [past_turn] = _round(registry, ctx, {"code": CODE})

    assert busy.kind == "no_capacity"
    assert not first.error and not second.error
    assert past_turn.kind == "call_limit" and "no more runs in this turn" in past_turn.text
    assert len(workbench.calls) == 2


@pytest.mark.parametrize("cause", ["no_time", "sealed"])
def test_a_run_never_sent_for_time_or_its_disclosure_is_no_run(cause: str) -> None:
    """A call that found too little time, or the turn's ledger sealed, sends nothing and spends no run."""
    workbench = FakeWorkbench()
    now = [TURN.deadline - 10.0 if cause == "no_time" else 880.0]
    tool = WorkbenchPythonTool(workbench, DISCOVERY, TURN, clock=lambda: now[0], slots=threading.BoundedSemaphore(2))
    ctx = _context()
    _three(ctx)
    registry = ToolRegistry([tool])
    sealed = DisclosureLedger()
    sealed.seal()

    [refused] = _round(registry, ctx, {"code": CODE}, ledger=sealed if cause == "sealed" else None)
    now[0] = 880.0
    [first] = _round(registry, ctx, {"code": CODE})
    [second] = _round(registry, ctx, {"code": CODE})
    [past_turn] = _round(registry, ctx, {"code": CODE})

    assert refused.kind == ("no_time" if cause == "no_time" else "unavailable")
    assert not first.error and not second.error
    assert past_turn.kind == "call_limit"
    assert len(workbench.calls) == 2


def test_a_run_that_reached_workbench_counts_even_when_it_failed() -> None:
    """A call workbench answered with a failure was sent, so it is one of the turn's two runs."""
    failure = RunFailure(kind="code_failed", token="not_completed", job_id=JOB, answered=True)
    workbench = FakeWorkbench(failure, run_result())
    tool, ctx = _tool(workbench), _context()
    _three(ctx)
    registry = ToolRegistry([tool])

    [failed] = _round(registry, ctx, {"code": CODE})
    [ran] = _round(registry, ctx, {"code": CODE})
    [past_turn] = _round(registry, ctx, {"code": CODE})

    assert failed.kind == "code_failed" and not ran.error
    assert past_turn.kind == "call_limit" and "no more runs in this turn" in past_turn.text
    assert len(workbench.calls) == 2


def test_a_repeated_call_runs_again() -> None:
    """The same arguments in two rounds reach workbench twice: a run is never answered from memory."""
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench), _context()
    _three(ctx)
    registry = ToolRegistry([tool])

    [first] = _round(registry, ctx, {"code": CODE, "title": TITLE})
    [again] = _round(registry, ctx, {"code": CODE, "title": TITLE})

    assert not first.error and not again.error
    assert len(workbench.calls) == 2
    assert [analysis["provenance"]["workbench"]["job_id"] for analysis in ctx.artifacts] == [JOB, JOB]


# --- passages.json -----------------------------------------------------------


def test_passages_json_holds_only_what_the_model_may_see() -> None:
    """Each entry reads the view's visible keys only; a hidden path never leaves docint."""
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench), _context()
    _held(ctx, _passage(1, filename="harbour-report.pdf", page=4, file_path="srv/hidden/report.pdf"))

    _run(tool, ctx, code=CODE)

    passages = json.loads(_sent(workbench)["passages.json"])
    assert passages == [
        {
            "n": 1,
            "text": TEXT,
            "file": "harbour-report.pdf",
            "page": 4,
            "time": None,
            "author": None,
            "network": None,
            "kind": None,
        }
    ]
    assert "srv/hidden" not in _sent(workbench)["passages.json"]


@pytest.mark.parametrize(
    ("metadata", "key", "value"),
    [
        ({"origin": {"filename": "minutes.docx"}}, "file", "minutes.docx"),
        ({"page_number": 7}, "page", 7),
        ({"origin": {"page_number": 9}}, "page", 9),
        ({"reference_metadata": {"timestamp": "2026-05-01T10:00:00Z"}}, "time", "2026-05-01T10:00:00Z"),
        ({"start_ts": "00:01:05"}, "time", "00:01:05"),
        ({"reference_metadata": {"author": "user-a"}}, "author", "user-a"),
        ({"speaker": "SPEAKER_01"}, "author", "SPEAKER_01"),
        ({"reference_metadata": {"network": "telegram"}}, "network", "telegram"),
        ({"reference_metadata": {"type": "posting"}}, "kind", "posting"),
        ({"docint_doc_kind": "transcript"}, "kind", "transcript"),
    ],
)
def test_each_field_falls_back_as_the_spec_lists(metadata: dict[str, Any], key: str, value: Any) -> None:
    """A field the first key leaves empty comes from the next one the spec lists."""
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench), _context()
    _held(ctx, _passage(1, **metadata))

    _run(tool, ctx, code=CODE)

    [entry] = json.loads(_sent(workbench)["passages.json"])
    assert entry[key] == value


def test_a_passages_text_is_cut_to_4000_characters() -> None:
    """A long passage goes with its first PASSAGE_TEXT_CHARS characters."""
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench), _context()
    _held(ctx, _passage(1, text="Hafen " * 1_000))

    _run(tool, ctx, code=CODE)

    [entry] = json.loads(_sent(workbench)["passages.json"])
    assert entry["text"] == ("Hafen " * 1_000)[:PASSAGE_TEXT_CHARS]


def test_left_out_passages_means_every_passage_shown() -> None:
    """Without passages, or with an empty list, every passage the turn holds goes."""
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench), _context()
    _three(ctx)
    registry = ToolRegistry([tool])

    _round(registry, ctx, {"code": CODE})
    left_out = [entry["n"] for entry in json.loads(_sent(workbench)["passages.json"])]
    _round(registry, ctx, {"code": CODE, "passages": []})
    empty = [entry["n"] for entry in json.loads(_sent(workbench)["passages.json"])]

    assert left_out == empty == [1, 2, 3]


def test_passage_numbers_read_as_the_model_writes_them() -> None:
    """Numbers as cited, as floats or as integers are read once each, in number order; anything else is refused."""
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench), _context()
    _three(ctx)
    registry = ToolRegistry([tool])

    _round(registry, ctx, {"code": CODE, "passages": ["[3]", 1.0, 3]})
    numbers = [entry["n"] for entry in json.loads(_sent(workbench)["passages.json"])]
    [boolean] = _round(registry, ctx, {"code": CODE, "passages": [True]})
    [word] = _round(registry, ctx, {"code": CODE, "passages": ["x"]})

    assert numbers == [1, 3]
    assert (boolean.kind, word.kind) == ("bad_argument", "bad_argument")
    assert len(workbench.calls) == 1


def test_a_number_the_evidence_does_not_hold_is_no_such_passage() -> None:
    """A number past the evidence is refused with the range it holds, and nothing is sent."""
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench), _context()
    _three(ctx)

    reply = _run(tool, ctx, code=CODE, passages=[9])

    assert reply.kind == "no_such_passage"
    assert "there is no passage [9]; the passages run from [1] to [3]." in reply.text
    assert workbench.calls == []


def test_code_must_be_given_and_fit() -> None:
    """No code is a missing argument and code past MAX_CODE_CHARS is too long; neither sends anything."""
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench), _context()
    _three(ctx)
    registry = ToolRegistry([tool])

    [missing] = _round(registry, ctx, {"title": TITLE})
    [blank] = _round(registry, ctx, {"code": "  \n"})
    [long] = _round(registry, ctx, {"code": "x" * (MAX_CODE_CHARS + 1)})

    assert (missing.kind, blank.kind, long.kind) == ("missing_argument", "missing_argument", "too_long")
    assert workbench.calls == []
    [fits] = _round(registry, ctx, {"code": "x" * MAX_CODE_CHARS})
    assert not fits.error and len(workbench.calls[0]["code"]) == MAX_CODE_CHARS


# --- keyword_checks.json and topics.json ------------------------------------


def test_keyword_checks_and_topics_come_from_the_turns_facts() -> None:
    """The turn's keyword checks and topic listings go beside the passages; other facts do not, nor empty files."""
    with_facts, without = FakeWorkbench(), FakeWorkbench()
    ctx = _context()
    _three(ctx)
    _run(_tool(without), ctx, code=CODE)
    ctx.facts.extend([KEYWORD_FACT, TOPICS_FACT, {"action": "search", "detail": "x"}])

    _run(_tool(with_facts), ctx, code=CODE)

    sent = _sent(with_facts)
    assert list(sent) == ["passages.json", "keyword_checks.json", "topics.json"]
    assert json.loads(sent["keyword_checks.json"]) == [KEYWORD_FACT]
    assert json.loads(sent["topics.json"]) == [TOPICS_FACT]
    assert list(_sent(without)) == ["passages.json"]


def test_keyword_checks_go_only_when_counted_and_once_per_check() -> None:
    """A failed count stays; checks the keyword tool tells apart stay apart, and a true repeat goes once."""
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench), _context()
    _three(ctx)
    failed = {"action": "keyword", "detail": "Feuerwerk", "field": "text", "matches": None}
    text_first = {"action": "keyword", "detail": "Regatta", "field": "text", "matches": 0}
    other_case = {"action": "keyword", "detail": "hafenfest", "field": "author", "matches": 5}
    text_again = {"action": "keyword", "detail": "regatta", "field": "text", "matches": 4}
    ctx.facts.extend([KEYWORD_FACT, failed, text_first, other_case, text_again])
    ledger = DisclosureLedger()

    _run(tool, ctx, ledger=ledger, code=CODE)

    assert check_key("Hafenfest", "author") != check_key("hafenfest", "author")
    assert check_key("Regatta", "text") == check_key("regatta", "text")
    assert json.loads(_sent(workbench)["keyword_checks.json"]) == [KEYWORD_FACT, other_case, text_again]
    [record] = ledger.take()
    assert record.keyword_checks == 3


def test_each_file_fills_in_order_while_the_next_entry_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entries go in order until the next would pass its file's cap; the reply names what stayed."""
    monkeypatch.setattr(wt, "PASSAGES_CAP", _size([_entry(1), _entry(2)]))
    second = {"action": "keyword", "detail": "Regatta", "field": "text", "matches": 0}
    monkeypatch.setattr(wt, "KEYWORD_CHECKS_CAP", _size([KEYWORD_FACT]))
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench), _context()
    _three(ctx)
    ctx.facts.extend([KEYWORD_FACT, second])
    ledger = DisclosureLedger()

    reply = _run(tool, ctx, ledger=ledger, code=CODE)

    assert [entry["n"] for entry in json.loads(_sent(workbench)["passages.json"])] == [1, 2]
    assert json.loads(_sent(workbench)["keyword_checks.json"]) == [KEYWORD_FACT]
    assert "passages [3]" in reply.text and "1 keyword check" in reply.text
    [record] = ledger.take()
    assert (record.point_ids, record.passages, record.keyword_checks) == (("p1", "p2"), 2, 1)
    assert [source["chunk_id"] for source in ctx.artifacts[0]["provenance"]["sources"]] == ["p1", "p2"]


def test_caps_count_utf8_bytes_not_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    """German text that fits the cap in characters but not in bytes is left out."""
    text = "Größe Übersicht " * 20
    as_written = json.dumps([_entry(1, text)], ensure_ascii=False)
    assert len(as_written) < len(as_written.encode("utf-8"))
    monkeypatch.setattr(wt, "PASSAGES_CAP", len(as_written))
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench), _context()
    _held(ctx, _passage(1, text=text))

    reply = _run(tool, ctx, code=CODE)

    assert json.loads(_sent(workbench)["passages.json"]) == []
    assert "passages [1]" in reply.text


def test_the_provenance_stays_in_docint() -> None:
    """Point ids and file hashes are kept with the analysis and never handed over."""
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench), _context()
    _held(ctx, _passage(1, filename="harbour-report.pdf", page=4, file_hash=HASH))

    _run(tool, ctx, code=CODE)

    assert ctx.artifacts[0]["provenance"]["sources"] == [
        {"citation_index": 1, "chunk_id": "p1", "file_hash": HASH, "filename": "harbour-report.pdf", "page": 4}
    ]
    sent = json.dumps([workbench.calls[0]["code"], *_sent(workbench).values()], ensure_ascii=False)
    assert "p1" not in sent and HASH not in sent


# --- time and capacity -------------------------------------------------------


def test_a_run_needs_twenty_seconds_and_sends_nothing_without_them() -> None:
    """With less than MIN_CALL_S left after the turn's reserve, nothing is sent or disclosed."""
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench, now=TURN.deadline - 34.9), _context()
    _three(ctx)
    ledger = DisclosureLedger()

    reply = _run(tool, ctx, ledger=ledger, code=CODE)

    assert reply.kind == "no_time"
    assert workbench.calls == []
    assert ledger.take() == []


@pytest.mark.parametrize(
    ("left", "time_limit", "timeout"), [(200.0, 60, 75.0), (90.0, 60, 75.0), (50.0, 23, 35.0), (35.0, 8, 20.0)]
)
def test_the_time_limit_leaves_room_for_start_up_and_the_turn(left: float, time_limit: int, timeout: float) -> None:
    """The call's deadline keeps the turn's reserve, and the run's limit keeps the start-up margin under the cap."""
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench, now=TURN.deadline - left), _context()
    _three(ctx)

    _run(tool, ctx, code=CODE)

    [call] = workbench.calls
    assert call["time_limit_s"] == time_limit
    assert call["timeout_s"] == pytest.approx(timeout)
    assert (call["owner"], call["session_id"]) == ("user-a", "session-1")


def test_the_tool_takes_its_timeout_from_its_caller_and_reads_no_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The call's deadline follows the timeout the tool was built with; a malformed environment is not its concern."""
    monkeypatch.setenv("WORKBENCH_TIMEOUT", "not-a-number")
    workbench = FakeWorkbench()
    tool = WorkbenchPythonTool(
        workbench,
        DISCOVERY,
        TURN,
        timeout_s=30.0,
        clock=lambda: TURN.deadline - 200.0,
        slots=threading.BoundedSemaphore(2),
    )
    ctx = _context()
    _three(ctx)

    _run(tool, ctx, code=CODE)

    [call] = workbench.calls
    assert (call["timeout_s"], call["time_limit_s"]) == (30.0, 18)


def test_a_refusal_has_a_text_for_every_kind_a_call_can_fail_with() -> None:
    """Every failure kind of the client, and no_time, has its refusal; a new kind must be added here and there."""
    assert set(wt.REFUSALS) == {*get_args(FailureKind), "no_time"}


def test_two_calls_at_once_are_the_most_a_process_sends() -> None:
    """With both of the process's slots held, a call is refused at once and discloses nothing."""
    workbench = FakeWorkbench()
    slots = threading.BoundedSemaphore(2)
    assert slots.acquire(blocking=False) and slots.acquire(blocking=False)
    tool, ctx = _tool(workbench, slots=slots), _context()
    _three(ctx)
    ledger = DisclosureLedger()

    reply = _run(tool, ctx, ledger=ledger, code=CODE)

    assert reply.kind == "no_capacity"
    assert workbench.calls == []
    assert ledger.take() == []


def test_the_slot_is_given_back_whatever_happened() -> None:
    """A run, a refused run and a refused disclosure each give their slot back."""
    slots = threading.BoundedSemaphore(1)
    workbench = FakeWorkbench(run_result(), RunFailure(kind="unavailable", token="timeout"))
    tool, ctx = _tool(workbench, slots=slots), _context()
    _three(ctx)
    registry = ToolRegistry([tool])
    sealed = DisclosureLedger()
    sealed.seal()

    [refused] = _round(registry, ctx, {"code": CODE}, ledger=sealed)
    [ran] = _round(registry, ctx, {"code": CODE})
    later = WorkbenchPythonTool(workbench, DISCOVERY, TURN, clock=lambda: 880.0, slots=slots)
    [failed] = _round(ToolRegistry([later]), ctx, {"code": CODE})

    assert (refused.kind, ran.error, failed.kind) == ("unavailable", False, "unavailable")
    assert slots.acquire(blocking=False)


# --- disclosure --------------------------------------------------------------


def test_the_disclosure_is_recorded_before_the_send_and_named_after() -> None:
    """The disclosure is in flight when the call goes out, and names workbench's job once the answer came."""
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench), _context()
    _held(ctx, _passage(1), _passage(2))
    ledger = DisclosureLedger()

    _run(tool, ctx, ledger=ledger, code=CODE)

    data_bytes = sum(len(content.encode("utf-8")) for content in _sent(workbench).values())
    assert workbench.ledger_states == [1]
    [record] = ledger.take()
    assert record == DisclosureRecord(
        point_ids=("p1", "p2"),
        passages=2,
        keyword_checks=0,
        topic_listings=0,
        data_bytes=data_bytes,
        job_id=JOB,
        outcome="ok",
        incomplete=False,
    )


def test_a_sealed_ledger_sends_nothing() -> None:
    """Once the turn's disclosures were written, a late call is refused and nothing leaves."""
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench), _context()
    _three(ctx)
    ledger = DisclosureLedger()
    ledger.seal()

    reply = _run(tool, ctx, ledger=ledger, code=CODE)

    assert reply.kind == "unavailable"
    assert workbench.calls == []


@pytest.mark.parametrize(
    ("kind", "token", "job_id", "answered"),
    [
        ("no_capacity", "queue_full", None, True),
        ("too_much_data", "too_large", None, True),
        ("code_failed", "not_completed", JOB, True),
        ("unavailable", "timeout", None, False),
        ("unavailable", "bad_result", None, True),
    ],
    ids=["no_capacity", "too_much_data", "code_failed", "unavailable", "unavailable_answered"],
)
def test_a_refused_run_is_an_error_reply_of_its_kind(
    kind: FailureKind, token: str, job_id: str | None, answered: bool
) -> None:
    """A failed run is an error of its kind, keeps no analysis, and is disclosed as answered or not."""
    workbench = FakeWorkbench(RunFailure(kind=kind, token=token, job_id=job_id, answered=answered))
    tool, ctx = _tool(workbench), _context()
    _three(ctx)
    ledger = DisclosureLedger()

    reply = _run(tool, ctx, ledger=ledger, code=CODE)

    assert (reply.error, reply.kind) == (True, kind)
    assert ctx.artifacts == [] and ctx.facts == []
    [record] = ledger.take()
    assert (record.outcome, record.job_id, record.incomplete) == ("ok" if answered else "error", job_id, False)


# --- the reply -----------------------------------------------------------------


def test_the_reply_fits_and_names_the_analysis() -> None:
    """The reply shows the table's shape and the chart within one reply, and ends naming the analysis."""
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench), _context()
    _held(ctx, _passage(1), _passage(2))

    reply = _run(tool, ctx, code=CODE, title=TITLE)

    assert len(reply.text) <= ctx.budget.result_chars
    assert reply.text.endswith(ANALYSIS_NOTE.format(k=1))
    assert "counts.csv: 2 rows × 2 columns" in reply.text and "chart.png" in reply.text  # noqa: RUF001
    assert "user-a,12" in reply.text and "2 authors" in reply.text
    assert (reply.found, reply.informative, reply.detail) == (0, True, TITLE)
    assert reply.trail == {"analysis": 1, "passages": 2, "tables": 1, "charts": 1}
    assert ctx.facts == [{"action": NAME, "detail": TITLE, "analysis": 1, "passages": 2, "tables": 1, "charts": 1}]


@pytest.mark.parametrize(
    ("held", "picked", "named"),
    [
        (64, None, r"64 passages \(\[1\] to \[64\]\)"),
        (800, list(range(1, 801, 2)), r"400 passages \(\[1\], \[3\], .+\[\d+\] and \d+ more\)"),
    ],
    ids=["all_64", "every_other_of_800"],
)
def test_the_note_ends_the_reply_however_many_passages_went(held: int, picked: list[int] | None, named: str) -> None:
    """A thinking model's shorter reply still ends naming the analysis, the passages named as ranges."""
    workbench = FakeWorkbench(run_result(stdout="z" * 10_000, stderr="w" * 2_000))
    tool = _tool(workbench)
    ctx = tool_context(reasoning=True, cap=held, view=agent_llm_view)
    _held(ctx, *(_passage(number, text="Hafen") for number in range(1, held + 1)))

    reply = _run(tool, ctx, code=CODE) if picked is None else _run(tool, ctx, code=CODE, passages=picked)

    assert ctx.budget.result_chars == 2_000
    assert len(reply.text) <= 2_000
    assert reply.text.endswith(ANALYSIS_NOTE.format(k=1))
    assert re.search(named, reply.text.splitlines()[0])


def test_an_analysis_is_kept_only_once_its_reply_is_written(monkeypatch: pytest.MonkeyPatch) -> None:
    """When writing the reply fails, the round reports the call failed and no analysis is left behind."""

    def _fails(*args: Any) -> str:
        """Fail as a broken reply would.

        Args:
            *args (Any): Unused.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("reply")

    monkeypatch.setattr(WorkbenchPythonTool, "_reply_text", staticmethod(_fails))
    tool, ctx = _tool(FakeWorkbench()), _context()
    _three(ctx)

    with pytest.raises(RuntimeError):
        _run(tool, ctx, code=CODE)

    assert ctx.artifacts == []


def test_the_second_analysis_of_a_turn_is_analysis_2() -> None:
    """Analyses are numbered in the order the turn made them."""
    tool, ctx = _tool(FakeWorkbench()), _context()
    _three(ctx)
    registry = ToolRegistry([tool])

    _round(registry, ctx, {"code": CODE})
    [second] = _round(registry, ctx, {"code": CODE})

    assert second.text.endswith(ANALYSIS_NOTE.format(k=2)) and second.trail is not None
    assert second.trail["analysis"] == 2 and len(ctx.artifacts) == 2


def test_a_long_stdout_is_cut_in_the_middle() -> None:
    """Output past what the reply holds keeps its start and its end around a marker."""
    workbench = FakeWorkbench(run_result(stdout="HEAD" + "x" * 50_000 + "TAIL"))
    tool, ctx = _tool(workbench), _context()
    _three(ctx)

    reply = _run(tool, ctx, code=CODE)

    assert len(reply.text) <= ctx.budget.result_chars
    assert "HEAD" in reply.text and "TAIL" in reply.text
    assert re.search(r"\[\.\.\. \d+ of 50008 characters not shown \.\.\.\]", reply.text)
    assert reply.text.endswith(ANALYSIS_NOTE.format(k=1))


def test_a_title_over_eighty_characters_is_cut_to_eighty() -> None:
    """The model's long title is normalised and cut to 80 characters, in the reply's detail and in the analysis."""
    title = "Posts per author  in   May and June\n" * 4
    expected = " ".join(title.split())[:80]
    tool, ctx = _tool(FakeWorkbench()), _context()
    _three(ctx)

    reply = _run(tool, ctx, code=CODE, title=title)

    assert len(expected) == 80
    assert (reply.detail, ctx.artifacts[0]["title"]) == (expected, expected)


@pytest.mark.parametrize(("stdout", "stderr"), [("2 authors", ""), ("", "warning"), ("2 authors", "warning")])
def test_a_cut_output_is_named_without_saying_which_stream(stdout: str, stderr: str) -> None:
    """Workbench says only that it cut an output, so the reply says that and never that stdout was cut."""
    workbench = FakeWorkbench(run_result(stdout=stdout, stderr=stderr, truncated=True))
    tool, ctx = _tool(workbench), _context()
    _three(ctx)

    reply = _run(tool, ctx, code=CODE)

    assert "Output cut by workbench." in reply.text
    assert "stdout (cut by workbench)" not in reply.text
    assert reply.text.endswith(ANALYSIS_NOTE.format(k=1))


def test_an_output_workbench_did_not_cut_is_not_said_to_be() -> None:
    """A whole output carries no mention of a cut."""
    tool, ctx = _tool(FakeWorkbench(run_result(truncated=False))), _context()
    _three(ctx)

    reply = _run(tool, ctx, code=CODE)

    assert "cut by workbench" not in reply.text.lower()


def test_ranges_name_runs_of_three_or_more_and_count_what_they_leave_out() -> None:
    """A run of three or more is a range, shorter runs are named one by one, and a limit counts the rest."""
    assert wt._ranges([*range(1, 11), 12, 14, 15]) == "[1] to [10], [12], [14], [15]"
    assert wt._ranges([1, 2, 3, 5, 6, 7, 8]) == "[1] to [3], [5] to [8]"
    assert wt._ranges([4]) == "[4]"
    cut = wt._ranges(range(1, 400, 2), most=30)
    assert cut == "[1], [3], [5], [7], [9], [11] and 194 more"


def test_the_end_of_stderr_is_shown() -> None:
    """A run that wrote to stderr shows the end of it, at most 500 characters."""
    workbench = FakeWorkbench(run_result(stderr="warn " * 300 + "LAST"))
    tool, ctx = _tool(workbench), _context()
    _three(ctx)

    reply = _run(tool, ctx, code=CODE)

    assert "LAST" in reply.text and "warn " * 120 not in reply.text


def test_only_a_run_with_a_table_or_a_chart_becomes_an_analysis() -> None:
    """A run that wrote only notes is named in the reply and kept as nothing."""
    notes = "n" * 120
    workbench = FakeWorkbench(
        run_result(
            files=(OutputFile(name="notes.txt", mime="text/plain", size=120, returned=True),),
            images=(),
            texts=(("notes.txt", "text/plain", notes),),
        )
    )
    tool, ctx = _tool(workbench), _context()
    _three(ctx)

    reply = _run(tool, ctx, code=CODE)

    assert ctx.artifacts == []
    assert "notes.txt (120 bytes)" in reply.text
    assert ANALYSIS_NOTE.format(k=1) not in reply.text
    assert reply.trail is not None and reply.trail["analysis"] is None


def test_an_empty_csv_is_no_table_and_the_reply_names_it() -> None:
    """An empty CSV file is named in the reply as no table, beside the table that was kept."""
    workbench = FakeWorkbench(
        run_result(images=(), texts=(("empty.csv", "text/csv", ""), ("counts.csv", "text/csv", CSV)))
    )
    tool, ctx = _tool(workbench), _context()
    _three(ctx)

    reply = _run(tool, ctx, code=CODE)

    assert [table["name"] for table in ctx.artifacts[0]["tables"]] == ["counts.csv"]
    assert "empty.csv (0 bytes, no table)" in reply.text


# --- tables and charts -------------------------------------------------------


def test_tables_read_what_pandas_writes() -> None:
    """The BOM goes, the index column's empty name stays, short rows are padded and long rows cut."""
    assert parse_table("x.csv", "\ufeff,author,posts\n0,user-a,12\n1,user-b\n") == {
        "name": "x.csv",
        "columns": ["", "author", "posts"],
        "rows": [["0", "user-a", "12"], ["1", "user-b", ""]],
        "row_count": 2,
        "truncated": False,
    }
    assert parse_table("x.csv", "author,posts\nuser-a,12,extra\n") == {
        "name": "x.csv",
        "columns": ["author", "posts"],
        "rows": [["user-a", "12"]],
        "row_count": 1,
        "truncated": True,
    }
    header_only = parse_table("x.csv", "author,posts\n")
    assert header_only is not None and (header_only["rows"], header_only["row_count"]) == ([], 0)
    assert parse_table("x.csv", "") is None
    assert parse_table("x.csv", "\ufeff") is None


def test_tables_and_charts_are_bounded() -> None:
    """Tables, rows, columns, cells and charts keep within their caps, the first ones first."""
    long_table = parse_table("rows.csv", "n\n" + "".join(f"{i}\n" for i in range(600)))
    assert long_table is not None
    assert (len(long_table["rows"]), long_table["row_count"], long_table["truncated"]) == (MAX_ROWS, 600, True)
    wide = parse_table("wide.csv", ",".join(f"c{i}" for i in range(40)) + "\n" + ",".join(["1"] * 40) + "\n")
    assert wide is not None and len(wide["columns"]) == MAX_COLUMNS and len(wide["rows"][0]) == MAX_COLUMNS
    assert wide["truncated"]
    cell = parse_table("cell.csv", "text\n" + "y" * 600 + "\n")
    assert cell is not None and cell["rows"] == [["y" * CELL_CHARS]] and cell["truncated"]

    tables = tuple((f"t{i}.csv", "text/csv", CSV) for i in range(5))
    charts = tuple((f"c{i}.png", tiny_png()) for i in range(5))
    workbench = FakeWorkbench(run_result(texts=tables, images=charts))
    tool, ctx = _tool(workbench), _context()
    _three(ctx)
    reply = _run(tool, ctx, code=CODE)
    [analysis] = ctx.artifacts
    assert [table["name"] for table in analysis["tables"]] == [f"t{i}.csv" for i in range(MAX_TABLES)]
    assert [chart["name"] for chart in analysis["charts"]] == [f"c{i}.png" for i in range(MAX_CHARTS)]
    assert "t4.csv" in reply.text and "c4.png" in reply.text

    heavy = tiny_png() + b"\x00" * (400 * 1024)
    workbench = FakeWorkbench(run_result(images=tuple((f"h{i}.png", heavy) for i in range(3))))
    tool, ctx = _tool(workbench), _context()
    _three(ctx)
    _run(tool, ctx, code=CODE)
    assert [chart["name"] for chart in ctx.artifacts[0]["charts"]] == ["h0.png", "h1.png"]


def test_a_chart_too_large_in_pixels_is_dropped() -> None:
    """A PNG wider or taller than MAX_CHART_SIDE is not stored, and the reply says why."""
    workbench = FakeWorkbench(run_result(images=(("huge.png", tiny_png(20_000, 20_000)),)))
    tool, ctx = _tool(workbench), _context()
    _three(ctx)

    reply = _run(tool, ctx, code=CODE)

    assert ctx.artifacts[0]["charts"] == []
    assert "huge.png" in reply.text and "too large in pixels" in reply.text


def _png_with(**changes: bytes) -> bytes:
    """Build the start of a PNG with one of its parts replaced.

    Args:
        **changes (bytes): ``length``, ``tag`` or ``header`` bytes to put in place of the valid ones.

    Returns:
        bytes: The image.
    """
    png = tiny_png()
    parts = {"length": png[8:12], "tag": png[12:16], "header": png[16:]}
    parts.update(changes)
    return png[:8] + parts["length"] + parts["tag"] + parts["header"]


@pytest.mark.parametrize(
    "png",
    [
        tiny_png()[:8],
        tiny_png()[:24],
        tiny_png()[:32],
        _png_with(tag=b"IDAT"),
        _png_with(length=b"\x00\x00\x00\x0c"),
        tiny_png(0, 480),
        tiny_png(640, 0),
    ],
    ids=[
        "signature_only",
        "cut_after_size",
        "cut_before_checksum",
        "not_ihdr",
        "length_not_13",
        "no_width",
        "no_height",
    ],
)
def test_a_chart_whose_header_is_not_a_readable_png_is_dropped(png: bytes) -> None:
    """A chart whose signature, first chunk or size does not read as a PNG header is dropped, and the reply says so."""
    workbench = FakeWorkbench(run_result(texts=(), images=(("bad.png", png),)))
    tool, ctx = _tool(workbench), _context()
    _three(ctx)

    reply = _run(tool, ctx, code=CODE)

    assert ctx.artifacts == []
    assert "bad.png (not a readable PNG)" in reply.text


def test_a_png_header_with_its_checksum_is_a_chart() -> None:
    """The smallest image that reads, 33 bytes with the header's checksum, is kept."""
    assert len(tiny_png()) == 33
    workbench = FakeWorkbench(run_result(texts=(), images=(("ok.png", tiny_png()),)))
    tool, ctx = _tool(workbench), _context()
    _three(ctx)

    _run(tool, ctx, code=CODE)

    assert [chart["name"] for chart in ctx.artifacts[0]["charts"]] == ["ok.png"]


def test_the_analysis_carries_its_code_output_and_provenance() -> None:
    """The analysis keeps its title, code, output, tables, charts, inputs and workbench's job and version."""
    workbench = FakeWorkbench()
    tool, ctx = _tool(workbench), _context()
    _held(ctx, _passage(1), _passage(2))
    ctx.facts.extend([KEYWORD_FACT, TOPICS_FACT])
    before = datetime.now().astimezone()

    _run(tool, ctx, code=CODE, title=TITLE)

    [analysis] = ctx.artifacts
    assert (analysis["schema"], analysis["title"], analysis["stdout"]) == (1, TITLE, "2 authors")
    assert analysis["code"] == CODE
    assert analysis["tables"] == [
        {
            "name": "counts.csv",
            "columns": ["author", "posts"],
            "rows": [["user-a", "12"], ["user-b", "7"]],
            "row_count": 2,
            "truncated": False,
        }
    ]
    [chart] = analysis["charts"]
    assert chart["name"] == "chart.png" and chart["bytes"] == len(tiny_png())
    assert chart["data_uri"].startswith("data:image/png;base64,")
    provenance = analysis["provenance"]
    assert provenance["inputs"] == {"passages": 2, "keyword_checks": 1, "topic_listings": 1}
    assert provenance["workbench"] == {"job_id": JOB, "version": "0.2.0", "tool": "run_python"}
    created = datetime.fromisoformat(provenance["created_at"])
    assert created.utcoffset() == timedelta(0)
    assert before - timedelta(seconds=1) <= created <= datetime.now().astimezone() + timedelta(seconds=1)


def test_the_description_names_no_other_tool_and_stays_short() -> None:
    """The model reads no other tool's name in the description, which stays within 60 words."""
    tool = _tool(FakeWorkbench())

    assert not re.search(r"\b(search|read|keyword|topics|search_images|keyframes)\b", tool.description)
    assert len(tool.description.split()) <= 60
    assert tool.name == NAME == "run_python"
    assert tool.parameters["required"] == ["code"]
    assert tool.parameters["properties"]["title"]["maxLength"] == 80
    passages = tool.parameters["properties"]["passages"]
    assert (passages["type"], passages["items"]) == ("array", {"type": "integer"})


def test_the_log_names_no_code_data_or_output(loguru_caplog_info: LogCaptureFixture) -> None:
    """Whatever happens to a run, the log holds no code, passage text or output."""
    secret_code = "print('CODE-MARK')"
    workbench = FakeWorkbench(run_result(stdout="OUT-MARK", stderr="ERR-MARK"))
    ctx = _context()
    _held(ctx, _passage(1, text="TEXT-MARK"))
    sealed = DisclosureLedger()
    sealed.seal()
    runs: list[Callable[[], ToolReply]] = [
        lambda: _run(_tool(workbench), ctx, code=secret_code),
        lambda: _run(_tool(FakeWorkbench(), now=990.0), ctx, code=secret_code),
        lambda: _run(_tool(FakeWorkbench()), ctx, ledger=sealed, code=secret_code),
    ]

    for run in runs:
        run()

    assert "run_python" in loguru_caplog_info.text
    for mark in ("CODE-MARK", "TEXT-MARK", "OUT-MARK", "ERR-MARK"):
        assert mark not in loguru_caplog_info.text
