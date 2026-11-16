"""The ``run_python`` tool of an agent turn: the turn's evidence handed to workbench, its tables and charts kept.

A call hands the passages the turn has shown, and the keyword checks and topic listings it
made, inline to workbench's sandboxed ``run_python``, with the model's code. The files are built
on the loop thread when the call is prepared, from the evidence as it stood then:
``passages.json`` reads only what the model may see of each passage (``ctx.llm_view``), and
``keyword_checks.json`` and ``topics.json`` the turn's facts. Each file fills in order while its
next entry fits its cap, counted in UTF-8 bytes. Which point, file and page each passage came
from stays in docint, as the provenance of the analysis.

The call runs off the loop thread. It takes one of the process's workbench slots without
waiting, records its disclosure before anything is sent (none is sent that cannot be
recorded), and is bounded by the turn's deadline less a reserve for the rest of the turn. A run
that returns a table or a chart becomes an *analysis*, kept in ``ctx.artifacts``, which the user
sees below the answer as Analysis k.

A run is limited to one per round and two per turn. A call refused before it was sent spends
neither: one refused at ``prepare`` is never counted, and one whose fetch sent nothing (no time,
no slot, no ledger to record it in) gets its run back when it is answered. A call the round
dropped after ``prepare`` stays counted, since its fetch may have sent. Its reply is
a plain :class:`~docint.agents.turn.tools.ToolReply`, which the registry never answers from
memory, so the same call made again runs again.

Nothing here logs or raises with code, data, ``stdout``, ``stderr`` or an output in its text:
only kinds and counts.
"""

from __future__ import annotations

import base64
import copy
import csv
import functools
import io
import json
import math
import struct
import threading
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from llama_index.core.schema import BaseNode, MetadataMode
from loguru import logger

from docint.agents.assessment.split import CITATION_NUMBERS_RE, neutralize_sentinel
from docint.agents.turn.keyword import DEFAULT_KEYWORD_FIELD, check_key
from docint.agents.turn.tools import (
    ToolContext,
    ToolReply,
    ToolWork,
    error_reply,
    passage_number,
    passage_range,
)
from docint.core.search.fields import UnknownSearchFieldError
from docint.core.summary.units import source_file_hash, source_file_name
from docint.utils.audit_disclose import DisclosureRefused, record_disclosure
from docint.utils.text_budget import cut_marker, fit_tail_with_marker
from docint.utils.workbench_client import (
    PNG_SIGNATURE,
    Discovery,
    FailureKind,
    InlineFile,
    RunFailure,
    RunResult,
)

#: The name the model calls the tool by, and workbench's tool it reaches.
NAME = "run_python"
#: Longest code a call may send, as workbench accepts it.
MAX_CODE_CHARS = 50_000
#: Longest title an analysis keeps.
TITLE_CHARS = 80
#: Longest text of one passage in ``passages.json``.
PASSAGE_TEXT_CHARS = 4_000

#: Caps on each file handed over, in UTF-8 bytes of the file as written.
PASSAGES_CAP = 384 * 1024
KEYWORD_CHECKS_CAP = 32 * 1024
TOPICS_CAP = 64 * 1024

#: Runs one turn may make; one per round.
CALLS_PER_TURN = 2
#: Calls one process may have waiting on workbench at once.
WORKBENCH_SLOTS = 2
WORKBENCH_SLOT_POOL = threading.BoundedSemaphore(WORKBENCH_SLOTS)

#: Seconds a call may take when its caller names no other, as ``WORKBENCH_TIMEOUT`` defaults.
DEFAULT_TIMEOUT_S = 75.0
#: Seconds of the turn kept for what follows a run.
TURN_RESERVE_S = 15.0
#: Fewest seconds a call needs; with less left nothing is sent.
MIN_CALL_S = 20.0
#: Seconds of a call kept for workbench's start-up and the answer, outside the run's own limit.
START_MARGIN_S = 12.0

#: What one analysis keeps.
MAX_TABLES = 4
MAX_CHARTS = 4
MAX_ROWS = 500
MAX_COLUMNS = 32
CELL_CHARS = 500
MAX_NAME_CHARS = 200
CHART_BYTES = 512 * 1024
CHARTS_BYTES = 1024 * 1024
MAX_CHART_SIDE = 4096

#: Characters the analyses may take together in the validator's prompt.
ANALYSIS_CHECKS_CHARS = 1_600
#: Characters of one analysis's ``stdout`` the validator reads.
STDOUT_CHECK_CHARS = 600

#: How the reply ends when the run became an analysis; ``k`` is its number in the turn.
ANALYSIS_NOTE = (
    "The user sees these tables and charts below your answer as Analysis {k}; refer to it by that name and cite "
    "the passages [n] the code used."
)

DESCRIPTION = (
    "Run Python (pandas, numpy, matplotlib) on the passages shown so far, to count, group, compare or chart them. "
    "passages picks them by number; left out, every passage shown is used. Tables written as CSV and charts as PNG "
    "into out/ are shown to the user."
)
PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "The Python to run."},
        "passages": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "The numbers [n] of the passages to use.",
        },
        "title": {"type": "string", "maxLength": TITLE_CHARS, "description": "A short title for the analysis."},
    },
    "required": ["code"],
}

# The replies to calls that did not run or did not finish; model-facing protocol, each prefixed
# ``Error: `` by ``error_reply``.
ROUND_LIMIT_REPLY = "run_python runs once per round; call it again next round."
TURN_LIMIT_REPLY = (
    "run_python has run twice in this turn, the most it may; no more runs in this turn, so answer from what it showed."
)
MISSING_CODE_REPLY = "run_python needs code: the Python to run on the passages."
BAD_CODE_REPLY = "run_python needs its code as text."
LONG_CODE_REPLY = f"the code is longer than {MAX_CODE_CHARS} characters; make it shorter."
BAD_PASSAGES_REPLY = (
    "passages must be a list of the numbers of passages shown before, such as [1, 3]; "
    "leave it out to use every passage shown."
)
#: Why a call sent nothing, or what a failed one means: the client's kinds, and ``no_time``.
RefusalKind = FailureKind | Literal["no_time"]
REFUSALS: Mapping[RefusalKind, str] = {
    "no_time": "not run: too little of this turn's time is left for a run; answer from the passages shown.",
    "no_capacity": "not run: workbench is busy; call it again in your next round, or answer from the passages shown.",
    "unavailable": "workbench did not answer; answer from the passages shown.",
    "too_much_data": "the data was too large for one run; pass fewer passages.",
    "code_failed": (
        "the code did not finish within its time and memory limits; make it simpler, or answer from the passages shown."
    ),
}

_BOM = "\ufeff"
#: A PNG's signature, the IHDR chunk's length, tag and 13 bytes of data, and its checksum.
_PNG_HEADER_BYTES = 33
_IHDR_LENGTH = b"\x00\x00\x00\x0d"
_STDERR_CHARS = 500
_ROWS_SHOWN = 5
_ROW_LINE_CHARS = 200
_HINT_CHARS = 300
# Longest list of passage numbers the head line names; past it, the rest are counted.
_HEAD_NUMBERS_CHARS = 200
_NAME_CHARS = 100
# Least room a stdout or stderr block is worth showing in.
_MIN_BLOCK_CHARS = 80


@dataclass(frozen=True)
class PythonTurn:
    """The stored turn a ``run_python`` tool works for.

    Attributes:
        owner (str): ``principal.effective_owner``, whose pseudonym workbench sees.
        session_id (str): The chat session the turn is stored in, sent as a pseudonym.
        deadline (float): When the turn's time is up, on ``time.monotonic()``'s scale.
    """

    owner: str
    session_id: str
    deadline: float


class WorkbenchCaller(Protocol):
    """What the tool needs of the workbench client: its ``run_python``."""

    def run_python(
        self,
        *,
        code: str,
        data: Sequence[InlineFile],
        time_limit_s: int,
        owner: str,
        session_id: str,
        timeout_s: float,
    ) -> RunResult | RunFailure:
        """Run ``code`` at workbench on ``data``, within one deadline, never raising.

        Args:
            code (str): The model's Python.
            data (Sequence[InlineFile]): The files written beside it.
            time_limit_s (int): The run's limit at workbench, within the cap.
            owner (str): The owner the call is made for.
            session_id (str): The chat session.
            timeout_s (float): The deadline for the whole call.

        Returns:
            RunResult | RunFailure: What the run returned, or why there is nothing.
        """
        ...


@dataclass(frozen=True)
class _Handover:
    """What one prepared call hands over, and what docint keeps of it.

    Attributes:
        code (str): The model's code.
        title (str): The analysis's title; empty for none.
        files (tuple[InlineFile, ...]): The files sent, ``passages.json`` first.
        numbers (tuple[int, ...]): The passages sent, by number.
        left_out (tuple[int, ...]): The passages asked for that did not fit.
        sources (tuple[Mapping[str, Any], ...]): Per passage sent, where it came from.
        keyword_checks (int): Keyword checks sent.
        keyword_left_out (int): Keyword checks that did not fit.
        topic_listings (int): Topic listings sent.
        topics_left_out (int): Topic listings that did not fit.
    """

    code: str
    title: str
    files: tuple[InlineFile, ...]
    numbers: tuple[int, ...]
    left_out: tuple[int, ...]
    sources: tuple[Mapping[str, Any], ...]
    keyword_checks: int
    keyword_left_out: int
    topic_listings: int
    topics_left_out: int

    @property
    def point_ids(self) -> tuple[str, ...]:
        """The points whose passages are sent, for the disclosure.

        Returns:
            tuple[str, ...]: Their ids, in number order.
        """
        return tuple(str(source["chunk_id"]) for source in self.sources)

    @property
    def data_bytes(self) -> int:
        """The bytes of data sent, for the disclosure.

        Returns:
            int: The UTF-8 bytes of every file.
        """
        return sum(len(item.content.encode("utf-8")) for item in self.files)


@dataclass(frozen=True)
class _NotSent:
    """A call whose fetch sent nothing, and why.

    Attributes:
        kind (RefusalKind): ``no_time``, ``no_capacity`` or ``unavailable``.
    """

    kind: RefusalKind


def _scalar(value: Any) -> str | int | float | None:
    """Return a metadata value that a file can carry as it is.

    Args:
        value (Any): The value.

    Returns:
        str | int | float | None: Non-blank text or a finite number; ``None`` for anything else.
    """
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value if math.isfinite(value) else None


def _first(*values: Any) -> str | int | float | None:
    """Return the first value a file can carry.

    Args:
        *values (Any): The candidates, in order.

    Returns:
        str | int | float | None: The first that :func:`_scalar` keeps; ``None`` when none is.
    """
    for value in values:
        kept = _scalar(value)
        if kept is not None:
            return kept
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return a metadata value as a mapping.

    Args:
        value (Any): The value.

    Returns:
        Mapping[str, Any]: It, when it is a dict; empty otherwise.
    """
    return value if isinstance(value, dict) else {}


def _passage_entry(number: int, node: BaseNode, view: Callable[[BaseNode], BaseNode]) -> dict[str, Any]:
    """Describe one passage for ``passages.json`` from what the model may see of it.

    Args:
        number (int): Its number in the turn's evidence.
        node (BaseNode): The passage.
        view (Callable[[BaseNode], BaseNode]): The turn's LLM view.

    Returns:
        dict[str, Any]: ``n``, ``text`` and the six fields, each read from a visible key only.
    """
    shown = view(node)
    hidden = set(shown.excluded_llm_metadata_keys)
    visible = {key: value for key, value in shown.metadata.items() if key not in hidden}
    origin = _mapping(visible.get("origin"))
    reference = _mapping(visible.get("reference_metadata"))
    return {
        "n": number,
        "text": shown.get_content(metadata_mode=MetadataMode.NONE)[:PASSAGE_TEXT_CHARS],
        "file": _first(visible.get("filename"), origin.get("filename")),
        "page": _first(visible.get("page"), visible.get("page_number"), origin.get("page_number")),
        "time": _first(reference.get("timestamp"), visible.get("start_ts")),
        "author": _first(reference.get("author"), visible.get("speaker")),
        "network": _first(reference.get("network")),
        "kind": _first(reference.get("type"), visible.get("docint_doc_kind")),
    }


def _source(number: int, node: BaseNode) -> dict[str, Any]:
    """Describe where a passage came from, for the analysis's provenance; never sent.

    Args:
        number (int): Its number in the turn's evidence.
        node (BaseNode): The passage.

    Returns:
        dict[str, Any]: Its number, point id, file hash, file name and page.
    """
    metadata = dict(node.metadata)
    origin = _mapping(metadata.get("origin"))
    pages = (metadata.get("page"), metadata.get("page_number"), origin.get("page_number"))
    page = next((value for value in pages if isinstance(value, int) and not isinstance(value, bool)), None)
    return {
        "citation_index": number,
        "chunk_id": node.node_id,
        "file_hash": source_file_hash(metadata) or None,
        "filename": source_file_name(metadata) or None,
        "page": page,
    }


def _fill(entries: Sequence[Any], cap: int) -> tuple[list[Any], str]:
    """Fill a file in order while its next entry fits.

    Args:
        entries (Sequence[Any]): The entries, in order.
        cap (int): The file's cap in UTF-8 bytes.

    Returns:
        tuple[list[Any], str]: The entries that fit, and the file as written.
    """
    kept: list[Any] = []
    size = 2
    for entry in entries:
        cost = len(json.dumps(entry, ensure_ascii=False).encode("utf-8")) + (2 if kept else 0)
        if size + cost > cap:
            break
        kept.append(entry)
        size += cost
    return kept, json.dumps(kept, ensure_ascii=False)


def _keyword_checks(facts: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return the turn's keyword checks whose count succeeded, each check once, as last made.

    Args:
        facts (Sequence[Mapping[str, Any]]): The turn's facts, in call order.

    Returns:
        list[Mapping[str, Any]]: The checks, in call order; of checks the keyword tool takes for
        one (:func:`~docint.agents.turn.keyword.check_key`), only the latest.
    """
    seen: set[Hashable] = set()
    latest: list[Mapping[str, Any]] = []
    for fact in reversed(facts):
        if fact.get("action") != "keyword" or type(fact.get("matches")) is not int:
            continue
        words, where = str(fact.get("detail") or ""), str(fact.get("field") or DEFAULT_KEYWORD_FIELD)
        try:
            key: Hashable = check_key(words, where)
        except UnknownSearchFieldError:
            key = (where, words)
        if key not in seen:
            seen.add(key)
            latest.append(fact)
    latest.reverse()
    return latest


def _numbers(raw: Any) -> tuple[int, ...] | None:
    """Read the passage numbers a call names, as the model writes them.

    Args:
        raw (Any): The ``passages`` argument; a single number counts as a list of one.

    Returns:
        tuple[int, ...] | None: Each number once, ascending; ``None`` when any is no number.
    """
    values = raw if isinstance(raw, list) else [raw]
    numbers: set[int] = set()
    for value in values:
        number = passage_number(value)
        if number is None:
            return None
        numbers.add(number)
    return tuple(sorted(numbers))


def _cell(value: str) -> tuple[str, bool]:
    """Cut a cell to ``CELL_CHARS``.

    Args:
        value (str): The cell.

    Returns:
        tuple[str, bool]: The cell, and whether it was cut.
    """
    return value[:CELL_CHARS], len(value) > CELL_CHARS


def parse_table(name: str, text: str) -> dict[str, Any] | None:
    """Read a CSV table as pandas writes it, within an analysis's bounds.

    A leading BOM is dropped, and an empty header name (pandas' index column) is kept as ``""``.
    The first row is the header; a short row is padded to its width and a long one cut, the
    table then marked ``truncated``. At most ``MAX_COLUMNS`` columns and ``MAX_ROWS`` rows are
    kept, each cell cut to ``CELL_CHARS``, and ``row_count`` and ``column_count`` keep how many rows and
    columns there were. The name is cut to ``MAX_NAME_CHARS``.

    Args:
        name (str): The file's name.
        text (str): The file's text.

    Returns:
        dict[str, Any] | None: ``name``, ``columns``, ``column_count``, ``rows``, ``row_count`` and
        ``truncated``; ``None`` for a file without a header, such as an empty one, or one the CSV reader
        refuses.
    """
    try:
        rows = list(csv.reader(io.StringIO(text.removeprefix(_BOM), newline="")))
    except csv.Error:
        return None
    if not rows or not rows[0]:
        return None
    header, body = rows[0], rows[1:]
    width = min(len(header), MAX_COLUMNS)
    truncated = len(header) > MAX_COLUMNS or len(body) > MAX_ROWS
    columns: list[str] = []
    for value in header[:width]:
        cell, cut = _cell(value)
        columns.append(cell)
        truncated = truncated or cut
    kept: list[list[str]] = []
    for row in body[:MAX_ROWS]:
        truncated = truncated or len(row) > len(header)
        cells: list[str] = []
        for value in (row + [""] * width)[:width]:
            cell, cut = _cell(value)
            cells.append(cell)
            truncated = truncated or cut
        kept.append(cells)
    return {
        "name": name[:MAX_NAME_CHARS],
        "columns": columns,
        "column_count": len(header),
        "rows": kept,
        "row_count": len(body),
        "truncated": truncated,
    }


def _png_size(png: bytes) -> tuple[int, int] | None:
    """Read a PNG's width and height from its IHDR chunk.

    Args:
        png (bytes): The image.

    Returns:
        tuple[int, int] | None: Width and height, both at least one; ``None`` when the bytes do not
        start as a PNG does: its signature, then a 13-byte IHDR chunk with its checksum.
    """
    if len(png) < _PNG_HEADER_BYTES or not png.startswith(PNG_SIGNATURE):
        return None
    if png[8:12] != _IHDR_LENGTH or png[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", png[16:24])
    return (width, height) if width and height else None


def _charts(images: Sequence[tuple[str, bytes]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep the charts an analysis may hold, in order, and say why each other one was dropped.

    Args:
        images (Sequence[tuple[str, bytes]]): ``(name, png)`` per returned chart.

    Returns:
        tuple[list[dict[str, Any]], list[str]]: The charts kept, and per dropped chart its name and why.
    """
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    total = 0
    for name, png in images:
        size = _png_size(png)
        shown = name[:_NAME_CHARS]
        if size is None:
            dropped.append(f"{shown} (not a readable PNG)")
        elif max(size) > MAX_CHART_SIDE:
            dropped.append(f"{shown} (too large in pixels; at most {MAX_CHART_SIDE} a side)")
        elif len(png) > CHART_BYTES:
            dropped.append(f"{shown} (larger than {CHART_BYTES // 1024} KiB)")
        elif len(kept) >= MAX_CHARTS:
            dropped.append(f"{shown} (at most {MAX_CHARTS} charts are kept)")
        elif total + len(png) > CHARTS_BYTES:
            dropped.append(f"{shown} (past {CHARTS_BYTES // (1024 * 1024)} MiB of charts in all)")
        else:
            total += len(png)
            data_uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
            kept.append({"name": name[:MAX_NAME_CHARS], "data_uri": data_uri, "bytes": len(png)})
    return kept, dropped


def _count(number: int, noun: str) -> str:
    """Count something in words.

    Args:
        number (int): How many.
        noun (str): What, in the singular.

    Returns:
        str: ``1 table``, ``2 tables``.
    """
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _and(parts: Sequence[str]) -> str:
    """Join phrases as a sentence lists them.

    Args:
        parts (Sequence[str]): The phrases.

    Returns:
        str: ``a``, ``a and b``, ``a, b and c``.
    """
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"


def _ranges(numbers: Sequence[int], *, most: int | None = None) -> str:
    """Name passages by number, runs of three or more as ranges.

    Args:
        numbers (Sequence[int]): Their numbers, ascending.
        most (int | None): The most characters the names may take; past it, the rest are
            counted instead. ``None`` for no limit.

    Returns:
        str: ``[1] to [10], [12], [14], [15]``, or with ``and 30 more`` at its end.
    """
    runs: list[tuple[int, int]] = []
    for number in numbers:
        if runs and number == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], number)
        else:
            runs.append((number, number))
    names: list[tuple[str, int]] = []
    for start, end in runs:
        if end - start >= 2:
            names.append((f"[{start}] to [{end}]", end - start + 1))
        else:
            names.extend((f"[{number}]", 1) for number in range(start, end + 1))
    shown: list[str] = []
    named = 0
    for name, count in names:
        if most is not None and len(", ".join([*shown, name])) > most:
            return f"{', '.join(shown)} and {len(numbers) - named} more"
        shown.append(name)
        named += count
    return ", ".join(shown)


def _csv_line(cells: Sequence[str]) -> str:
    """Write one row as a CSV line of at most ``_ROW_LINE_CHARS``, on one line.

    Args:
        cells (Sequence[str]): The row.

    Returns:
        str: The line, indented under its table.
    """
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow([" ".join(cell.split()) for cell in cells])
    line = "  " + buffer.getvalue()
    return line if len(line) <= _ROW_LINE_CHARS else line[: _ROW_LINE_CHARS - 3] + "..."


def _middle(text: str, room: int) -> str:
    """Cut a text to its start and end around a marker, within ``room``.

    Args:
        text (str): The text.
        room (int): Characters it may take, the marker included.

    Returns:
        str: The text, or its start and end around a marker on a line of its own; empty when
        not even the marker fits.
    """
    if len(text) <= room:
        return text
    keep = room - len(cut_marker(len(text), len(text))) - 2
    if keep <= 0:
        return ""
    head = (keep + 1) // 2
    tail = keep - head
    marker = cut_marker(len(text) - keep, len(text))
    return f"{text[:head]}\n{marker}\n{text[len(text) - tail :]}"


def _tail(text: str, room: int) -> str:
    """Keep a text's end within ``room``, a marker ahead of it.

    Args:
        text (str): The text.
        room (int): Characters it may take, the marker included.

    Returns:
        str: The text, or a marker and its end; empty when not even the marker fits.
    """
    if len(text) <= room:
        return text
    keep = room - len(cut_marker(len(text), len(text))) - 1
    return fit_tail_with_marker(text, keep)[0] if keep > 0 else ""


def _check_text(raw: str, limit: int) -> str:
    """Defuse a title the model wrote before the validator reads it.

    Args:
        raw (str): The text.
        limit (int): The most characters to keep.

    Returns:
        str: The text on one line, without double quotes, citation numbers or sentinels, cut to ``limit``.
    """
    text = neutralize_sentinel(CITATION_NUMBERS_RE.sub("", raw.replace('"', "'")))
    return " ".join(text.split())[:limit].rstrip()


def _check_output(raw: str, limit: int) -> str:
    """Defuse what the code printed or named before the validator reads it, keeping every figure.

    A citation number is not removed, for a printed list of figures looks like one: the sentinel is
    neutralized first, then square brackets become parentheses, so nothing reads as a citation.

    Args:
        raw (str): The text.
        limit (int): The most characters to keep.

    Returns:
        str: The text on one line, without square brackets, double quotes or the sentinel, cut to ``limit``.
    """
    text = neutralize_sentinel(raw).replace("[", "(").replace("]", ")").replace('"', "'")
    return " ".join(text.split())[:limit].rstrip()


def _check_passages(provenance: Mapping[str, Any]) -> str:
    """Name the passages an analysis was computed from.

    Args:
        provenance (Mapping[str, Any]): The analysis's provenance.

    Returns:
        str: The numbers of its sources as ranges; empty when it names none.
    """
    sources = provenance.get("sources")
    numbers = {
        source["citation_index"]
        for source in (sources if isinstance(sources, list) else [])
        if isinstance(source, Mapping) and type(source.get("citation_index")) is int and source["citation_index"] > 0
    }
    return _ranges(sorted(numbers), most=_HEAD_NUMBERS_CHARS)


def _column_count(table: Mapping[str, Any]) -> int:
    """Count the columns a stored table's run had.

    Args:
        table (Mapping[str, Any]): The table, its ``columns`` a list.

    Returns:
        int: Its ``column_count``, never fewer than the columns it keeps; an older table has none.
    """
    kept, count = len(table["columns"]), table.get("column_count")
    return max(kept, count) if type(count) is int else kept


def _check_table(table: Any) -> str | None:
    """Describe a table's name and shape.

    Args:
        table (Any): The table as the analysis stores it.

    Returns:
        str | None: Its name with its row and column counts in brackets; ``None`` for a table without either.
    """
    if not isinstance(table, Mapping):
        return None
    name, columns, rows = table.get("name"), table.get("columns"), table.get("row_count")
    if not isinstance(name, str) or not isinstance(columns, list) or type(rows) is not int or rows < 0:
        return None
    label = _check_output(name, _NAME_CHARS)
    return f"{label} ({_count(rows, 'row')} × {_count(_column_count(table), 'column')})" if label else None  # noqa: RUF001


def _analysis_check(number: int, analysis: Any) -> str | None:
    """Write the line one analysis gives its validator.

    Args:
        number (int): The analysis's place in the turn, from 1, which is the name the user sees.
        analysis (Any): The analysis as it was stored.

    Returns:
        str | None: The line; ``None`` for an analysis without a title text, a table list or the passages it was
        computed from.
    """
    if not isinstance(analysis, Mapping):
        return None
    title, tables, provenance = analysis.get("title"), analysis.get("tables"), analysis.get("provenance")
    if not isinstance(title, str) or not isinstance(tables, list) or not isinstance(provenance, Mapping):
        return None
    passages = _check_passages(provenance)
    if not passages:
        return None
    label = _check_text(title, TITLE_CHARS)
    line = f'- Analysis {number} "{label}"' if label else f"- Analysis {number}"
    line += f": from passages {passages}"
    shapes = [shape for shape in map(_check_table, tables) if shape]
    if shapes:
        line += f"; tables {', '.join(shapes)}"
    stdout = analysis.get("stdout")
    shown = _check_output(stdout[:STDOUT_CHECK_CHARS], STDOUT_CHECK_CHARS) if isinstance(stdout, str) else ""
    return f"{line}; stdout: {shown}" if shown else line


def render_analysis_checks(analyses: Sequence[Mapping[str, Any]], *, max_chars: int = ANALYSIS_CHECKS_CHARS) -> str:
    """Render a turn's analyses for its validator, which reads them as facts computed from the passages they name.

    The analyses are stored and read back as data, so a malformed one is skipped, and what the model
    or its code wrote is defused. Each keeps the number the user sees it under, whether or not
    another before it was skipped. Lines are kept whole, in order, while they fit.

    Args:
        analyses (Sequence[Mapping[str, Any]]): The turn's analyses, in order.
        max_chars (int): Characters the block may take.

    Returns:
        str: One line per analysis, naming its title, the passages it came from, its tables' shapes and the start
        of its output; empty when there is none.
    """
    lines: list[str] = []
    used = 0
    for number, analysis in enumerate(analyses, start=1):
        line = _analysis_check(number, analysis)
        if line is None:
            continue
        cost = len(line) + (1 if lines else 0)
        if used + cost > max_chars:
            break
        lines.append(line)
        used += cost
    return "\n".join(lines)


@dataclass
class _Slot:
    """One part of a reply: its place, how much it matters, and its text or how to fit it.

    Attributes:
        priority (int): Parts of a lower number are given room first.
        text (str): The part as it stands; used when ``fit`` is ``None``.
        fit (Callable[[int], str] | None): Fits the part into the characters left; empty for none.
        after (_Slot | None): A part this one is shown only beside.
        shown (str | None): What the reply shows of it, once decided.
    """

    priority: int
    text: str = ""
    fit: Callable[[int], str] | None = None
    after: _Slot | None = None
    shown: str | None = field(default=None, init=False)


def _compose(slots: Sequence[_Slot], limit: int) -> str:
    """Write the parts that fit within ``limit``, the more important first, in their order.

    Args:
        slots (Sequence[_Slot]): The parts, in reply order.
        limit (int): Characters the reply may take.

    Returns:
        str: The parts given room, one per line or block.
    """
    room = limit + 1
    for priority in sorted({slot.priority for slot in slots}):
        for slot in slots:
            if slot.priority != priority or (slot.after is not None and slot.after.shown is None):
                continue
            text = slot.text if slot.fit is None else slot.fit(room - 1)
            if text and len(text) + 1 <= room:
                slot.shown = text
                room -= len(text) + 1
    return "\n".join(slot.shown for slot in slots if slot.shown is not None)


def _block(label: str, text: str, cut: Callable[[str, int], str], most: int) -> Callable[[int], str]:
    """Fit a labelled block of output into the room left.

    Args:
        label (str): Its first line.
        text (str): The output.
        cut (Callable[[str, int], str]): How to cut it to a room.
        most (int): The most characters the output may take.

    Returns:
        Callable[[int], str]: The fitter; it gives nothing when the room is under ``_MIN_BLOCK_CHARS``.
    """

    def fit(room: int) -> str:
        allowance = min(most, room - len(label) - 1)
        if allowance < min(_MIN_BLOCK_CHARS, len(text)):
            return ""
        shown = cut(text, allowance)
        return f"{label}\n{shown}" if shown else ""

    return fit


def _lines(lines: Sequence[str]) -> Callable[[int], str]:
    """Fit whole lines into the room left, in order.

    Args:
        lines (Sequence[str]): The lines.

    Returns:
        Callable[[int], str]: The fitter.
    """

    def fit(room: int) -> str:
        kept: list[str] = []
        used = -1
        for line in lines:
            if used + len(line) + 1 > room:
                break
            kept.append(line)
            used += len(line) + 1
        return "\n".join(kept)

    return fit


class WorkbenchPythonTool:
    """``run_python(code[, passages][, title])``: run the model's Python at workbench on the turn's evidence."""

    def __init__(
        self,
        client: WorkbenchCaller,
        discovery: Discovery,
        turn: PythonTurn,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
        slots: threading.Semaphore = WORKBENCH_SLOT_POOL,
    ) -> None:
        """Bind the client, what discovery found and the turn; the call counts start at zero.

        Args:
            client (WorkbenchCaller): The workbench client.
            discovery (Discovery): What discovery found: the cap on a run's limit and the version.
            turn (PythonTurn): The stored turn the calls are made for, with its deadline.
            timeout_s (float): The most a call may take, as ``WORKBENCH_TIMEOUT`` sets it; the caller reads it.
            clock (Callable[[], float]): Reads the time on the deadline's scale.
            slots (threading.Semaphore): The process's workbench slots, taken without waiting.
        """
        self.name = NAME
        self.description = DESCRIPTION
        self.parameters = copy.deepcopy(PARAMETERS)
        self._client = client
        self._discovery = discovery
        self._turn = turn
        self._clock = clock
        self._slots = slots
        self._timeout_s = timeout_s
        self._round_calls = 0
        self._turn_calls = 0

    def begin_round(self) -> None:
        """Start counting the calls of a new round."""
        self._round_calls = 0

    def prepare(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolReply | ToolWork:
        """Check a call within the limits and build what it hands over from the evidence as it stands.

        The round's call is counted first, then the turn's limit checked, then the code and the
        passage numbers. Only a call that passes all of them builds its files and is counted
        against the turn's two; :meth:`_answer` gives the run back when its fetch sent nothing.

        Args:
            arguments (Mapping[str, Any]): ``code``; optionally ``passages``, the numbers to use
                (left out or empty, every passage shown), and ``title``.
            ctx (ToolContext): The turn's evidence, LLM view and facts, read only.

        Returns:
            ToolReply | ToolWork: An error reply when the call may not run; otherwise its work.
        """
        self._round_calls += 1
        raw_title = arguments.get("title")
        title = " ".join(raw_title.split())[:TITLE_CHARS] if isinstance(raw_title, str) else ""
        if self._round_calls > 1:
            return error_reply(self.name, ROUND_LIMIT_REPLY, kind="call_limit", detail=title)
        if self._turn_calls >= CALLS_PER_TURN:
            return error_reply(self.name, TURN_LIMIT_REPLY, kind="call_limit", detail=title)
        code = arguments.get("code")
        if code is None or (isinstance(code, str) and not code.strip()):
            return error_reply(self.name, MISSING_CODE_REPLY, kind="missing_argument", detail=title)
        if not isinstance(code, str):
            return error_reply(self.name, BAD_CODE_REPLY, kind="bad_argument", detail=title)
        if len(code) > MAX_CODE_CHARS:
            return error_reply(self.name, LONG_CODE_REPLY, kind="too_long", detail=title)
        held = len(ctx.ledger)
        raw = arguments.get("passages")
        numbers = tuple(range(1, held + 1)) if raw is None or raw == [] else _numbers(raw)
        if numbers is None:
            return error_reply(self.name, BAD_PASSAGES_REPLY, kind="bad_argument", detail=title)
        unheld = [number for number in numbers if number > held]
        if unheld:
            message = f"there is no passage [{unheld[0]}]; {passage_range(held)}."
            return error_reply(self.name, message, kind="no_such_passage", detail=title)
        handover = self._handover(code, title, numbers, ctx)
        self._turn_calls += 1
        return ToolWork(
            detail=title,
            fetch=functools.partial(self._fetch, handover),
            answer=functools.partial(self._answer, handover),
        )

    def _handover(self, code: str, title: str, numbers: Sequence[int], ctx: ToolContext) -> _Handover:
        """Build the files a call sends, and the provenance docint keeps.

        Args:
            code (str): The model's code.
            title (str): The analysis's title.
            numbers (Sequence[int]): The passages to send, ascending, each held.
            ctx (ToolContext): The turn's evidence, LLM view and facts.

        Returns:
            _Handover: The files, each filled in order within its cap, and what was left out.
        """
        entries = ctx.ledger.entries()
        nodes = {number: entries[number - 1].node for number in numbers}
        passages, passages_file = _fill(
            [_passage_entry(number, node, ctx.llm_view) for number, node in nodes.items()], PASSAGES_CAP
        )
        sent = tuple(entry["n"] for entry in passages)
        sent_set = set(sent)
        files = [InlineFile(name="passages.json", content=passages_file)]
        checks = _keyword_checks(ctx.facts)
        kept_checks, checks_file = _fill(checks, KEYWORD_CHECKS_CAP)
        if kept_checks:
            files.append(InlineFile(name="keyword_checks.json", content=checks_file))
        listings = [fact for fact in ctx.facts if fact.get("action") == "topics"]
        kept_listings, listings_file = _fill(listings, TOPICS_CAP)
        if kept_listings:
            files.append(InlineFile(name="topics.json", content=listings_file))
        return _Handover(
            code=code,
            title=title,
            files=tuple(files),
            numbers=sent,
            left_out=tuple(number for number in numbers if number not in sent_set),
            sources=tuple(_source(number, nodes[number]) for number in sent),
            keyword_checks=len(kept_checks),
            keyword_left_out=len(checks) - len(kept_checks),
            topic_listings=len(kept_listings),
            topics_left_out=len(listings) - len(kept_listings),
        )

    def _fetch(self, handover: _Handover) -> RunResult | RunFailure | _NotSent:
        """Run the call at workbench, off the loop thread; it reads neither the context nor its evidence.

        The call's deadline is the turn's time left less ``TURN_RESERVE_S``, at most the client's
        timeout; under ``MIN_CALL_S`` nothing is sent. A slot is taken without waiting, the
        disclosure recorded before the send, and settled after it: answered whenever an answer
        came, unanswered otherwise. The slot is given back whatever happened.

        Args:
            handover (_Handover): What the call sends.

        Returns:
            RunResult | RunFailure | _NotSent: What workbench answered, or why nothing was sent.
        """
        t = min(self._timeout_s, self._turn.deadline - self._clock() - TURN_RESERVE_S)
        if t < MIN_CALL_S:
            logger.info("A run_python call was not sent: too little of its turn's time was left")
            return _NotSent("no_time")
        if not self._slots.acquire(blocking=False):
            logger.info("A run_python call was not sent: every workbench slot of this process was taken")
            return _NotSent("no_capacity")
        try:
            try:
                disclosure = record_disclosure(
                    point_ids=handover.point_ids,
                    passages=len(handover.numbers),
                    keyword_checks=handover.keyword_checks,
                    topic_listings=handover.topic_listings,
                    data_bytes=handover.data_bytes,
                )
            except DisclosureRefused:
                logger.warning("A run_python call was not sent: its disclosure could not be recorded")
                return _NotSent("unavailable")
            outcome = self._client.run_python(
                code=handover.code,
                data=handover.files,
                time_limit_s=min(math.floor(t - START_MARGIN_S), self._discovery.cap_s),
                owner=self._turn.owner,
                session_id=self._turn.session_id,
                timeout_s=t,
            )
            if isinstance(outcome, RunResult) or outcome.answered:
                disclosure.answered(outcome.job_id)
            else:
                disclosure.unanswered()
            return outcome
        finally:
            self._slots.release()

    def _answer(self, handover: _Handover, fetched: Any, ctx: ToolContext) -> ToolReply:
        """Reply with what the run returned, and keep its tables and charts as an analysis.

        A call whose fetch sent nothing gets its run back, on the loop thread, before the next
        round is prepared. The analysis is kept only once its reply is written.

        Args:
            handover (_Handover): What the call sent.
            fetched (Any): What :meth:`_fetch` returned.
            ctx (ToolContext): The turn's budget and artifacts.

        Returns:
            ToolReply: An error reply of its kind for a run that sent nothing or returned
            nothing; otherwise the reply, within ``result_chars``, ending in
            :data:`ANALYSIS_NOTE` when the run became an analysis. It admits nothing and is
            informative; its trail names the analysis (``None`` without one) and the counts.
        """
        if isinstance(fetched, _NotSent):
            self._turn_calls -= 1
            return error_reply(self.name, REFUSALS[fetched.kind], kind=fetched.kind, detail=handover.title)
        if isinstance(fetched, RunFailure):
            return error_reply(self.name, REFUSALS[fetched.kind], kind=fetched.kind, detail=handover.title)
        if not isinstance(fetched, RunResult):
            return error_reply(self.name, REFUSALS["unavailable"], kind="unavailable", detail=handover.title)
        tables: list[dict[str, Any]] = []
        others: list[str] = []
        not_kept: list[str] = []
        for name, _mime, text in fetched.texts:
            shown = name[:_NAME_CHARS]
            size = len(text.encode("utf-8"))
            if not name.lower().endswith(".csv"):
                others.append(f"{shown} ({size} bytes)")
                continue
            table = parse_table(name, text)
            if table is None:
                others.append(f"{shown} ({size} bytes, no table)")
            elif len(tables) >= MAX_TABLES:
                not_kept.append(f"{shown} (at most {MAX_TABLES} tables are kept)")
            else:
                tables.append(table)
        charts, dropped = _charts(fetched.images)
        not_kept.extend(dropped)
        number = len(ctx.artifacts) + 1 if tables or charts else None
        text = self._reply_text(handover, fetched, tables, charts, others, not_kept, number, ctx.budget.result_chars)
        if number is not None:
            ctx.artifacts.append(self._analysis(handover, fetched, tables, charts))
        trail = {"analysis": number, "passages": len(handover.numbers), "tables": len(tables), "charts": len(charts)}
        return ToolReply(text=text, found=0, action=self.name, detail=handover.title, informative=True, trail=trail)

    def _analysis(
        self,
        handover: _Handover,
        result: RunResult,
        tables: list[dict[str, Any]],
        charts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the analysis a run became.

        Args:
            handover (_Handover): What the call sent.
            result (RunResult): What the run returned.
            tables (list[dict[str, Any]]): The tables kept.
            charts (list[dict[str, Any]]): The charts kept.

        Returns:
            dict[str, Any]: The analysis, schema 1, with its code, output and provenance.
        """
        return {
            "schema": 1,
            "title": handover.title,
            "code": handover.code,
            "stdout": result.stdout,
            "tables": tables,
            "charts": charts,
            "provenance": {
                "sources": [dict(source) for source in handover.sources],
                "inputs": {
                    "passages": len(handover.numbers),
                    "keyword_checks": handover.keyword_checks,
                    "topic_listings": handover.topic_listings,
                },
                "workbench": {
                    "job_id": result.job_id,
                    "version": result.server_version or self._discovery.server_version,
                    "tool": NAME,
                },
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        }

    @staticmethod
    def _reply_text(
        handover: _Handover,
        result: RunResult,
        tables: Sequence[Mapping[str, Any]],
        charts: Sequence[Mapping[str, Any]],
        others: Sequence[str],
        not_kept: Sequence[str],
        number: int | None,
        limit: int,
    ) -> str:
        """Write the reply within ``limit``, its parts given room in order of importance.

        The closing note comes first, then the head line, its passages named as ranges; then the
        tables' shapes, the charts, the
        other outputs, what was left out or dropped and workbench's hint; then the end of
        ``stderr``; then a few rows of each table; and ``stdout`` fills what is left.

        Args:
            handover (_Handover): What the call sent.
            result (RunResult): What the run returned.
            tables (Sequence[Mapping[str, Any]]): The tables kept.
            charts (Sequence[Mapping[str, Any]]): The charts kept.
            others (Sequence[str]): The other outputs, each named with its size.
            not_kept (Sequence[str]): The tables and charts dropped, each named with why.
            number (int | None): The analysis's number in the turn; ``None`` without one.
            limit (int): Characters the reply may take.

        Returns:
            str: The reply, a look-alike of the assessment sentinel in what the code wrote defused.
        """
        named = _ranges(handover.numbers, most=_HEAD_NUMBERS_CHARS)
        inputs = [_count(len(handover.numbers), "passage") + (f" ({named})" if handover.numbers else "")]
        if handover.keyword_checks:
            inputs.append(_count(handover.keyword_checks, "keyword check"))
        if handover.topic_listings:
            inputs.append(_count(handover.topic_listings, "topic listing"))
        made = (
            f" It made {_count(len(tables), 'table')} and {_count(len(charts), 'chart')}."
            if number is not None
            else " It made no table or chart, so the user sees no analysis."
        )
        slots = [_Slot(1, text=f"run_python ran on {_and(inputs)}.{made}")]
        stdout, stderr = result.stdout.rstrip(), result.stderr.rstrip()
        if stdout.strip():
            slots.append(_Slot(5, fit=_block("stdout:", stdout, _middle, limit)))
        for table in tables:
            rows, columns = int(table["row_count"]), _column_count(table)
            size = f"{_count(rows, 'row')} × {_count(columns, 'column')}"  # noqa: RUF001
            shape = f"Table {str(table['name'])[:_NAME_CHARS]}: {size}"
            shape_slot = _Slot(2, text=shape + ("; the user sees only part of it" if table["truncated"] else ""))
            lines = [_csv_line(table["columns"]), *(_csv_line(row) for row in table["rows"][:_ROWS_SHOWN])]
            slots += [shape_slot, _Slot(4, fit=_lines(lines), after=shape_slot)]
        info: list[str] = []
        if charts:
            info.append("Charts: " + ", ".join(str(chart["name"])[:_NAME_CHARS] for chart in charts) + ".")
        if others:
            info.append("Other outputs: " + ", ".join(others) + ".")
        not_returned = [f"{item.name[:_NAME_CHARS]} ({item.size} bytes)" for item in result.files if not item.returned]
        if not_returned:
            info.append("Not returned by workbench: " + ", ".join(not_returned) + ".")
        if result.truncated:
            info.append("Output cut by workbench.")
        if not_kept:
            info.append("Not kept: " + ", ".join(not_kept) + ".")
        left: list[str] = []
        if handover.left_out:
            left.append(f"passages {_ranges(handover.left_out)}")
        if handover.keyword_left_out:
            left.append(_count(handover.keyword_left_out, "keyword check"))
        if handover.topics_left_out:
            left.append(_count(handover.topics_left_out, "topic listing"))
        if left:
            info.append("Left out, too large for one run: " + "; ".join(left) + ".")
        if result.hint:
            info.append("workbench: " + " ".join(result.hint.split())[:_HINT_CHARS])
        slots += [_Slot(2, text=line) for line in info]
        if stderr.strip():
            slots.append(_Slot(3, fit=_block("stderr, its end:", stderr, _tail, _STDERR_CHARS)))
        if number is not None:
            slots.append(_Slot(0, text=ANALYSIS_NOTE.format(k=number)))
        return neutralize_sentinel(_compose(slots, limit))
