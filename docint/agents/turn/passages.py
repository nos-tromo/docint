"""Admit what a search found into a turn's evidence, and render it as one tool reply.

A reply never numbers a passage it does not show: a new passage enters the ledger only while
every passage the reply shows still keeps at least its floor. Passages shown before are named
by number, with their own note when they grew past what the reply could show. Sizes are in
characters: the reply keeps within ``result_chars``, room for its notes reserved first, and
the text of each passage within ``passage_chars``; a header shows whole up to ``HEADER_CHARS``.
A calling tool may say what to do about the passages it had no room for, and add a note of
its own; both are among the notes, its own note last, or ahead of the passages when the tool
asks for it first.

Everything here reads or writes the turn's ledger, so it runs on the loop thread.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from llama_index.core.schema import NodeWithScore
from loguru import logger

from docint.agents.assessment.ledger import EvidenceLedger, passage_costs, passage_parts, render_context, stamp
from docint.utils.text_budget import allocate_with_floors

if TYPE_CHECKING:
    from docint.agents.turn.tools import ToolContext

#: Longest header a reply shows whole; a longer one keeps its citation line and ``COMPACT_HEADER_CHARS`` of the rest.
HEADER_CHARS = 500
#: What a search reply says when it found nothing the turn had not shown.
NO_NEW_PASSAGES = "No new passages."
#: How a reply joins its passages, and its passages to the notes after them; ``render_context`` reserves it.
_JOIN = "\n\n"


class RenderedPassages(NamedTuple):
    """A search's passages, admitted into the turn's evidence and rendered as one reply.

    Attributes:
        text (str): The reply: the passages shown, then the notes; a calling tool's own note
            leads instead when it asked for it first.
        found (int): Passages admitted into the evidence or extended there.
        numbers (tuple[int, ...]): The ledger numbers the reply showed or named, ascending.
    """

    text: str
    found: int
    numbers: tuple[int, ...]


def refs(numbers: Iterable[int]) -> str:
    """Name passages by number, as the answer cites them.

    Args:
        numbers (Iterable[int]): Their ledger numbers.

    Returns:
        str: ``[1], [4]`` in ascending order.
    """
    return ", ".join(f"[{number}]" for number in sorted(numbers))


def search_hint(tool: str) -> str:
    """Say what a search reply suggests for the matching passages it had no room for: the same search, narrower.

    The hint names the tool that searched, so a reply never points at a tool its turn does
    not offer: a visual turn has ``search_images`` and no ``search``.

    Args:
        tool (str): The searching tool's name.

    Returns:
        str: The hint, the end of the reply's "did not fit in this reply" note.
    """
    return f"call {tool} again with narrower words to see them"


def _numbered(entry: NodeWithScore, number: int) -> NodeWithScore:
    """Stamp an entry with its number, its citation line first, so every passage a reply shows opens with it.

    Args:
        entry (NodeWithScore): The ledger entry; never written to.
        number (int): Its ledger number.

    Returns:
        NodeWithScore: A stamped clone (see :func:`~docint.agents.assessment.ledger.stamp`).
    """
    stamped = stamp(entry, number)
    metadata = {"citation_index": number, **stamped.node.metadata}
    return NodeWithScore(node=stamped.node.model_copy(update={"metadata": metadata}), score=stamped.score)


def _compact(stamped: NodeWithScore, ctx: ToolContext) -> bool:
    """Return whether a passage's header is too long to show whole.

    Args:
        stamped (NodeWithScore): The numbered passage.
        ctx (ToolContext): The turn's LLM view.

    Returns:
        bool: ``True`` when its header runs past ``HEADER_CHARS``.
    """
    headers, _, _ = passage_parts([stamped], ctx.llm_view)
    return len(headers[0]) > HEADER_CHARS


@dataclass(frozen=True)
class _Planned:
    """A passage a reply may show, priced as the reply renders it.

    Attributes:
        number (int): Its ledger number.
        passage (NodeWithScore): The ledger entry, numbered (see :func:`_numbered`).
        compact (bool): Whether its header is compacted (see :func:`_compact`).
        reserve (int): What its header, joins and cut marker take.
        floor (int): The least text it shows: its legible floor, or ``passage_chars`` if less.
        need (int): The most text it shows: all of it, or ``passage_chars`` if less.
    """

    number: int
    passage: NodeWithScore
    compact: bool
    reserve: int
    floor: int
    need: int


def _plan(entry: NodeWithScore, number: int, ctx: ToolContext) -> _Planned:
    """Price one entry under its number, with the header a reply gives it.

    Args:
        entry (NodeWithScore): The ledger entry, or the entry a new passage would become.
        number (int): Its ledger number.
        ctx (ToolContext): The turn's budget and LLM view.

    Returns:
        _Planned: The numbered entry and its costs.
    """
    numbered = _numbered(entry, number)
    compact = _compact(numbered, ctx)
    headers, bodies, anchors = passage_parts([numbered], ctx.llm_view, compact=compact)
    reserve, floor = passage_costs(headers[0], bodies[0], anchors[0])
    cap = ctx.budget.passage_chars
    return _Planned(
        number=number,
        passage=numbered,
        compact=compact,
        reserve=reserve,
        floor=min(floor, cap),
        need=min(len(bodies[0]), cap),
    )


def _fits(planned: Sequence[_Planned], room: int) -> bool:
    """Return whether passages can all be shown, each at least at its floor, in the room.

    Args:
        planned (Sequence[_Planned]): The passages.
        room (int): Characters they may take together.

    Returns:
        bool: ``True`` when text room remains after their reserves and covers their floors.
    """
    text_room = room - sum(passage.reserve for passage in planned)
    return text_room > 0 and text_room >= sum(passage.floor for passage in planned)


def _render(planned: Sequence[_Planned], room: int, ctx: ToolContext) -> str:
    """Render passages in number order, sharing the room with each kept within its cap.

    Each passage first gets its floor; the rest is shared shortest-first up to its need.
    Each is then rendered alone, so its text is cut window by window around its matches
    to exactly its share, which no shared packing would cap. ``render_context`` defuses
    sentinels in each, and each starts with its citation line, so no join completes one.

    Args:
        planned (Sequence[_Planned]): Passages that fit together (see :func:`_fits`).
        room (int): Characters they may take together.
        ctx (ToolContext): The turn's LLM view.

    Returns:
        str: The passages, each its header and its text, joined by blank lines.
    """
    ordered = sorted(planned, key=lambda passage: passage.number)
    text_room = room - sum(passage.reserve for passage in ordered)
    shares = allocate_with_floors(
        [passage.need for passage in ordered], [passage.floor for passage in ordered], max(0, text_room)
    )
    return _JOIN.join(
        render_context(
            [passage.passage],
            llm_view=ctx.llm_view,
            budget_tokens=passage.reserve + share,
            # The budget is in characters: a reply is sized in characters.
            char_token_ratio=1.0,
            compact=passage.compact,
        ).context
        for passage, share in zip(ordered, shares, strict=True)
    )


def render_entry(entry: NodeWithScore, number: int, ctx: ToolContext) -> str:
    """Render one ledger entry alone under its number, as much of it as one reply holds.

    Args:
        entry (NodeWithScore): The ledger entry; never written to.
        number (int): Its ledger number.
        ctx (ToolContext): The turn's budget and LLM view.

    Returns:
        str: Its header, whole up to ``HEADER_CHARS``, and its text, cut around its
        matches only past ``result_chars``.
    """
    numbered = _numbered(entry, number)
    return render_context(
        [numbered],
        llm_view=ctx.llm_view,
        budget_tokens=ctx.budget.result_chars,
        # The budget is in characters, as a search reply's is.
        char_token_ratio=1.0,
        compact=_compact(numbered, ctx),
    ).context


def _sections(found: Iterable[NodeWithScore]) -> list[list[NodeWithScore]]:
    """Group a search's passages by node id, so the windows of one section are one passage.

    Args:
        found (Iterable[NodeWithScore]): The passages, best first.

    Returns:
        list[list[NodeWithScore]]: Each section's windows in arrival order, sections in the
        order they first appear.
    """
    sections: dict[str, list[NodeWithScore]] = {}
    for scored in found:
        sections.setdefault(scored.node.node_id, []).append(scored)
    return list(sections.values())


def _merged(windows: Sequence[NodeWithScore]) -> NodeWithScore:
    """Return the entry a section not yet held becomes once its windows are admitted.

    Args:
        windows (Sequence[NodeWithScore]): The section's windows, in arrival order.

    Returns:
        NodeWithScore: The entry, merged by the ledger's own rules on a scratch ledger.
    """
    scratch = EvidenceLedger(cap=1)
    scratch.add(windows)
    return scratch.entries()[0]


def _left_out(count: int, *, full: bool, hint: str) -> str:
    """Say how many new passages a reply did not show, and what to do about them.

    Args:
        count (int): How many it left out.
        full (bool): Whether the evidence was full, rather than the reply.
        hint (str): What to do about passages the reply had no room for.

    Returns:
        str: One sentence.
    """
    passages = "passage" if count == 1 else "passages"
    if full:
        return (
            f"{count} more matching {passages} left out: the evidence holds all it can, "
            "so answer from the passages shown."
        )
    return f"{count} more matching {passages} did not fit in this reply; {hint}."


def _trailer(
    *, shown: bool, again: Sequence[int], grown: Sequence[int], left_out: int, full: bool, hint: str, note: str
) -> str:
    """Write the notes after a search reply's passages.

    Args:
        shown (bool): Whether the reply shows any passage.
        again (Sequence[int]): Passages shown before that matched again unchanged, by number.
        grown (Sequence[int]): Passages shown before that the search extended with text the
            reply had no room to show, by number.
        left_out (int): New passages the reply did not show.
        full (bool): Whether they were left out because the evidence was full.
        hint (str): What to do about passages the reply had no room for.
        note (str): The calling tool's own note, last; empty for none.

    Returns:
        str: The notes, one per line; empty when there is nothing to note.
    """
    lines: list[str] = []
    if not shown and not left_out:
        lines.append(NO_NEW_PASSAGES)
    if again:
        lines.append(f"Also matched, shown before: {refs(again)}.")
    if grown:
        lines.append(f"Also matched, with new text not shown here: {refs(grown)}; read(n) shows it.")
    if left_out:
        lines.append(_left_out(left_out, full=full, hint=hint))
    if note:
        lines.append(note)
    return "\n".join(lines)


def answer_passages(
    found: Iterable[NodeWithScore],
    ctx: ToolContext,
    *,
    action: str,
    hint: str | None = None,
    note: str = "",
    note_first: bool = False,
) -> RenderedPassages:
    """Admit what a search found into the turn's evidence and render it as one reply.

    Passages are taken best first, the windows of one section as one passage. A passage
    held before keeps its number: found unchanged, it is named as shown before; extended
    by the search, its grown entry is shown if it fits, and otherwise named as having new
    text that ``read(n)`` shows. A new passage is admitted, under the next number, only
    while the reply can still show every passage at least at its floor; one that does not
    fit, or finds the evidence full, stays unnumbered and the reply says how many more
    matched. The reply keeps within ``result_chars``, room for its notes reserved first
    (every held passage counted in both notes naming them, an upper bound), and each
    passage's text within ``passage_chars``.

    Args:
        found (Iterable[NodeWithScore]): The search's passages, best first.
        ctx (ToolContext): The turn's evidence, budget and LLM view.
        action (str): The calling tool's name, for the log line and the default hint.
        hint (str | None): What the reply suggests for matching passages it had no room for,
            the end of its "did not fit in this reply" note; ``None`` for the calling tool's
            :func:`search_hint`.
        note (str): A note of the calling tool's own, after the others unless ``note_first``;
            empty for none.
        note_first (bool): Whether that note leads the reply, a paragraph ahead of the
            passages, rather than ending it. Its room is reserved first either way; the
            overflow retry keeps a reply's head, so a note that must survive it leads.

    Returns:
        RenderedPassages: The reply, what it admitted or extended, and the numbers it showed or named.
    """
    hint = search_hint(action) if hint is None else hint
    lead, tail = (note, "") if note_first else ("", note)
    ledger = ctx.ledger
    sections = _sections(found)
    held = [number for windows in sections if (number := ledger.number_of(windows[0].node.node_id)) is not None]
    fresh = len(sections) - len(held)
    notes = max(
        len(_trailer(shown=False, again=held, grown=held, left_out=fresh, full=full, hint=hint, note=tail))
        for full in (False, True)
    )
    room = ctx.budget.result_chars - (len(lead) + len(_JOIN) if lead else 0) - notes - len(_JOIN)
    shown: list[_Planned] = []
    again: list[int] = []
    grown: list[int] = []
    left_out, full, added = 0, False, 0
    for windows in sections:
        number = ledger.number_of(windows[0].node.node_id)
        if number is not None:
            if not ledger.add(windows):
                again.append(number)
                continue
            added += 1
            planned = _plan(ledger.entries()[number - 1], number, ctx)
            if _fits([*shown, planned], room):
                shown.append(planned)
            else:
                grown.append(number)
        elif ledger.full:
            left_out, full = left_out + 1, True
        else:
            planned = _plan(_merged(windows), len(ledger) + 1, ctx)
            if _fits([*shown, planned], room) and ledger.add(windows):
                added += 1
                shown.append(planned)
            else:
                left_out += 1
    trailer = _trailer(shown=bool(shown), again=again, grown=grown, left_out=left_out, full=full, hint=hint, note=tail)
    logger.debug(
        "The '{}' tool showed {} passages, named {} shown before and {} grown, and left {} out",
        action,
        len(shown),
        len(again),
        len(grown),
        left_out,
    )
    return RenderedPassages(
        text=_JOIN.join(part for part in (lead, _render(shown, room, ctx), trailer) if part),
        found=added,
        numbers=tuple(sorted({passage.number for passage in shown} | set(again) | set(grown))),
    )
