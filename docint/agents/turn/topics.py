"""The topics of an agent turn's collection: what it is about, how much of it each subject takes, and its evidence.

A collection's active topic run groups its chunks by meaning and labels each group. ``topics()``
lists them, largest first, each with its label, its first keywords and its share of the
collection's chunks, so a question about what the collection holds, or how much of it concerns
a subject, is answered by a count rather than by sampling passages. ``topics(topic=T<n>)``
shows that topic's passages nearest its centre, numbered like any search's. A share is a fact
about the collection, not a source, which the turn's guide line says. The run, its counts and
its passages are read in RAG, which owns the store and the request's filters; this module holds
what the turn does with them, and imports nothing from :mod:`docint.core.rag` or
:mod:`docint.core.state`.
"""

from __future__ import annotations

import copy
import functools
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from llama_index.core.schema import NodeWithScore

from docint.agents.assessment.split import CITATION_NUMBERS_RE, neutralize_sentinel
from docint.agents.turn.tools import ToolContext, ToolReply, ToolWork, error_reply, passage_reply, trail_detail

TOPICS_DESCRIPTION = (
    "List the collection's topics, each with its label, keywords and share of chunks, or pass topic=T<n> "
    "for that topic's most central passages under [n]. Use it for what the collection is about, how much of "
    "it concerns a subject, or for one subject's evidence."
)
TOPICS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"topic": {"type": "string", "description": "A topic id such as T12; omit it to list the topics."}},
    "required": [],
    "additionalProperties": False,
}
#: Longest label a listing line shows.
LABEL_CHARS = 48
#: Keywords a listing line shows per topic.
KEYWORDS_SHOWN = 3
#: What a topic reply suggests for central passages it had no room for.
TOPIC_HINT = "search for the topic's keywords to see them"
#: The recall key of a listing: every listing of one turn is one call.
LIST_KEY = "list"

#: A topic id as a listing writes it, ``T12``, in either case, or as the bare number.
_TOPIC_REF_RE = re.compile(r"\s*[Tt]?(\d{1,9})\s*")


@dataclass(frozen=True)
class TopicLine:
    """One topic as a listing shows it.

    Attributes:
        topic_id (int): The topic's id in its run.
        label (str): Its label, as the run stores it.
        keywords (tuple[str, ...]): Its keywords, most telling first.
        size (int): Chunks the run places in it.
    """

    topic_id: int
    label: str
    keywords: tuple[str, ...]
    size: int


@dataclass(frozen=True)
class TopicsListing:
    """A collection's topics, largest first.

    Attributes:
        lines (tuple[TopicLine, ...]): The topics, sorted by size descending, then id.
        total (int): Chunks in the collection a topic share is taken of.
    """

    lines: tuple[TopicLine, ...]
    total: int


@dataclass(frozen=True)
class TopicsSource:
    """The topics an agent turn offers its model, bound to its collection and filters.

    Attributes:
        listing (Callable[[], TopicsListing]): Reads the collection's topics and their sizes.
        passages (Callable[[int], list[NodeWithScore] | None]): Reads one topic's passages nearest its
            centre, under the turn's filters; ``None`` when the run has no such topic.
    """

    listing: Callable[[], TopicsListing]
    passages: Callable[[int], list[NodeWithScore] | None]


def topic_ref(topic_id: int) -> str:
    """Write a topic id as a listing shows it.

    Args:
        topic_id (int): The topic id.

    Returns:
        str: ``T`` and the id, such as ``T12``.
    """
    return f"T{topic_id}"


def parse_topic_ref(value: object) -> int | None:
    """Read a topic id as a model may write it.

    Args:
        value (object): The ``topic`` the model sent.

    Returns:
        int | None: The id, from ``T12``, ``t12``, ``12`` or the integer 12; ``None`` for anything
        else, a boolean included.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    found = _TOPIC_REF_RE.fullmatch(value)
    return int(found.group(1)) if found else None


def _clean(text: str) -> str:
    """Put model-written text on one line, without citation numbers or sentinels.

    Args:
        text (str): A label or keyword.

    Returns:
        str: The text, safe to show the model beside numbered passages.
    """
    return " ".join(neutralize_sentinel(CITATION_NUMBERS_RE.sub("", text)).split())


def _label(line: TopicLine) -> str:
    """Return a topic's label as a reply shows it.

    Args:
        line (TopicLine): The topic.

    Returns:
        str: The label on one line, cut to ``LABEL_CHARS``.
    """
    return _clean(line.label)[:LABEL_CHARS]


def _share(size: int, total: int) -> float:
    """Return a part of the collection's chunks in percent.

    Args:
        size (int): Chunks in the part.
        total (int): Chunks in the collection.

    Returns:
        float: The percentage; 0 when the collection counts no chunks.
    """
    return size / total * 100 if total else 0.0


def _topic_line(line: TopicLine, total: int) -> str:
    """Write one topic's listing line.

    Args:
        line (TopicLine): The topic.
        total (int): Chunks in the collection.

    Returns:
        str: ``T<id> · label · keywords · size chunks (share%)``, empty parts left out.
    """
    keywords = ", ".join(word for word in (_clean(word) for word in line.keywords[:KEYWORDS_SHOWN]) if word)
    parts = [topic_ref(line.topic_id), _label(line), keywords, f"{line.size} chunks ({_share(line.size, total):.1f}%)"]
    return " · ".join(part for part in parts if part)


def render_listing(listing: TopicsListing, max_chars: int) -> str:
    """Write a listing reply: a count line, whole topic lines while they fit, then what was left out.

    Args:
        listing (TopicsListing): The topics, largest first.
        max_chars (int): Characters the reply may take.

    Returns:
        str: The count line, one line per topic shown, and, when topics were left out, a line
        counting them and their share of the chunks; within ``max_chars`` unless the count line
        alone is longer.
    """
    head = f"{len(listing.lines)} topics over {listing.total} chunks:"
    rows = [_topic_line(line, listing.total) for line in listing.lines]

    def _render(shown: int) -> str:
        """Write the reply showing the first ``shown`` topics.

        Args:
            shown (int): Topic lines to show.

        Returns:
            str: The reply.
        """
        lines = [head, *rows[:shown]]
        omitted = listing.lines[shown:]
        if omitted:
            share = _share(sum(line.size for line in omitted), listing.total)
            lines.append(f"+{len(omitted)} more topics ({share:.1f}% of chunks)")
        return "\n".join(lines)

    shown, used = 0, len(head)
    for row in rows:
        if used + 1 + len(row) > max_chars:
            break
        shown, used = shown + 1, used + 1 + len(row)
    # The remainder line needs room of its own: give up topic lines until it fits.
    while shown > 0 and len(_render(shown)) > max_chars:
        shown -= 1
    return _render(shown)


def _requested(arguments: Mapping[str, Any]) -> tuple[bool, Any]:
    """Read whether a call lists the topics, and the topic it names otherwise.

    Args:
        arguments (Mapping[str, Any]): The call's arguments.

    Returns:
        tuple[bool, Any]: ``True`` and ``None`` for a listing (no topic, or a blank one); otherwise
        ``False`` and the topic as sent.
    """
    raw = arguments.get("topic")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return True, None
    return False, raw


class TopicsTool:
    """``topics([topic])``: list the collection's topics, or show one topic's central passages."""

    def __init__(self, source: TopicsSource) -> None:
        """Bind the source.

        Args:
            source (TopicsSource): The turn's topics, bound to its collection and filters.
        """
        self.name = "topics"
        self.description = TOPICS_DESCRIPTION
        self.parameters = copy.deepcopy(TOPICS_PARAMETERS)
        self._source = source
        self._lock = threading.Lock()
        self._listing: TopicsListing | None = None

    def prepare(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolReply | ToolWork:
        """Check the topic; the listing or the topic's passages are the call's fetch.

        Args:
            arguments (Mapping[str, Any]): ``topic``, optionally: the topic to read.
            ctx (ToolContext): Unused; a read needs nothing of the evidence to start.

        Returns:
            ToolReply | ToolWork: An error reply for a topic that is no topic id; otherwise the
            read to run and the answer to give with what it found.
        """
        listing, raw = _requested(arguments)
        if listing:
            return ToolWork(detail="", fetch=self._listed, answer=self._answer_listing)
        topic_id = parse_topic_ref(raw)
        if topic_id is None:
            shown = trail_detail(str(raw))
            return error_reply(
                self.name, f"{shown!r} is not a topic id such as T12.", kind="bad_argument", detail=shown
            )
        return ToolWork(
            detail=topic_ref(topic_id),
            fetch=functools.partial(self._read, topic_id),
            answer=functools.partial(self._answer_topic, topic_id),
        )

    def recall_key(self, arguments: Mapping[str, Any]) -> str | None:
        """Name a call by what it reads, so a topic asked for in another spelling is known.

        Args:
            arguments (Mapping[str, Any]): The call's arguments, of any types.

        Returns:
            str | None: ``list`` for a listing, the topic's ``T<n>`` for a topic; ``None`` for a topic
            that is no topic id, which is left to the default key.
        """
        listing, raw = _requested(arguments)
        if listing:
            return LIST_KEY
        topic_id = parse_topic_ref(raw)
        return None if topic_id is None else topic_ref(topic_id)

    def _listed(self) -> TopicsListing:
        """Read the listing once per turn; later calls reuse it.

        Returns:
            TopicsListing: The collection's topics.
        """
        with self._lock:
            kept = self._listing
        if kept is not None:
            return kept
        read = self._source.listing()
        with self._lock:
            self._listing = read
        return read

    def _read(self, topic_id: int) -> tuple[list[NodeWithScore], str] | None:
        """Read one topic's passages and the label its note names it by.

        Args:
            topic_id (int): The topic.

        Returns:
            tuple[list[NodeWithScore], str] | None: The passages and the topic's label, empty when the
            listing no longer holds it; ``None`` when the run has no such topic.
        """
        nodes = self._source.passages(topic_id)
        if nodes is None:
            return None
        label = next((_label(line) for line in self._listed().lines if line.topic_id == topic_id), "")
        return nodes, label

    def _answer_listing(self, fetched: TopicsListing, ctx: ToolContext) -> ToolReply:
        """List the topics within one reply.

        Args:
            fetched (TopicsListing): The collection's topics.
            ctx (ToolContext): The turn's budget, which sizes the reply.

        Returns:
            ToolReply: The listing, which admits nothing but is news, keeping its count line when cleared.
        """
        text = render_listing(fetched, ctx.budget.result_chars)
        return ToolReply(text=text, found=0, action=self.name, detail="", informative=True, keep=text.split("\n", 1)[0])

    def _answer_topic(
        self, topic_id: int, fetched: tuple[list[NodeWithScore], str] | None, ctx: ToolContext
    ) -> ToolReply:
        """Show a topic's central passages after a note naming it.

        Args:
            topic_id (int): The topic.
            fetched (tuple[list[NodeWithScore], str] | None): Its passages and label; ``None`` for no such topic.
            ctx (ToolContext): The turn's evidence, budget and LLM view.

        Returns:
            ToolReply: An error reply for a topic the run lacks; otherwise the note, then the
            passages, as a :class:`~docint.agents.turn.tools.PassageReply` that is news and keeps
            its note when cleared.
        """
        ref = topic_ref(topic_id)
        if fetched is None:
            return error_reply(
                self.name, f"there is no topic {ref}; omit topic to list them.", kind="no_such_topic", detail=ref
            )
        nodes, label = fetched
        note = f"{ref} · {label}" if label else ref
        shown = passage_reply(nodes, ctx, action=self.name, detail=ref, hint=TOPIC_HINT, note=note, note_first=True)
        return replace(shown, informative=True, keep=note)
