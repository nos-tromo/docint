"""The topics of an agent turn's collection: what it is about, how much of it each subject takes, and its evidence.

A collection's active topic run groups its chunks by meaning and labels each group. ``topics()``
lists them, largest first, each with its label, its first keywords and its share of the
collection's chunks, so a question about what the collection holds, or how much of it concerns
a subject, is answered by a count rather than by sampling passages. Under the turn's filters the
counts and shares are those of the chunks the filters select, and the listing says so. A turn
lists once: a second listing is answered with a note pointing at the first, which stays in the
transcript, since a listing is never cleared. ``topics(topic=T<n>)`` shows that topic's passages
nearest its centre, numbered like any search's. A share is a fact about the collection, not a
source, which the turn's guide line says; the counts a listing showed reach the turn's validator
as topic checks (:func:`render_topic_checks`), as a keyword count does. The run, its counts and
its passages are read in RAG, which owns the store and the request's filters; this module holds
what the turn does with them, and imports nothing from :mod:`docint.core.rag` or
:mod:`docint.core.state`.
"""

from __future__ import annotations

import copy
import functools
import re
import threading
from collections.abc import Callable, Mapping, Sequence
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
#: What a second listing in one turn says instead of the topics, which the first listing shows above it.
LISTED_AGAIN = "The topics are already listed above; pass topic=T<n> to read one."
#: What a listing counted under the turn's filters says in its count line, and its topic checks end in.
FILTERED_MARK = "within the turn's filters"
#: Characters the topic checks may take in the validator's prompt; whole lines only.
TOPIC_CHECKS_CHARS = 800

#: A topic id as a listing writes it, ``T12``, in either case, or as the bare number.
_TOPIC_REF_RE = re.compile(r"\s*[Tt]?(\d{1,9})\s*")


@dataclass(frozen=True)
class TopicLine:
    """One topic as a listing shows it.

    Attributes:
        topic_id (int): The topic's id in its run.
        label (str): Its label, as the run stores it.
        keywords (tuple[str, ...]): Its keywords, most telling first.
        size (int): Chunks the run places in it, within the turn's filters when the listing is filtered.
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
        total (int): Chunks a topic's share is taken of.
        filtered (bool): Whether the sizes and the total count only the chunks the turn's filters
            select; a filtered listing leaves out a topic none of whose chunks they select.
    """

    lines: tuple[TopicLine, ...]
    total: int
    filtered: bool = False


@dataclass(frozen=True)
class TopicPassages:
    """One topic's passages nearest its centre, and the label its reply names it by.

    Attributes:
        label (str): The topic's label in the run the passages were read from; empty when it has none.
        nodes (list[NodeWithScore]): The passages, closest first.
    """

    label: str
    nodes: list[NodeWithScore]


@dataclass(frozen=True)
class TopicsSource:
    """The topics an agent turn offers its model, bound to its collection and filters.

    Attributes:
        listing (Callable[[], TopicsListing]): Reads the collection's topics and their sizes.
        passages (Callable[[int], TopicPassages | None]): Reads one topic's passages nearest its
            centre, under the turn's filters, with its label; ``None`` when the run has no such topic.
    """

    listing: Callable[[], TopicsListing]
    passages: Callable[[int], TopicPassages | None]


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


def _cut_label(label: str) -> str:
    """Return a label as a reply shows it.

    Args:
        label (str): The label, as the run stores it.

    Returns:
        str: The label on one line, cut to ``LABEL_CHARS`` and without trailing space.
    """
    return _clean(label)[:LABEL_CHARS].rstrip()


def _share(size: int, total: int) -> float:
    """Return a part of the counted chunks in percent.

    Args:
        size (int): Chunks in the part.
        total (int): Chunks counted.

    Returns:
        float: The percentage; 0 when no chunk was counted.
    """
    return size / total * 100 if total else 0.0


def _topic_line(line: TopicLine, total: int) -> str:
    """Write one topic's listing line.

    Args:
        line (TopicLine): The topic.
        total (int): Chunks counted.

    Returns:
        str: ``T<id> · label · keywords · size chunks (share%)``, empty parts left out.
    """
    keywords = ", ".join(word for word in (_clean(word) for word in line.keywords[:KEYWORDS_SHOWN]) if word)
    size = f"{line.size} chunks ({_share(line.size, total):.1f}%)"
    parts = [topic_ref(line.topic_id), _cut_label(line.label), keywords, size]
    return " · ".join(part for part in parts if part)


def render_listing(listing: TopicsListing, max_chars: int) -> tuple[str, int]:
    """Write a listing reply: a count line, whole topic lines while they fit, then what was left out.

    Args:
        listing (TopicsListing): The topics, largest first.
        max_chars (int): Characters the reply may take.

    Returns:
        tuple[str, int]: The count line (naming the turn's filters when the listing counted under
        them), one line per topic shown, and, when topics were left out, a line counting them and
        their share of the chunks; then how many topics it shows. The reply keeps within
        ``max_chars`` unless the count line, with the remainder line when a topic is left out, is
        longer on its own; the turn then cuts it as it cuts any reply.
    """
    within = f" {FILTERED_MARK}" if listing.filtered else ""
    head = f"{len(listing.lines)} topics over {listing.total} chunks{within}:"
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
    return _render(shown), shown


def _check_label(raw: str) -> str:
    """Defuse a stored label before the validator reads it.

    Args:
        raw (str): The label as the turn's trail recorded it.

    Returns:
        str: The label on one line, without citation numbers, sentinels or double quotes, at most
        ``LABEL_CHARS`` long.
    """
    return _cut_label(raw.replace('"', "'"))


def render_topic_checks(actions: Sequence[Mapping[str, Any]], *, max_chars: int) -> str:
    """Render the topic counts a turn's listing showed, for its validator, which reads them as facts beside the sources.

    Only a listing step carries them (its ``topics``, ``total`` and ``filtered`` keys), and only
    the topics it showed. The trail is stored and read back as data, so a malformed entry is
    skipped, and a label is defused again. Lines are kept whole, mark included, in trail order,
    while they fit.

    Args:
        actions (Sequence[Mapping[str, Any]]): The turn's trail (``agent.actions``), in order.
        max_chars (int): Characters the block may take.

    Returns:
        str: One ``- T<id> "label": size of total chunks (share%)`` line per topic shown, ending in
        ``; within the turn's filters`` when the listing counted under them; empty when no listing
        showed a topic.
    """
    lines: list[str] = []
    used = 0
    for action in actions:
        listed, total = action.get("topics"), action.get("total")
        if action.get("action") != "topics" or not isinstance(listed, list) or type(total) is not int:
            continue
        within = f"; {FILTERED_MARK}" if action.get("filtered") is True else ""
        for item in listed:
            valid = isinstance(item, (list, tuple)) and len(item) == 3
            if not valid or type(item[0]) is not int or not isinstance(item[1], str) or type(item[2]) is not int:
                continue
            topic_id, label, size = item
            line = f'- {topic_ref(topic_id)} "{_check_label(label)}": {size} of {total} chunks '
            line += f"({_share(size, total):.1f}%){within}"
            cost = len(line) + (1 if lines else 0)
            if used + cost > max_chars:
                return "\n".join(lines)
            lines.append(line)
            used += cost
    return "\n".join(lines)


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
    """``topics([topic])``: list the collection's topics, or show one topic's central passages.

    Build one per turn: it remembers that it listed the topics, and answers a second listing
    with :data:`LISTED_AGAIN` instead of reading them again.
    """

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
        self._listed = False

    def prepare(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolReply | ToolWork:
        """Check the topic; the listing or the topic's passages are the call's fetch.

        Args:
            arguments (Mapping[str, Any]): ``topic``, optionally: the topic to read.
            ctx (ToolContext): Unused; a read needs nothing of the evidence to start.

        Returns:
            ToolReply | ToolWork: The note for a listing the turn already showed, or an error reply
            for a topic that is no topic id; otherwise the read to run and the answer to give with
            what it found.
        """
        listing, raw = _requested(arguments)
        if listing:
            if self._listed:
                return self._listed_again()
            return ToolWork(detail="", fetch=self._read_listing, answer=self._answer_listing)
        topic_id = parse_topic_ref(raw)
        if topic_id is None:
            shown = trail_detail(str(raw))
            return error_reply(
                self.name, f"{shown!r} is not a topic id such as T12.", kind="bad_argument", detail=shown
            )
        return ToolWork(
            detail=topic_ref(topic_id),
            fetch=functools.partial(self._source.passages, topic_id),
            answer=functools.partial(self._answer_topic, topic_id),
        )

    def recall_key(self, arguments: Mapping[str, Any]) -> str | None:
        """Name a topic read by its topic, so the same topic asked for in another spelling is known.

        A listing is left to the default key: it is no passage reply, so the registry never
        remembers it, and the tool answers a second one itself.

        Args:
            arguments (Mapping[str, Any]): The call's arguments, of any types.

        Returns:
            str | None: The topic's ``T<n>`` for a topic read; ``None`` for a listing, or for a topic
            that is no topic id.
        """
        listing, raw = _requested(arguments)
        if listing:
            return None
        topic_id = parse_topic_ref(raw)
        return None if topic_id is None else topic_ref(topic_id)

    def _listed_again(self) -> ToolReply:
        """Answer a listing the turn already showed.

        Returns:
            ToolReply: :data:`LISTED_AGAIN`, which tells the model nothing new, so a round of it is idle.
        """
        return ToolReply(text=LISTED_AGAIN, found=0, action=self.name, detail="")

    def _read_listing(self) -> TopicsListing:
        """Read the listing at most once per turn, even when two calls of one round ask for it at once.

        Returns:
            TopicsListing: The collection's topics.
        """
        with self._lock:
            if self._listing is None:
                self._listing = self._source.listing()
            return self._listing

    def _answer_listing(self, fetched: TopicsListing, ctx: ToolContext) -> ToolReply:
        """List the topics within one reply, or point at the listing the turn already showed.

        Args:
            fetched (TopicsListing): The collection's topics.
            ctx (ToolContext): The turn's budget, which sizes the reply.

        Returns:
            ToolReply: The listing, which admits nothing but is news, its step's trail holding the
            ``total``, whether it was ``filtered`` and the ``topics`` it showed (id, label as shown,
            size); :data:`LISTED_AGAIN` when an earlier call of the turn already listed them.
        """
        if self._listed:
            return self._listed_again()
        text, shown = render_listing(fetched, ctx.budget.result_chars)
        self._listed = True
        trail: dict[str, Any] = {
            "total": fetched.total,
            "filtered": fetched.filtered,
            "topics": [[line.topic_id, _cut_label(line.label), line.size] for line in fetched.lines[:shown]],
        }
        return ToolReply(text=text, found=0, action=self.name, detail="", informative=True, trail=trail)

    def _answer_topic(self, topic_id: int, fetched: TopicPassages | None, ctx: ToolContext) -> ToolReply:
        """Show a topic's central passages after a note naming it.

        Args:
            topic_id (int): The topic.
            fetched (TopicPassages | None): Its passages and label; ``None`` for no such topic.
            ctx (ToolContext): The turn's evidence, budget and LLM view.

        Returns:
            ToolReply: An error reply for a topic the run lacks; otherwise the note, the topic's id
            and label or its id alone, then the passages, as a
            :class:`~docint.agents.turn.tools.PassageReply` that is news and keeps its note when cleared.
        """
        ref = topic_ref(topic_id)
        if fetched is None:
            return error_reply(
                self.name, f"there is no topic {ref}; omit topic to list them.", kind="no_such_topic", detail=ref
            )
        label = _cut_label(fetched.label)
        note = f"{ref} · {label}" if label else ref
        shown = passage_reply(
            fetched.nodes, ctx, action=self.name, detail=ref, hint=TOPIC_HINT, note=note, note_first=True
        )
        return replace(shown, informative=True, keep=note)
