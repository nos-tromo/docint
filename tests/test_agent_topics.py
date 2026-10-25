"""The topics tool of an agent turn: a collection's topics listed, and one topic's central passages read.

Every topic, label, keyword and passage here is invented.
"""

from __future__ import annotations

import re

import pytest
from agent_fakes import programme_passage, shown_numbers, tool_call, tool_context
from llama_index.core.schema import NodeWithScore

from docint.agents.turn.tools import PassageReply, ToolRegistry
from docint.agents.turn.topics import (
    KEYWORDS_SHOWN,
    LABEL_CHARS,
    TOPICS_DESCRIPTION,
    TopicLine,
    TopicsListing,
    TopicsSource,
    TopicsTool,
    parse_topic_ref,
    topic_ref,
)

#: The invented topics of a small collection, largest first.
LINES = (
    TopicLine(topic_id=4, label="Hafenfest und Stände", keywords=("hafenfest", "stände", "musik"), size=120),
    TopicLine(topic_id=0, label="Regatta am Sonntag", keywords=("regatta", "segel", "wind"), size=60),
    TopicLine(topic_id=2, label="Anmeldung am Kai", keywords=("anmeldung", "kai", "frist"), size=20),
)
TOTAL = 400


class _Source:
    """A topics source double: a fixed listing, and the programme passage for topic 4 alone."""

    def __init__(self, listing: TopicsListing) -> None:
        """Hold the listing and start counting reads.

        Args:
            listing (TopicsListing): What every listing returns.
        """
        self._listing = listing
        self.listed = 0
        self.read: list[int] = []

    def listing(self) -> TopicsListing:
        """Return the listing.

        Returns:
            TopicsListing: The listing.
        """
        self.listed += 1
        return self._listing

    def passages(self, topic_id: int) -> list[NodeWithScore] | None:
        """Return a topic's central passages.

        Args:
            topic_id (int): The topic asked for.

        Returns:
            list[NodeWithScore] | None: The programme for topic 4; ``None`` for any other.
        """
        self.read.append(topic_id)
        return [programme_passage()] if topic_id == 4 else None

    def source(self) -> TopicsSource:
        """Bind the double as a topics source.

        Returns:
            TopicsSource: The source.
        """
        return TopicsSource(listing=self.listing, passages=self.passages)


def _registry(listing: TopicsListing | None = None) -> tuple[ToolRegistry, _Source]:
    """Offer the topics tool alone, over a source double.

    Args:
        listing (TopicsListing | None): The listing; the three invented topics when not given.

    Returns:
        tuple[ToolRegistry, _Source]: The registry and the double behind it.
    """
    double = _Source(TopicsListing(lines=LINES, total=TOTAL) if listing is None else listing)
    return ToolRegistry([TopicsTool(double.source())]), double


def test_the_tool_is_named_topics_and_describes_itself_briefly() -> None:
    """The tool is ``topics``, its one optional argument a topic id, and its description short."""
    tool = TopicsTool(_Source(TopicsListing(lines=(), total=0)).source())

    assert (tool.name, tool.description) == ("topics", TOPICS_DESCRIPTION)
    assert len(TOPICS_DESCRIPTION.split()) <= 60
    assert tool.parameters == {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "A topic id such as T12; omit it to list the topics."}
        },
        "required": [],
        "additionalProperties": False,
    }


def _many(count: int) -> TopicsListing:
    """Build a listing of more topics than one reply holds.

    Args:
        count (int): How many topics.

    Returns:
        TopicsListing: Topics of falling size, each with a long label.
    """
    lines = tuple(
        TopicLine(
            topic_id=index,
            label=f"Abschnitt {index} der Hafenordnung über Liegeplätze und Gebühren",
            keywords=("liegeplatz", "gebühr", "hafen"),
            size=1000 - index,
        )
        for index in range(count)
    )
    return TopicsListing(lines=lines, total=sum(line.size for line in lines) + 500)


def test_a_listing_adds_whole_lines_while_they_fit_then_a_remainder_line() -> None:
    """A listing longer than one reply keeps whole topic lines, largest first, then counts the rest."""
    listing = _many(200)
    registry, _ = _registry(listing)
    ctx = tool_context()

    reply = registry.call(tool_call("topics"), ctx)

    lines = reply.text.split("\n")
    assert len(reply.text) <= ctx.budget.result_chars
    assert lines[0] == f"200 topics over {listing.total} chunks:"
    shown = [line for line in lines[1:] if line.startswith("T")]
    assert shown == [
        f"{topic_ref(item.topic_id)} · {item.label[:LABEL_CHARS]} · liegeplatz, gebühr, hafen · {item.size} chunks "
        f"({item.size / listing.total * 100:.1f}%)"
        for item in listing.lines[: len(shown)]
    ]
    omitted = listing.lines[len(shown) :]
    pct = sum(item.size for item in omitted) / listing.total * 100
    assert omitted
    assert lines[-1] == f"+{len(omitted)} more topics ({pct:.1f}% of chunks)"
    assert len(lines) == len(shown) + 2


def test_a_listing_that_fits_names_every_topic_and_no_remainder() -> None:
    """Three topics fit one reply: each is listed, in the listing's order, with no remainder line."""
    registry, _ = _registry()

    reply = registry.call(tool_call("topics"), tool_context())

    assert reply.text.split("\n") == [
        f"3 topics over {TOTAL} chunks:",
        "T4 · Hafenfest und Stände · hafenfest, stände, musik · 120 chunks (30.0%)",
        "T0 · Regatta am Sonntag · regatta, segel, wind · 60 chunks (15.0%)",
        "T2 · Anmeldung am Kai · anmeldung, kai, frist · 20 chunks (5.0%)",
    ]


def test_a_listing_cuts_labels_to_48_characters_and_shows_three_keywords() -> None:
    """A long label is cut to its first 48 characters, on one line, and only the first three keywords show."""
    label = "Ein sehr langer Titel über den Hafenverein,\nseine Satzung, Beiträge und die Mitgliederversammlung"
    line = TopicLine(topic_id=7, label=label, keywords=("verein", "satzung", "beitrag", "versammlung"), size=5)
    registry, _ = _registry(TopicsListing(lines=(line,), total=10))

    reply = registry.call(tool_call("topics"), tool_context())

    listed = reply.text.split("\n")[1]
    cut = " ".join(label.split())[:LABEL_CHARS]
    assert listed == f"T7 · {cut} · verein, satzung, beitrag · 5 chunks (50.0%)"
    assert (len(cut), KEYWORDS_SHOWN) == (LABEL_CHARS, 3)
    assert "versammlung" not in listed


def test_a_listing_of_an_empty_collection_shares_nothing() -> None:
    """With no chunks counted, a topic's share is 0, never a division by zero."""
    line = TopicLine(topic_id=1, label="Kai", keywords=("kai",), size=0)
    registry, _ = _registry(TopicsListing(lines=(line,), total=0))

    reply = registry.call(tool_call("topics"), tool_context())

    assert reply.text.split("\n")[1] == "T1 · Kai · kai · 0 chunks (0.0%)"


def test_a_listing_is_informative_and_keeps_its_first_line() -> None:
    """A listing admits nothing yet tells the model something, and keeps its count line when cleared."""
    registry, _ = _registry()

    reply = registry.call(tool_call("topics", topic=" "), tool_context())

    assert (reply.found, reply.action, reply.detail, reply.informative, reply.error) == (0, "topics", "", True, False)
    assert reply.keep == f"3 topics over {TOTAL} chunks:"
    assert not isinstance(reply, PassageReply)


def test_a_topic_returns_its_central_passages_under_numbers() -> None:
    """A topic's passages are admitted and shown under their numbers, after a note naming the topic."""
    registry, double = _registry()

    reply = registry.call(tool_call("topics", topic="T4"), tool_context())

    assert isinstance(reply, PassageReply)
    assert double.read == [4]
    assert shown_numbers(reply.text) == [1]
    assert reply.text.startswith("T4 · Hafenfest und Stände")
    assert (reply.found, reply.action, reply.detail, reply.informative) == (1, "topics", "T4", True)
    assert reply.keep == "T4 · Hafenfest und Stände"


def test_an_unknown_topic_is_an_error_of_kind_no_such_topic() -> None:
    """A topic the run lacks is refused with a pointer to the listing, and nothing is admitted."""
    registry, double = _registry()
    ctx = tool_context()

    reply = registry.call(tool_call("topics", topic="T9"), ctx)

    assert reply.text == "Error: there is no topic T9; omit topic to list them."
    assert (reply.error, reply.kind, reply.detail, reply.found) == (True, "no_such_topic", "T9", 0)
    assert double.read == [9]
    assert ctx.ledger.entries() == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [("T12", 12), ("t12", 12), ("12", 12), (12, 12), (" T3 ", 3), ("T", None), ("T1.5", None), ("zwölf", None)],
)
def test_topic_refs_parse(value: object, expected: int | None) -> None:
    """A topic id reads as the listing writes it, in either case or as a bare number; anything else is not one.

    Args:
        value (object): What a model may send.
        expected (int | None): The topic id it names, or ``None``.
    """
    assert parse_topic_ref(value) == expected
    if expected is not None:
        assert topic_ref(expected) == f"T{expected}"


@pytest.mark.parametrize("value", ["Regatta", True, 1.5, ["T1"]])
def test_topic_refs_parse_and_garbage_is_a_bad_argument(value: object) -> None:
    """A topic that is not an id is refused as a bad argument, naming what was sent, and reads nothing.

    Args:
        value (object): What a model sent as the topic.
    """
    registry, double = _registry()

    reply = registry.call(tool_call("topics", topic=value), tool_context())

    assert reply.text == f"Error: {str(value)!r} is not a topic id such as T12."
    assert (reply.kind, reply.detail) == ("bad_argument", str(value))
    assert (double.read, double.listed) == ([], 0)


def test_the_same_call_is_answered_from_memory() -> None:
    """A topic read again, in either spelling, is answered by number and never fetched twice."""
    registry, double = _registry()
    ctx = tool_context()

    first = registry.call(tool_call("topics", topic="T4"), ctx)
    again = registry.call(tool_call("topics", topic="4"), ctx)

    assert shown_numbers(first.text) == [1]
    assert again.text == "Already called topics with these arguments; see [1]."
    assert (again.found, again.informative) == (0, False)
    assert double.read == [4]


def test_the_recall_keys_name_the_listing_and_the_topic() -> None:
    """A listing is keyed ``list``, a topic by its id, and garbage is left to the default key."""
    tool = TopicsTool(_Source(TopicsListing(lines=(), total=0)).source())

    calls: list[dict[str, object]] = [{}, {"topic": ""}, {"topic": "t12"}, {"topic": 12}]

    assert [tool.recall_key(arguments) for arguments in calls] == ["list", "list", "T12", "T12"]
    assert tool.recall_key({"topic": "Regatta"}) is None


def test_a_label_cannot_forge_a_citation() -> None:
    """A label is model-written, so a citation number inside it is dropped before the model reads it."""
    line = TopicLine(topic_id=3, label="Gebühren [2] am Kai", keywords=("gebühr",), size=1)
    registry, _ = _registry(TopicsListing(lines=(line,), total=1))

    reply = registry.call(tool_call("topics"), tool_context())

    assert not re.search(r"\[\d+\]", reply.text)
