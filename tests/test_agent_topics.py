"""The topics tool of an agent turn: a collection's topics listed, and one topic's central passages read.

Every topic, label, keyword and passage here is invented.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from agent_fakes import (
    LEDGER_CAP,
    answer_round,
    call_round,
    programme_passage,
    scripted_loop,
    shown_numbers,
    streaming_tool_llm,
    tool_call,
    tool_context,
    two_phase_round,
)

from docint.agents.assessment.ledger import EvidenceLedger
from docint.agents.turn.history import ConversationWindow
from docint.agents.turn.loop import AgentResult
from docint.agents.turn.tools import PassageReply, ToolRegistry
from docint.agents.turn.topics import (
    FILTERED_MARK,
    KEYWORDS_SHOWN,
    LABEL_CHARS,
    LISTED_AGAIN,
    TOPIC_CHECKS_CHARS,
    TOPICS_DESCRIPTION,
    TopicLine,
    TopicPassages,
    TopicsListing,
    TopicsSource,
    TopicsTool,
    parse_topic_ref,
    render_topic_checks,
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

    def __init__(self, listing: TopicsListing, *, label: str = "Hafenfest und Stände") -> None:
        """Hold the listing and start counting reads.

        Args:
            listing (TopicsListing): What every listing returns.
            label (str): The label the run stores for topic 4.
        """
        self._listing = listing
        self._label = label
        self.listed = 0
        self.read: list[int] = []

    def listing(self) -> TopicsListing:
        """Return the listing.

        Returns:
            TopicsListing: The listing.
        """
        self.listed += 1
        return self._listing

    def passages(self, topic_id: int) -> TopicPassages | None:
        """Return a topic's central passages with its label in the run.

        Args:
            topic_id (int): The topic asked for.

        Returns:
            TopicPassages | None: The programme for topic 4; ``None`` for any other.
        """
        self.read.append(topic_id)
        return TopicPassages(label=self._label, nodes=[programme_passage()]) if topic_id == 4 else None

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
        f"{topic_ref(item.topic_id)} · {item.label[:LABEL_CHARS].rstrip()} · liegeplatz, gebühr, hafen · "
        f"{item.size} chunks ({item.size / listing.total * 100:.1f}%)"
        for item in listing.lines[: len(shown)]
    ]
    omitted = listing.lines[len(shown) :]
    pct = sum(item.size for item in omitted) / listing.total * 100
    assert omitted
    assert lines[-1] == f"+{len(omitted)} more topics ({pct:.1f}% of chunks)"
    assert len(lines) == len(shown) + 2
    assert reply.trail is not None
    assert [item[0] for item in reply.trail["topics"]] == [item.topic_id for item in listing.lines[: len(shown)]]


def test_a_label_cut_at_a_space_keeps_no_trailing_space() -> None:
    """A label whose 48th character is a space is shown without it, in the listing and in its step's trail."""
    label = f"{'Hafen' * 9}ab Liegeplätze"
    assert label[LABEL_CHARS - 1] == " "
    registry, _ = _registry(TopicsListing(lines=(TopicLine(topic_id=5, label=label, keywords=(), size=1),), total=1))

    reply = registry.call(tool_call("topics"), tool_context())

    assert reply.text.split("\n")[1] == f"T5 · {label[: LABEL_CHARS - 1]} · 1 chunks (100.0%)"
    assert reply.trail is not None
    assert reply.trail["topics"] == [[5, label[: LABEL_CHARS - 1], 1]]


def test_a_filtered_listing_says_so_in_its_count_line_and_its_trail() -> None:
    """A listing counted within the turn's filters names them in its count line, and its step records it."""
    registry, _ = _registry(TopicsListing(lines=LINES[:2], total=200, filtered=True))

    reply = registry.call(tool_call("topics"), tool_context())

    assert reply.text.split("\n") == [
        f"2 topics over 200 chunks {FILTERED_MARK}:",
        "T4 · Hafenfest und Stände · hafenfest, stände, musik · 120 chunks (60.0%)",
        "T0 · Regatta am Sonntag · regatta, segel, wind · 60 chunks (30.0%)",
    ]
    assert reply.trail is not None
    assert (reply.trail["total"], reply.trail["filtered"]) == (200, True)


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
    cut = " ".join(label.split())[:LABEL_CHARS].rstrip()
    assert listed == f"T7 · {cut} · verein, satzung, beitrag · 5 chunks (50.0%)"
    assert (len(cut), KEYWORDS_SHOWN) == (LABEL_CHARS, 3)
    assert "versammlung" not in listed


def test_a_listing_of_an_empty_collection_shares_nothing() -> None:
    """With no chunks counted, a topic's share is 0, never a division by zero."""
    line = TopicLine(topic_id=1, label="Kai", keywords=("kai",), size=0)
    registry, _ = _registry(TopicsListing(lines=(line,), total=0))

    reply = registry.call(tool_call("topics"), tool_context())

    assert reply.text.split("\n")[1] == "T1 · Kai · kai · 0 chunks (0.0%)"


def test_a_listing_is_informative_and_its_step_carries_the_topics_it_showed() -> None:
    """A listing admits nothing yet tells the model something; its step records each topic shown, for the validator."""
    registry, _ = _registry()

    reply = registry.call(tool_call("topics", topic=" "), tool_context())

    assert (reply.found, reply.action, reply.detail, reply.informative, reply.error) == (0, "topics", "", True, False)
    assert reply.keep == ""
    assert not isinstance(reply, PassageReply)
    assert reply.trail == {
        "total": TOTAL,
        "filtered": False,
        "topics": [[4, "Hafenfest und Stände", 120], [0, "Regatta am Sonntag", 60], [2, "Anmeldung am Kai", 20]],
    }


def test_a_repeat_listing_is_a_note_and_reads_the_source_once() -> None:
    """A second listing in one turn points at the first, tells the model nothing new and reads nothing."""
    registry, double = _registry()
    ctx = tool_context()

    first = registry.call(tool_call("topics"), ctx)
    again = registry.call(tool_call("topics", topic=""), ctx)
    prepared = registry.prepare(tool_call("topics"), ctx)

    assert first.informative
    assert again.text == LISTED_AGAIN
    assert (again.found, again.informative, again.error, again.trail, again.keep) == (0, False, False, None, "")
    assert not isinstance(again, PassageReply)
    assert double.listed == 1
    # Answered when prepared: no fetch to run, and no full reply's room to reserve.
    assert (prepared.work, prepared.reply) == (None, again)


def test_a_turns_listing_step_keeps_what_it_showed_and_repeat_listings_leave_it_idle() -> None:
    """In a turn, the listing's step stores the topics it showed as JSON; two repeat listings are idle rounds."""
    double = _Source(TopicsListing(lines=LINES, total=TOTAL))
    rounds = [call_round((f"call_{n}", "topics", {})) for n in range(3)]
    llm = streaming_tool_llm([*rounds, answer_round("Die Sammlung behandelt drei Themen.")])
    loop = scripted_loop(llm, target="documents", topics=double.source())

    items = list(loop.run(window=ConversationWindow(), message="Worum geht es?", ledger=EvidenceLedger(cap=LEDGER_CAP)))

    result = items[-1]
    assert isinstance(result, AgentResult)
    listing_step = {
        "action": "topics",
        "detail": "",
        "found": 0,
        "total": TOTAL,
        "filtered": False,
        "topics": [[4, "Hafenfest und Stände", 120], [0, "Regatta am Sonntag", 60], [2, "Anmeldung am Kai", 20]],
    }
    repeat_step = {"action": "topics", "detail": "", "found": 0}
    assert result.actions == [listing_step, repeat_step, repeat_step]
    assert json.loads(json.dumps(result.actions)) == result.actions
    assert double.listed == 1
    assert (result.reason, result.steps) == ("idle", 3)
    assert llm.stream_chat.call_args_list[3].kwargs["tool_choice"] == "none"
    assert len(render_topic_checks(result.actions, max_chars=TOPIC_CHECKS_CHARS).splitlines()) == 3


def test_two_listings_in_one_round_read_the_source_once_and_the_second_is_the_note() -> None:
    """Prepared side by side, both listings fetch, yet the source is read once and only the first lists."""
    registry, double = _registry()

    first, second = two_phase_round(registry, [tool_call("topics"), tool_call("topics")], tool_context())

    assert first.text.startswith(f"3 topics over {TOTAL} chunks:")
    assert (second.text, second.informative) == (LISTED_AGAIN, False)
    assert double.listed == 1


def test_a_topic_returns_its_central_passages_under_numbers() -> None:
    """A topic's passages are admitted and shown under their numbers, after a note naming it by the run's label."""
    registry, double = _registry()

    reply = registry.call(tool_call("topics", topic="T4"), tool_context())

    assert isinstance(reply, PassageReply)
    assert double.read == [4]
    assert double.listed == 0
    assert shown_numbers(reply.text) == [1]
    assert reply.text.startswith("T4 · Hafenfest und Stände")
    assert (reply.found, reply.action, reply.detail, reply.informative) == (1, "topics", "T4", True)
    assert reply.keep == "T4 · Hafenfest und Stände"


def test_a_topic_without_a_label_is_named_by_its_id_and_keeps_its_passages() -> None:
    """With no label to name it by, a topic's note is its id alone, and its passages are shown all the same."""
    double = _Source(TopicsListing(lines=LINES, total=TOTAL), label="")
    registry = ToolRegistry([TopicsTool(double.source())])

    reply = registry.call(tool_call("topics", topic="T4"), tool_context())

    assert reply.text.startswith("T4\n")
    assert reply.keep == "T4"
    assert shown_numbers(reply.text) == [1]


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


def test_the_recall_keys_name_a_topic_and_leave_a_listing_to_the_default() -> None:
    """A topic read is keyed by its id; a listing, which the tool answers again itself, and garbage are not keyed."""
    tool = TopicsTool(_Source(TopicsListing(lines=(), total=0)).source())

    calls: list[dict[str, object]] = [{}, {"topic": ""}, {"topic": "t12"}, {"topic": 12}, {"topic": "Regatta"}]

    assert [tool.recall_key(arguments) for arguments in calls] == [None, None, "T12", "T12", None]


def test_a_label_cannot_forge_a_citation() -> None:
    """A label is model-written, so a citation number inside it is dropped before the model reads it."""
    line = TopicLine(topic_id=3, label="Gebühren [2] am Kai", keywords=("gebühr",), size=1)
    registry, _ = _registry(TopicsListing(lines=(line,), total=1))

    reply = registry.call(tool_call("topics"), tool_context())

    assert not re.search(r"\[\d+\]", reply.text)


def _listing_step(*, filtered: bool = False, topics: list[Any] | None = None, total: int = 12000) -> dict[str, Any]:
    """Build a listing step as a turn's trail stores it.

    Args:
        filtered (bool): Whether the listing counted within the turn's filters.
        topics (list[Any] | None): The topics it showed, as ``[id, label, size]``; two invented ones by default.
        total (int): The chunks it counted.

    Returns:
        dict[str, Any]: The step.
    """
    shown = [[3, "Hafenfest und Stände", 410], [0, "Regatta am Sonntag", 120]] if topics is None else topics
    return {"action": "topics", "detail": "", "found": 0, "total": total, "filtered": filtered, "topics": shown}


def test_topic_checks_name_each_topic_the_listing_showed_with_its_count_and_share() -> None:
    """Each topic a listing showed becomes one check line, in the listing's order; other steps add nothing."""
    actions = [
        {"action": "search", "detail": "Hafenfest", "found": 2},
        _listing_step(),
        {"action": "topics", "detail": "", "found": 0},
        {"action": "topics", "detail": "T3", "found": 1},
    ]

    assert render_topic_checks(actions, max_chars=TOPIC_CHECKS_CHARS).splitlines() == [
        '- T3 "Hafenfest und Stände": 410 of 12000 chunks (3.4%)',
        '- T0 "Regatta am Sonntag": 120 of 12000 chunks (1.0%)',
    ]


def test_topic_checks_of_a_filtered_listing_say_so_on_every_line() -> None:
    """A share counted within the turn's filters is no share of the whole collection, and each line says so."""
    block = render_topic_checks([_listing_step(filtered=True, total=0)], max_chars=TOPIC_CHECKS_CHARS)

    assert block.splitlines() == [
        f'- T3 "Hafenfest und Stände": 410 of 0 chunks (0.0%); {FILTERED_MARK}',
        f'- T0 "Regatta am Sonntag": 120 of 0 chunks (0.0%); {FILTERED_MARK}',
    ]


def test_topic_checks_keep_whole_lines_within_their_budget() -> None:
    """Lines are kept whole while they fit, in trail order: one that does not fit ends the block."""
    actions = [_listing_step()]
    first = '- T3 "Hafenfest und Stände": 410 of 12000 chunks (3.4%)'

    assert render_topic_checks(actions, max_chars=len(first)) == first
    assert render_topic_checks(actions, max_chars=len(first) - 1) == ""
    assert len(render_topic_checks([_listing_step(topics=[[n, "Kai", n] for n in range(100)])], max_chars=200)) <= 200


def test_topic_checks_skip_what_the_trail_stores_malformed_and_defuse_labels() -> None:
    """The trail is read back as data: malformed entries are skipped, and a label cannot quote or cite."""
    topics: list[Any] = [
        [1, 'Ein "Zitat" mit [2] darin', 5],
        ["2", "Kai", 5],
        [3, None, 5],
        [4, "Kai", "5"],
        [True, "Kai", 5],
        [5, "Kai"],
        "T6",
    ]
    actions: list[dict[str, Any]] = [
        _listing_step(topics=topics),
        {**_listing_step(), "total": "12000"},
        {**_listing_step(), "topics": "T3"},
    ]

    assert (
        render_topic_checks(actions, max_chars=TOPIC_CHECKS_CHARS)
        == "- T1 \"Ein 'Zitat' mit darin\": 5 of 12000 chunks (0.0%)"
    )


def test_without_a_listing_there_are_no_topic_checks() -> None:
    """A turn that never listed its topics hands the validator none, even when it read one."""
    actions = [{"action": "topics", "detail": "T3", "found": 1}, {"action": "keyword", "detail": "Kai", "matches": 0}]

    assert render_topic_checks(actions, max_chars=TOPIC_CHECKS_CHARS) == ""
    assert render_topic_checks([], max_chars=TOPIC_CHECKS_CHARS) == ""


def test_an_empty_filtered_listing_still_grounds_the_validator() -> None:
    """A listing with no matching topics supplies a zero-coverage check without inventing a topic."""
    source = _Source(TopicsListing(lines=(), total=12, filtered=True))
    registry = ToolRegistry([TopicsTool(source.source())])
    reply = registry.call(tool_call("topics"), tool_context())
    assert reply.trail is not None
    step = {"action": "topics", **reply.trail}
    assert render_topic_checks([step], max_chars=TOPIC_CHECKS_CHARS) == (
        f"- No topics cover any of 12 chunks; {FILTERED_MARK}."
    )
    assert render_topic_checks([step], max_chars=5) == ""
    assert render_topic_checks([_listing_step(topics=[])], max_chars=TOPIC_CHECKS_CHARS) == ""


def test_checks_cover_every_topic_the_listing_shows() -> None:
    """A full listing can be validated through its final topic rather than stopping at the old 800-char cap."""
    lines = tuple(TopicLine(i, f"Synthetic topic {i}", ("port",), 1) for i in range(40))
    source = _Source(TopicsListing(lines=lines, total=40))
    registry = ToolRegistry([TopicsTool(source.source())])
    reply = registry.call(tool_call("topics"), tool_context())
    assert reply.trail is not None
    checks = render_topic_checks([{"action": "topics", **reply.trail}], max_chars=TOPIC_CHECKS_CHARS)
    assert len(checks.splitlines()) == len(reply.trail["topics"])
