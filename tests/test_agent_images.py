"""The pictures a visual turn shows the model: which, in what message, at what cost, and how they are taken back.

Every passage, caption and thumbnail here is invented.
"""

from __future__ import annotations

from typing import Any

import pytest
from _pytest.logging import LogCaptureFixture
from agent_fakes import ThumbnailStore, frame_passage, programme_passage, thumbnail, thumbnail_url
from llama_index.core.base.llms.types import ChatMessage, ImageBlock, MessageRole, TextBlock
from llama_index.core.schema import NodeWithScore

from docint.agents.assessment.ledger import EvidenceLedger
from docint.agents.turn.budget import message_cost
from docint.agents.turn.images import TurnImages
from docint.core.retrieval.visual import (
    VISUAL_IMAGE_TOKEN_ESTIMATE,
    VISUAL_IMAGES_ATTACHED_KEY,
    VISUAL_LEGEND_TOKEN_ALLOWANCE,
    render_legend,
)

LEGEND = "Die Bilder unten sind Quellen, als Bilder beigefügt:\n{image_legend}"
#: An invented legend as long as the shipped ones: its message costs more than the flat allowance.
LONG_LEGEND = (
    "Die folgenden Bilder sind Quellen dieser Antwort und als Bilder beigefügt, damit du sie siehst und "
    "nicht nur ihre Beschreibungen liest. Jede Zeile nennt die Nummer, unter der ein Bild zitiert wird.\n\n"
    "{image_legend}\n\n"
    "Beschreibe nur, was die beigefügten Bilder zeigen, und nenne zu jeder Aussage die Nummer des Bildes. "
    "Weicht ein Bild von seiner Beschreibung ab, folge dem Bild und sage es."
)
#: Room enough for every picture the tests offer.
WIDE = 10_000
#: What a picture message retired to make room says, for the passages whose pictures it showed.
RETIRED = "The pictures of {refs} were shown above and are no longer attached, to make room; read(n) shows one again."


def _images(store: ThumbnailStore | None, *, max_images: int = 6, round_room: int = WIDE) -> TurnImages:
    """Build one turn's pictures over a thumbnail store.

    Args:
        store (ThumbnailStore | None): The fetcher; ``None`` for a turn that shows no pictures.
        max_images (int): The most pictures one prompt may carry.
        round_room (int): The turn's room for tool replies, which caps the pictures too.

    Returns:
        TurnImages: The turn's pictures, none shown yet.
    """
    return TurnImages(
        char_token_ratio=3.5, fetch=store, legend_template=LEGEND, max_images=max_images, round_room=round_room
    )


def _numbered(*passages: NodeWithScore, first: int = 1) -> list[tuple[int, NodeWithScore]]:
    """Number passages as a round's new ledger entries are numbered.

    Args:
        *passages (NodeWithScore): The passages, in ledger order.
        first (int): The first one's number.

    Returns:
        list[tuple[int, NodeWithScore]]: Each passage under its number.
    """
    return list(enumerate(passages, start=first))


def _parts(message: ChatMessage) -> tuple[str, list[str]]:
    """Read an image message as the model receives it: its legend, then its pictures' data URLs.

    Args:
        message (ChatMessage): The message.

    Returns:
        tuple[str, list[str]]: The legend's text, and each picture's URL in order.
    """
    first, *rest = message.blocks
    assert isinstance(first, TextBlock)
    assert all(isinstance(block, ImageBlock) for block in rest)
    return first.text, [str(block.url) for block in rest if isinstance(block, ImageBlock)]


def _price(*numbers: int) -> int:
    """Return what one message showing pictures under these numbers costs as it is sent.

    Args:
        *numbers (int): The numbers its pictures are cited by.

    Returns:
        int: Its legend with the message's envelope, and each picture's estimate.
    """
    legend = render_legend(LEGEND, [(number, "image/jpeg", "") for number in numbers])
    return message_cost(legend, 3.5) + VISUAL_IMAGE_TOKEN_ESTIMATE * len(numbers)


def test_a_picture_that_fits_the_flat_allowance_but_not_its_message_as_sent_is_not_shown() -> None:
    """A picture message is priced as it is sent: its legend and envelope, and each picture's estimate."""
    store = ThumbnailStore(0)
    images = TurnImages(char_token_ratio=3.5, fetch=store, legend_template=LONG_LEGEND, max_images=6, round_room=WIDE)
    flat = VISUAL_LEGEND_TOKEN_ALLOWANCE + VISUAL_IMAGE_TOKEN_ESTIMATE
    as_sent = message_cost(render_legend(LONG_LEGEND, [(1, *thumbnail(0))]), 3.5) + VISUAL_IMAGE_TOKEN_ESTIMATE

    refused = images.attach(_numbered(frame_passage(0)), room=flat, after_round=1)
    shown = images.attach(_numbered(frame_passage(0)), room=as_sent, after_round=1)

    assert flat < as_sent
    assert (refused, store.fetches) == (None, [["frame-0"]])
    assert shown is not None and shown[1] == as_sent


def test_a_rounds_new_pictures_are_shown_after_a_legend_in_number_order() -> None:
    """A round's pictures are fetched at once and shown in one user message: the legend, then each picture."""
    store = ThumbnailStore(0, 1)
    images = _images(store)

    shown = images.attach(_numbered(programme_passage(), frame_passage(0), frame_passage(1)), room=WIDE, after_round=1)

    assert shown is not None
    message, cost = shown
    legend = render_legend(LEGEND, [(2, *thumbnail(0)), (3, *thumbnail(1))])
    assert (message.role, _parts(message), cost) == (
        MessageRole.USER,
        (legend, [thumbnail_url(0), thumbnail_url(1)]),
        _price(2, 3),
    )
    assert store.fetches == [["frame-0", "frame-1"]]
    assert images.count == 2


@pytest.mark.parametrize(
    ("store", "max_images"), [(None, 6), (ThumbnailStore(0), 0)], ids=["no-fetcher", "no-pictures-allowed"]
)
def test_a_turn_without_a_fetcher_or_an_allowance_shows_no_picture(
    store: ThumbnailStore | None, max_images: int
) -> None:
    """With no fetcher, or no picture allowed, nothing is fetched and nothing is shown.

    Args:
        store (ThumbnailStore | None): The fetcher, if any.
        max_images (int): The most pictures one prompt may carry.
    """
    images = _images(store, max_images=max_images)

    assert images.attach(_numbered(frame_passage(0)), room=WIDE, after_round=1) is None
    assert (images.count, store.fetches if store else []) == (0, [])


@pytest.mark.parametrize(
    ("max_images", "round_room", "shown"),
    [(2, WIDE, 2), (6, 2 * VISUAL_IMAGE_TOKEN_ESTIMATE + VISUAL_IMAGE_TOKEN_ESTIMATE - 1, 2)],
    ids=["by-max-images", "by-the-turns-room"],
)
def test_a_prompt_carries_no_more_pictures_than_its_cap(max_images: int, round_room: int, shown: int) -> None:
    """The cap is the fewer of ``max_images`` and the pictures the turn's room could hold, less those still attached.

    Pictures attached in an earlier round count until they are retired (see the retiring tests below).

    Args:
        max_images (int): The most pictures one prompt may carry.
        round_room (int): The turn's room for tool replies.
        shown (int): The pictures attached after both rounds, none of them retired.
    """
    store = ThumbnailStore(0, 1, 2, 3)
    images = _images(store, max_images=max_images, round_room=round_room)

    first = images.attach(_numbered(frame_passage(0)), room=WIDE, after_round=1)
    second = images.attach(
        _numbered(frame_passage(1), frame_passage(2), frame_passage(3), first=2), room=WIDE, after_round=2
    )

    assert first is not None and second is not None
    assert (len(_parts(first[0])[1]), _parts(second[0])[1], images.count) == (1, [thumbnail_url(1)], shown)


@pytest.mark.parametrize(
    ("room", "shown"),
    [(_price(1), 1), (_price(1) - 1, 0), (_price(1, 2) - 1, 1), (_price(1, 2), 2)],
    ids=["one-exactly", "one-short", "two-short", "two-exactly"],
)
def test_each_picture_must_fit_the_room_left(room: int, shown: int) -> None:
    """A message pays its legend as sent, envelope included, and each picture's estimate; pictures may use the room up.

    Args:
        room (int): The tokens the transcript may still take.
        shown (int): The pictures the message shows.
    """
    store = ThumbnailStore(0, 1)
    images = _images(store)

    attached = images.attach(_numbered(frame_passage(0), frame_passage(1)), room=room, after_round=1)

    assert (0 if attached is None else len(_parts(attached[0])[1]), images.count) == (shown, shown)
    assert store.fetches == ([["frame-0", "frame-1"]] if shown else [])


def test_the_fetch_waits_only_for_room_for_the_first_new_picture_alone() -> None:
    """The cheapest message shows the first new picture alone, whose number is the shortest: room for it is enough."""
    first = next(number for number in range(1, 1000) if _price(number) < _price(number + 1))
    store = ThumbnailStore(0, 1)
    images = _images(store)

    shown = images.attach(_numbered(frame_passage(0), frame_passage(1), first=first), room=_price(first), after_round=1)

    assert shown is not None
    assert (_parts(shown[0])[1], shown[1], store.fetches) == (
        [thumbnail_url(0)],
        _price(first),
        [["frame-0", "frame-1"]],
    )


def test_a_failed_fetch_shows_nothing_and_names_only_its_type(loguru_caplog: LogCaptureFixture) -> None:
    """A store outage leaves the round's pictures unshown: the turn answers from their captions.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    images = _images(ThumbnailStore(0, failure=RuntimeError("zzsecretzz invented store outage")))

    assert images.attach(_numbered(frame_passage(0)), room=WIDE, after_round=1) is None
    assert images.count == 0
    assert loguru_caplog.messages == ["An agent round's thumbnails could not be fetched (RuntimeError); none is shown"]


def test_a_picture_without_a_stored_thumbnail_is_passed_over() -> None:
    """Only pictures with a stored thumbnail are shown; one without leaves its turn to the next."""
    images = _images(ThumbnailStore(1), max_images=1)

    shown = images.attach(_numbered(frame_passage(0), frame_passage(1)), room=WIDE, after_round=1)

    assert shown is not None
    assert (_parts(shown[0]), images.count) == ((render_legend(LEGEND, [(2, *thumbnail(1))]), [thumbnail_url(1)]), 1)


def test_a_picture_shown_before_is_not_shown_again() -> None:
    """A picture the prompt carries is not shown again, even when another passage carries it too."""
    store = ThumbnailStore(0, 1)
    images = _images(store)
    again = frame_passage(0)
    again.node.id_ = "frame-1"

    first = images.attach(_numbered(frame_passage(0)), room=WIDE, after_round=1)
    second = images.attach(_numbered(again, first=2), room=WIDE, after_round=2)

    assert first is not None
    assert (second, images.count, store.fetches) == (None, 1, [["frame-0"]])


def test_two_passages_of_one_picture_in_a_round_show_it_once() -> None:
    """Two passages a round admits that carry one picture show it once, under the first one's number."""
    images = _images(ThumbnailStore(0, 1))
    again = frame_passage(0)
    again.node.id_ = "frame-1"

    shown = images.attach(_numbered(frame_passage(0), again), room=WIDE, after_round=1)

    assert shown is not None
    assert (_parts(shown[0]), shown[1], images.count) == (
        (render_legend(LEGEND, [(1, *thumbnail(0))]), [thumbnail_url(0)]),
        _price(1),
        1,
    )


def test_a_drop_takes_every_picture_back_and_shows_no_more() -> None:
    """A refused turn takes back each picture message and its room, keeps every other message, and shows no more."""
    store = ThumbnailStore(0, 1, 2)
    images = _images(store)
    first = images.attach(_numbered(frame_passage(0)), room=WIDE, after_round=1)
    second = images.attach(_numbered(frame_passage(1), first=2), room=WIDE, after_round=2)
    assert first is not None and second is not None
    reply = ChatMessage(
        role=MessageRole.TOOL, content="citation_index: 1", additional_kwargs={"tool_call_id": "call_a"}
    )
    transcript = [ChatMessage(role=MessageRole.ASSISTANT, content=""), reply, first[0], reply, second[0]]

    kept, freed = images.drop(transcript)

    assert (kept, freed, images.count) == ([transcript[0], reply, reply], _price(1) + _price(2), 0)
    assert images.attach(_numbered(frame_passage(2), first=3), room=WIDE, after_round=3) is None
    assert len(store.fetches) == 2


def test_a_drop_takes_back_only_the_messages_it_built() -> None:
    """A drop takes back the messages ``attach`` built, by identity: other picture messages stay, even an equal one."""
    images = _images(ThumbnailStore(0))
    built = images.attach(_numbered(frame_passage(0)), room=WIDE, after_round=1)
    assert built is not None
    foreign = ChatMessage(
        role=MessageRole.USER, blocks=[TextBlock(text="Ein Bild vom Anleger."), ImageBlock(url=thumbnail_url(1))]
    )
    equal = built[0].model_copy(deep=True)

    kept, freed = images.drop([foreign, equal, built[0]])

    assert ([message is foreign for message in kept], kept[1:] == [equal], freed) == ([True, False], True, _price(1))


def _ledger(*passages: NodeWithScore) -> list[NodeWithScore]:
    """Number passages as a turn's result returns them.

    Args:
        *passages (NodeWithScore): The passages, in ledger order.

    Returns:
        list[NodeWithScore]: Their numbered copies.
    """
    ledger = EvidenceLedger(cap=10)
    ledger.add(passages)
    return ledger.numbered()


def _stamped(nodes: list[NodeWithScore]) -> list[Any]:
    """Read which passages carry the stamp that the model saw their picture.

    Args:
        nodes (list[NodeWithScore]): The result's passages.

    Returns:
        list[Any]: Each passage's stamp, ``None`` where it carries none.
    """
    return [scored.node.metadata.get(VISUAL_IMAGES_ATTACHED_KEY) for scored in nodes]


def test_the_stamp_marks_the_passages_whose_pictures_were_shown_until_a_drop() -> None:
    """The result's copies of the passages shown are stamped; after a drop none is, and the ledger is never written."""
    passages = (programme_passage(), frame_passage(0), frame_passage(1))
    images = _images(ThumbnailStore(1))
    images.attach(_numbered(*passages), room=WIDE, after_round=1)

    before = _stamped(images.stamp(_ledger(*passages)))
    images.drop([])
    after = _stamped(images.stamp(_ledger(*passages)))

    assert (before, after) == ([None, None, True], [None, None, None])
    assert _stamped([NodeWithScore(node=scored.node) for scored in passages]) == [None, None, None]


def test_the_stamp_finds_each_passage_by_its_number_not_its_place() -> None:
    """The stamp reads the number each copy carries, so copies in any order are stamped right."""
    passages = (programme_passage(), frame_passage(0), frame_passage(1))
    images = _images(ThumbnailStore(1))
    images.attach(_numbered(*passages), room=WIDE, after_round=1)

    stamped = images.stamp(list(reversed(_ledger(*passages))))

    assert [(scored.node.metadata["citation_index"], _stamped([scored])[0]) for scored in stamped] == [
        (3, True),
        (2, None),
        (1, None),
    ]


def _retired_cost(refs: str) -> int:
    """Return what the line a retired picture message leaves costs in the prompt.

    Args:
        refs (str): The passages it names, as ``[2], [3]``.

    Returns:
        int: Its text with the message's envelope.
    """
    return message_cost(RETIRED.format(refs=refs), 3.5)


def _reply(call_id: str) -> ChatMessage:
    """Build a tool reply, as a round adds one ahead of its pictures.

    Args:
        call_id (str): The id of the call it answers.

    Returns:
        ChatMessage: The reply.
    """
    return ChatMessage(role=MessageRole.TOOL, content="citation_index: 1", additional_kwargs={"tool_call_id": call_id})


def test_retiring_replaces_the_oldest_seen_picture_message_where_it_stands() -> None:
    """The oldest picture message the model has seen becomes a line naming its passages; every other message stays."""
    images = _images(ThumbnailStore(0, 1, 2))
    first = images.attach(_numbered(frame_passage(0)), room=WIDE, after_round=1)
    second = images.attach(_numbered(frame_passage(1), frame_passage(2), first=2), room=WIDE, after_round=2)
    assert first is not None and second is not None
    opening = ChatMessage(role=MessageRole.ASSISTANT, content="")
    transcript = [opening, _reply("call_a"), first[0], _reply("call_b"), second[0]]

    retirable = images.retirable(seen_through=1)
    retired = images.retire_oldest(transcript, seen_through=1)

    assert retired is not None
    kept, freed = retired
    assert retirable == freed == _price(1) - _retired_cost("[1]")
    assert [after is before for after, before in zip(kept, transcript, strict=True)] == [True, True, False, True, True]
    assert (kept[2].role, kept[2].blocks) == (MessageRole.USER, [TextBlock(text=RETIRED.format(refs="[1]"))])
    assert transcript[2] is first[0]
    assert (images.count, images.shown) == (2, 3)


def test_only_picture_messages_the_model_has_seen_are_retired_oldest_first() -> None:
    """A picture message is seen once a round up to ``seen_through`` attached it; each retirement takes the oldest."""
    images = _images(ThumbnailStore(0, 1, 2))
    first = images.attach(_numbered(frame_passage(0)), room=WIDE, after_round=1)
    second = images.attach(_numbered(frame_passage(1), frame_passage(2), first=2), room=WIDE, after_round=2)
    assert first is not None and second is not None

    unseen = (images.retirable(seen_through=0), images.retire_oldest([first[0], second[0]], seen_through=0))
    both = images.retirable(seen_through=2)
    once = images.retire_oldest([first[0], second[0]], seen_through=2)
    assert once is not None
    twice = images.retire_oldest(once[0], seen_through=2)
    assert twice is not None

    assert unseen == (0, None)
    assert [message.content for message in twice[0]] == [RETIRED.format(refs="[1]"), RETIRED.format(refs="[2], [3]")]
    assert both == once[1] + twice[1] == _price(1) + _price(2, 3) - _retired_cost("[1]") - _retired_cost("[2], [3]")
    assert (images.retirable(seen_through=2), images.retire_oldest(twice[0], seen_through=2)) == (0, None)
    assert (images.count, images.shown) == (0, 3)


def test_a_retired_picture_gives_its_place_under_the_cap_back_and_is_not_shown_again_unread() -> None:
    """The cap counts a prompt's pictures: retiring one frees its place, and a mention does not bring it back."""
    store = ThumbnailStore(0, 1, 2)
    images = _images(store, max_images=2)
    first = images.attach(_numbered(frame_passage(0), frame_passage(1)), room=WIDE, after_round=1)
    assert first is not None
    full = images.attach(_numbered(frame_passage(2), first=3), room=WIDE, after_round=2)
    assert images.retire_oldest([first[0]], seen_through=1) is not None

    later = images.attach([(1, frame_passage(0)), (3, frame_passage(2))], room=WIDE, after_round=3)

    assert full is None
    assert later is not None
    assert (_parts(later[0])[1], store.fetches) == ([thumbnail_url(2)], [["frame-0", "frame-1"], ["frame-2"]])
    assert (images.count, images.shown) == (1, 3)


def test_a_read_shows_a_retired_picture_again_and_it_counts_once() -> None:
    """A read of a passage whose picture was retired fetches the picture again and shows it; the turn counts it once."""
    store = ThumbnailStore(0)
    images = _images(store)
    first = images.attach(_numbered(frame_passage(0)), room=WIDE, after_round=1)
    assert first is not None
    assert images.retire_oldest([first[0]], seen_through=1) is not None

    again = images.attach(_numbered(frame_passage(0)), room=WIDE, after_round=2, read={1})

    assert again is not None
    assert (_parts(again[0]), again[1]) == (
        (render_legend(LEGEND, [(1, *thumbnail(0))]), [thumbnail_url(0)]),
        _price(1),
    )
    assert (store.fetches, images.count, images.shown) == ([["frame-0"], ["frame-0"]], 1, 1)
    assert _stamped(images.stamp(_ledger(frame_passage(0)))) == [True]


def test_a_read_of_a_picture_still_attached_shows_it_no_second_time() -> None:
    """Only a retired picture comes back for a read: one the prompt carries now is not fetched or shown twice."""
    store = ThumbnailStore(0)
    images = _images(store)
    assert images.attach(_numbered(frame_passage(0)), room=WIDE, after_round=1) is not None

    again = images.attach(_numbered(frame_passage(0)), room=WIDE, after_round=2, read={1})

    assert (again, store.fetches, images.count, images.shown) == (None, [["frame-0"]], 1, 1)


def test_a_drop_takes_back_only_the_pictures_still_attached() -> None:
    """A refusal takes back the attached pictures; a retired one was seen, so its line, its count and its stamp stay."""
    passages = (frame_passage(0), frame_passage(1))
    images = _images(ThumbnailStore(0, 1))
    first = images.attach(_numbered(passages[0]), room=WIDE, after_round=1)
    second = images.attach(_numbered(passages[1], first=2), room=WIDE, after_round=2)
    assert first is not None and second is not None
    retired = images.retire_oldest([_reply("call_a"), first[0], _reply("call_b"), second[0]], seen_through=1)
    assert retired is not None

    kept, freed = images.drop(retired[0])

    assert [message.role for message in kept] == [MessageRole.TOOL, MessageRole.USER, MessageRole.TOOL]
    assert (kept[1] is retired[0][1], kept[1].content, freed) == (True, RETIRED.format(refs="[1]"), _price(2))
    assert (images.count, images.shown, _stamped(images.stamp(_ledger(*passages)))) == (0, 1, [True, None])
    assert images.attach(_numbered(passages[1], first=2), room=WIDE, after_round=3) is None
