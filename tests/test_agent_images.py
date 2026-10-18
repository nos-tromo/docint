"""The pictures a visual turn shows the model: which, in what message, at what cost, and how they are taken back.

Every passage, caption and thumbnail here is invented.
"""

from __future__ import annotations

from typing import Any

import pytest
from _pytest.logging import LogCaptureFixture
from agent_fakes import ThumbnailStore, frame_passage, programme_passage, thumbnail
from llama_index.core.base.llms.types import ChatMessage, ImageBlock, MessageRole, TextBlock
from llama_index.core.schema import NodeWithScore

from docint.agents.assessment.ledger import EvidenceLedger
from docint.agents.turn.images import TurnImages
from docint.core.retrieval.visual import (
    VISUAL_IMAGE_TOKEN_ESTIMATE,
    VISUAL_IMAGES_ATTACHED_KEY,
    VISUAL_LEGEND_TOKEN_ALLOWANCE,
    render_legend,
)

LEGEND = "Die Bilder unten sind Quellen, als Bilder beigefügt:\n{image_legend}"
#: Room enough for every picture the tests offer.
WIDE = 10_000


def _images(store: ThumbnailStore | None, *, max_images: int = 6, round_room: int = WIDE) -> TurnImages:
    """Build one turn's pictures over a thumbnail store.

    Args:
        store (ThumbnailStore | None): The fetcher; ``None`` for a turn that shows no pictures.
        max_images (int): The most pictures the turn may show.
        round_room (int): The turn's room for tool replies, which caps the pictures too.

    Returns:
        TurnImages: The turn's pictures, none shown yet.
    """
    return TurnImages(fetch=store, legend_template=LEGEND, max_images=max_images, round_room=round_room)


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


def _url(index: int) -> str:
    """Return the data URL a frame's thumbnail is shown by.

    Args:
        index (int): The frame's place in the clip.

    Returns:
        str: The URL.
    """
    mime, data = thumbnail(index)
    return f"data:{mime};base64,{data}"


def _cost(count: int) -> int:
    """Return what one message of ``count`` pictures costs.

    Args:
        count (int): The pictures in it.

    Returns:
        int: The legend's allowance and each picture's estimate.
    """
    return VISUAL_LEGEND_TOKEN_ALLOWANCE + count * VISUAL_IMAGE_TOKEN_ESTIMATE


def test_a_rounds_new_pictures_are_shown_after_a_legend_in_number_order() -> None:
    """A round's pictures are fetched at once and shown in one user message: the legend, then each picture."""
    store = ThumbnailStore(0, 1)
    images = _images(store)

    shown = images.attach(_numbered(programme_passage(), frame_passage(0), frame_passage(1)), room=WIDE)

    assert shown is not None
    message, cost = shown
    legend = render_legend(LEGEND, [(2, *thumbnail(0)), (3, *thumbnail(1))])
    assert (message.role, _parts(message), cost) == (MessageRole.USER, (legend, [_url(0), _url(1)]), _cost(2))
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
        max_images (int): The most pictures the turn may show.
    """
    images = _images(store, max_images=max_images)

    assert images.attach(_numbered(frame_passage(0)), room=WIDE) is None
    assert (images.count, store.fetches if store else []) == (0, [])


@pytest.mark.parametrize(
    ("max_images", "round_room", "shown"),
    [(2, WIDE, 2), (6, 2 * VISUAL_IMAGE_TOKEN_ESTIMATE + VISUAL_IMAGE_TOKEN_ESTIMATE - 1, 2)],
    ids=["by-max-images", "by-the-turns-room"],
)
def test_a_turn_shows_no_more_pictures_than_its_cap(max_images: int, round_room: int, shown: int) -> None:
    """The cap is the fewer of ``max_images`` and the pictures the turn's whole room could hold, across rounds.

    Args:
        max_images (int): The most pictures the turn may show.
        round_room (int): The turn's room for tool replies.
        shown (int): The pictures the turn shows in all.
    """
    store = ThumbnailStore(0, 1, 2, 3)
    images = _images(store, max_images=max_images, round_room=round_room)

    first = images.attach(_numbered(frame_passage(0)), room=WIDE)
    second = images.attach(_numbered(frame_passage(1), frame_passage(2), frame_passage(3), first=2), room=WIDE)

    assert first is not None and second is not None
    assert (len(_parts(first[0])[1]), _parts(second[0])[1], images.count) == (1, [_url(1)], shown)


@pytest.mark.parametrize(("room", "shown"), [(_cost(1), 1), (_cost(1) - 1, 0), (_cost(2) - 1, 1), (_cost(2), 2)])
def test_each_picture_must_fit_the_room_left(room: int, shown: int) -> None:
    """A message pays the legend's allowance once and each picture's estimate; pictures may use the room up.

    Args:
        room (int): The tokens the transcript may still take.
        shown (int): The pictures the message shows.
    """
    store = ThumbnailStore(0, 1)
    images = _images(store)

    attached = images.attach(_numbered(frame_passage(0), frame_passage(1)), room=room)

    assert (0 if attached is None else len(_parts(attached[0])[1]), images.count) == (shown, shown)
    assert store.fetches == ([["frame-0", "frame-1"]] if shown else [])


def test_a_failed_fetch_shows_nothing_and_names_only_its_type(loguru_caplog: LogCaptureFixture) -> None:
    """A store outage leaves the round's pictures unshown: the turn answers from their captions.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    images = _images(ThumbnailStore(0, failure=RuntimeError("zzsecretzz invented store outage")))

    assert images.attach(_numbered(frame_passage(0)), room=WIDE) is None
    assert images.count == 0
    assert loguru_caplog.messages == ["An agent round's thumbnails could not be fetched (RuntimeError); none is shown"]


def test_a_picture_without_a_stored_thumbnail_is_passed_over() -> None:
    """Only pictures with a stored thumbnail are shown; one without leaves its turn to the next."""
    images = _images(ThumbnailStore(1), max_images=1)

    shown = images.attach(_numbered(frame_passage(0), frame_passage(1)), room=WIDE)

    assert shown is not None
    assert (_parts(shown[0]), images.count) == ((render_legend(LEGEND, [(2, *thumbnail(1))]), [_url(1)]), 1)


def test_a_picture_shown_before_is_not_shown_again() -> None:
    """A picture is shown once a turn, even when another passage carries it too."""
    store = ThumbnailStore(0, 1)
    images = _images(store)
    again = frame_passage(0)
    again.node.id_ = "frame-1"

    first = images.attach(_numbered(frame_passage(0)), room=WIDE)
    second = images.attach(_numbered(again, first=2), room=WIDE)

    assert first is not None
    assert (second, images.count, store.fetches) == (None, 1, [["frame-0"]])


def test_two_passages_of_one_picture_in_a_round_show_it_once() -> None:
    """Two passages a round admits that carry one picture show it once, under the first one's number."""
    images = _images(ThumbnailStore(0, 1))
    again = frame_passage(0)
    again.node.id_ = "frame-1"

    shown = images.attach(_numbered(frame_passage(0), again), room=WIDE)

    assert shown is not None
    assert (_parts(shown[0]), shown[1], images.count) == (
        (render_legend(LEGEND, [(1, *thumbnail(0))]), [_url(0)]),
        _cost(1),
        1,
    )


def test_a_drop_takes_every_picture_back_and_shows_no_more() -> None:
    """A refused turn takes back each picture message and its room, keeps every other message, and shows no more."""
    store = ThumbnailStore(0, 1, 2)
    images = _images(store)
    first = images.attach(_numbered(frame_passage(0)), room=WIDE)
    second = images.attach(_numbered(frame_passage(1), first=2), room=WIDE)
    assert first is not None and second is not None
    reply = ChatMessage(
        role=MessageRole.TOOL, content="citation_index: 1", additional_kwargs={"tool_call_id": "call_a"}
    )
    transcript = [ChatMessage(role=MessageRole.ASSISTANT, content=""), reply, first[0], reply, second[0]]

    kept, freed = images.drop(transcript)

    assert (kept, freed, images.count) == ([transcript[0], reply, reply], _cost(1) * 2, 0)
    assert images.attach(_numbered(frame_passage(2), first=3), room=WIDE) is None
    assert len(store.fetches) == 2


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
    images.attach(_numbered(*passages), room=WIDE)

    before = _stamped(images.stamp(_ledger(*passages)))
    images.drop([])
    after = _stamped(images.stamp(_ledger(*passages)))

    assert (before, after) == ([None, None, True], [None, None, None])
    assert _stamped([NodeWithScore(node=scored.node) for scored in passages]) == [None, None, None]
