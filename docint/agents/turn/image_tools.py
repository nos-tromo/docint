"""The image tools an agent turn offers: find pictures by what they show, and look through one clip's frames.

A video's keyframes are stored as pictures beside the collection, each captioned and labelled
with its place in the clip: ``#n @ mm:ss``, its frame number and when it appears, or ``#n``
alone for a clip stored without frame times. ``search_images(description)`` finds pictures by
what they show, as ``search`` finds passages by their words, and answers as a search does.
``keyframes(file)`` shows one clip's frames in time order: every frame up to ``MAX_FRAMES``,
and of a longer clip that many evenly spaced, its first and last among them, with a note of
how many there are. ``keyframes(file, index)`` opens one frame by the number of its label.
Every frame is a passage of the turn's evidence, numbered like any other, so an answer cites
a frame as it cites a page, and a frame both tools find is one passage.

``file`` is a clip's file name, as a passage header shows it, or the number of a passage from
the clip, which names the clip by that passage's media hash (else its file name) as the
evidence stood when the round's calls were prepared. A clip with no stored frames is answered
with the clips the turn's passages came from, so the model can pick one. A file name is not
an identity: two clips from different folders can share one, and such a name is answered
with a way to each clip, a passage from it or its id, rather than with both clips' frames.
"""

from __future__ import annotations

import copy
import functools
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llama_index.core.schema import NodeWithScore

from docint.agents.assessment.split import neutralize_sentinel
from docint.agents.turn.tools import (
    DETAIL_CHARS,
    SearchFn,
    ToolContext,
    ToolReply,
    ToolWork,
    error_reply,
    passage_number,
    passage_range,
    passage_reply,
    search_work,
    trail_detail,
)

KeyframesFn = Callable[[str], list[NodeWithScore]]
"""List one clip's frames, by its file name or its content hash, in time order."""

#: Most frames one keyframes reply shows of a clip.
MAX_FRAMES = 8
#: What a keyframes reply suggests for frames it had no room for.
KEYFRAMES_HINT = "call keyframes(file, index) for one"

SEARCH_IMAGES_DESCRIPTION = (
    "Search the collection's pictures and video frames for what can be seen: who appears and how many people, "
    "clothing, weather and sky, places, objects, visible text. Returns the pictures new to this turn, each under "
    "the number [n] to cite it by, and names pictures shown before by their number; describe what the picture "
    "would show."
)
SEARCH_IMAGES_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"description": {"type": "string", "description": "What the picture should show, in a few words."}},
    "required": ["description"],
}
KEYFRAMES_DESCRIPTION = (
    "Show one video clip's frames in time order; use it when a question concerns a specific clip or follows up a "
    "passage from one. Each frame is numbered [n] to cite it by and labelled #n @ mm:ss (frame number, time); "
    "a long clip shows evenly spaced frames, and index opens one frame."
)
KEYFRAMES_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file": {
            "type": "string",
            "description": "The clip's filename, as a passage header shows it, or the number of a passage from "
            "the clip, such as 3.",
        },
        "index": {"type": "integer", "description": "One frame's number, the n of its #n label."},
    },
    "required": ["file"],
}


class SearchImagesTool:
    """``search_images(description)``: find pictures and video frames by what they show, answered as a search is."""

    def __init__(self, search_images: SearchFn) -> None:
        """Bind the image search.

        Args:
            search_images (SearchFn): Finds the pictures that show a description, with the
                turn's filters and rerank, and returns them best first.
        """
        self.name = "search_images"
        self.description = SEARCH_IMAGES_DESCRIPTION
        self.parameters = copy.deepcopy(SEARCH_IMAGES_PARAMETERS)
        self._search_images = search_images

    def prepare(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolReply | ToolWork:
        """Check the description; the image search is the call's fetch, and its answer a search's.

        Args:
            arguments (Mapping[str, Any]): ``description``, what the picture should show.
            ctx (ToolContext): Unused; a search needs nothing of the evidence to start.

        Returns:
            ToolReply | ToolWork: An error reply when the description is missing, blank or
            not text; otherwise the search to run and the answer to give with what it found.
        """
        return search_work(
            self.name,
            arguments.get("description"),
            self._search_images,
            missing="search_images needs a description: what the picture should show.",
        )


@dataclass(frozen=True)
class _Clip:
    """The clip a keyframes call names.

    Attributes:
        key (str): What its frames are listed by: its file name or its content hash.
        name (str): How replies and the trail name it: its file name, or its hash when no
            file name is known.
    """

    key: str
    name: str


def _text(value: Any) -> str:
    """Read a metadata value as trimmed text.

    Args:
        value (Any): The value.

    Returns:
        str: The text; empty for anything that is not text.
    """
    return value.strip() if isinstance(value, str) else ""


def _quoted(name: str) -> str:
    """Name a clip in a reply: on one line, cut to ``DETAIL_CHARS``, sentinels defused, in quotes.

    A file name is data, whether the model sent it or a document carries it, so a reply
    repeats it bounded and defused, as it does a query.

    Args:
        name (str): The clip's name.

    Returns:
        str: The quoted name.
    """
    return repr(neutralize_sentinel(" ".join(name.split())[:DETAIL_CHARS]))


def _held_clips(ctx: ToolContext) -> dict[str, tuple[str, int]]:
    """Find the clips the turn's passages came from, each once, in evidence order.

    A passage came from a clip when it carries the clip's media hash, as its transcript
    segments and frames do. The hash tells clips apart, since a file name need not.

    Args:
        ctx (ToolContext): The turn's evidence.

    Returns:
        dict[str, tuple[str, int]]: By media hash, the clip's ``source_file`` (its hash
        when it has none) and the number of its first passage.
    """
    clips: dict[str, tuple[str, int]] = {}
    for number, entry in enumerate(ctx.ledger.entries(), start=1):
        metadata = entry.node.metadata
        media_hash = _text(metadata.get("media_file_hash"))
        if media_hash and media_hash not in clips:
            clips[media_hash] = (_text(metadata.get("source_file")) or media_hash, number)
    return clips


def _clips_seen(ctx: ToolContext, *, refused: str = "") -> str:
    """Name the clips the turn's passages came from, for a reply that needs a clip.

    Each clip is named by its ``source_file``, in evidence order; a clip whose name another
    clip seen also carries, the refused one included, adds the number of a passage from it,
    so a reply never shows two clips as one.

    Args:
        ctx (ToolContext): The turn's evidence.
        refused (str): The file name or id of a clip the reply refuses, which it never
            offers back; empty when it refuses none.

    Returns:
        str: ``the clips seen so far are 'a.webm', 'b.mp4'``, or what to do when there are none.
    """
    held = _held_clips(ctx)
    clips = [
        (_quoted(name), number) for media_hash, (name, number) in held.items() if refused not in (media_hash, name)
    ]
    if not clips:
        other = "other " if held else ""
        return f"no {other}clip has been seen yet, so search for one first"
    names = Counter(_quoted(name) for name, _ in held.values())
    return "the clips seen so far are " + ", ".join(
        f"{name} (passage [{number}])" if names[name] > 1 else name for name, number in clips
    )


def _ambiguous(name: str, hashes: Sequence[str], ctx: ToolContext) -> str:
    """Say that several clips carry one file name, and name each so the model can pick one.

    Args:
        name (str): The file name they share.
        hashes (Sequence[str]): Their media hashes.
        ctx (ToolContext): The turn's evidence, for a passage from each.

    Returns:
        str: The refusal, naming each clip by its first passage in the evidence, or by its
        id when the evidence holds none of its passages: held passages first, by number.
    """
    held = _held_clips(ctx)
    ways = sorted(
        (0, held[media_hash][1], f"[{held[media_hash][1]}]") if media_hash in held else (1, 0, _quoted(media_hash))
        for media_hash in hashes
    )
    return (
        f"{len(hashes)} clips are named {_quoted(name)}; call keyframes with file set to the number of a passage "
        "from the clip you mean, or to its id if none of its passages was shown, one per clip: "
        + ", ".join(way for _, _, way in ways)
        + "."
    )


def _frame_number(frame: NodeWithScore) -> int | None:
    """Return a frame's number as its label shows it: its ``keyframe_index`` plus one.

    Args:
        frame (NodeWithScore): A frame, as the keyframe listing returns it.

    Returns:
        int | None: The number; ``None`` for a frame without an index.
    """
    index = frame.node.metadata.get("keyframe_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        return None
    return index + 1


def _asked_frame(value: Any) -> int | None:
    """Read the frame number a model sent as ``index``.

    Args:
        value (Any): The ``index`` sent: ``3``, ``"3"``, or ``"#3"`` as its label reads.

    Returns:
        int | None: The number; ``None`` for anything that is no positive whole number.
    """
    if isinstance(value, str):
        value = value.strip().removeprefix("#")
    return passage_number(value)


def _evenly_spaced(frames: Sequence[NodeWithScore], count: int) -> list[NodeWithScore]:
    """Pick at most ``count`` frames, evenly spaced, the first and the last among them.

    Args:
        frames (Sequence[NodeWithScore]): The clip's frames, in time order.
        count (int): The most frames to pick, at least two.

    Returns:
        list[NodeWithScore]: The picked frames, in time order; all of them when they are few enough.
    """
    if len(frames) <= count:
        return list(frames)
    last = len(frames) - 1
    return [frames[round(step * last / (count - 1))] for step in range(count)]


def _frame_range(numbers: Sequence[int]) -> str:
    """Name the frame numbers a clip has, for a reply to one it lacks.

    Nextext numbers frames before it drops near-duplicates, so the numbers have gaps: a clip
    of at most ``MAX_FRAMES`` frames has them listed, a longer one its range.

    Args:
        numbers (Sequence[int]): The numbers of its frames, at least one.

    Returns:
        str: ``its frames are #1, #2, #4``, ``its frames run from #1 to #20``, or ``its only
        frame is #1``.
    """
    if len(numbers) == 1:
        return f"its only frame is #{numbers[0]}"
    if len(numbers) <= MAX_FRAMES:
        return "its frames are " + ", ".join(f"#{number}" for number in sorted(numbers))
    return f"its frames run from #{min(numbers)} to #{max(numbers)}"


class KeyframesTool:
    """``keyframes(file, index?)``: show one clip's frames in time order, or open one by its number."""

    def __init__(self, list_keyframes: KeyframesFn) -> None:
        """Bind the keyframe listing.

        Args:
            list_keyframes (KeyframesFn): Lists one clip's frames, with the turn's filters,
                by its file name or content hash, in time order.
        """
        self.name = "keyframes"
        self.description = KEYFRAMES_DESCRIPTION
        self.parameters = copy.deepcopy(KEYFRAMES_PARAMETERS)
        self._list_keyframes = list_keyframes

    def prepare(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolReply | ToolWork:
        """Read which clip the call names, and the frame it asks for; the listing is the call's fetch.

        A passage number names its clip as the evidence stood when the round's calls were
        prepared, so a number the same round admits is not known yet.

        Args:
            arguments (Mapping[str, Any]): ``file``, a clip's file name or the number of a
                passage from it, and optionally ``index``, one frame's number.
            ctx (ToolContext): The turn's evidence, read for a passage number.

        Returns:
            ToolReply | ToolWork: An error reply when ``file`` names no clip or ``index`` no
            frame; otherwise the listing to run and the answer to give with the frames.
        """
        raw = arguments.get("file")
        clip = self._clip(raw, ctx)
        if isinstance(clip, ToolReply):
            return clip
        index: int | None = None
        if arguments.get("index") is not None:
            index = _asked_frame(arguments.get("index"))
            if index is None:
                return error_reply(
                    self.name,
                    "index is the number of one frame, the n of its #n label, such as 3.",
                    detail=trail_detail(clip.name),
                )
        detail = trail_detail(clip.name if index is None else f"{clip.name} #{index}")
        return ToolWork(
            detail=detail,
            fetch=functools.partial(self._list_keyframes, clip.key),
            answer=functools.partial(self._answer, clip, index, detail),
        )

    def _clip(self, raw: Any, ctx: ToolContext) -> _Clip | ToolReply:
        """Read which clip a call names: by its file name, or by a passage of the turn's evidence.

        Args:
            raw (Any): The ``file`` the model sent.
            ctx (ToolContext): The turn's evidence.

        Returns:
            _Clip | ToolReply: The clip; an error reply when ``file`` is missing, names no
            passage of the evidence, or names a passage from no clip.
        """
        number = passage_number(raw)
        if number is None:
            name = raw.strip() if isinstance(raw, str) else ""
            if not name:
                return error_reply(
                    self.name,
                    "keyframes needs file: a clip's filename, or the number of a passage from it, such as 3.",
                    detail=trail_detail(raw),
                )
            return _Clip(key=name, name=name)
        held = len(ctx.ledger)
        if not 1 <= number <= held:
            return error_reply(
                self.name, f"there is no passage [{number}]; {passage_range(held)}.", detail=trail_detail(raw)
            )
        metadata = ctx.ledger.entries()[number - 1].node.metadata
        media_hash, source_file = _text(metadata.get("media_file_hash")), _text(metadata.get("source_file"))
        if not (media_hash or source_file):
            return error_reply(
                self.name, f"passage [{number}] is not from a clip; {_clips_seen(ctx)}.", detail=trail_detail(raw)
            )
        return _Clip(key=media_hash or source_file, name=source_file or media_hash)

    def _answer(self, clip: _Clip, index: int | None, detail: str, fetched: Any, ctx: ToolContext) -> ToolReply:
        """Show the clip's frames, at most ``MAX_FRAMES`` evenly spaced, or the one asked for.

        Args:
            clip (_Clip): The clip.
            index (int | None): The frame asked for, by its number; ``None`` for the clip's frames.
            detail (str): The call's main argument, for the trail.
            fetched (Any): The clip's frames, in time order, as the listing returned them.
            ctx (ToolContext): The turn's evidence, budget and LLM view.

        Returns:
            ToolReply: The frames as passages under their numbers, with a note of how many
            the clip has when it has more than ``MAX_FRAMES``; an error reply when the clip
            has no stored frames or none of that number, or when its frames come from more
            than one clip, which a file name two clips share lists, so it is never shown as one.
        """
        numbered = [(number, frame) for frame in fetched or () if (number := _frame_number(frame)) is not None]
        if not numbered:
            seen = _clips_seen(ctx, refused=clip.key)
            return error_reply(self.name, f"no keyframes are stored for {_quoted(clip.name)}; {seen}.", detail=detail)
        hashes = list(
            dict.fromkeys(
                media_hash for _, frame in numbered if (media_hash := _text(frame.node.metadata.get("media_file_hash")))
            )
        )
        if len(hashes) > 1:
            return error_reply(self.name, _ambiguous(clip.name, hashes, ctx), detail=detail)
        if index is not None:
            chosen = [frame for number, frame in numbered if number == index]
            if not chosen:
                known = _frame_range([number for number, _ in numbered])
                return error_reply(self.name, f"{_quoted(clip.name)} has no frame #{index}; {known}.", detail=detail)
            return passage_reply(chosen[:1], ctx, action=self.name, detail=detail, hint=KEYFRAMES_HINT)
        frames = [frame for _, frame in numbered]
        note = f"{len(frames)} frames in all; {KEYFRAMES_HINT}." if len(frames) > MAX_FRAMES else ""
        return passage_reply(
            _evenly_spaced(frames, MAX_FRAMES), ctx, action=self.name, detail=detail, hint=KEYFRAMES_HINT, note=note
        )
