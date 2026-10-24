"""The pictures a visual turn shows the model, beside the captions its tools return.

A caption says what was written about a picture when it was stored, which answered a different
question than the one asked now, so the visual target shows the model the pictures themselves.
After a round's calls are answered, the passages it admitted, showed or named that are pictures
(they carry an ``image_id``, as :func:`~docint.core.retrieval.visual.select_answer_images` reads
them) and that the prompt does not carry have their stored thumbnails fetched by node id, in one
fetch, and are shown in one user message after the round's tool replies: a legend tying each
picture to the number it is cited by, then the pictures. A picture the prompt carries is never
attached twice, one retired to make room comes back only for a read (see below), and a failed
fetch shows none of the round's.

Room. A prompt carries at most ``min(max_images, round_room // VISUAL_IMAGE_TOKEN_ESTIMATE)``
pictures. A message is priced as it is sent: its rendered legend with the message's envelope
(:func:`~docint.agents.turn.budget.message_cost`), and ``VISUAL_IMAGE_TOKEN_ESTIMATE`` a picture.
Pictures are added one at a time while that price fits the room left, down to nothing; the loop's
next check then makes room or makes the model answer. Nothing is fetched when not even the first
new picture alone would fit. To make room, the loop retires the picture messages the model has
seen, oldest first, once it has cleared every reply it may (:meth:`TurnImages.retire_oldest`).
A message is seen once a model call that began carried it: one attached after a round up to
``seen_through``, as a reply is read. Each becomes a line naming its passages
(:data:`RETIRED_PICTURES`) and its pictures leave the prompt's count, so later ones may follow.
A retired picture is attached again only when a round reads its passage, fresh or from memory,
in that round's message and within its cap and room; a search or listing that names it does not
bring it back, so the room just freed is not spent again on every mention. Before a forced
answer, the loop puts retired messages back where they stood, newest first, as far as the room
left and the cap allow (:meth:`TurnImages.restore`).

Stamps. The passages whose pictures the turn showed, retired ones included, are stamped
``VISUAL_IMAGES_ATTACHED_KEY`` on the numbered copies the turn's result returns, as
:class:`~docint.core.retrieval.visual.VisualImagesMixin` stamps its nodes, so the client can say
the model saw them; the ledger's passages are never written.

Refusal. An endpoint may refuse a request that carries pictures: it caps how many it accepts, or
has no vision tower. :meth:`TurnImages.drop` then takes back every picture still attached, its
message, stamp and room, and the turn shows no more, as ``VisualImagesMixin._drop_attached_images``
gives them up. A retired picture was seen by calls that began, so it stays shown and stamped, and
its line stays; so does one a restore put back, whose line returns where it stood.

Nothing here logs a caption, a thumbnail or a passage: only exception types.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from llama_index.core.base.llms.types import ChatMessage, ImageBlock, MessageRole, TextBlock
from llama_index.core.schema import NodeWithScore
from loguru import logger

from docint.agents.turn.budget import message_cost
from docint.agents.turn.passages import refs
from docint.core.retrieval.visual import (
    VISUAL_IMAGE_TOKEN_ESTIMATE,
    VISUAL_IMAGES_ATTACHED_KEY,
    ThumbnailFetcher,
    render_legend,
)

#: What a picture message retired to make room says instead, by the passages whose pictures it showed.
RETIRED_PICTURES = (
    "The pictures of {refs} were shown above and are no longer attached, to make room; read(n) shows one again."
)


@dataclass
class _Attached:
    """One picture message a turn attached, which the loop may retire to make room and put back.

    Attributes:
        message (ChatMessage): The message the transcript holds: the pictures, or the line that
            replaced them while retired.
        built (ChatMessage): The picture message as it was attached, which a restore puts back.
        numbers (tuple[int, ...]): The passages whose pictures it shows, by number, ascending.
        pictures (tuple[str, ...]): Their pictures' ``image_id`` values.
        cost (int): What the picture message cost as it was sent.
        after_round (int): The tool round after which it was attached.
        line (ChatMessage | None): The line that stands in for its pictures while retired;
            ``None`` until it is first retired. A message with one was seen by calls that began.
        retired (bool): Whether the transcript holds its line in place of its pictures.
    """

    message: ChatMessage
    built: ChatMessage
    numbers: tuple[int, ...]
    pictures: tuple[str, ...]
    cost: int
    after_round: int
    line: ChatMessage | None = None
    retired: bool = False


def _picture(scored: NodeWithScore) -> str:
    """Name the picture a passage shows, as ``select_answer_images`` reads it.

    Args:
        scored (NodeWithScore): A passage.

    Returns:
        str: Its ``image_id``; empty for a passage that shows no picture.
    """
    value = scored.node.metadata.get("image_id")
    return str(value) if value else ""


def _retired_text(record: _Attached) -> str:
    """Write the line a retired picture message leaves in the transcript.

    Args:
        record (_Attached): The picture message.

    Returns:
        str: :data:`RETIRED_PICTURES`, naming the passages whose pictures it showed.
    """
    return RETIRED_PICTURES.format(refs=refs(record.numbers))


def _require(record: _Attached, transcript: Sequence[ChatMessage]) -> None:
    """Check that the transcript holds what a picture message stands as now, found by identity.

    Args:
        record (_Attached): The picture message.
        transcript (Sequence[ChatMessage]): The turn's rounds so far.

    Raises:
        RuntimeError: When the transcript does not hold it, which the turn's records never allow.
    """
    if not any(message is record.message for message in transcript):
        raise RuntimeError("A picture message the turn attached is missing from its transcript.")


class TurnImages:
    """The pictures one turn shows the model: which passages, at what cost, and whether more may follow."""

    def __init__(
        self,
        *,
        char_token_ratio: float,
        fetch: ThumbnailFetcher | None = None,
        legend_template: str = "",
        max_images: int = 0,
        round_room: int = 0,
    ) -> None:
        """Bind the thumbnail store and the turn's allowance; with no store, the turn shows no picture.

        Args:
            char_token_ratio (float): Characters per token, the ratio the turn's room was measured with.
            fetch (ThumbnailFetcher | None): Fetches stored thumbnails, ``point_id -> (mime,
                base64)``; ``None`` for a turn that shows no picture.
            legend_template (str): Tells the model the pictures are its sources, with an
                ``{image_legend}`` slot for the lines tying each to its number.
            max_images (int): The most pictures one prompt may carry.
            round_room (int): The turn's room for tool replies (see
                :attr:`~docint.agents.turn.budget.TurnBudget.round_room`), which caps the pictures too.
        """
        self._ratio = char_token_ratio
        self._fetch = fetch
        self._template = legend_template
        self._cap = max(0, min(max_images, round_room // VISUAL_IMAGE_TOKEN_ESTIMATE)) if fetch is not None else 0
        #: The picture messages :meth:`attach` built, in order; :meth:`drop` takes back those still attached.
        self._attached: list[_Attached] = []
        self._dropped = False

    @property
    def count(self) -> int:
        """How many pictures the transcript carries now, which the cap bounds.

        Returns:
            int: The pictures attached and neither retired nor taken back.
        """
        return sum(len(record.numbers) for record in self._attached if not record.retired)

    @property
    def shown(self) -> int:
        """How many pictures count as shown this turn: those attached now and those retired, each once.

        Returns:
            int: The pictures attached now and those retired, which calls that began carried; a
            picture retired and shown again by a read counts once. Pictures a refusal took back
            are not counted, even when an earlier call carried them.
        """
        return len({picture for record in self._attached for picture in record.pictures})

    @property
    def retired(self) -> int:
        """How many picture messages are retired now, which a forced answer may put back.

        Returns:
            int: The picture messages whose lines the transcript holds.
        """
        return sum(record.retired for record in self._attached)

    def _saving(self, record: _Attached) -> int:
        """Return what a picture message frees when its line stands in for it.

        Args:
            record (_Attached): The picture message.

        Returns:
            int: Its cost as it was sent, less its line's.
        """
        return record.cost - message_cost(_retired_text(record), self._ratio)

    def _seen(self, seen_through: int) -> list[_Attached]:
        """Return the picture messages still attached that the model has seen, oldest first.

        Args:
            seen_through (int): The last tool round whose pictures a model call that began carried.

        Returns:
            list[_Attached]: The messages attached after a round up to ``seen_through`` and not retired.
        """
        return [record for record in self._attached if not record.retired and record.after_round <= seen_through]

    def _price(self, shown: Sequence[tuple[int, str, str]]) -> int:
        """Price a picture message as it is sent.

        Args:
            shown (Sequence[tuple[int, str, str]]): ``(number, mime, base64)`` per picture in it;
                only the numbers reach the legend.

        Returns:
            int: Its rendered legend with the message's envelope, and each picture's estimate.
        """
        legend = render_legend(self._template, shown)
        return message_cost(legend, self._ratio) + VISUAL_IMAGE_TOKEN_ESTIMATE * len(shown)

    def attach(
        self,
        entries: Sequence[tuple[int, NodeWithScore]],
        *,
        room: int,
        after_round: int,
        read: Collection[int] = (),
    ) -> tuple[ChatMessage, int] | None:
        """Show a round's new pictures in one message, within the prompt's cap and the room left.

        The pictures the prompt does not carry are fetched at once, those retired to make room
        only when the round read their passages, unless the cheapest message, the first of them
        alone, would not fit. Those with a stored thumbnail are then added in number order while
        the cap, less the pictures attached now, holds and the message, priced as it is sent,
        fits the room left.

        Args:
            entries (Sequence[tuple[int, NodeWithScore]]): The passages the round admitted,
                showed or named, under their numbers, in number order.
            room (int): Tokens the transcript may still take.
            after_round (int): The tool round after which the message is sent, by which the loop
                tells whether the model has seen it.
            read (Collection[int]): The passages the round's reads showed, fresh or from memory,
                by number: a picture of theirs retired to make room is shown again, and any other
                retired picture is not.

        Returns:
            tuple[ChatMessage, int] | None: The user message showing the pictures and what it
            costs; ``None`` when it shows none: the turn shows no picture or no more, none of the
            passages is a new picture, none fits, or the fetch failed.
        """
        free = self._cap - self.count
        carried = {picture for record in self._attached if not record.retired for picture in record.pictures}
        retired = {picture for record in self._attached if record.retired for picture in record.pictures}
        new = [
            (number, scored)
            for number, scored in entries
            if (picture := _picture(scored)) and picture not in carried and (picture not in retired or number in read)
        ]
        # The cheapest message shows the first new picture alone: its legend line is the shortest.
        if self._fetch is None or self._dropped or free <= 0 or not new or self._price([(new[0][0], "", "")]) > room:
            return None
        try:
            stored = self._fetch([scored.node.node_id for _, scored in new])
        except Exception as exc:
            logger.warning("An agent round's thumbnails could not be fetched ({}); none is shown", type(exc).__name__)
            return None
        shown: list[tuple[int, str, str]] = []
        pictures: list[str] = []
        for number, scored in new:
            picture, thumbnail = _picture(scored), stored.get(scored.node.node_id)
            if thumbnail is None or picture in carried:
                continue
            # A later picture's number is no shorter, so once one does not fit, none does.
            if len(shown) == free or self._price([*shown, (number, *thumbnail)]) > room:
                break
            carried.add(picture)
            pictures.append(picture)
            shown.append((number, *thumbnail))
        if not shown:
            return None
        blocks: list[Any] = [TextBlock(text=render_legend(self._template, shown))]
        blocks.extend(ImageBlock(url=f"data:{mime};base64,{data}") for _, mime, data in shown)
        message = ChatMessage(role=MessageRole.USER, blocks=blocks)
        cost = self._price(shown)
        numbers = tuple(number for number, _, _ in shown)
        self._attached.append(
            _Attached(
                message=message,
                built=message,
                numbers=numbers,
                pictures=tuple(pictures),
                cost=cost,
                after_round=after_round,
            )
        )
        return message, cost

    def retirable(self, *, seen_through: int) -> int:
        """Return what retiring every picture message the model has seen would free.

        Args:
            seen_through (int): The last tool round whose pictures a model call that began carried.

        Returns:
            int: Tokens: each such message still attached, less the line that would replace it.
        """
        return sum(self._saving(record) for record in self._seen(seen_through))

    def retire_oldest(
        self, transcript: Sequence[ChatMessage], *, seen_through: int
    ) -> tuple[list[ChatMessage], int] | None:
        """Replace the oldest picture message the model has seen with a line naming its passages.

        The line (:data:`RETIRED_PICTURES`) is a user message where the pictures stood, so the
        transcript keeps its order; its pictures stay shown and stamped, and leave the prompt's count.

        Args:
            transcript (Sequence[ChatMessage]): The turn's rounds so far; never changed.
            seen_through (int): The last tool round whose pictures a model call that began carried.

        Returns:
            tuple[list[ChatMessage], int] | None: The transcript with the message replaced, every
            other message the same object, and the tokens that frees; ``None`` when no picture
            message the model has seen is still attached.

        Raises:
            RuntimeError: When the message is not in the transcript, which the turn's records never allow.
        """
        seen = self._seen(seen_through)
        if not seen:
            return None
        record = seen[0]
        _require(record, transcript)
        if record.line is None:
            record.line = ChatMessage(role=MessageRole.USER, content=_retired_text(record))
        line = record.line
        kept = [line if message is record.message else message for message in transcript]
        record.message, record.retired = line, True
        return kept, self._saving(record)

    def restore(self, transcript: Sequence[ChatMessage], *, room: int) -> tuple[list[ChatMessage], int, int]:
        """Put retired picture messages back where they stood, newest first, while each fits the room and the cap.

        The first that does not fit, in the room left or under the prompt's cap beside the
        pictures attached now, stops it. A message holding a picture the prompt carries again,
        which a read showed after it was retired, is passed over: putting it back would show that
        picture twice. Nothing comes back after a drop: the turn shows no more pictures.

        Args:
            transcript (Sequence[ChatMessage]): The turn's rounds so far; never changed.
            room (int): Tokens the transcript may still take.

        Returns:
            tuple[list[ChatMessage], int, int]: The transcript with the messages put back, every
            other message the same object; the tokens that takes; and how many came back.

        Raises:
            RuntimeError: When a retired message's line is not in the transcript, which the
                turn's records never allow.
        """
        kept = list(transcript)
        grown = back = 0
        if self._dropped:
            return kept, grown, back
        for record in reversed(self._attached):
            if not record.retired:
                continue
            carried = {picture for other in self._attached if not other.retired for picture in other.pictures}
            if carried.intersection(record.pictures):
                continue
            if self.count + len(record.numbers) > self._cap or self._saving(record) > room - grown:
                break
            _require(record, kept)
            kept = [record.built if message is record.message else message for message in kept]
            record.message, record.retired = record.built, False
            grown, back = grown + self._saving(record), back + 1
        return kept, grown, back

    def drop(self, transcript: Sequence[ChatMessage]) -> tuple[list[ChatMessage], int]:
        """Take back every picture still attached, its message, stamp and room, and show no more this turn.

        A retired picture was seen by calls that began, so it stays shown and stamped, and its
        line stays in the transcript. So was one a restore put back: its line returns where it
        stood, and it stays shown and stamped too.

        Args:
            transcript (Sequence[ChatMessage]): The turn's rounds so far; never changed.

        Returns:
            tuple[list[ChatMessage], int]: The transcript without the picture messages still
            attached, found by identity, a restored one's line in its place; and the tokens they
            took.
        """
        kept = list(transcript)
        freed = 0
        for record in self._attached:
            if record.retired:
                continue
            if record.line is None:
                built = record.message
                kept = [message for message in kept if message is not built]
                freed += record.cost
            else:
                line = record.line
                kept = [line if message is record.message else message for message in kept]
                record.message, record.retired = line, True
                freed += self._saving(record)
        self._attached = [record for record in self._attached if record.retired]
        self._dropped = True
        return kept, freed

    def stamp(self, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """Stamp the passages whose pictures the model was shown this turn, retired ones included.

        Args:
            nodes (list[NodeWithScore]): The turn's numbered copies, as the result returns them,
                each found by the ``citation_index`` it carries; the ledger's own passages are
                never written.

        Returns:
            list[NodeWithScore]: The same copies, those shown stamped ``VISUAL_IMAGES_ATTACHED_KEY``.
        """
        shown = {number for record in self._attached for number in record.numbers}
        for scored in nodes:
            if scored.node.metadata.get("citation_index") in shown:
                scored.node.metadata[VISUAL_IMAGES_ATTACHED_KEY] = True
        return nodes
