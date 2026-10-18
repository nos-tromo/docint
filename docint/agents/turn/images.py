"""The pictures a visual turn shows the model, beside the captions its tools return.

A caption says what was written about a picture when it was stored, which answered a different
question than the one asked now, so the visual target shows the model the pictures themselves.
After a round's calls are answered, the passages it newly admitted that are pictures (they carry
an ``image_id``, as :func:`~docint.core.retrieval.visual.select_answer_images` reads them) have
their stored thumbnails fetched by node id, in one fetch, and are shown in one user message after
the round's tool replies: a legend tying each picture to the number it is cited by, then the
pictures. A picture is shown once a turn, and a failed fetch shows none of the round's.

Room. A turn shows at most ``min(max_images, round_room // VISUAL_IMAGE_TOKEN_ESTIMATE)``
pictures. A message is priced as it is sent: its rendered legend with the message's envelope
(:func:`~docint.agents.turn.budget.message_cost`), and ``VISUAL_IMAGE_TOKEN_ESTIMATE`` a picture.
Pictures are added one at a time while that price fits the room left, down to nothing; the loop's
next check then makes the model answer. Nothing is fetched when not even the first new picture
alone would fit.

Stamps. The passages whose pictures were shown are stamped ``VISUAL_IMAGES_ATTACHED_KEY`` on the
numbered copies the turn's result returns, as :class:`~docint.core.retrieval.visual.VisualImagesMixin`
stamps its nodes, so the client can say the model saw them; the ledger's passages are never written.

Refusal. An endpoint may refuse a request that carries pictures: it caps how many it accepts, or
has no vision tower. :meth:`TurnImages.drop` then takes every picture back, its message, stamp and
room, and the turn shows no more, as ``VisualImagesMixin._drop_attached_images`` gives them up.

Nothing here logs a caption, a thumbnail or a passage: only exception types.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llama_index.core.base.llms.types import ChatMessage, ImageBlock, MessageRole, TextBlock
from llama_index.core.schema import NodeWithScore
from loguru import logger

from docint.agents.turn.budget import message_cost
from docint.core.retrieval.visual import (
    VISUAL_IMAGE_TOKEN_ESTIMATE,
    VISUAL_IMAGES_ATTACHED_KEY,
    ThumbnailFetcher,
    render_legend,
)


def _picture(scored: NodeWithScore) -> str:
    """Name the picture a passage shows, as ``select_answer_images`` reads it.

    Args:
        scored (NodeWithScore): A passage.

    Returns:
        str: Its ``image_id``; empty for a passage that shows no picture.
    """
    value = scored.node.metadata.get("image_id")
    return str(value) if value else ""


def _shows_pictures(message: ChatMessage) -> bool:
    """Say whether a message carries pictures.

    Args:
        message (ChatMessage): A message of the transcript.

    Returns:
        bool: Whether any of its blocks is a picture.
    """
    return any(isinstance(block, ImageBlock) for block in message.blocks)


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
            max_images (int): The most pictures the turn may show.
            round_room (int): The turn's room for tool replies (see
                :attr:`~docint.agents.turn.budget.TurnBudget.round_room`), which caps the pictures too.
        """
        self._ratio = char_token_ratio
        self._fetch = fetch
        self._template = legend_template
        self._cap = max(0, min(max_images, round_room // VISUAL_IMAGE_TOKEN_ESTIMATE)) if fetch is not None else 0
        #: The pictures shown, by the number of the passage that shows each.
        self._shown: dict[int, str] = {}
        self._cost = 0
        self._dropped = False

    @property
    def count(self) -> int:
        """How many pictures the transcript carries.

        Returns:
            int: The pictures shown and not taken back.
        """
        return len(self._shown)

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

    def attach(self, entries: Sequence[tuple[int, NodeWithScore]], *, room: int) -> tuple[ChatMessage, int] | None:
        """Show a round's new pictures in one message, within the turn's cap and the room left.

        The pictures not shown before are fetched at once, unless the cheapest message, the first
        of them alone, would not fit. Those with a stored thumbnail are then added in number
        order while the cap holds and the message, priced as it is sent, fits the room left.

        Args:
            entries (Sequence[tuple[int, NodeWithScore]]): The passages the round admitted, under
                their numbers, in number order.
            room (int): Tokens the transcript may still take.

        Returns:
            tuple[ChatMessage, int] | None: The user message showing the pictures and what it
            costs; ``None`` when it shows none: the turn shows no picture or no more, none of the
            passages is a new picture, none fits, or the fetch failed.
        """
        free = self._cap - self.count
        seen = set(self._shown.values())
        new = [(number, scored) for number, scored in entries if _picture(scored) and _picture(scored) not in seen]
        # The cheapest message shows the first new picture alone: its legend line is the shortest.
        if self._fetch is None or self._dropped or free <= 0 or not new or self._price([(new[0][0], "", "")]) > room:
            return None
        try:
            stored = self._fetch([scored.node.node_id for _, scored in new])
        except Exception as exc:
            logger.warning("An agent round's thumbnails could not be fetched ({}); none is shown", type(exc).__name__)
            return None
        shown: list[tuple[int, str, str]] = []
        for number, scored in new:
            picture, thumbnail = _picture(scored), stored.get(scored.node.node_id)
            if thumbnail is None or picture in seen:
                continue
            # A later picture's number is no shorter, so once one does not fit, none does.
            if len(shown) == free or self._price([*shown, (number, *thumbnail)]) > room:
                break
            seen.add(picture)
            self._shown[number] = picture
            shown.append((number, *thumbnail))
        if not shown:
            return None
        blocks: list[Any] = [TextBlock(text=render_legend(self._template, shown))]
        blocks.extend(ImageBlock(url=f"data:{mime};base64,{data}") for _, mime, data in shown)
        cost = self._price(shown)
        self._cost += cost
        return ChatMessage(role=MessageRole.USER, blocks=blocks), cost

    def drop(self, transcript: Sequence[ChatMessage]) -> tuple[list[ChatMessage], int]:
        """Take every picture back, its message, stamp and room, and show no more this turn.

        Args:
            transcript (Sequence[ChatMessage]): The turn's rounds so far; never changed.

        Returns:
            tuple[list[ChatMessage], int]: The transcript without its picture messages, and the
            tokens they took.
        """
        freed, self._cost = self._cost, 0
        self._shown.clear()
        self._dropped = True
        return [message for message in transcript if not _shows_pictures(message)], freed

    def stamp(self, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """Stamp the passages whose pictures the model was shown.

        Args:
            nodes (list[NodeWithScore]): The turn's numbered copies, in number order, as the
                result returns them; the ledger's own passages are never written.

        Returns:
            list[NodeWithScore]: The same copies, those shown stamped ``VISUAL_IMAGES_ATTACHED_KEY``.
        """
        for number, scored in enumerate(nodes, start=1):
            if number in self._shown:
                scored.node.metadata[VISUAL_IMAGES_ATTACHED_KEY] = True
        return nodes
