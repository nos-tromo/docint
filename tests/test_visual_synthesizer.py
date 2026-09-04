"""Tests for putting the retrieved imagery in front of the model.

Every other retrieval path answers from text, so a picture is only ever as
useful as the caption written at ingest time. The visual target attaches the
stored thumbnails to the same synthesis call — and must do so without letting
those pixels leak onto a node, into ``sources``, or into a persisted turn.
"""

from typing import Any, cast

import pytest
from llama_index.core.base.llms.types import ChatMessage, ImageBlock, MessageRole, TextBlock
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.llms.openai.utils import to_openai_message_dict
from typing_extensions import override

from docint.core.retrieval.visual import (
    VISUAL_IMAGE_TOKEN_ESTIMATE,
    VISUAL_IMAGES_ATTACHED_KEY,
    VisualImagesMixin,
    citation_index_of,
    image_token_reserve,
    render_legend,
    select_answer_images,
)

THUMBNAIL_B64 = "QUJDREVG"


class _Prompt:
    """Prompt-template stand-in."""


class _Program:
    """Refine-program stand-in carrying a prompt."""

    def __init__(self) -> None:
        """Initialize the stand-in."""
        self._prompt = _Prompt()


class _LLM:
    """LLM stand-in recording the messages it was handed."""

    def __init__(self) -> None:
        """Initialize the stand-in."""
        self.seen: list[list[ChatMessage]] = []

    def _get_messages(self, _prompt: Any, **_kwargs: Any) -> list[ChatMessage]:
        """Render the prompt as one user message.

        Args:
            _prompt (Any): Unused.
            **_kwargs (Any): Unused.

        Returns:
            list[ChatMessage]: One user message.
        """
        return [ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text="context and question")])]

    def chat(self, messages: list[ChatMessage]) -> Any:
        """Record the messages and answer.

        Args:
            messages (list[ChatMessage]): The rendered messages.

        Returns:
            Any: A response carrying a fixed answer.
        """
        self.seen.append(messages)
        return type("R", (), {"message": ChatMessage(role=MessageRole.ASSISTANT, content="answer")})()

    def stream_chat(self, messages: list[ChatMessage]) -> Any:
        """Record the messages and stream one delta.

        Args:
            messages (list[ChatMessage]): The rendered messages.

        Returns:
            Any: A generator of chat responses.
        """
        self.seen.append(messages)

        def _gen() -> Any:
            """Yield one streamed chunk.

            Yields:
                Any: A chat response carrying a delta.
            """
            yield type(
                "C",
                (),
                {"message": ChatMessage(role=MessageRole.ASSISTANT, content="answer"), "delta": "answer"},
            )()

        return _gen()


class _Upstream:
    """Synthesizer stand-in recording whether upstream was used."""

    def __init__(self) -> None:
        """Initialize the stand-in."""
        self.upstream_calls = 0

    def _update_response(self, program: Any, program_kwargs: dict[str, Any], response_kwargs: dict[str, Any]) -> Any:
        """Answer the way the non-visual synthesizer would.

        Args:
            program (Any): Unused.
            program_kwargs (dict[str, Any]): Unused.
            response_kwargs (dict[str, Any]): Unused.

        Returns:
            Any: A marker answer.
        """
        self.upstream_calls += 1
        return "upstream"

    def synthesize(self, query: Any, nodes: Any, *args: Any, **kwargs: Any) -> Any:
        """Record an upstream synthesis.

        Args:
            query (Any): Unused.
            nodes (Any): Unused.
            *args (Any): Unused.
            **kwargs (Any): Unused.

        Returns:
            Any: A marker response.
        """
        self.upstream_calls += 1
        return "synthesized"

    async def asynthesize(self, query: Any, nodes: Any, *args: Any, **kwargs: Any) -> Any:
        """Record an upstream async synthesis.

        Args:
            query (Any): Unused.
            nodes (Any): Unused.
            *args (Any): Unused.
            **kwargs (Any): Unused.

        Returns:
            Any: A marker response.
        """
        self.upstream_calls += 1
        return "synthesized"

    async def _aupdate_response(
        self,
        program: Any,
        program_kwargs: dict[str, Any],
        response_kwargs: dict[str, Any],
    ) -> Any:
        """Answer the way the non-visual synthesizer would, asynchronously.

        Args:
            program (Any): Unused.
            program_kwargs (dict[str, Any]): Unused.
            response_kwargs (dict[str, Any]): Unused.

        Returns:
            Any: A marker answer.
        """
        self.upstream_calls += 1
        return "upstream"


class _Synth(VisualImagesMixin, _Upstream):
    """The mixin over a recording stand-in."""

    def __init__(
        self,
        *,
        streaming: bool = False,
        max_images: int = 6,
        thumbnails: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the synthesizer.

        Args:
            streaming (bool): Whether to stream.
            max_images (int): Attachment cap.
            thumbnails (dict[str, Any] | None): Thumbnails to serve.
        """
        super().__init__()
        self._llm = _LLM()
        self._streaming = streaming
        self._structured_answer_filtering = False
        self._output_cls: type | None = None
        self._max_images = max_images
        self._legend_template = "Attached:\n{image_legend}"
        self._attached: list[tuple[int, str, str]] = []
        served = {"point-1": ("image/jpeg", THUMBNAIL_B64)} if thumbnails is None else thumbnails
        self._fetch_thumbnails = lambda ids: {key: served[key] for key in ids if key in served}


def _image_node(point_id: str, *, citation: int | None = None) -> NodeWithScore:
    """Build an image source node.

    Args:
        point_id (str): The companion point id.
        citation (int | None): Citation number to stamp.

    Returns:
        NodeWithScore: The node.
    """
    metadata: dict[str, Any] = {"image_id": f"img-{point_id}"}
    if citation is not None:
        metadata["citation_index"] = citation
    return NodeWithScore(node=TextNode(id_=point_id, text="a caption", metadata=metadata), score=0.3)


def _text_node() -> NodeWithScore:
    """Build a text source node.

    Returns:
        NodeWithScore: The node.
    """
    return NodeWithScore(node=TextNode(id_="chunk-1", text="prose", metadata={"file_hash": "abc"}), score=0.5)


def test_only_image_sources_are_attached() -> None:
    """A text chunk has no pixels to show."""
    selected = select_answer_images([_text_node(), _image_node("point-1")], max_images=6)

    assert [node.node.node_id for node in selected] == ["point-1"]


def test_selection_stops_at_the_configured_cap() -> None:
    """Each attached image costs prompt budget, so the cap is a real limit."""
    nodes = [_image_node(f"point-{index}") for index in range(10)]

    assert len(select_answer_images(nodes, max_images=3)) == 3


def test_synthesize_attaches_the_thumbnails_in_citation_order() -> None:
    """The model must be able to say which picture it is describing."""
    synth = _Synth(thumbnails={"point-1": ("image/jpeg", THUMBNAIL_B64), "point-2": ("image/png", THUMBNAIL_B64)})

    synth.synthesize(None, [_image_node("point-1", citation=4), _image_node("point-2", citation=7)])

    assert [citation for citation, _, _ in synth._attached] == [4, 7]


def test_attached_nodes_are_stamped_for_the_response() -> None:
    """A degraded visual turn must be reportable, so the stamp is on the node."""
    synth = _Synth()
    node = _image_node("point-1")

    synth.synthesize(None, [node])

    assert node.node.metadata[VISUAL_IMAGES_ATTACHED_KEY] is True


def test_a_thumbnail_outage_answers_from_captions_alone() -> None:
    """A dead companion read degrades the answer; it never fails the turn."""

    def _boom(_ids: Any) -> dict[str, Any]:
        """Fail the way an unreachable Qdrant does.

        Args:
            _ids (Any): Unused.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("qdrant unreachable")

    synth = _Synth()
    synth._fetch_thumbnails = _boom
    node = _image_node("point-1")

    synth.synthesize(None, [node])

    assert synth._attached == []
    assert VISUAL_IMAGES_ATTACHED_KEY not in node.node.metadata


def test_a_source_without_a_stored_thumbnail_is_not_attached() -> None:
    """Only the pictures actually shown may be reported as shown."""
    synth = _Synth(thumbnails={})

    synth.synthesize(None, [_image_node("point-1")])

    assert synth._attached == []


def test_non_streaming_answers_through_chat_with_image_blocks() -> None:
    """The pixels ride the same synthesis call as the captions."""
    synth = _Synth(streaming=False)
    synth.synthesize(None, [_image_node("point-1")])

    answer = synth._update_response(_Program(), {}, {})

    assert answer == "answer"
    blocks = synth._llm.seen[0][-1].blocks
    assert sum(isinstance(block, ImageBlock) for block in blocks) == 1


def test_streaming_answers_through_stream_chat_with_image_blocks() -> None:
    """Streaming is unchanged by the attachment; tokens still flow."""
    synth = _Synth(streaming=True)
    synth.synthesize(None, [_image_node("point-1")])

    tokens = list(synth._update_response(_Program(), {}, {}))

    assert "".join(tokens) == "answer"
    assert any(isinstance(block, ImageBlock) for block in synth._llm.seen[0][-1].blocks)


def test_image_blocks_render_as_data_uri_image_url_parts() -> None:
    """This is the wire format vLLM accepts for an image."""
    synth = _Synth()
    synth.synthesize(None, [_image_node("point-1")])
    synth._update_response(_Program(), {}, {})

    rendered = to_openai_message_dict(synth._llm.seen[0][-1])

    content = cast(list[dict[str, Any]], rendered["content"])
    parts = [part for part in content if part.get("type") == "image_url"]
    assert parts[0]["image_url"]["url"] == f"data:image/jpeg;base64,{THUMBNAIL_B64}"


def test_a_turn_with_nothing_attached_defers_to_upstream() -> None:
    """No pixels means the ordinary text synthesis, not a second code path."""
    synth = _Synth(thumbnails={})
    synth.synthesize(None, [_image_node("point-1")])

    assert synth._update_response(_Program(), {}, {}) == "upstream"
    assert synth.upstream_calls == 2


@pytest.mark.anyio
async def test_async_answers_through_achat_with_image_blocks() -> None:
    """The async path attaches the same way the sync one does."""
    synth = _Synth()

    async def _achat(messages: list[ChatMessage]) -> Any:
        """Record the messages and answer.

        Args:
            messages (list[ChatMessage]): The rendered messages.

        Returns:
            Any: A response carrying a fixed answer.
        """
        synth._llm.seen.append(messages)
        return type("R", (), {"message": ChatMessage(role=MessageRole.ASSISTANT, content="answer")})()

    synth._llm.achat = _achat  # type: ignore[attr-defined]
    await synth.asynthesize(None, [_image_node("point-1")])

    assert await synth._aupdate_response(_Program(), {}, {}) == "answer"
    assert any(isinstance(block, ImageBlock) for block in synth._llm.seen[0][-1].blocks)


def test_the_legend_maps_images_to_citation_numbers() -> None:
    """Without it the model sees pictures it cannot cite."""
    legend = render_legend("Attached:\n{image_legend}", [(3, "image/jpeg", "x"), (7, "image/jpeg", "y")])

    assert legend == "Attached:\nImage 1: source [3]\nImage 2: source [7]"


def test_the_legend_is_empty_when_nothing_is_attached() -> None:
    """A legend for no images would describe evidence that is not there."""
    assert render_legend("Attached:\n{image_legend}", []) == ""


def test_a_node_without_a_citation_number_falls_back_to_its_position() -> None:
    """Numbering runs before synthesis, but the answer must never be unciteable."""
    assert citation_index_of(_image_node("point-1"), 2) == 2


def test_the_image_budget_covers_every_attachment_and_the_legend() -> None:
    """The caption context has to fit beside the pixels, not instead of them."""
    assert image_token_reserve(3) > 3 * VISUAL_IMAGE_TOKEN_ESTIMATE


class _RefusingLLM(_LLM):
    """LLM stand-in that refuses any prompt carrying images.

    The shape a deployment produces when the endpoint's own images-per-prompt
    cap (vLLM's ``--limit-mm-per-prompt``) is lower than
    ``VISUAL_ANSWER_MAX_IMAGES``.
    """

    @staticmethod
    def _refuse_if_images(messages: list[ChatMessage]) -> None:
        """Raise the way the endpoint does when images are attached.

        Args:
            messages (list[ChatMessage]): The rendered messages.

        Raises:
            ValueError: When any message carries an image block.
        """
        if any(isinstance(block, ImageBlock) for message in messages for block in (message.blocks or [])):
            raise ValueError("At most 1 image(s) may be provided in one prompt.")

    @override
    def chat(self, messages: list[ChatMessage]) -> Any:
        """Refuse an image prompt, else answer.

        Args:
            messages (list[ChatMessage]): The rendered messages.

        Returns:
            Any: The upstream stand-in's answer.
        """
        self._refuse_if_images(messages)
        return super().chat(messages)

    @override
    def stream_chat(self, messages: list[ChatMessage]) -> Any:
        """Refuse an image prompt on the first pull, else stream.

        A streaming client sends the request when the generator is first
        advanced, not when it is created, which is where the refusal lands.

        Args:
            messages (list[ChatMessage]): The rendered messages.

        Returns:
            Any: A generator of chat responses.
        """
        streamed = super().stream_chat(messages)

        def _gen() -> Any:
            """Refuse, then yield what the parent would.

            Yields:
                Any: A chat response carrying a delta.
            """
            self._refuse_if_images(messages)
            yield from streamed

        return _gen()


def test_a_refused_attachment_answers_from_captions_instead_of_failing() -> None:
    """A model that will not take the pictures must not lose the answer.

    The endpoint's images-per-prompt cap is set independently of
    ``VISUAL_ANSWER_MAX_IMAGES``, so the two can disagree on any deployment.
    Failing the turn would throw away an answer the captions could carry.
    """
    synth = _Synth()
    synth._llm = _RefusingLLM()
    node = _image_node("point-1")
    synth.synthesize(None, [node])
    assert synth._attached

    answer = synth._update_response(_Program(), {}, {})

    assert answer == "upstream"
    # The turn reports what the model saw, which is now nothing.
    assert synth._attached == []
    assert VISUAL_IMAGES_ATTACHED_KEY not in node.node.metadata


def test_a_refused_attachment_degrades_the_stream_too() -> None:
    """The streaming path refuses on the first pull, not on the call."""
    synth = _Synth(streaming=True)
    synth._llm = _RefusingLLM()
    node = _image_node("point-1")
    synth.synthesize(None, [node])

    result = synth._update_response(_Program(), {}, {})

    assert result == "upstream"
    assert synth._attached == []
    assert VISUAL_IMAGES_ATTACHED_KEY not in node.node.metadata


def test_a_streamed_answer_keeps_its_first_token() -> None:
    """The token pulled to detect a refusal must still reach the reader."""
    synth = _Synth(streaming=True)
    synth.synthesize(None, [_image_node("point-1")])

    tokens = list(synth._update_response(_Program(), {}, {}))

    assert "".join(tokens) == "answer"


@pytest.mark.anyio
async def test_a_refused_attachment_degrades_the_async_path_too() -> None:
    """The async path takes the same degradation as the sync one."""
    synth = _Synth()

    async def _achat(_messages: list[ChatMessage]) -> Any:
        """Refuse the way the endpoint does.

        Args:
            _messages (list[ChatMessage]): Unused.

        Raises:
            ValueError: Always.
        """
        raise ValueError("At most 1 image(s) may be provided in one prompt.")

    synth._llm = cast(Any, type("L", (_LLM,), {"achat": staticmethod(_achat)})())
    node = _image_node("point-1")
    await synth.asynthesize(None, [node])

    answer = await synth._aupdate_response(_Program(), {}, {})

    assert answer == "upstream"
    assert synth._attached == []
    assert VISUAL_IMAGES_ATTACHED_KEY not in node.node.metadata
