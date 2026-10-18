"""The RAG side of an agent turn: the model it runs on, the view of its passages, and the turn the factory builds.

Every message, passage, clip, filter and collection here is invented.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock

import httpx
import openai
import pytest
from agent_fakes import (
    CLIP,
    chunk,
    programme_passage,
    streaming_tool_llm,
    text_chunks,
    tool_call,
    tool_call_delta,
    turn_budget,
)
from assessment_fakes import scored_node
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import BaseNode, MetadataMode, NodeWithScore, TextNode

from docint.agents.assessment.ledger import EvidenceLedger, render_context, stamp
from docint.agents.assessment.split import SENTINEL
from docint.agents.turn.budget import MESSAGE_OVERHEAD_TOKENS, schema_tokens
from docint.agents.turn.calls import SEARCH_THREAD_NAME
from docint.agents.turn.history import NO_HISTORY, ConversationTurn, ConversationWindow
from docint.agents.turn.loop import AgentResult
from docint.agents.turn.tools import ReadTool, SearchTool, ToolContext, ToolRegistry
from docint.agents.turn.toolsets import toolset_for
from docint.core.rag import (
    AGENT_LEDGER_CAP,
    AGENT_METADATA_VALUE_CHARS,
    RAG,
    AgentTurn,
    ParentContextPostprocessor,
    agent_llm_view,
)
from docint.core.retrieval.evidence import EveryChildQuery
from docint.core.retrieval.visual import RETRIEVAL_TARGETS, RetrievalTarget
from docint.core.retrieval_filters import build_qdrant_filter
from docint.utils.embed_chunking import estimate_tokens
from docint.utils.env_cfg import OpenAIConfig
from docint.utils.prompt_loader import load_localized_prompt

MESSAGE = "Wann öffnen die Stände beim Hafenfest, und hat jemand das Wetter am Sonntag erwähnt?"
#: The physical name a request's collection scope binds.
PHYSICAL = "u0123456789ab__hafen"
#: An invented request filter: video frames only.
RULES = [{"field": "source_type", "operator": "eq", "value": "video_keyframe"}]
#: The post an invented comment answers, as a social export repeats it on the comment's row.
ANCHOR = ("Der Hafenverein verschiebt das Hafenfest wegen der Sturmwarnung auf Sonntag. " * 25)[:1500]
SOCIAL_FILE = "example_profile/dossier.json"
AUTHOR = "Person A"
TIMESTAMP = "2026-03-14T09:31:00Z"
#: Invented social texts: a post and a comment of the length a real one can run to, and short ones.
LONG_POST = ("Der Hafenverein lädt zum Hafenfest am Sonntag ein, mit Ständen, Musik und einer Regatta. " * 20)[:1500]
LONG_COMMENT = ("Weiß jemand, ob die Regatta trotzdem stattfindet oder ob sie wegen des Winds ausfällt? " * 20)[:1500]
SHORT_POST = "Das Hafenfest beginnt am Sonntag um zehn Uhr am Kai."
SHORT_COMMENT = "Danke für die Info, wir kommen dann am Sonntag mit den Kindern vorbei."


def _rag(
    monkeypatch: pytest.MonkeyPatch,
    *,
    code: str = "en",
    ctx_window: int = 8192,
    provider: str = "vllm",
    thinking: bool = False,
    max_retries: int = 0,
) -> RAG:
    """Build a RAG whose chat model targets an OpenAI-compatible endpoint, its prompts in one locale.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pins the locale before the engine reads it.
        code (str): The locale the prompts load from.
        ctx_window (int): The chat model's context window in tokens.
        provider (str): ``INFERENCE_PROVIDER``.
        thinking (bool): ``OPENAI_ENABLE_THINKING``, the default when no request toggle is bound.
        max_retries (int): ``OPENAI_MAX_RETRIES``; two is the production default.

    Returns:
        RAG: The engine, with only the OpenAI knobs, the ratio and the locale set.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", code)
    rag = RAG(qdrant_collection="hafen")
    rag.text_model_id = "invented-model"
    rag.openai_config = OpenAIConfig(
        api_base="http://vllm-router:4000/v1",
        api_key="sk-test",
        ctx_window=ctx_window,
        dimensions=1024,
        max_retries=max_retries,
        num_output=256,
        inference_provider=provider,
        reuse_client=False,
        seed=42,
        temperature=0.0,
        thinking_effort="medium",
        thinking_enabled=thinking,
        timeout=5.0,
        top_p=0.0,
    )
    for key in (
        "api_base",
        "api_key",
        "ctx_window",
        "max_retries",
        "inference_provider",
        "reuse_client",
        "seed",
        "temperature",
        "timeout",
        "top_p",
    ):
        setattr(rag, f"openai_{key}", getattr(rag.openai_config, key))
    rag.embed_char_token_ratio = 3.5
    return rag


def _stalled_endpoint(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Make every request time out at the HTTP transport, recording each attempt that reaches it.

    The SDK's own retries and llama-index's both sit above the transport, so the count is
    every attempt a call makes. Backoff sleeps are skipped.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.

    Returns:
        list[httpx.Request]: The attempts, appended to as they happen.
    """
    attempts: list[httpx.Request] = []

    def _stall(_transport: Any, request: httpx.Request) -> httpx.Response:
        """Record one attempt and time it out.

        Args:
            _transport (Any): The transport, unused.
            request (httpx.Request): The attempt.

        Raises:
            httpx.ReadTimeout: Always.
        """
        attempts.append(request)
        raise httpx.ReadTimeout("The endpoint stalled.", request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _stall)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    return attempts


def test_the_agent_model_follows_the_reasoning_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    """The toggle picks the reasoning model or the plain one, each built once; the reasoning one asks to think.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    rag = _rag(monkeypatch)

    with rag.reasoning_scope(True):
        thinking = rag.agent_text_model
        again = rag.agent_text_model
    with rag.reasoning_scope(False):
        plain = rag.agent_text_model

    assert again is thinking
    assert plain is rag.bounded_text_model
    assert thinking is not plain
    assert thinking is not rag.text_model
    assert thinking.reasoning_effort == "medium"
    assert thinking.additional_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert plain.additional_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.parametrize("thinking", [False, True])
def test_without_a_toggle_the_agent_model_follows_the_process_default(
    monkeypatch: pytest.MonkeyPatch, thinking: bool
) -> None:
    """With no request toggle bound, ``OPENAI_ENABLE_THINKING`` decides, as it does for the plain answer.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        thinking (bool): The process default.
    """
    rag = _rag(monkeypatch, thinking=thinking)

    assert (rag.agent_text_model is not rag.bounded_text_model) is thinking
    assert rag.reasoning_requested is thinking
    assert (rag.post_retrieval_text_model is not rag.text_model) is thinking


@pytest.mark.parametrize("reasoning", [False, True])
def test_the_agent_model_makes_one_attempt_per_call(monkeypatch: pytest.MonkeyPatch, reasoning: bool) -> None:
    """Retries are off for both kinds: the loop's time budget and model slots assume a call is one attempt.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        reasoning (bool): The request's toggle.
    """
    attempts = _stalled_endpoint(monkeypatch)
    rag = _rag(monkeypatch, max_retries=2)

    with rag.reasoning_scope(reasoning), pytest.raises(openai.APITimeoutError):
        next(iter(rag.agent_text_model.stream_chat([ChatMessage(role=MessageRole.USER, content=MESSAGE)])))

    assert len(attempts) == 1


def _comment_row() -> NodeWithScore:
    """Build an invented social comment row: the 1,500-character post it answers, then its author, then its file.

    Returns:
        NodeWithScore: The row, with a hidden description as long as the post.
    """
    metadata = {
        "file_hash": "f" * 64,
        "source": "table",
        "llm_description": ANCHOR,
        "reference_metadata": {"type": "comment", "anchor_text": ANCHOR, "author": AUTHOR},
        "filename": SOCIAL_FILE,
    }
    text = "Danke für die Info, wir kommen am Sonntag mit den Kindern vorbei."
    return NodeWithScore(node=TextNode(id_="comment-4", text=text, metadata=metadata), score=1.0)


def _search_reply(view: Callable[[BaseNode], BaseNode], row: NodeWithScore | None = None) -> str:
    """Show a passage as an agent turn's search reply does, through a view of its passages.

    Args:
        view (Callable[[BaseNode], BaseNode]): The LLM view the turn renders passages through.
        row (NodeWithScore | None): The passage the search finds; the comment row by default.

    Returns:
        str: The reply the model reads.
    """
    found = _comment_row() if row is None else row
    registry = ToolRegistry([SearchTool(lambda _query: [found]), ReadTool()])
    ctx = ToolContext(ledger=EvidenceLedger(cap=8), budget=turn_budget(), llm_view=view)
    return registry.call(tool_call("search", query="Hafenfest Sonntag"), ctx).text


def test_the_agent_view_keeps_a_social_rows_author_and_file_in_its_header() -> None:
    """The post a comment answers pushes the chat view's header into compaction; cut short, it leaves both shown."""
    chat = _search_reply(ParentContextPostprocessor._emit_with_llm_exclusion)
    agent = _search_reply(agent_llm_view)

    assert AUTHOR not in chat
    assert SOCIAL_FILE not in chat
    assert AUTHOR in agent
    assert SOCIAL_FILE in agent


def test_the_agent_view_cuts_only_the_visible_strings_of_a_clone() -> None:
    """Every LLM-visible string is cut to its limit with an ellipsis; hidden keys, the text and the stored node stay."""
    row = _comment_row()

    view = agent_llm_view(row.node)

    cut = view.metadata["reference_metadata"]["anchor_text"]
    assert len(cut) == AGENT_METADATA_VALUE_CHARS == 200
    assert cut.endswith("…")
    assert ANCHOR.startswith(cut[:-1])
    assert view.metadata["reference_metadata"]["author"] == AUTHOR
    assert view.metadata["filename"] == SOCIAL_FILE
    chat = ParentContextPostprocessor._emit_with_llm_exclusion(row.node)
    assert view.metadata["llm_description"] == chat.metadata["llm_description"]
    assert view.excluded_llm_metadata_keys == chat.excluded_llm_metadata_keys
    assert view.node_id == row.node.node_id
    assert view.get_content(metadata_mode=MetadataMode.NONE) == row.node.get_content(metadata_mode=MetadataMode.NONE)
    assert row.node.metadata["reference_metadata"]["anchor_text"] == ANCHOR
    assert row.node.metadata["llm_description"] == ANCHOR


def test_the_agent_view_keeps_a_string_at_the_limit_and_cuts_one_in_a_list() -> None:
    """A string of exactly the limit stays whole; one a character longer, or one inside a list, is cut."""
    at_limit = "a" * AGENT_METADATA_VALUE_CHARS
    metadata = {
        "filename": at_limit,
        "reference_metadata": {"author": "b" * (AGENT_METADATA_VALUE_CHARS + 1), "tags": ["c" * 300, "Hafen"]},
    }

    view = agent_llm_view(TextNode(id_="row-limit", text="Notiz zum Hafenfest.", metadata=metadata))

    reference = view.metadata["reference_metadata"]
    assert view.metadata["filename"] == at_limit
    assert reference["author"] == "b" * (AGENT_METADATA_VALUE_CHARS - 1) + "…"
    assert reference["tags"] == ["c" * (AGENT_METADATA_VALUE_CHARS - 1) + "…", "Hafen"]


def _dossier_row(kind: str, text: str, *, anchor: str | None = None, parent: str | None = None) -> NodeWithScore:
    """Build an invented social row as the dossier linker writes it: every reference key, ``None`` included.

    Args:
        kind (str): ``posting`` or ``comment``.
        text (str): The row's own text.
        anchor (str | None): The post a comment answers.
        parent (str | None): The comment a reply answers.

    Returns:
        NodeWithScore: The row.
    """
    reference = {
        "network": "examplenet",
        "type": kind,
        "uuid": "3f2b8c1e-9d4a-4c6b-8e21-5a7f0c9d2b14",
        "url": "https://social.invalid/p/9876543214",
        "timestamp": TIMESTAMP,
        "author": AUTHOR,
        "author_id": "1234567890",
        "vanity": "person_a",
        "text": text,
        "text_id": "9876543214",
        "anchor_text": anchor,
        "parent_text": parent,
    }
    metadata = {
        "file_path": f"uploads/{SOCIAL_FILE}",
        "file_name": SOCIAL_FILE,
        "filename": SOCIAL_FILE,
        "file_hash": "f" * 64,
        "origin": {"filename": SOCIAL_FILE, "filetype": "application/json"},
        "source": "table",
        "table": {"style": f"{kind}s", "row_index": 4, "n_rows": 120},
        "reference_metadata": reference,
    }
    return NodeWithScore(node=TextNode(id_=f"{kind}-4", text=text, metadata=metadata), score=1.0)


@pytest.mark.parametrize(
    "row",
    [
        _dossier_row("posting", SHORT_POST),
        _dossier_row("posting", LONG_POST),
        _dossier_row("comment", SHORT_COMMENT, anchor=LONG_POST),
        _dossier_row("comment", SHORT_COMMENT, anchor=LONG_POST, parent=LONG_COMMENT),
    ],
    ids=["short-posting", "long-posting", "comment", "reply"],
)
def test_a_social_rows_header_names_who_wrote_it_when_and_where(row: NodeWithScore) -> None:
    """Whether its header shows whole or is compacted, a social row opens with its file and its author's identity.

    Args:
        row (NodeWithScore): An invented dossier row.
    """
    reply = _search_reply(agent_llm_view, row)

    assert reply.startswith(f"citation_index: 1\nfilename: {SOCIAL_FILE}\n")
    assert AUTHOR in reply
    assert TIMESTAMP in reply


def _social_frame(rag: RAG) -> NodeWithScore:
    """Build a frame of an invented social clip as the image companion returns it, its post's long text included.

    Args:
        rag (RAG): The engine whose caption-node builder makes the frame.

    Returns:
        NodeWithScore: The frame, labelled ``#3 @ 00:12``.
    """
    link = {
        "posting_uuid": "3f2b8c1e-9d4a-4c6b-8e21-5a7f0c9d2b14",
        "posting_id": "1234567890123456",
        "media_id": "9876543210987654",
    }
    media = {"source_file": CLIP, "media_file_hash": "c" * 64}
    posting = {
        "posting_network": "examplenet",
        "posting_author": AUTHOR,
        "posting_author_id": "1234567890",
        "posting_vanity": "person_a",
        "posting_timestamp": TIMESTAMP,
        "posting_url": "https://social.invalid/p/1234567890123456",
        "posting_text": LONG_POST,
    }
    payload = {
        "image_id": f"{CLIP}-image-2",
        **link,
        "source_type": "social_media",
        **media,
        "source_path": f"example_profile/{CLIP}",
        **posting,
        "reference_metadata": {"type": "keyframe", **link, **media, **posting},
        "keyframe_index": 2,
        "keyframe_time_sec": 12.0,
        "llm_description": "Zwei Personen in Regenjacken am Hafen, bedeckter Himmel",
        "node_id": "social-frame-3",
    }
    return rag._image_caption_node(payload, RAG._image_caption_text(payload))


def test_a_social_frames_header_keeps_its_label_then_its_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """The frame's label still leads, the clip it came from follows, and its compacted header names who posted it, when.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    reply = _search_reply(agent_llm_view, _social_frame(_rag(monkeypatch)))

    assert reply.startswith(f"citation_index: 1\nkeyframe: #3 @ 00:12\nfilename: {CLIP}\n")
    assert "characters not shown" in reply.partition("\n\n")[0]
    assert AUTHOR in reply
    assert TIMESTAMP in reply


def _social_segment() -> NodeWithScore:
    """Build a transcript segment of an invented social clip as the transcript reader writes it.

    Its reference block holds the segment's own identity, then its text, then the post the
    clip hangs off, merged in the reader's order.

    Returns:
        NodeWithScore: The segment, the clip's fourth, spoken twelve seconds in.
    """
    link = {
        "posting_uuid": "3f2b8c1e-9d4a-4c6b-8e21-5a7f0c9d2b14",
        "posting_id": "1234567890123456",
        "media_id": "9876543210987654",
    }
    posting = {
        "posting_network": "examplenet",
        "posting_author": AUTHOR,
        "posting_author_id": "1234567890",
        "posting_vanity": "person_a",
        "posting_timestamp": TIMESTAMP,
        "posting_url": "https://social.invalid/p/1234567890123456",
        "posting_text": LONG_POST,
    }
    text = "Wir stehen hier am Hafen, und gleich fängt es an zu regnen."
    reference = {
        "type": "transcript_segment",
        "network": "nextext",
        "author": "SPEAKER_01",
        "text": text,
        "text_id": f"{CLIP}:3",
        "timestamp": "00:00:12",
        "language": "de",
        "source_file": CLIP,
        "speaker": "SPEAKER_01",
        **link,
        **posting,
    }
    metadata = {
        "file_path": f"example_profile/{CLIP}",
        "file_name": CLIP,
        "filename": CLIP,
        "file_type": "application/json",
        "mimetype": "application/json",
        "source": "transcript",
        "origin": {"filename": f"{CLIP}.nextext.jsonl", "mimetype": "application/json"},
        **link,
        "source_file": CLIP,
        "media_file_hash": "c" * 64,
        **posting,
        "file_hash": "d" * 64,
        "sentence_index": 3,
        "docint_doc_kind": "transcript_segment",
        "start_seconds": 12.0,
        "end_seconds": 15.5,
        "start_ts": "00:00:12",
        "end_ts": "00:00:15",
        "speaker": "SPEAKER_01",
        "reference_metadata": reference,
    }
    return NodeWithScore(node=TextNode(id_="segment-3", text=text, metadata=metadata), score=1.0)


def test_a_social_clips_segment_names_who_posted_the_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """A segment's compacted header keeps its own start, then the post's author, ahead of the post's long text.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    reply = _search_reply(agent_llm_view, _social_segment())

    assert reply.startswith(f"citation_index: 1\nfilename: {CLIP}\n")
    assert "characters not shown" in reply.partition("\n\n")[0]
    assert "00:00:12" in reply
    assert AUTHOR in reply


@pytest.mark.parametrize("passage", ["frame", "segment"])
def test_a_reference_block_leads_with_its_own_identity_then_its_posts(
    monkeypatch: pytest.MonkeyPatch, passage: str
) -> None:
    """A row's own identity comes first, then the post it hangs off, then the rest in the order it was written.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        passage (str): A social clip's ``frame`` or transcript ``segment``.
    """
    node = _social_frame(_rag(monkeypatch)) if passage == "frame" else _social_segment()
    posts = ["posting_author", "posting_author_id", "posting_vanity", "posting_timestamp", "posting_network"]
    expected = {
        "frame": [
            "type",
            *posts,
            "posting_uuid",
            "posting_id",
            "media_id",
            "source_file",
            "media_file_hash",
            "posting_url",
            "posting_text",
        ],
        "segment": [
            "type",
            "author",
            "timestamp",
            "network",
            "text_id",
            *posts,
            "text",
            "language",
            "source_file",
            "speaker",
            "posting_uuid",
            "posting_id",
            "media_id",
            "posting_url",
            "posting_text",
        ],
    }[passage]

    view = agent_llm_view(node.node)

    assert list(view.metadata["reference_metadata"]) == expected


def _pdf_chunk() -> NodeWithScore:
    """Build an invented chunk of a PDF with the metadata the document reader stores.

    Returns:
        NodeWithScore: The chunk; its ``origin`` repeats its file name, as a real chunk's does.
    """
    name = "hafenfest_programm.pdf"
    metadata: dict[str, Any] = {
        "chunk_id": "programm-chunk-3",
        "point_id": "5d0c1f5e-2b8e-5c47-9a0e-3f1d2c4b6a71",
        "doc_id": "e" * 64,
        "file_hash": "e" * 64,
        "file_path": f"uploads/{name}",
        "file_name": name,
        "filename": name,
        "file_type": "application/pdf",
        "mimetype": "application/pdf",
        "source": "document",
        "origin": {"filename": name, "mimetype": "application/pdf", "file_hash": "e" * 64, "page_number": 2},
        "pipeline_name": "core_pipeline",
        "pipeline_version": "1",
        "page_range": [2],
        "block_ids": ["block-7"],
        "section_path": ["Programm"],
        "table_ids": [],
        "image_ids": [],
        "source_mix": "text",
        "bbox_refs": [],
        "page": 2,
        "page_number": 2,
    }
    text = "Die Stände öffnen am Sonntag um zehn Uhr, die Musik beginnt am Mittag."
    return NodeWithScore(node=TextNode(id_="programm-chunk-3", text=text, metadata=metadata), score=1.0)


def test_a_plain_pdf_chunk_renders_as_the_chat_prompt_renders_it() -> None:
    """A chunk with no reference block is shown exactly as before, in a search reply and in a pinned scope."""
    chunk = _pdf_chunk()
    chat = ParentContextPostprocessor._emit_with_llm_exclusion

    assert _search_reply(agent_llm_view, chunk) == _search_reply(chat, chunk)
    scoped = [
        render_context([stamp(chunk, 1)], llm_view=view, budget_tokens=10_000, char_token_ratio=1.0).context
        for view in (agent_llm_view, chat)
    ]
    assert scoped[0] == scoped[1]


@pytest.mark.parametrize("placement", ["first", "behind-hidden-keys", "last"])
def test_a_reference_row_leads_with_its_file_then_its_identity(placement: str) -> None:
    """The citation line keeps its place; the file and the reference block lead the rest, who and when first in it.

    Args:
        placement (str): Where the citation line sits: first, as a search reply numbers a passage; first of
            the shown keys behind hidden ones; or last, as the ledger stamps a pinned passage.
    """
    metadata = dict(_dossier_row("comment", SHORT_COMMENT, anchor=LONG_POST).node.metadata)
    stamped = {
        "first": {"citation_index": 1, **metadata},
        "behind-hidden-keys": {"file_hash": "f" * 64, "citation_index": 1, **metadata},
        "last": {**metadata, "citation_index": 1},
    }[placement]

    view = agent_llm_view(TextNode(id_="comment-4", text=SHORT_COMMENT, metadata=stamped))

    visible = [key for key in view.metadata if key not in view.excluded_llm_metadata_keys]
    rest = ["filename", "reference_metadata", "table"]
    assert visible == ([*rest, "citation_index"] if placement == "last" else ["citation_index", *rest])
    assert list(view.metadata["reference_metadata"]) == [
        "type",
        "author",
        "author_id",
        "vanity",
        "timestamp",
        "network",
        "text_id",
        "uuid",
        "url",
        "text",
        "anchor_text",
    ]


def test_the_agent_view_drops_visible_none_leaves_and_leaves_the_rest_in_order() -> None:
    """``None`` says nothing, so a visible one is dropped at any depth; hidden values and the order stay."""
    metadata = {
        "file_path": None,
        "filename": "hafenfest_programm.pdf",
        "page": None,
        "table": {"columns": ["Stand", None, "Zeit"], "row_index": None, "n_rows": 12},
    }

    view = agent_llm_view(TextNode(id_="row-none", text="Die Stände öffnen um zehn Uhr.", metadata=metadata))

    assert list(view.metadata) == ["file_path", "filename", "table"]
    assert view.metadata["file_path"] is None
    assert view.metadata["table"] == {"columns": ["Stand", "Zeit"], "n_rows": 12}


@pytest.mark.parametrize(
    ("origin_file", "filename", "kept"),
    [
        (SOCIAL_FILE, SOCIAL_FILE, False),
        ("other_profile/dossier.json", SOCIAL_FILE, True),
        (SOCIAL_FILE, None, True),
        (None, None, True),
    ],
    ids=["repeats-the-file", "names-another-file", "only-file-locator", "no-file-at-all"],
)
def test_a_reference_rows_origin_is_dropped_only_where_it_repeats_the_file(
    origin_file: str | None, filename: str | None, kept: bool
) -> None:
    """An ``origin`` that repeats the row's file goes; one that names another, or the only file the row names, stays.

    Args:
        origin_file (str | None): The file the row's ``origin`` names; ``None`` for one that names none.
        filename (str | None): The row's own ``filename``; ``None`` for a row without one.
        kept (bool): Whether the view keeps the ``origin``.
    """
    metadata: dict[str, Any] = {
        "origin": {"filename": origin_file, "filetype": "application/json"},
        "reference_metadata": {"type": "posting", "author": AUTHOR},
    }
    if filename is not None:
        metadata["filename"] = filename

    view = agent_llm_view(TextNode(id_="row-origin", text=SHORT_POST, metadata=metadata))

    assert ("origin" in view.metadata) is kept


def _engine(*found: NodeWithScore) -> MagicMock:
    """Build a turn engine double whose ``retrieve`` returns the passages given.

    Args:
        *found (NodeWithScore): What every retrieval returns.

    Returns:
        MagicMock: The engine; ``retrieve.call_args_list`` records each bundle.
    """
    engine = MagicMock()
    engine.retrieve.return_value = list(found)
    return engine


def _turn(
    rag: RAG,
    *,
    target: str = "all",
    scoped: bool = False,
    pinned_count: int = 0,
    window: ConversationWindow | None = None,
    message: str = MESSAGE,
    engine: Any = None,
    metadata_filter_rules: list[dict[str, Any]] | None = None,
    metadata_filters_active: bool = False,
    vector_store_kwargs: dict[str, Any] | None = None,
) -> AgentTurn:
    """Build an agent turn the budget has room for.

    Args:
        rag (RAG): The engine.
        target (str): The turn's retrieval target.
        scoped (bool): Whether the turn answers a pinned scope.
        pinned_count (int): How many distinct passages the scope pinned.
        window (ConversationWindow | None): The conversation before the message; empty by default.
        message (str): The user's message.
        engine (Any): The turn engine; one that finds nothing by default.
        metadata_filter_rules (list[dict[str, Any]] | None): The request's raw filter rules.
        metadata_filters_active (bool): Whether the request carries filters.
        vector_store_kwargs (dict[str, Any] | None): Native query kwargs, the compiled filter in them.

    Returns:
        AgentTurn: The turn.
    """
    turn = rag.agent_turn(
        _engine() if engine is None else engine,
        retrieval_target=cast(RetrievalTarget, target),
        scoped=scoped,
        pinned_count=pinned_count,
        window=ConversationWindow() if window is None else window,
        message=message,
        metadata_filter_rules=metadata_filter_rules,
        metadata_filters_active=metadata_filters_active,
        vector_store_kwargs=vector_store_kwargs,
    )
    assert turn is not None
    return turn


def _ctx(turn: AgentTurn) -> ToolContext:
    """Build the context the turn's tools answer in, as its loop does.

    Args:
        turn (AgentTurn): The turn.

    Returns:
        ToolContext: Its ledger, budget and the agent view.
    """
    return ToolContext(ledger=turn.ledger, budget=turn.budget, llm_view=agent_llm_view)


def _prompt(name: str, code: str) -> str:
    """Load a prompt from a locale, apart from the engine's own loading.

    Args:
        name (str): The prompt's file name without ``.txt``.
        code (str): The locale.

    Returns:
        str: The prompt, stripped.
    """
    return load_localized_prompt(name, default="", required=True, lang=code)


def _system_prompt(code: str, target: str, *, rule: bool, summary: str) -> str:
    """Render the system prompt a turn should be sent.

    Args:
        code (str): The locale.
        target (str): The retrieval target, which picks the tools guide.
        rule (bool): Whether the assessment rule is in it.
        summary (str): The window's summary block.

    Returns:
        str: The rendered prompt.
    """
    return _prompt("agent_system", code).format(
        tools_guide=_prompt(f"agent_tools_{target}", code),
        assessment_rule=_prompt("agent_assessment_rule", code) if rule else "",
        conversation_summary=summary,
    )


def _nothing(_key: str) -> list[NodeWithScore]:
    """Find nothing, for a tool set built only for its schemas.

    Args:
        _key (str): The query or clip, unused.

    Returns:
        list[NodeWithScore]: No passage.
    """
    return []


def _schemas(target: str) -> list[dict[str, Any]]:
    """Return the schemas a target's tools are offered with.

    Args:
        target (str): The retrieval target.

    Returns:
        list[dict[str, Any]]: The schemas, as the endpoint receives them.
    """
    return toolset_for(target, search=_nothing, search_images=_nothing, list_keyframes=_nothing).schemas()


@pytest.mark.parametrize("target", RETRIEVAL_TARGETS)
@pytest.mark.parametrize("code", ["en", "de"])
def test_the_real_prompts_leave_a_thinking_model_room_for_a_turn_at_8k(
    monkeypatch: pytest.MonkeyPatch, code: str, target: str
) -> None:
    """Measured on the real prompts, the assessment rule in, and the target's real schemas, 8k is enough.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        code (str): The locale.
        target (str): The retrieval target.
    """
    monkeypatch.setenv("ASSESSMENT_ENABLED", "true")
    rag = _rag(monkeypatch, code=code, ctx_window=8192)

    with rag.reasoning_scope(True):
        turn = rag.agent_turn(
            _engine(),
            retrieval_target=cast(RetrievalTarget, target),
            scoped=False,
            pinned_count=0,
            window=ConversationWindow(),
            message=MESSAGE,
            metadata_filter_rules=None,
            metadata_filters_active=False,
            vector_store_kwargs=None,
        )

    assert turn is not None
    budget = turn.budget
    assert budget.available, f"{code}/{target}: {budget.round_room} tokens of room for tool replies"
    assert (budget.ctx_window, budget.reasoning, budget.history_tokens) == (8192, True, 0)
    assert budget.system_tokens == estimate_tokens(_system_prompt(code, target, rule=True, summary=NO_HISTORY), 3.5)
    assert budget.tool_schema_tokens == schema_tokens(_schemas(target), 3.5)
    assert budget.message_tokens == estimate_tokens(MESSAGE, 3.5)


@pytest.mark.parametrize(
    ("ctx_window", "message"),
    [(4096, MESSAGE), (8192, "Bitte fasse jede Stellungnahme zum Hafenfest einzeln zusammen. " * 300)],
    ids=["small-window", "long-message"],
)
def test_a_budget_without_room_for_tool_replies_builds_no_turn(
    monkeypatch: pytest.MonkeyPatch, ctx_window: int, message: str
) -> None:
    """Too small a window, or a message that fills it, leaves no room: no turn, and nothing raised.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        ctx_window (int): The chat model's context window.
        message (str): The user's message.
    """
    rag = _rag(monkeypatch, ctx_window=ctx_window)

    turn = rag.agent_turn(
        _engine(),
        retrieval_target="all",
        scoped=False,
        pinned_count=0,
        window=ConversationWindow(),
        message=message,
        metadata_filter_rules=None,
        metadata_filters_active=False,
        vector_store_kwargs=None,
    )

    assert turn is None


@pytest.mark.parametrize("reasoning", [False, True])
def test_every_round_of_a_turn_runs_on_the_agent_model(monkeypatch: pytest.MonkeyPatch, reasoning: bool) -> None:
    """The turn holds the model the toggle picked when it was built, and says which.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        reasoning (bool): The request's toggle.
    """
    rag = _rag(monkeypatch)

    with rag.reasoning_scope(reasoning):
        turn = _turn(rag)
        model = rag.agent_text_model

    assert turn.loop._llm is model
    assert turn.reasoning is reasoning
    assert turn.budget.reasoning is reasoning


@pytest.mark.parametrize("reasoning", [False, True])
def test_a_turns_round_makes_one_attempt_at_a_stalled_endpoint(
    monkeypatch: pytest.MonkeyPatch, reasoning: bool
) -> None:
    """The loop's own calls go out once each, whatever ``OPENAI_MAX_RETRIES`` says.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        reasoning (bool): The request's toggle.
    """
    attempts = _stalled_endpoint(monkeypatch)
    rag = _rag(monkeypatch, max_retries=2)
    with rag.reasoning_scope(reasoning):
        turn = _turn(rag)

    with pytest.raises(openai.APITimeoutError):
        list(turn.loop.run(window=ConversationWindow(), message=MESSAGE, ledger=turn.ledger))

    assert len(attempts) == 1


@pytest.mark.parametrize(
    ("provider", "reasoning", "opened"),
    [("vllm", True, True), ("vllm", False, False), ("ollama", True, False), ("openai", True, False)],
)
def test_a_reply_may_open_in_reasoning_only_where_thinking_is_switched_on(
    monkeypatch: pytest.MonkeyPatch, provider: str, reasoning: bool, opened: bool
) -> None:
    """Only vLLM is sent ``enable_thinking``, so only there can a reply open inside a think block.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        provider (str): ``INFERENCE_PROVIDER``.
        reasoning (bool): The request's toggle.
        opened (bool): Whether the loop reads a reply as opening in reasoning.
    """
    rag = _rag(monkeypatch, provider=provider)

    with rag.reasoning_scope(reasoning):
        turn = _turn(rag)

    assert turn.loop._reasoning_open is opened


@pytest.mark.parametrize(("configured", "ratio"), [(3.0, 3.0), (0.0, 3.5)])
def test_the_budget_and_the_loop_measure_with_one_ratio(
    monkeypatch: pytest.MonkeyPatch, configured: float, ratio: float
) -> None:
    """The configured characters per token, 3.5 when unset, sizes the budget and the loop alike.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        configured (float): ``EMBED_CHAR_TOKEN_RATIO``.
        ratio (float): The ratio the turn is measured with.
    """
    rag = _rag(monkeypatch)
    rag.embed_char_token_ratio = configured

    turn = _turn(rag)

    assert rag.agent_char_token_ratio == ratio
    assert turn.budget.char_token_ratio == ratio
    assert turn.loop._ratio == ratio
    assert turn.budget.message_tokens == estimate_tokens(MESSAGE, ratio)


@pytest.mark.parametrize("assessment", [False, True])
@pytest.mark.parametrize("code", ["en", "de"])
def test_the_prompts_are_the_locales_with_the_targets_guide_and_the_windows_summary(
    monkeypatch: pytest.MonkeyPatch, code: str, assessment: bool
) -> None:
    """The system prompt carries the target's guide, the rule while assessments are on, and the summary last.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        code (str): The locale.
        assessment (bool): ``ASSESSMENT_ENABLED``.
    """
    monkeypatch.setenv("ASSESSMENT_ENABLED", str(assessment).lower())
    rag = _rag(monkeypatch, code=code)
    window = ConversationWindow(
        summary="Person A fragte, wann das Hafenfest stattfindet.",
        earlier_questions=("Wer organisiert das Hafenfest?",),
    )

    turn = _turn(rag, target="documents", window=window)

    expected = _system_prompt(code, "documents", rule=assessment, summary=window.summary_block())
    assert turn.loop._system_prompt == expected
    assert turn.loop._system_prompt.endswith(window.summary_block())
    assert (SENTINEL in turn.loop._system_prompt) is assessment
    assert turn.budget.system_tokens == estimate_tokens(expected, 3.5)
    assert turn.loop._final_note == _prompt("agent_final", code)
    assert turn.loop._scope_note == _prompt("agent_scope", code)
    assert turn.loop._llm_view is agent_llm_view


def test_the_history_counts_each_recent_message_with_its_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every message the window sends costs its text and the envelope a chat template wraps it in.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    window = ConversationWindow(
        recent=(
            ConversationTurn("Wann ist das Hafenfest?", "Am Sonntag, wegen der Sturmwarnung."),
            ConversationTurn("Und die Regatta?", "Sie fällt aus."),
        )
    )
    texts = ("Wann ist das Hafenfest?", "Am Sonntag, wegen der Sturmwarnung.", "Und die Regatta?", "Sie fällt aus.")

    turn = _turn(_rag(monkeypatch), window=window)

    assert turn.budget.history_tokens == sum(estimate_tokens(text, 3.5) + MESSAGE_OVERHEAD_TOKENS for text in texts)


@pytest.mark.parametrize("target", RETRIEVAL_TARGETS)
def test_a_scoped_turn_counts_no_tool_schemas(monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    """A pinned round offers no tools, so its budget pays for none; an open turn pays for its own.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        target (str): The retrieval target.
    """
    rag = _rag(monkeypatch)

    scoped = _turn(rag, target=target, scoped=True, pinned_count=3)
    unscoped = _turn(rag, target=target)

    assert scoped.budget.tool_schema_tokens == 0
    assert unscoped.budget.tool_schema_tokens == schema_tokens(_schemas(target), 3.5) > 0
    assert scoped.budget.round_room == unscoped.budget.round_room + unscoped.budget.tool_schema_tokens


def _pinned(count: int) -> list[NodeWithScore]:
    """Build distinct invented passages a user pinned.

    Args:
        count (int): How many.

    Returns:
        list[NodeWithScore]: The passages.
    """
    return [scored_node(f"pinned-{number}", f"Notiz {number} zum Hafenfest am Sonntag.") for number in range(count)]


def test_a_scoped_ledger_holds_every_pinned_passage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pinned evidence is never refused: a scope larger than the cap raises the cap to hold it all.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    pinned = _pinned(AGENT_LEDGER_CAP + 6)

    turn = _turn(_rag(monkeypatch), scoped=True, pinned_count=len(pinned))
    turn.ledger.add(pinned)

    assert len(turn.ledger) == len(pinned)


@pytest.mark.parametrize(("scoped", "pinned_count"), [(False, 0), (True, 3)])
def test_a_turns_ledger_holds_the_cap_otherwise(
    monkeypatch: pytest.MonkeyPatch, scoped: bool, pinned_count: int
) -> None:
    """An open turn, or a scope smaller than the cap, admits ``AGENT_LEDGER_CAP`` passages.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        scoped (bool): Whether the turn answers a pinned scope.
        pinned_count (int): How many distinct passages it pinned.
    """
    turn = _turn(_rag(monkeypatch), scoped=scoped, pinned_count=pinned_count)
    turn.ledger.add(_pinned(AGENT_LEDGER_CAP + 6))

    assert len(turn.ledger) == AGENT_LEDGER_CAP == 64


def test_the_limits_come_from_the_agent_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rounds, the time and the calls per round are the deployment's ``AGENT_*`` settings.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("AGENT_MAX_ROUNDS", "3")
    monkeypatch.setenv("AGENT_TIME_BUDGET_S", "45")
    monkeypatch.setenv("AGENT_MAX_CALLS_PER_ROUND", "2")

    loop = _turn(_rag(monkeypatch)).loop

    assert (loop._max_rounds, loop._time_budget_s, loop._calls._max_calls) == (3, 45.0, 2)


@pytest.mark.parametrize("code", ["en", "de"])
def test_only_a_visual_turn_shows_the_model_its_pictures(monkeypatch: pytest.MonkeyPatch, code: str) -> None:
    """The visual target gets the thumbnail fetcher, the agent legend and the picture cap; the others get none.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        code (str): The locale.
    """
    monkeypatch.setenv("VISUAL_ANSWER_MAX_IMAGES", "3")
    rag = _rag(monkeypatch, code=code)

    visual = _turn(rag, target="visual")
    others = [_turn(rag, target=target) for target in ("all", "documents")]

    assert visual.loop._fetch_thumbnails == rag._fetch_companion_thumbnails
    assert visual.loop._legend_template == _prompt("agent_image_legend", code)
    assert visual.loop._max_images == 3
    for other in others:
        assert (other.loop._fetch_thumbnails, other.loop._legend_template, other.loop._max_images) == (None, "", 0)


def test_each_search_expands_its_own_query_for_retrieval_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Graph terms reach the engine, which asks for every child; the model's reply and trail show its own words.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    rag = _rag(monkeypatch)
    monkeypatch.setattr(
        rag, "expand_query_with_graph", lambda query: f"{query}\n\nRelated entities for retrieval: Person B"
    )
    engine = _engine(programme_passage())
    turn = _turn(rag, target="documents", engine=engine)

    reply = turn.loop._registry.call(tool_call("search", query="Hafenfest Sonntag"), _ctx(turn))

    (bundle,) = [call.args[0] for call in engine.retrieve.call_args_list]
    assert isinstance(bundle, EveryChildQuery)
    assert bundle.query_str == "Hafenfest Sonntag\n\nRelated entities for retrieval: Person B"
    assert reply.detail == "Hafenfest Sonntag"
    assert "Person B" not in reply.text


def _record_image_tools(monkeypatch: pytest.MonkeyPatch, rag: RAG) -> tuple[list[Any], list[Any]]:
    """Replace the engine's image searches with doubles that record the filter each call carries.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        rag (RAG): The engine, before the turn is built.

    Returns:
        tuple[list[Any], list[Any]]: The filters ``search_images`` and ``keyframes`` were called with.
    """
    searched: list[Any] = []
    listed: list[Any] = []

    def search_images(description: str, *, qdrant_filter: Any = None) -> list[NodeWithScore]:
        """Record the filter an image search carries.

        Args:
            description (str): What the pictures should show.
            qdrant_filter (Any): The companion filter.

        Returns:
            list[NodeWithScore]: No pictures.
        """
        searched.append(qdrant_filter)
        return []

    def list_keyframes(file_or_hash: str, *, qdrant_filter: Any = None) -> list[NodeWithScore]:
        """Record the filter a keyframe listing carries.

        Args:
            file_or_hash (str): The clip.
            qdrant_filter (Any): The companion filter.

        Returns:
            list[NodeWithScore]: No frames.
        """
        listed.append(qdrant_filter)
        return []

    monkeypatch.setattr(rag, "_agent_search_images", search_images)
    monkeypatch.setattr(rag, "_agent_list_keyframes", list_keyframes)
    return searched, listed


@pytest.mark.parametrize(("with_rules", "with_compiled"), [(True, True), (True, False), (False, True)])
def test_the_image_tools_search_the_companion_under_the_requests_filter(
    monkeypatch: pytest.MonkeyPatch, with_rules: bool, with_compiled: bool
) -> None:
    """The filter the request compiled, or else its rules compiled here, reaches both image tools.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        with_rules (bool): Whether the request carries its raw rules.
        with_compiled (bool): Whether the request carries its compiled filter.
    """
    rag = _rag(monkeypatch)
    searched, listed = _record_image_tools(monkeypatch, rag)
    compiled = build_qdrant_filter(RULES)
    turn = _turn(
        rag,
        target="visual",
        metadata_filter_rules=RULES if with_rules else None,
        metadata_filters_active=True,
        vector_store_kwargs={"qdrant_filters": compiled} if with_compiled else None,
    )

    turn.loop._registry.call(tool_call("search_images", description="Regenjacken am Hafen"), _ctx(turn))
    turn.loop._registry.call(tool_call("keyframes", file=CLIP), _ctx(turn))

    assert compiled is not None
    assert searched == [compiled]
    assert listed == [compiled]


def test_an_unfiltered_turn_searches_the_whole_companion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a request filter the image tools carry none.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    rag = _rag(monkeypatch)
    searched, listed = _record_image_tools(monkeypatch, rag)
    turn = _turn(rag, target="all")

    turn.loop._registry.call(tool_call("search_images", description="Regenjacken am Hafen"), _ctx(turn))
    turn.loop._registry.call(tool_call("keyframes", file=CLIP), _ctx(turn))

    assert searched == [None]
    assert listed == [None]


def test_filters_that_reached_the_turn_in_neither_form_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unfiltered imagery must never answer a filtered question, so the turn refuses as the visual target does.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    rag = _rag(monkeypatch)

    with pytest.raises(ValueError, match="filters are active"):
        _turn(rag, target="all", metadata_filters_active=True)


def test_a_search_runs_on_a_search_thread_inside_the_requests_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loop starts each search on its own thread in a copy of the request's context, expansion included.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("AGENT_MAX_ROUNDS", "6")
    rag = _rag(monkeypatch)
    rag.graphrag_enabled = True
    expanded_in: list[str] = []
    searched_in: list[tuple[str, str]] = []

    def aggregate(refresh: bool = False) -> dict[str, Any]:
        """Record which collection the expansion reads its entities from.

        Args:
            refresh (bool): Unused.

        Returns:
            dict[str, Any]: No entities, so the query is not expanded.
        """
        expanded_in.append(rag.qdrant_collection)
        return {"entities": []}

    def retrieve(bundle: Any) -> list[NodeWithScore]:
        """Record which collection and thread a search ran in.

        Args:
            bundle (Any): The query bundle.

        Returns:
            list[NodeWithScore]: The programme passage.
        """
        searched_in.append((rag.qdrant_collection, threading.current_thread().name))
        return [programme_passage()]

    monkeypatch.setattr(rag, "_get_collection_ner_aggregate", aggregate)
    engine = MagicMock()
    engine.retrieve.side_effect = retrieve
    rag._bounded_text_model = cast(
        Any,
        streaming_tool_llm(
            [
                [
                    chunk(
                        tool_calls=[
                            tool_call_delta(
                                index=0, call_id="call_1", name="search", arguments='{"query": "Hafenfest Sonntag"}'
                            )
                        ],
                        finish_reason="tool_calls",
                    )
                ],
                [*text_chunks("Die Stände öffnen am Sonntag um zehn Uhr [1].", 12), chunk(finish_reason="stop")],
            ]
        ),
    )

    with rag.collection_scope(PHYSICAL):
        turn = _turn(rag, target="documents", engine=engine)
        items = list(turn.loop.run(window=ConversationWindow(), message=MESSAGE, ledger=turn.ledger))

    assert expanded_in == [PHYSICAL]
    assert searched_in == [(PHYSICAL, SEARCH_THREAD_NAME)]
    result = items[-1]
    assert isinstance(result, AgentResult)
    assert [scored.node.node_id for scored in result.nodes] == ["programme"]
    assert result.steps == 1
