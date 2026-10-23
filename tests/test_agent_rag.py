"""The RAG side of an agent turn: the model it runs on, the view of its passages, and the turn the factory builds.

Every message, passage, clip, filter and collection here is invented.
"""

from __future__ import annotations

import json
import threading
import types
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, cast
from unittest.mock import MagicMock

import httpx
import openai
import pytest
from agent_fakes import (
    CLIP,
    KeywordClient,
    chunk,
    keyword_coverage,
    programme_passage,
    ready_schema,
    streaming_tool_llm,
    text_chunks,
    tool_call,
    tool_call_delta,
    turn_budget,
)
from assessment_fakes import scored_node
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import BaseNode, MetadataMode, NodeWithScore, QueryBundle, TextNode
from loguru import logger

from docint.agents.assessment.ledger import EvidenceLedger, render_context, stamp
from docint.agents.assessment.split import SENTINEL
from docint.agents.turn.budget import MESSAGE_OVERHEAD_TOKENS, MIN_ROUND_ROOM, TurnBudget, schema_tokens
from docint.agents.turn.calls import SEARCH_THREAD_NAME
from docint.agents.turn.history import NO_HISTORY, ConversationTurn, ConversationWindow
from docint.agents.turn.keyword import KeywordMatcher
from docint.agents.turn.loop import AgentResult
from docint.agents.turn.probe import PROBE_SYSTEM_PROMPT, PROBE_TOOL, probe_tools
from docint.agents.turn.tools import ReadTool, SearchTool, ToolContext, ToolRegistry
from docint.agents.turn.toolsets import toolset_for
from docint.core.rag import (
    AGENT_LEDGER_CAP,
    AGENT_METADATA_VALUE_CHARS,
    IMAGE_LANE_METADATA_KEY,
    RAG,
    AgentTurn,
    ParentContextPostprocessor,
    agent_llm_view,
)
from docint.core.retrieval.evidence import AgentSearchQuery, EveryChildQuery
from docint.core.retrieval.visual import BLOB_PAYLOAD_KEYS, RETRIEVAL_TARGETS, RetrievalTarget
from docint.core.retrieval_filters import build_qdrant_filter
from docint.core.search.index import image_companion_name
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
#: The file an older table export names a row by, in its ``origin`` alone.
LEGACY_FILE = "example_profile/comments.csv"
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
        RAG: The engine, with only the OpenAI knobs, the ratio and the locale set, over an empty store.
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
    # A store holding no collection, so no turn is offered keyword unless a test readies one.
    rag._qdrant_client = cast(Any, KeywordClient({}))
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


def test_a_probe_on_the_agent_model_makes_one_attempt_within_ten_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tools probe sends the fixed request once, streamed, waiting ten seconds, whatever retries are configured.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    attempts = _stalled_endpoint(monkeypatch)
    rag = _rag(monkeypatch, max_retries=2)

    assert probe_tools(rag.agent_text_model) == "inconclusive"

    assert len(attempts) == 1
    sent = json.loads(attempts[0].content)
    assert attempts[0].extensions["timeout"]["read"] == 10.0
    assert [(message["role"], message["content"]) for message in sent["messages"]] == [
        ("system", PROBE_SYSTEM_PROMPT),
        ("user", "ping"),
    ]
    assert (sent["tools"], sent["tool_choice"], sent["stream"]) == ([PROBE_TOOL], "auto", True)


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


def _legacy_row(*, empty_name: bool = False) -> NodeWithScore:
    """Build an invented comment row of an older table export, which names its file only in ``origin``, first.

    Args:
        empty_name (bool): Whether the row carries a ``filename`` that is empty, rather than none.

    Returns:
        NodeWithScore: The row, with no file name of its own and the 1,500-character post it answers.
    """
    reference = {
        "network": "examplenet",
        "type": "comment",
        "uuid": "3f2b8c1e-9d4a-4c6b-8e21-5a7f0c9d2b14",
        "url": "https://social.invalid/p/9876543214",
        "timestamp": TIMESTAMP,
        "author": AUTHOR,
        "author_id": "1234567890",
        "vanity": "person_a",
        "text": SHORT_COMMENT,
        "text_id": "9876543214",
        "anchor_text": LONG_POST,
        "parent_text": None,
    }
    metadata = {
        "origin": {"filename": LEGACY_FILE, "filetype": "text/csv"},
        "source": "table",
        "table": {"columns": ["id", "author", "text", "published_at"], "n_rows": 120, "row_index": 4},
        "reference_metadata": reference,
    }
    if empty_name:
        metadata["filename"] = ""
    return NodeWithScore(node=TextNode(id_="legacy-comment-4", text=SHORT_COMMENT, metadata=metadata), score=1.0)


@pytest.mark.parametrize("empty_name", [False, True], ids=["no-file-name", "empty-file-name"])
def test_a_row_without_a_file_name_keeps_its_origin_in_the_files_place(empty_name: bool) -> None:
    """An older export names a row's file only in its ``origin``, which takes the file's place: both show.

    Args:
        empty_name (bool): Whether the row carries an empty ``filename`` rather than none.
    """
    row = _legacy_row(empty_name=empty_name)

    reply = _search_reply(agent_llm_view, row)

    assert reply.startswith(f"citation_index: 1\norigin: {{'filename': '{LEGACY_FILE}'")
    assert "characters not shown" in reply.partition("\n\n")[0]
    assert AUTHOR in reply
    view = agent_llm_view(row.node)
    visible = [key for key in view.metadata if key not in view.excluded_llm_metadata_keys]
    assert visible[:2] == ["origin", "reference_metadata"]


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
    assert turn.loop._search_note == _prompt("agent_search_first", code)
    assert turn.loop._llm_view is agent_llm_view


@pytest.mark.parametrize("target", RETRIEVAL_TARGETS)
def test_only_an_unscoped_turn_asks_a_first_round_that_looked_nothing_up_again(
    monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """Every unscoped turn gets the search note; a pinned scope, which offers no tools, gets none.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        target (str): The retrieval target.
    """
    rag = _rag(monkeypatch)

    scoped = _turn(rag, target=target, scoped=True, pinned_count=2)
    unscoped = _turn(rag, target=target)

    assert (scoped.loop._search_note, unscoped.loop._search_note) == ("", _prompt("agent_search_first", "en"))


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
    # An agent's own search: its sub-floor images get no description re-score.
    assert isinstance(bundle, AgentSearchQuery)
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


#: Every keyword field, as a collection whose indexes are all ready offers them.
KEYWORD_FIELDS = ("text", "author", "network", "uuid")
#: The line that says a turn was built without keyword for want of room, up to its numbers.
NO_ROOM_FOR_KEYWORD = "The keyword tool was withheld from an agent turn: "
#: The words that tell that line from the others: the DEBUG line naming a readiness reason shares its prefix, and
#: the warning that no turn fits shares these words but not the prefix.
NO_ROOM_MARK = "tokens of room"
#: The line that says a turn was offered keyword on the text alone for want of room, up to its numbers.
TEXT_ALONE_FOR_ROOM = "The keyword tool was offered on the text alone in an agent turn: "


def _withheld_for_room(messages: Sequence[str]) -> list[str]:
    """Pick out the lines that say a turn was built without keyword for want of room.

    Args:
        messages (Sequence[str]): The captured log messages.

    Returns:
        list[str]: The matching lines, in order.
    """
    return [line for line in messages if line.startswith(NO_ROOM_FOR_KEYWORD) and NO_ROOM_MARK in line]


def _keyword_ready(
    monkeypatch: pytest.MonkeyPatch,
    rag: RAG,
    *,
    fields: tuple[str, ...] = ("text",),
    collection: str = "hafen",
    matches: int = 2,
    found: Sequence[NodeWithScore] = (),
) -> tuple[KeywordClient, list[dict[str, Any]]]:
    """Ready a collection for keyword: its indexes built, its matches counted by a store double, retrieved by another.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        rag (RAG): The engine, before the turn is built.
        fields (tuple[str, ...]): The fields whose indexes are ready, the text first.
        collection (str): The physical collection the turn runs in.
        matches (int): What every count returns.
        found (Sequence[NodeWithScore]): What every keyword retrieval returns.

    Returns:
        tuple[KeywordClient, list[dict[str, Any]]]: The store double, and each retriever build's arguments
        with the collection (``collection``) and thread (``thread``) it ran in.
    """
    client = KeywordClient({collection: ready_schema(*fields)}, counts={collection: matches})
    rag._qdrant_client = cast(Any, client)
    keyword_coverage(monkeypatch, {collection: (10, 10)})
    built: list[dict[str, Any]] = []

    class _Retriever:
        """Return the passages given."""

        def retrieve(self, bundle: QueryBundle) -> list[NodeWithScore]:
            """Return the passages.

            Args:
                bundle (QueryBundle): The query bundle.

            Returns:
                list[NodeWithScore]: The passages given.
            """
            return list(found)

    def _build(**kwargs: Any) -> _Retriever:
        """Record a retriever build, where and on which thread it ran.

        Args:
            **kwargs (Any): The build arguments.

        Returns:
            _Retriever: The double.
        """
        built.append({**kwargs, "collection": rag.qdrant_collection, "thread": threading.current_thread().name})
        return _Retriever()

    monkeypatch.setattr(rag, "_build_retriever", _build)
    monkeypatch.setattr(rag, "_text_postprocessors", lambda **kwargs: [])
    return client, built


def _window_with_room(budget: TurnBudget, room: int) -> int:
    """Find the smallest context window that leaves a turn shaped like ``budget`` at least ``room`` tokens.

    Args:
        budget (TurnBudget): A turn's budget, whose own window leaves at least ``room``.
        room (int): The room for tool replies wanted.

    Returns:
        int: The window.
    """
    assert budget.round_room >= room
    window = budget.ctx_window
    while replace(budget, ctx_window=window - 1).round_room >= room:
        window -= 1
    return window


@pytest.mark.parametrize("target", ["all", "documents"])
@pytest.mark.parametrize("code", ["en", "de"])
def test_a_turn_over_text_is_offered_keyword_where_its_collection_is_ready(
    monkeypatch: pytest.MonkeyPatch, code: str, target: str
) -> None:
    """The tool is offered and its schema paid for, and the prompt carries its line and the fields it takes.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        code (str): The locale.
        target (str): The retrieval target.
    """
    rag = _rag(monkeypatch, code=code)
    _keyword_ready(monkeypatch, rag, fields=KEYWORD_FIELDS)

    turn = _turn(rag, target=target)

    registry = turn.loop._registry
    prompt = turn.loop._system_prompt
    assert "keyword" in registry.names
    assert prompt == rag._agent_system_prompt(target, ConversationWindow(), keyword_fields=KEYWORD_FIELDS)
    assert _prompt("agent_tools_keyword", code) in prompt
    assert _prompt("agent_tools_keyword_fields", code).format(fields=", ".join(KEYWORD_FIELDS)) in prompt
    assert turn.budget.tool_schema_tokens == schema_tokens(registry.schemas(), 3.5)
    assert turn.budget.system_tokens == estimate_tokens(prompt, 3.5)


def test_a_turn_offered_the_text_alone_names_no_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """With only the text ready the prompt carries the keyword line and no fields line.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    rag = _rag(monkeypatch)
    _keyword_ready(monkeypatch, rag)

    turn = _turn(rag, target="documents")

    assert "keyword" in turn.loop._registry.names
    expected = rag._agent_system_prompt("documents", ConversationWindow(), keyword_fields=("text",))
    assert turn.loop._system_prompt == expected
    assert _prompt("agent_tools_keyword", "en") in turn.loop._system_prompt
    assert "keyword also takes field" not in turn.loop._system_prompt


@pytest.mark.parametrize(("target", "scoped"), [("visual", False), ("all", True), ("documents", True)])
def test_a_visual_or_scoped_turn_is_never_offered_keyword(
    monkeypatch: pytest.MonkeyPatch, target: str, scoped: bool
) -> None:
    """No matcher is even asked for, so neither gets the tool or its lines, though the collection is ready.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        target (str): The retrieval target.
        scoped (bool): Whether the turn answers a pinned scope.
    """
    rag = _rag(monkeypatch)
    _keyword_ready(monkeypatch, rag, fields=KEYWORD_FIELDS)
    ready = rag._agent_keyword_matcher(
        metadata_filter_rules=None, metadata_filters_active=False, retrieval_target="all"
    )
    assert ready is not None
    asked: list[dict[str, Any]] = []

    def _matcher(**kwargs: Any) -> KeywordMatcher:
        """Record that a matcher was asked for, and hand over the ready one.

        Args:
            **kwargs (Any): The request it was asked for.

        Returns:
            KeywordMatcher: The matcher of the ready collection.
        """
        asked.append(kwargs)
        return ready

    monkeypatch.setattr(rag, "_agent_keyword_matcher", _matcher)

    turn = _turn(rag, target=target, scoped=scoped, pinned_count=2 if scoped else 0)

    assert asked == []
    assert "keyword" not in turn.loop._registry.names
    assert turn.loop._system_prompt == _system_prompt("en", target, rule=False, summary=NO_HISTORY)


def _keyword_budgets(monkeypatch: pytest.MonkeyPatch) -> dict[str, TurnBudget]:
    """Measure an ``all`` turn's budget with keyword on every field, on the text alone, and without keyword.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.

    Returns:
        dict[str, TurnBudget]: The budgets by ``every``, ``text`` and ``none``, at the default window.
    """
    every = _rag(monkeypatch)
    _keyword_ready(monkeypatch, every, fields=KEYWORD_FIELDS)
    text = _rag(monkeypatch)
    _keyword_ready(monkeypatch, text)
    return {"every": _turn(every).budget, "text": _turn(text).budget, "none": _turn(_rag(monkeypatch)).budget}


@pytest.mark.parametrize(
    ("boundary", "offset", "offered"),
    [
        ("every", 0, KEYWORD_FIELDS),
        ("every", -1, ("text",)),
        ("text", 0, ("text",)),
        ("text", -1, ()),
    ],
    ids=["every-field-fits", "every-field-one-short", "text-fits", "nothing-fits"],
)
def test_keyword_gives_up_its_other_fields_before_the_tool_when_room_is_short(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog_info: Any, boundary: str, offset: int, offered: tuple[str, ...]
) -> None:
    """At each window boundary a turn offers what fits: every field, the text alone, or no keyword at all.

    Its tools, prompt and budget agree on what is offered, and each step back logs its numbers alone.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog_info (Any): Bridged INFO capture.
        boundary (str): The offer whose smallest window with room the turn is built at, or one token short of.
        offset (int): Tokens of window added to that smallest window.
        offered (tuple[str, ...]): The fields keyword is offered with; none when it is withheld.
    """
    budgets = _keyword_budgets(monkeypatch)
    window = _window_with_room(budgets[boundary], MIN_ROUND_ROOM) + offset
    loguru_caplog_info.clear()
    rag = _rag(monkeypatch, ctx_window=window)
    _keyword_ready(monkeypatch, rag, fields=KEYWORD_FIELDS)

    turn = _turn(rag)

    tools = {schema["function"]["name"]: schema["function"]["parameters"] for schema in turn.loop._registry.schemas()}
    field = tools["keyword"]["properties"].get("field") if "keyword" in tools else None
    assert ("keyword" in tools) is bool(offered)
    assert (field or {}).get("enum") == (list(offered) if len(offered) > 1 else None)
    assert turn.loop._system_prompt == rag._agent_system_prompt("all", ConversationWindow(), keyword_fields=offered)
    kinds: dict[tuple[str, ...], str] = {KEYWORD_FIELDS: "every", ("text",): "text", (): "none"}
    assert turn.budget == replace(budgets[kinds[offered]], ctx_window=window)
    assert turn.budget.available
    every_room = replace(budgets["every"], ctx_window=window).round_room
    text_room = replace(budgets["text"], ctx_window=window).round_room
    messages = loguru_caplog_info.messages
    assert [line for line in messages if line.startswith(TEXT_ALONE_FOR_ROOM)] == (
        []
        if offered == KEYWORD_FIELDS
        else [
            f"{TEXT_ALONE_FOR_ROOM}{every_room} tokens of room with its {len(KEYWORD_FIELDS)} fields, "
            f"under the {MIN_ROUND_ROOM} one needs"
        ]
    )
    assert _withheld_for_room(messages) == (
        [f"{NO_ROOM_FOR_KEYWORD}{text_room} tokens of room with it, under the {MIN_ROUND_ROOM} one needs"]
        if not offered
        else []
    )
    assert MESSAGE not in loguru_caplog_info.text


def test_a_turn_without_room_even_without_keyword_builds_none(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog_info: Any
) -> None:
    """Withholding keyword is tried first; when the room is short without it too, no turn is built.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog_info (Any): Bridged INFO capture.
    """
    rag = _rag(monkeypatch, ctx_window=4096)
    _keyword_ready(monkeypatch, rag, fields=KEYWORD_FIELDS)

    turn = rag.agent_turn(
        _engine(),
        retrieval_target="all",
        scoped=False,
        pinned_count=0,
        window=ConversationWindow(),
        message=MESSAGE,
        metadata_filter_rules=None,
        metadata_filters_active=False,
        vector_store_kwargs=None,
    )

    assert turn is None
    messages = loguru_caplog_info.messages
    assert _withheld_for_room(messages) != []
    assert [line for line in messages if line.startswith("An agent turn would have ")] != []
    assert MESSAGE not in loguru_caplog_info.text


@pytest.mark.parametrize("with_compiled", [True, False], ids=["compiled", "rules-only"])
def test_a_keyword_check_counts_and_retrieves_under_the_requests_filter(
    monkeypatch: pytest.MonkeyPatch, with_compiled: bool
) -> None:
    """The request's rules, compiled for the match, narrow the count and the retrieval alike.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        with_compiled (bool): Whether the request also carries its compiled filter.
    """
    rag = _rag(monkeypatch)
    client, built = _keyword_ready(monkeypatch, rag, found=[programme_passage()])
    compiled = build_qdrant_filter(RULES)
    assert compiled is not None
    turn = _turn(
        rag,
        target="documents",
        metadata_filter_rules=RULES,
        metadata_filters_active=True,
        vector_store_kwargs={"qdrant_filters": compiled} if with_compiled else None,
    )

    reply = turn.loop._registry.call(tool_call("keyword", words="Hafenfest"), _ctx(turn))

    ((counted_in, counted_filter),) = client.counted
    assert counted_in == "hafen"
    assert cast(list[Any], compiled.must)[0] in cast(list[Any], counted_filter.must)
    assert "search_text" in repr(counted_filter)
    assert built[0]["vector_store_kwargs"] == {"qdrant_filters": counted_filter}
    assert (built[0]["metadata_filter_rules"], built[0]["metadata_filters_active"]) == (RULES, True)
    assert reply.found == 1


def test_a_turn_whose_filter_arrived_compiled_alone_is_offered_no_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keyword narrows its count by the request's rules; without them it could count outside the filter.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    rag = _rag(monkeypatch)
    _keyword_ready(monkeypatch, rag)

    turn = _turn(
        rag,
        target="documents",
        metadata_filters_active=True,
        vector_store_kwargs={"qdrant_filters": build_qdrant_filter(RULES)},
    )

    assert "keyword" not in turn.loop._registry.names


def test_a_keyword_check_runs_on_a_search_thread_inside_the_requests_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loop runs a keyword check's count and retrieval on a search thread, in a copy of the request's context.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("AGENT_MAX_ROUNDS", "6")
    rag = _rag(monkeypatch)
    client, built = _keyword_ready(monkeypatch, rag, collection=PHYSICAL, found=[programme_passage()])
    counted_on: list[str] = []
    count = client.count

    def _count(collection_name: str, count_filter: Any = None, exact: bool = True) -> Any:
        """Record the thread a count ran on, then count.

        Args:
            collection_name (str): The collection.
            count_filter (Any): The filter.
            exact (bool): Whether the count is exact.

        Returns:
            Any: The count.
        """
        counted_on.append(threading.current_thread().name)
        return count(collection_name, count_filter=count_filter, exact=exact)

    monkeypatch.setattr(client, "count", _count)
    rag._bounded_text_model = cast(
        Any,
        streaming_tool_llm(
            [
                [
                    chunk(
                        tool_calls=[
                            tool_call_delta(
                                index=0, call_id="call_1", name="keyword", arguments='{"words": "Hafenfest"}'
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
        turn = _turn(rag, target="documents")
        items = list(turn.loop.run(window=ConversationWindow(), message=MESSAGE, ledger=turn.ledger))

    assert [name for name, _ in client.counted] == [PHYSICAL]
    assert counted_on == [SEARCH_THREAD_NAME]
    assert [(build["collection"], build["thread"]) for build in built] == [(PHYSICAL, SEARCH_THREAD_NAME)]
    result = items[-1]
    assert isinstance(result, AgentResult)
    assert [scored.node.node_id for scored in result.nodes] == ["programme"]
    assert result.steps == 1


#: The image companion of the request's collection.
PHYSICAL_IMAGES = image_companion_name(PHYSICAL)
#: An invented section too long to embed whole, which ingestion keeps only in the docstore.
SECTION = " ".join(
    f"Absatz {line}: Die Hafenordnung regelt Liegeplätze, Gebühren und Fristen für Anmeldungen am Kai."
    for line in range(1, 61)
)
SECTION_ID = "5a1d7c3e-2b4f-4e8a-9c6d-0f1e2d3c4b5a"
#: A chunk the main collection stores as a point; Qdrant hands its id back as an integer.
CHUNK_ID = "7"
#: A frame of the invented clip, stored in the image companion.
FRAME_ID = "8c2e4a6f-1b3d-4f5e-a7c9-0d2b4f6e8a1c"


def _section_node() -> TextNode:
    """Build the invented section as the docstore keeps it.

    Returns:
        TextNode: The section, on page four of its file.
    """
    metadata = {"file_name": "hafenordnung.pdf", "filename": "hafenordnung.pdf", "page": 4, "source_type": "pdf"}
    return TextNode(id_=SECTION_ID, text=SECTION, metadata=metadata)


def _chunk_payload() -> dict[str, Any]:
    """Build the payload the invented programme chunk is stored with in the main collection.

    Returns:
        dict[str, Any]: The payload.
    """
    return {
        "text": programme_passage().node.get_content(),
        "file_name": "hafenfest_programm.pdf",
        "filename": "hafenfest_programm.pdf",
        "page": 1,
        "source_type": "pdf",
    }


def _frame_payload() -> dict[str, Any]:
    """Build the payload a frame of the invented clip is stored with in the image companion, its blobs included.

    Returns:
        dict[str, Any]: The payload of the clip's third frame, twelve seconds in.
    """
    return {
        "image_id": f"{CLIP}-image-2",
        "source_type": "video_keyframe",
        "source_doc_id": "c" * 64,
        "source_file": CLIP,
        "source_path": f"uploads/{CLIP}",
        "media_file_hash": "c" * 64,
        "keyframe_index": 2,
        "keyframe_time_sec": 12.0,
        "llm_description": "Zwei Personen in Regenjacken am Hafen, bedeckter Himmel",
        "llm_tags": ["hafen"],
        "thumbnail_b64": "QUJD",
        "thumbnail_mime": "image/jpeg",
        "search_text": "Zwei Personen in Regenjacken am Hafen",
    }


class _PointStore:
    """A Qdrant double for carried passages: points by collection, read by id, every read recorded."""

    def __init__(self, points: dict[str, dict[Any, dict[str, Any]]], *, failing: str | None = None) -> None:
        """Hold the points.

        Args:
            points (dict[str, dict[Any, dict[str, Any]]]): Per existing collection, each point's stored payload by
                its id as Qdrant returns it.
            failing (str | None): A collection whose reads fail as an unreachable store's do.
        """
        self.points = points
        self.failing = failing
        self.retrieved: list[dict[str, Any]] = []

    def collection_exists(self, collection_name: str) -> bool:
        """Report whether a collection exists.

        Args:
            collection_name (str): The collection.

        Returns:
            bool: Whether points were given for it.
        """
        return collection_name in self.points

    def retrieve(
        self, collection_name: str, ids: Sequence[Any], with_payload: Any = True, with_vectors: bool = False
    ) -> list[types.SimpleNamespace]:
        """Read points by id, honouring the payload selector as Qdrant does.

        Args:
            collection_name (str): The collection.
            ids (Sequence[Any]): The point ids, in the types Qdrant takes.
            with_payload (Any): ``True``, or a selector excluding keys.
            with_vectors (bool): Whether vectors are read.

        Returns:
            list[types.SimpleNamespace]: The points held, in no particular order.

        Raises:
            ConnectionError: For the failing collection, naming what it was asked for.
        """
        self.retrieved.append(
            {
                "collection_name": collection_name,
                "ids": list(ids),
                "with_payload": with_payload,
                "with_vectors": with_vectors,
            }
        )
        if collection_name == self.failing:
            raise ConnectionError(f"invented outage while reading {list(ids)}")
        held = self.points.get(collection_name, {})
        excluded = set(getattr(with_payload, "exclude", None) or ())
        return [
            types.SimpleNamespace(id=point_id, payload={k: v for k, v in held[point_id].items() if k not in excluded})
            for point_id in reversed(ids)
            if point_id in held
        ]


def _store(*, failing: str | None = None) -> _PointStore:
    """Hold the programme chunk in the main collection and the frame in the companion.

    Args:
        failing (str | None): A collection whose reads fail.

    Returns:
        _PointStore: The store.
    """
    return _PointStore(
        {PHYSICAL: {7: _chunk_payload()}, PHYSICAL_IMAGES: {FRAME_ID: _frame_payload()}}, failing=failing
    )


class _Docstore:
    """A docstore double: ``get_node`` answers the nodes it holds and ``None`` for any other id, recording each id."""

    def __init__(self, *nodes: BaseNode, failing: str | None = None) -> None:
        """Hold the nodes.

        Args:
            *nodes (BaseNode): The nodes, by their own ids.
            failing (str | None): An id whose entry cannot be read, as a corrupt one cannot.
        """
        self.nodes = {node.node_id: node for node in nodes}
        self.failing = failing
        self.asked: list[str] = []

    def get_node(self, node_id: str, raise_error: bool = True) -> BaseNode | None:
        """Look one node up, as the docstore does when told not to raise.

        Args:
            node_id (str): The node id.
            raise_error (bool): Must be ``False``: a carried id the docstore lacks is no error.

        Returns:
            BaseNode | None: The node, or ``None`` when it is not held.

        Raises:
            ValueError: For the failing id.
        """
        assert raise_error is False
        self.asked.append(node_id)
        if node_id == self.failing:
            raise ValueError(f"invented corrupt entry {node_id}")
        return self.nodes.get(node_id)


def _carrying(monkeypatch: pytest.MonkeyPatch, store: _PointStore, docstore: _Docstore) -> RAG:
    """Build a RAG over the given store and docstore.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patches the index the docstore is reached through.
        store (_PointStore): The Qdrant double.
        docstore (_Docstore): The docstore double.

    Returns:
        RAG: The engine.
    """
    rag = _rag(monkeypatch)
    rag._qdrant_client = cast(Any, store)
    monkeypatch.setattr(RAG, "index", property(lambda self: types.SimpleNamespace(docstore=docstore)))
    return rag


def _carry(
    rag: RAG,
    node_ids: Sequence[str],
    *,
    target: RetrievalTarget = "all",
    rules: Sequence[Any] | None = None,
    active: bool = False,
) -> list[NodeWithScore]:
    """Load carried passages inside the request's collection scope.

    Args:
        rag (RAG): The engine.
        node_ids (Sequence[str]): The cited ids.
        target (RetrievalTarget): The turn's target.
        rules (Sequence[Any] | None): The request's raw filter rules.
        active (bool): Whether the request carries filters.

    Returns:
        list[NodeWithScore]: What the turn carries.
    """
    with rag.collection_scope(PHYSICAL):
        return rag.agent_carried_evidence(
            node_ids, retrieval_target=target, metadata_filter_rules=rules, metadata_filters_active=active
        )


@pytest.mark.parametrize("target", ["documents", "all"])
def test_carried_evidence_reads_a_section_kept_only_in_the_docstore(
    monkeypatch: pytest.MonkeyPatch, target: RetrievalTarget
) -> None:
    """An oversize section lives in the docstore alone; it is carried whole, and the store is never asked for it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        target (RetrievalTarget): A target that reads the main collection.
    """
    store = _PointStore({PHYSICAL: {}, PHYSICAL_IMAGES: {}})
    rag = _carrying(monkeypatch, store, _Docstore(_section_node()))

    (carried,) = _carry(rag, [SECTION_ID], target=target)

    assert (carried.node.node_id, carried.score) == (SECTION_ID, None)
    assert carried.node.get_content(metadata_mode=MetadataMode.NONE) == SECTION
    assert store.retrieved == []


@pytest.mark.parametrize("target", ["all", "visual"])
def test_carried_evidence_reads_a_picture_as_the_image_tools_build_it(
    monkeypatch: pytest.MonkeyPatch, target: RetrievalTarget
) -> None:
    """A cited frame is built as search_images and keyframes build it: its caption, its label first, and no blobs.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        target (RetrievalTarget): A target that reads the image companion.
    """
    store = _store()
    rag = _carrying(monkeypatch, store, _Docstore())

    (carried,) = _carry(rag, [FRAME_ID], target=target)

    node = carried.node
    assert node.get_content(metadata_mode=MetadataMode.NONE) == RAG._image_caption_text(_frame_payload())
    assert next(iter(node.metadata)) == "keyframe"
    assert node.metadata["keyframe"] == "#3 @ 00:12"
    assert node.metadata[IMAGE_LANE_METADATA_KEY] is True
    assert not set(BLOB_PAYLOAD_KEYS) & set(node.metadata)
    assert (node.node_id, carried.score) == (FRAME_ID, None)
    assert store.retrieved[-1]["collection_name"] == PHYSICAL_IMAGES
    assert store.retrieved[-1]["with_vectors"] is False


@pytest.mark.parametrize(
    ("target", "carried", "docstore_asked", "collections_read"),
    [
        ("all", [SECTION_ID, CHUNK_ID, FRAME_ID], [SECTION_ID, CHUNK_ID, FRAME_ID], [PHYSICAL, PHYSICAL_IMAGES]),
        ("documents", [SECTION_ID, CHUNK_ID], [SECTION_ID, CHUNK_ID, FRAME_ID], [PHYSICAL]),
        ("visual", [FRAME_ID], [], [PHYSICAL_IMAGES]),
    ],
)
def test_a_documents_turn_carries_no_picture_and_a_visual_turn_no_text(
    monkeypatch: pytest.MonkeyPatch,
    target: RetrievalTarget,
    carried: list[str],
    docstore_asked: list[str],
    collections_read: list[str],
) -> None:
    """Each target reads its own lanes alone: a documents turn never the companion, a visual turn nothing else.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        target (RetrievalTarget): The turn's target.
        carried (list[str]): The ids it carries.
        docstore_asked (list[str]): The ids it looks up in the docstore.
        collections_read (list[str]): The collections it reads, in order.
    """
    store, docstore = _store(), _Docstore(_section_node())
    rag = _carrying(monkeypatch, store, docstore)

    found = _carry(rag, [SECTION_ID, CHUNK_ID, FRAME_ID], target=target)

    assert [scored.node.node_id for scored in found] == carried
    assert docstore.asked == docstore_asked
    assert [read["collection_name"] for read in store.retrieved] == collections_read


def test_carried_evidence_drops_passages_outside_the_request_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    """With rules, a passage is carried only where it matches them, wherever it was read from.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    rag = _carrying(monkeypatch, _store(), _Docstore(_section_node()))
    asked = [SECTION_ID, CHUNK_ID, FRAME_ID]

    frames = _carry(rag, asked, rules=RULES, active=True)
    later_pages = _carry(rag, asked, rules=[{"field": "page", "operator": "gte", "value": 2}], active=True)

    assert [scored.node.node_id for scored in frames] == [FRAME_ID]
    assert [scored.node.node_id for scored in later_pages] == [SECTION_ID]


@pytest.mark.parametrize("rules", [None, []], ids=["none", "empty"])
def test_filters_without_rules_carry_nothing(monkeypatch: pytest.MonkeyPatch, rules: list[Any] | None) -> None:
    """Filters that arrived without their rules cannot be checked, so nothing is read or carried.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        rules (list[Any] | None): The rules the request arrived with.
    """
    store, docstore = _store(), _Docstore(_section_node())
    rag = _carrying(monkeypatch, store, docstore)
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="DEBUG")
    try:
        carried = _carry(rag, [SECTION_ID, CHUNK_ID, FRAME_ID], rules=rules, active=True)
    finally:
        logger.remove(sink)

    assert carried == []
    assert (store.retrieved, docstore.asked) == ([], [])
    assert [line for line in messages if line.startswith("Carried passages")] == [
        "Carried passages were withheld from an agent turn: filters"
    ]


def _broken_index(self: RAG) -> Any:
    """Fail as an index that cannot be loaded does.

    Args:
        self (RAG): The engine.

    Raises:
        RuntimeError: Always.
    """
    raise RuntimeError(f"invented index failure for {SECTION_ID}")


@pytest.mark.parametrize(
    ("failing", "target", "error"),
    [
        ("main", "all", "ConnectionError"),
        ("companion", "all", "ConnectionError"),
        ("companion", "visual", "ConnectionError"),
        ("index", "documents", "RuntimeError"),
    ],
)
def test_a_store_outage_carries_nothing(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: Any, failing: str, target: RetrievalTarget, error: str
) -> None:
    """A failed read carries nothing, not the part read before it, and is logged by the exception's type alone.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (Any): Bridged WARNING capture.
        failing (str): What fails: a collection's reads, or the index.
        target (RetrievalTarget): The turn's target, one that reaches what fails.
        error (str): The type of the exception raised.
    """
    store = _store(failing={"main": PHYSICAL, "companion": PHYSICAL_IMAGES}.get(failing))
    rag = _carrying(monkeypatch, store, _Docstore(_section_node()))
    if failing == "index":
        monkeypatch.setattr(RAG, "index", property(_broken_index))
    loguru_caplog.clear()

    carried = _carry(rag, [SECTION_ID, CHUNK_ID, FRAME_ID], target=target)

    assert carried == []
    assert loguru_caplog.messages == [f"Carried passages could not be loaded ({error}); the turn carries none"]


def test_carried_evidence_keeps_the_order_asked_and_skips_missing_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passages come back in the order asked, each once, under the id asked for; each lane reads what is still missing.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    store = _store()
    rag = _carrying(monkeypatch, store, _Docstore(_section_node()))

    carried = _carry(rag, [FRAME_ID, "gone", CHUNK_ID, SECTION_ID, FRAME_ID])

    assert [scored.node.node_id for scored in carried] == [FRAME_ID, CHUNK_ID, SECTION_ID]
    assert [scored.score for scored in carried] == [None, None, None]
    assert [(read["collection_name"], read["ids"]) for read in store.retrieved] == [
        (PHYSICAL, [FRAME_ID, "gone", 7]),
        (PHYSICAL_IMAGES, [FRAME_ID, "gone"]),
    ]


def test_a_docstore_entry_that_cannot_be_read_is_looked_up_in_the_store(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: Any
) -> None:
    """One unreadable docstore entry is a miss, not an outage: its id goes on to the store, and the rest are carried.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (Any): Bridged WARNING capture.
    """
    store = _store()
    rag = _carrying(monkeypatch, store, _Docstore(_section_node(), failing=CHUNK_ID))
    loguru_caplog.clear()

    carried = _carry(rag, [SECTION_ID, CHUNK_ID], target="documents")

    assert [scored.node.node_id for scored in carried] == [SECTION_ID, CHUNK_ID]
    assert [(read["collection_name"], read["ids"]) for read in store.retrieved] == [(PHYSICAL, [7])]
    assert loguru_caplog.messages == []


def test_a_collection_without_an_image_companion_carries_its_text_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a companion, a cited picture is not found, and nothing is read from a collection that does not exist.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    store = _PointStore({PHYSICAL: {7: _chunk_payload()}})
    rag = _carrying(monkeypatch, store, _Docstore())

    carried = _carry(rag, [CHUNK_ID, FRAME_ID])

    assert [scored.node.node_id for scored in carried] == [CHUNK_ID]
    assert [read["collection_name"] for read in store.retrieved] == [PHYSICAL]


def test_a_carry_is_logged_by_its_counts_alone(monkeypatch: pytest.MonkeyPatch, loguru_caplog_info: Any) -> None:
    """One INFO line counts what was carried, not found and left out; no id, label or text is logged.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog_info (Any): Bridged INFO capture.
    """
    rag = _carrying(monkeypatch, _store(), _Docstore(_section_node()))
    loguru_caplog_info.clear()

    _carry(rag, [SECTION_ID, "gone", CHUNK_ID, FRAME_ID], rules=RULES, active=True)

    assert loguru_caplog_info.messages == [
        "An agent turn carries 1 of the 4 passages its earlier answers cite "
        "(1 not found, 2 outside its target or filters)"
    ]
    for secret in (SECTION_ID, FRAME_ID, "gone", "Hafenordnung", "Regenjacken", CLIP):
        assert secret not in loguru_caplog_info.text


def test_a_carried_passage_is_labelled_by_its_header_as_the_agent_reads_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The window lists a carried passage by the header an agent turn's model reads, on one line.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    rag = _carrying(monkeypatch, _store(), _Docstore(_section_node()))

    section, frame = _carry(rag, [SECTION_ID, FRAME_ID])

    assert rag.agent_passage_label(section) == "filename: hafenordnung.pdf; page: 4"
    assert rag.agent_passage_label(frame) == f"keyframe: #3 @ 00:12; docint_doc_kind: image; filename: {CLIP}"
