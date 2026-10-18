"""The RAG side of an agent turn: the model it runs on, the view of its passages, and the turn the factory builds.

Every message, passage, clip, filter and collection here is invented.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import openai
import pytest
from agent_fakes import tool_call, turn_budget
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import BaseNode, MetadataMode, NodeWithScore, TextNode

from docint.agents.assessment.ledger import EvidenceLedger
from docint.agents.turn.tools import ReadTool, SearchTool, ToolContext, ToolRegistry
from docint.core.rag import AGENT_METADATA_VALUE_CHARS, RAG, ParentContextPostprocessor, agent_llm_view
from docint.utils.env_cfg import OpenAIConfig

MESSAGE = "Wann öffnen die Stände beim Hafenfest, und hat jemand das Wetter am Sonntag erwähnt?"
#: The post an invented comment answers, as a social export repeats it on the comment's row.
ANCHOR = ("Der Hafenverein verschiebt das Hafenfest wegen der Sturmwarnung auf Sonntag. " * 25)[:1500]
SOCIAL_FILE = "example_profile/dossier.json"
AUTHOR = "Person A"


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


def _search_reply(view: Callable[[BaseNode], BaseNode]) -> str:
    """Show the comment row as an agent turn's search reply does, through a view of its passages.

    Args:
        view (Callable[[BaseNode], BaseNode]): The LLM view the turn renders passages through.

    Returns:
        str: The reply the model reads.
    """
    registry = ToolRegistry([SearchTool(lambda _query: [_comment_row()]), ReadTool()])
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
    assert len(cut) == AGENT_METADATA_VALUE_CHARS
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
