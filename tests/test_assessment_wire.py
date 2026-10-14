"""What a research step actually sends to an OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from assessment_fakes import scored_node
from llama_index.core.schema import QueryBundle
from openai.resources.chat.completions import Completions
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)

from docint.agents.assessment.ledger import EvidenceLedger
from docint.agents.assessment.research import ResearchLoop
from docint.core.rag import RAG
from docint.utils.env_cfg import OpenAIConfig


def _rag() -> RAG:
    """Build a RAG whose text model targets a vLLM-style endpoint.

    Returns:
        RAG: The engine, with only the OpenAI knobs set.
    """
    rag = RAG(qdrant_collection="test")
    rag.text_model_id = "test-model"
    rag.openai_config = OpenAIConfig(
        api_base="http://vllm-router:4000/v1",
        api_key="sk-test",
        ctx_window=32768,
        dimensions=1024,
        max_retries=0,
        num_output=256,
        inference_provider="vllm",
        reuse_client=False,
        seed=42,
        temperature=0.0,
        thinking_effort="medium",
        thinking_enabled=False,
        timeout=5.0,
        top_p=0.0,
    )
    for key in (
        "api_base",
        "api_key",
        "ctx_window",
        "max_retries",
        "reuse_client",
        "seed",
        "temperature",
        "timeout",
        "top_p",
    ):
        setattr(rag, f"openai_{key}", getattr(rag.openai_config, key))
    return rag


def _completion(name: str, arguments: str, call_id: str) -> ChatCompletion:
    """Build a chat completion in which the model calls one tool.

    Args:
        name (str): The tool the model calls.
        arguments (str): The call's arguments, as the JSON string the API carries.
        call_id (str): The id the endpoint gives the call.

    Returns:
        ChatCompletion: The completion.
    """
    call = ChatCompletionMessageFunctionToolCall(
        id=call_id, type="function", function=Function(name=name, arguments=arguments)
    )
    message = ChatCompletionMessage(role="assistant", content=None, tool_calls=[call])
    return ChatCompletion(
        id="c1",
        created=int(time.time()),
        model="test-model",
        object="chat.completion",
        choices=[Choice(index=0, finish_reason="tool_calls", message=message)],
    )


def _serve(monkeypatch: pytest.MonkeyPatch, replies: list[ChatCompletion]) -> list[dict[str, Any]]:
    """Replace the SDK's completions call with one that replays ``replies``.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        replies (list[ChatCompletion]): One completion per request, in order.

    Returns:
        list[dict[str, Any]]: Every request body, as the SDK received it.
    """
    sent: list[dict[str, Any]] = []
    queue = list(replies)

    def _create(self: Any, **kwargs: Any) -> ChatCompletion:
        """Record one request body and answer it with the next completion.

        Args:
            self (Any): The SDK's ``Completions`` resource, unused.
            **kwargs (Any): The request body.

        Returns:
            ChatCompletion: The next scripted completion.
        """
        sent.append(kwargs)
        return queue.pop(0)

    monkeypatch.setattr(Completions, "create", _create)
    return sent


def test_a_research_step_sends_required_single_non_strict_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """``tool_choice="required"`` needs no server flags on vLLM; thinking stays off.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    sent = _serve(monkeypatch, [_completion("finish", "{}", "call_1")])
    loop = ResearchLoop(
        llm=_rag().text_model, search=lambda _query: [], max_steps=3, ctx_window=32768, char_token_ratio=3.5
    )

    list(loop.run(question="What could be improved?", prior_turn=None, ledger=EvidenceLedger(cap=5)))

    body = sent[0]
    assert body["tool_choice"] == "required"
    assert body["parallel_tool_calls"] is False
    assert body["max_tokens"] == 512
    assert body["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert {tool["function"]["name"] for tool in body["tools"]} == {"search", "finish"}
    assert all(tool["function"]["strict"] is False for tool in body["tools"])


def test_the_second_request_answers_the_first_call_by_its_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool reply reaches the endpoint as a ``tool`` message keyed by the call's id.

    A vLLM endpoint validates exactly this pairing; a mocked model cannot show it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    sent = _serve(
        monkeypatch, [_completion("search", '{"query": "method"}', "call_7"), _completion("finish", "{}", "call_8")]
    )
    found = [scored_node("m", "Method section.", filename="study.pdf")]
    loop = ResearchLoop(
        llm=_rag().text_model, search=lambda _query: found, max_steps=3, ctx_window=32768, char_token_ratio=3.5
    )

    list(loop.run(question="What could be improved?", prior_turn=None, ledger=EvidenceLedger(cap=5)))

    assert len(sent) == 2
    call_message, reply_message = sent[1]["messages"][1:3]
    assert call_message["role"] == "assistant"
    assert call_message["tool_calls"][0]["id"] == "call_7"
    assert call_message["tool_calls"][0]["function"]["name"] == "search"
    assert json.loads(call_message["tool_calls"][0]["function"]["arguments"]) == {"query": "method"}
    assert reply_message["role"] == "tool"
    assert reply_message["tool_call_id"] == "call_7"
    assert "[1] (study.pdf) Method section." in reply_message["content"]


def test_only_an_unscoped_turn_gets_a_researcher_that_searches_the_turns_engine() -> None:
    """A pinned scope forbids searching beyond it; an open turn researches through its own engine."""
    found = [scored_node("m", "Method section.")]
    engine = MagicMock()
    engine.retrieve.return_value = found
    rag = _rag()

    scoped = rag.assessment_runner(engine, scoped=True)
    unscoped = rag.assessment_runner(engine, scoped=False)

    assert scoped._researcher is None
    researcher = unscoped._researcher
    assert isinstance(researcher, ResearchLoop)
    assert researcher._search("method") == found
    engine.retrieve.assert_called_once()
    bundle = engine.retrieve.call_args.args[0]
    assert isinstance(bundle, QueryBundle)
    assert bundle.query_str == "method"


def test_the_research_step_cap_is_assessment_max_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented ``ASSESSMENT_MAX_STEPS`` knob reaches the loop.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("ASSESSMENT_MAX_STEPS", "4")

    researcher = _rag().assessment_runner(MagicMock(), scoped=False)._researcher

    assert isinstance(researcher, ResearchLoop)
    assert researcher._max_steps == 4
