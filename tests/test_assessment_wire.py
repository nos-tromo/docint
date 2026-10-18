"""What a research step actually sends to an OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import time
from typing import Any, cast
from unittest.mock import MagicMock

import httpx
import openai
import pytest
from assessment_fakes import FixedRetriever, RecordingPostprocessor, scored_node
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import Refine
from llama_index.core.schema import QueryBundle
from openai.resources.chat.completions import Completions
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)

import docint.core.api as api_module
from docint.agents.assessment.classify import classify_turn
from docint.agents.assessment.ledger import EvidenceLedger
from docint.agents.assessment.research import ResearchLoop
from docint.core.rag import RAG
from docint.core.retrieval.evidence import EveryChildQuery
from docint.utils.env_cfg import OpenAIConfig


def _rag(*, max_retries: int = 0) -> RAG:
    """Build a RAG whose text model targets a vLLM-style endpoint.

    Args:
        max_retries (int): ``OPENAI_MAX_RETRIES``; two is the production default.

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
        max_retries=max_retries,
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


def _text_completion(content: str) -> ChatCompletion:
    """Build a chat completion in which the model answers in text.

    Args:
        content (str): The reply.

    Returns:
        ChatCompletion: The completion.
    """
    message = ChatCompletionMessage(role="assistant", content=content)
    return ChatCompletion(
        id="c2",
        created=int(time.time()),
        model="test-model",
        object="chat.completion",
        choices=[Choice(index=0, finish_reason="stop", message=message)],
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
    # The SDK's per-request timeout: the research budget still left, two minutes at most.
    assert 0 < body["timeout"] <= 120.0
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


def test_the_turn_classifier_is_bounded_in_length_and_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """It runs before every turn, so a verbose or slow model must fail soft quickly.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    sent = _serve(monkeypatch, [_text_completion('{"mode": "assessment"}')])

    mode = classify_turn(_rag().text_model, question="What could be improved?", prior_turn=None)

    assert mode == "assessment"
    assert sent[0]["max_tokens"] == 64
    assert sent[0]["timeout"] == 15.0


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
    assert isinstance(bundle, EveryChildQuery)
    assert bundle.query_str == "method"


def test_a_research_search_asks_a_real_engine_for_every_child() -> None:
    """The flag reaches the turn engine's postprocessors on every research search, as it does on the seed."""
    recorder = RecordingPostprocessor()
    engine = RetrieverQueryEngine(
        retriever=FixedRetriever([scored_node("m", "Method section.")]),
        response_synthesizer=Refine(llm=MockLLM()),
        node_postprocessors=[recorder],
    )
    researcher = _rag().assessment_runner(engine, scoped=False)._researcher
    assert isinstance(researcher, ResearchLoop)

    found = researcher._search("method")

    assert [scored.node.node_id for scored in found] == ["m"]
    assert [type(bundle) for bundle in recorder.bundles] == [EveryChildQuery]
    assert [cast(QueryBundle, bundle).query_str for bundle in recorder.bundles] == ["method"]


def test_the_research_step_cap_is_assessment_max_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented ``ASSESSMENT_MAX_STEPS`` knob reaches the loop.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("ASSESSMENT_MAX_STEPS", "4")

    researcher = _rag().assessment_runner(MagicMock(), scoped=False)._researcher

    assert isinstance(researcher, ResearchLoop)
    assert researcher._max_steps == 4


def test_the_runner_counts_with_the_embedding_tokenizer_when_one_is_loaded() -> None:
    """Measured density needs a tokenizer; without one the runner keeps the configured ratio."""
    rag = _rag()
    rag._embed_token_counter = lambda text: list(range(len(text) // 2))

    with_tokenizer = rag.assessment_runner(MagicMock(), scoped=True)
    rag._embed_token_counter = None
    without = rag.assessment_runner(MagicMock(), scoped=True)

    assert with_tokenizer._count_tokens is not None
    assert with_tokenizer._count_tokens("abcdefgh") == 4
    assert without._count_tokens is None


def test_research_picks_queries_on_the_plain_model_and_the_toggle_chooses_the_answer_model() -> None:
    """The reasoning toggle reaches the answer only; a thinking model never drives research."""
    rag = _rag()

    with rag.reasoning_scope(True):
        thinking = rag.assessment_runner(MagicMock(), scoped=False)
        thinking_answer_llm = rag.post_retrieval_text_model
    with rag.reasoning_scope(False):
        plain = rag.assessment_runner(MagicMock(), scoped=False)

    for runner in (thinking, plain):
        assert isinstance(runner._researcher, ResearchLoop)
        assert runner._researcher._llm is rag.bounded_text_model
        # Only research is bounded; the answer keeps the retries.
        assert runner._answer_llm is not rag.bounded_text_model
    assert thinking._answer_llm is thinking_answer_llm
    assert thinking._answer_llm is not rag.text_model
    assert thinking._reasoning is True
    assert plain._answer_llm is rag.text_model
    assert plain._reasoning is False


def _stalled_endpoint(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Make every request time out at the HTTP transport, recording each attempt that reaches it.

    The SDK's own retries and llama-index's both sit above the transport, so the
    count is every attempt a call makes. Backoff sleeps are skipped.

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
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    return attempts


def test_the_bounded_model_makes_one_attempt_at_a_stalled_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries are off for the bounded model, in the SDK and in llama-index alike.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    attempts = _stalled_endpoint(monkeypatch)

    with pytest.raises(openai.APITimeoutError):
        _rag(max_retries=2).bounded_text_model.complete("Is this a judgement?", timeout=15.0)

    assert len(attempts) == 1


def test_the_plain_model_retries_a_stalled_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control: the same stall costs the plain model several attempts, which a time bound cannot afford.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    attempts = _stalled_endpoint(monkeypatch)

    with pytest.raises(openai.APITimeoutError):
        _rag(max_retries=2).text_model.complete("Is this a judgement?", timeout=15.0)

    assert len(attempts) > 1


def test_the_turn_classifier_makes_one_attempt_at_a_stalled_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """One attempt per call, so a classification abandoned at its deadline frees its slot at its own timeout.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    attempts = _stalled_endpoint(monkeypatch)
    monkeypatch.setenv("ASSESSMENT_ENABLED", "true")
    monkeypatch.setattr(api_module, "rag", _rag(max_retries=2))

    assert api_module._classify_turn("Is the method convincing?", None, "all") == "lookup"
    assert len(attempts) == 1


def test_research_makes_one_attempt_per_call_at_a_stalled_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """One attempt per call, so a research call abandoned at the budget frees its slot at its own timeout.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    attempts = _stalled_endpoint(monkeypatch)
    researcher = _rag(max_retries=2).assessment_runner(MagicMock(), scoped=False)._researcher
    assert isinstance(researcher, ResearchLoop)

    items = list(researcher.run(question="Is the method convincing?", prior_turn=None, ledger=EvidenceLedger(cap=5)))

    assert getattr(items[-1], "reason", None) == "research_failed"
    assert len(attempts) == 1


def test_the_bounded_model_is_the_plain_one_with_retries_off_and_the_answer_keeps_its_retries() -> None:
    """Only the retry count differs from the plain model; the answer models keep ``OPENAI_MAX_RETRIES``."""
    rag = _rag(max_retries=2)

    bounded = rag.bounded_text_model
    with rag.reasoning_scope(True):
        thinking_answer = rag.post_retrieval_text_model
    with rag.reasoning_scope(False):
        plain_answer = rag.post_retrieval_text_model

    assert bounded is rag.bounded_text_model
    assert bounded is not rag.text_model
    assert bounded.max_retries == 0
    assert bounded.additional_kwargs == rag.text_model.additional_kwargs
    assert bounded.additional_kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert (bounded.model, bounded.api_base, bounded.timeout) == (
        rag.text_model.model,
        rag.text_model.api_base,
        rag.text_model.timeout,
    )
    assert rag.text_model.max_retries == 2
    assert plain_answer is rag.text_model
    assert thinking_answer.max_retries == 2
