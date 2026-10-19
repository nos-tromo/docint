"""What an agent turn's model client makes of a stalled endpoint, and what its searches ask a real engine."""

from __future__ import annotations

import time
from typing import Any, cast

import httpx
import openai
import pytest
from agent_fakes import KeywordClient, tool_call
from assessment_fakes import FixedRetriever, RecordingPostprocessor, scored_node
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import Refine
from llama_index.core.schema import QueryBundle

from docint.agents.turn.history import ConversationWindow
from docint.agents.turn.tools import ToolContext
from docint.core.rag import RAG, agent_llm_view
from docint.core.retrieval.evidence import AgentSearchQuery, EveryChildQuery
from docint.utils.env_cfg import OpenAIConfig


def _rag(*, max_retries: int = 0) -> RAG:
    """Build a RAG whose text model targets a vLLM-style endpoint.

    Args:
        max_retries (int): ``OPENAI_MAX_RETRIES``; two is the production default.

    Returns:
        RAG: The engine, with only the OpenAI knobs set, over an empty store.
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
    # A store holding no collection, so a turn reads no live Qdrant to decide on keyword.
    rag._qdrant_client = cast(Any, KeywordClient({}))
    return rag


def test_an_agent_search_asks_a_real_engine_for_every_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flag reaches the turn engine's postprocessors through llama-index's own retrieve, on every search.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pins the prompts' locale and leaves the query unexpanded.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    recorder = RecordingPostprocessor()
    engine = RetrieverQueryEngine(
        retriever=FixedRetriever([scored_node("m", "Method section.")]),
        response_synthesizer=Refine(llm=MockLLM()),
        node_postprocessors=[recorder],
    )
    rag = _rag()
    monkeypatch.setattr(rag, "expand_query_with_graph", lambda query: query)
    turn = rag.agent_turn(
        engine,
        retrieval_target="documents",
        scoped=False,
        pinned_count=0,
        window=ConversationWindow(),
        message="What does the method section say?",
        metadata_filter_rules=None,
        metadata_filters_active=False,
        vector_store_kwargs=None,
    )
    assert turn is not None
    ctx = ToolContext(ledger=turn.ledger, budget=turn.budget, llm_view=agent_llm_view)

    reply = turn.loop._registry.call(tool_call("search", query="method"), ctx)

    assert reply.found == 1
    assert [type(bundle) for bundle in recorder.bundles] == [AgentSearchQuery]
    assert isinstance(recorder.bundles[0], EveryChildQuery)
    assert [cast(QueryBundle, bundle).query_str for bundle in recorder.bundles] == ["method"]


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
