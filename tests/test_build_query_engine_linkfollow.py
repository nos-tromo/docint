"""Tests for LinkFollowingPostprocessor wiring in build_query_engine."""

import threading
from collections import OrderedDict
from typing import Any

import pytest


def test_link_following_added_for_social_collections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert that LinkFollowingPostprocessor is wired into the social-table query engine.

    RAG is @dataclass(slots=True) so instance-level attribute patching is not
    possible for methods or properties.  All seams are therefore patched at the
    class level; monkeypatch restores the originals automatically after the test.
    """
    from docint.core import rag as rag_module

    captured: dict[str, Any] = {}

    class _FakeQE:
        @classmethod
        def from_args(cls, **kwargs: Any) -> "_FakeQE":
            captured.update(kwargs)
            return cls()

    monkeypatch.setattr(rag_module, "RetrieverQueryEngine", _FakeQE)

    # Patch methods at the class level (slots=True forbids per-instance override).
    monkeypatch.setattr(rag_module.RAG, "create_index", lambda self: None)
    monkeypatch.setattr(rag_module.RAG, "_infer_collection_profile", lambda self: {"is_social_table": True})
    monkeypatch.setattr(
        rag_module.RAG,
        "_resolve_runtime_retrieval_settings",
        lambda self, **k: {"parent_context_enabled": False},
    )
    monkeypatch.setattr(rag_module.RAG, "_resolve_chat_response_mode", lambda self: "compact")
    monkeypatch.setattr(rag_module.RAG, "_build_retriever", lambda self, **k: object())
    monkeypatch.setattr(rag_module.RAG, "_build_grounded_text_qa_template", lambda self, **k: None)
    monkeypatch.setattr(rag_module.RAG, "_build_grounded_refine_template", lambda self, **k: None)
    # post_retrieval_text_model is a @property — patch as a descriptor at class
    # level. A real (mock) LLM is required: build_query_engine constructs the
    # response synthesizer eagerly, and a None llm would make llama-index
    # resolve Settings.llm — a real OpenAI client that fails on keyless CI.
    from llama_index.core.llms import MockLLM

    fake_llm = MockLLM(max_tokens=8)
    monkeypatch.setattr(rag_module.RAG, "post_retrieval_text_model", property(lambda self: fake_llm))

    rag = rag_module.RAG.__new__(rag_module.RAG)
    # index is a property backed by the _index_cache slot (guarded by
    # _retrieval_cache_lock) keyed by qdrant_collection. RAG.__new__ skips
    # __init__, so seed the backing slots before assigning through the index
    # setter. SocialSourceDiversityPostprocessor is constructed with its own
    # default diversity_limit, so no RAG-level knob needs seeding here.
    rag.qdrant_collection = "c"
    rag._retrieval_cache_lock = threading.Lock()
    rag._index_cache = OrderedDict()
    rag._resolved_index_cache = {}
    rag.index = object()  # type: ignore[assignment]

    rag.build_query_engine()
    kinds = [type(p).__name__ for p in captured["node_postprocessors"]]
    assert "LinkFollowingPostprocessor" in kinds
