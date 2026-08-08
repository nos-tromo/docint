"""Tests that the dense embedding client targets the configured endpoint.

Production (vLLM) sets no EMBED_* vars and must keep inheriting
OPENAI_API_BASE exactly; a CPU dev host overrides EMBED_API_BASE to
reach the embed-only container without moving chat.
"""

import pytest

from docint.core.rag import RAG


@pytest.fixture()
def _clear_embed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every EMBED_* var so each test controls its own state."""
    for name in ("EMBED_API_BASE", "EMBED_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_embed_config_inherits_openai_base_when_unset(
    monkeypatch: pytest.MonkeyPatch,
    _clear_embed_env: None,
) -> None:
    """Production path: unset EMBED_API_BASE must resolve to OPENAI_API_BASE."""
    monkeypatch.setenv("OPENAI_API_BASE", "http://vllm-router:4000/v1")
    rag = RAG(qdrant_collection="test")
    assert rag.embed_client_config is not None
    assert rag.embed_client_config.api_base == "http://vllm-router:4000/v1"
    assert rag.embed_client_config.api_base == rag.openai_api_base


def test_embed_config_uses_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
    _clear_embed_env: None,
) -> None:
    """Dev path: EMBED_API_BASE moves dense without moving chat."""
    monkeypatch.setenv("OPENAI_API_BASE", "http://ollama:11434/v1")
    monkeypatch.setenv("EMBED_API_BASE", "http://embed-only:8000/v1")
    rag = RAG(qdrant_collection="test")
    assert rag.embed_client_config is not None
    assert rag.embed_client_config.api_base == "http://embed-only:8000/v1"
    assert rag.openai_api_base == "http://ollama:11434/v1"


def test_embed_model_client_targets_the_embed_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    _clear_embed_env: None,
) -> None:
    """The constructed embedding client must use the embed base and key, not the chat ones."""
    monkeypatch.setenv("OPENAI_API_BASE", "http://ollama:11434/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-chat")
    monkeypatch.setenv("EMBED_API_BASE", "http://embed-only:8000/v1")
    monkeypatch.setenv("EMBED_API_KEY", "sk-embed")
    captured: dict[str, object] = {}

    from docint.core import rag as rag_module

    def _capture(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(rag_module, "BudgetedOpenAIEmbedding", _capture)
    rag = RAG(qdrant_collection="test")
    _ = rag.embed_model

    assert captured["api_base"] == "http://embed-only:8000/v1"
    assert captured["api_key"] == "sk-embed"


def _rag_against_a_bare_embed_host(monkeypatch: pytest.MonkeyPatch) -> RAG:
    """Build a RAG whose dense embeddings target a ``/v1``-less base URL.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        RAG: Configured the way the bug report's deployment was.
    """
    monkeypatch.setenv("OPENAI_API_BASE", "http://ollama:11434/v1")
    monkeypatch.setenv("EMBED_API_BASE", "http://embed-only:8000")
    monkeypatch.setenv("EMBED_API_KEY", "sk-embed")
    return RAG(qdrant_collection="test")


def test_probe_raises_when_the_embed_endpoint_cannot_serve(
    monkeypatch: pytest.MonkeyPatch,
    _clear_embed_env: None,
) -> None:
    """A dead dense endpoint must fail the run before any file work.

    Document parsing runs first and is the expensive part — vision OCR
    alone costs minutes per page. Without this probe the operator learns
    the endpoint was misconfigured only when the first embed batch is
    submitted, hours in.
    """
    from docint.utils.openai_cfg import EmbeddingEndpointError

    def _not_found(self: object, query: str) -> list[float]:
        """Raise the 404 a ``/v1``-less base URL produces.

        Args:
            self: Embedding instance.
            query: Probe text.

        Raises:
            NotFoundError: Always.
        """
        _ = (self, query)
        import httpx
        from openai import NotFoundError

        request = httpx.Request("POST", "http://embed-only:8000/embeddings")
        body = {"detail": "Not Found"}
        raise NotFoundError(
            f"Error code: 404 - {body}",
            response=httpx.Response(404, request=request, json=body),
            body=body,
        )

    monkeypatch.setattr(
        "llama_index.embeddings.openai.base.OpenAIEmbedding._get_query_embedding",
        _not_found,
    )

    with pytest.raises(EmbeddingEndpointError, match="EMBED_API_BASE"):
        _rag_against_a_bare_embed_host(monkeypatch).probe_embed_endpoint()


def test_probe_passes_when_the_embed_endpoint_answers(
    monkeypatch: pytest.MonkeyPatch,
    _clear_embed_env: None,
) -> None:
    """A healthy endpoint lets the ingest proceed."""
    calls: list[str] = []

    def _ok(self: object, query: str) -> list[float]:
        """Return a vector, recording the probe text.

        Args:
            self: Embedding instance.
            query: Probe text.

        Returns:
            list[float]: A stand-in embedding.
        """
        _ = self
        calls.append(query)
        return [0.1, 0.2]

    monkeypatch.setattr(
        "llama_index.embeddings.openai.base.OpenAIEmbedding._get_query_embedding",
        _ok,
    )

    _rag_against_a_bare_embed_host(monkeypatch).probe_embed_endpoint()

    assert len(calls) == 1


def test_ingest_docs_probes_the_embed_endpoint_before_any_file_work(
    monkeypatch: pytest.MonkeyPatch,
    _clear_embed_env: None,
) -> None:
    """``ingest_docs`` must run the dense probe before staging any file.

    Pins the ordering at the call site: collection resolution, then both
    endpoint probes, then — only if they pass — file staging. Without
    this test a refactor of the preamble could drop the call with the
    rest of the suite still green, since the probe tests above only ever
    call the method directly.
    """

    class _SentinelError(Exception):
        """Distinguishes the probe's failure from any other exception."""

    def _raise_sentinel(self: RAG) -> None:
        """Stand in for a probe that finds the endpoint dead.

        Args:
            self: The RAG instance.

        Raises:
            _SentinelError: Always.
        """
        _ = self
        raise _SentinelError("embed probe sentinel")

    prepare_calls: list[object] = []

    def _record_prepare(self: RAG, data_dir: object) -> object:
        """Record that file staging ran.

        Args:
            self: The RAG instance.
            data_dir: The directory being staged.

        Returns:
            object: The directory, unchanged.
        """
        _ = self
        prepare_calls.append(data_dir)
        return data_dir

    monkeypatch.setattr(RAG, "create_collection_if_missing", lambda self: None)
    monkeypatch.setattr(RAG, "probe_sparse_endpoint", lambda self: None)
    monkeypatch.setattr(RAG, "probe_embed_endpoint", _raise_sentinel)
    monkeypatch.setattr(RAG, "_prepare_sources_dir", _record_prepare)

    rag = RAG(qdrant_collection="test")
    with pytest.raises(_SentinelError, match="embed probe sentinel"):
        rag.ingest_docs("/unused/path", build_query_engine=False)

    assert prepare_calls == []
