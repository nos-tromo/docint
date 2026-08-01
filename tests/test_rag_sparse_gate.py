"""Tests for the remote sparse encoder gate.

Sparse embedding is remote on every provider. These tests pin the
selection logic and the frozen wire format; the encoder's request and
response handling must not drift, because production collections were
ingested with it.
"""

from unittest.mock import MagicMock

import pytest

from docint.core import rag as rag_module
from docint.core.rag import RAG, RemoteSparseEncoder

# `RemoteSparseEncoder` is `@dataclass(slots=True)`, so an instance cannot
# take on a `_request_json` attribute that shadows the class method (no
# `__dict__`, and the slot descriptor is read-only). Patch the class method
# instead of the instance; each fake takes `self` as its first argument.


def test_encoder_appends_pooling_and_tokenize_to_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """The encoder owns the route suffixes; the config carries only the base."""
    encoder = RemoteSparseEncoder(api_base="http://sparse-only:8000", model="BAAI/bge-m3")
    captured: list[tuple[str, dict[str, object]]] = []

    def _fake_request(self: RemoteSparseEncoder, url: str, payload: dict[str, object]) -> object:
        captured.append((url, payload))
        if url.endswith("/pooling"):
            return {"data": [{"data": [0.0, 0.7, 0.0]}]}
        return {"tokens": [0, 42, 2]}

    monkeypatch.setattr(RemoteSparseEncoder, "_request_json", _fake_request)
    indices, values = encoder.encode_texts(["alpha"])

    urls = [url for url, _ in captured]
    assert "http://sparse-only:8000/pooling" in urls
    assert "http://sparse-only:8000/tokenize" in urls
    assert indices == [[42]]
    assert values == [[pytest.approx(0.7)]]


def test_encoder_strips_v1_suffix_for_router_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """Against the router the base ends in /v1, but the routes sit at the root."""
    encoder = RemoteSparseEncoder(api_base="http://vllm-router:4000/v1", model="BAAI/bge-m3")
    captured: list[str] = []

    def _fake_request(self: RemoteSparseEncoder, url: str, payload: dict[str, object]) -> object:
        captured.append(url)
        if url.endswith("/pooling"):
            return {"data": [{"data": [0.5]}]}
        return {"tokens": [7]}

    monkeypatch.setattr(RemoteSparseEncoder, "_request_json", _fake_request)
    encoder.encode_texts(["alpha"])

    assert captured[0] == "http://vllm-router:4000/pooling"


def test_encoder_drops_non_positive_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    """ReLU zeroes most tokens; those must not enter the sparse vector."""
    encoder = RemoteSparseEncoder(api_base="http://sparse-only:8000", model="BAAI/bge-m3")

    def _fake_request(self: RemoteSparseEncoder, url: str, payload: dict[str, object]) -> object:
        if url.endswith("/pooling"):
            return {"data": [{"data": [0.0, 0.9, 0.0, 0.4]}]}
        return {"tokens": [0, 11, 2, 12]}

    monkeypatch.setattr(RemoteSparseEncoder, "_request_json", _fake_request)
    indices, values = encoder.encode_texts(["alpha beta"])

    assert indices == [[11, 12]]
    assert values == [[pytest.approx(0.9), pytest.approx(0.4)]]


@pytest.fixture()
def rag_instance() -> RAG:
    """A RAG built the way tests/test_rag_unit.py builds one (no live Qdrant)."""
    return RAG(qdrant_collection="test")


def _vector_store_kwargs(monkeypatch: pytest.MonkeyPatch, rag: RAG) -> dict[str, object]:
    """Capture the kwargs RAG passes to QdrantVectorStore."""
    captured: dict[str, object] = {}

    def _fake_store(**kwargs: object) -> object:
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(rag_module, "QdrantVectorStore", _fake_store)
    rag._vector_store()
    return captured


def test_hybrid_on_wires_remote_encoder_for_docs_and_queries(
    monkeypatch: pytest.MonkeyPatch,
    rag_instance: RAG,
) -> None:
    """Both sparse callbacks come from the remote encoder — never fastembed."""
    rag_instance.enable_hybrid = True
    kwargs = _vector_store_kwargs(monkeypatch, rag_instance)

    assert "fastembed_sparse_model" not in kwargs
    assert callable(kwargs["sparse_doc_fn"])
    assert callable(kwargs["sparse_query_fn"])
    assert kwargs["enable_hybrid"] is True


@pytest.mark.parametrize("provider", ["ollama", "openai", "vllm"])
def test_remote_encoder_used_on_every_provider(
    monkeypatch: pytest.MonkeyPatch,
    rag_instance: RAG,
    provider: str,
) -> None:
    """The provider no longer selects the encoder — only ENABLE_HYBRID does."""
    rag_instance.enable_hybrid = True
    rag_instance.openai_inference_provider = provider
    kwargs = _vector_store_kwargs(monkeypatch, rag_instance)

    assert "fastembed_sparse_model" not in kwargs
    assert callable(kwargs["sparse_doc_fn"])


def test_hybrid_off_wires_no_sparse_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    rag_instance: RAG,
) -> None:
    """Dense-only deployments send no sparse kwargs at all."""
    rag_instance.enable_hybrid = False
    kwargs = _vector_store_kwargs(monkeypatch, rag_instance)

    assert "fastembed_sparse_model" not in kwargs
    assert "sparse_doc_fn" not in kwargs
    assert kwargs["enable_hybrid"] is False
