"""Tests for the remote sparse encoder gate.

Sparse embedding is remote on every provider. These tests pin the
selection logic and the frozen wire format; the encoder's request and
response handling must not drift, because production collections were
ingested with it.
"""

import pytest

from docint.core.rag import RemoteSparseEncoder

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
