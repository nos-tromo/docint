import pytest

pytest.importorskip("magic", reason="libmagic is required for RAG device tests")

from docint.core import rag as rag_module
from docint.core.rag import RAG


class _EmbedSentinel:
    pass


def test_embed_model_falls_back_to_cpu(monkeypatch):
    rag = RAG(qdrant_collection="unit-test")
    rag._device = "mps"

    attempts: list[str] = []

    def _failing_embedding(*, model_name, normalize, device):
        attempts.append(device)
        if len(attempts) == 1:
            raise RuntimeError("Cannot copy out of meta tensor; no data!")
        return _EmbedSentinel()

    monkeypatch.setattr(rag_module, "HuggingFaceEmbedding", _failing_embedding)

    result = rag.embed_model

    assert isinstance(result, _EmbedSentinel)
    assert attempts == ["mps", "cpu"]
    assert rag.device == "cpu"


class _RerankerSentinel:
    pass


def test_reranker_falls_back_to_cpu(monkeypatch):
    rag = RAG(qdrant_collection="unit-test")
    rag._device = "mps"

    attempts: list[str] = []

    def _failing_reranker(*, top_n, model, device):
        attempts.append(device)
        if len(attempts) == 1:
            raise RuntimeError("Cannot copy out of meta tensor; no data!")
        return _RerankerSentinel()

    monkeypatch.setattr(rag_module, "SentenceTransformerRerank", _failing_reranker)

    result = rag.reranker

    assert isinstance(result, _RerankerSentinel)
    assert attempts == ["mps", "cpu"]
    assert rag.device == "cpu"
