"""Tests for model configuration loading utilities."""

from pathlib import Path
from types import SimpleNamespace

from docint.utils import model_cfg as model_cfg_module


def test_main_treats_vllm_as_remote_provider(tmp_path: Path, monkeypatch) -> None:
    """vLLM should skip provider-side local model provisioning.

    Args:
        tmp_path: Temporary cache root.
        monkeypatch: Pytest monkeypatch fixture.
    """
    calls: list[tuple[str, str]] = []

    path_config = SimpleNamespace(
        hf_hub_cache=tmp_path / "hf",
    )
    path_config.__dataclass_fields__ = {
        "hf_hub_cache": object(),
    }

    model_config = SimpleNamespace(
        image_embed_model="openai/clip-vit-base-patch32",
        ner_model="gliner-community/gliner_large-v2.5",
        sparse_model="Qdrant/all_miniLM_L6_v2_with_attentions",
        rerank_model="BAAI/bge-reranker-v2-m3",
        embed_model="BAAI/bge-m3",
        text_model="Qwen/Qwen3.5-2B",
        vision_model="Qwen/Qwen3.5-2B",
        whisper_model="base",
    )
    model_config.__dataclass_fields__ = {
        "image_embed_model": object(),
        "ner_model": object(),
        "sparse_model": object(),
        "rerank_model": object(),
        "embed_model": object(),
        "text_model": object(),
        "vision_model": object(),
        "whisper_model": object(),
    }

    monkeypatch.setattr(
        model_cfg_module,
        "load_path_env",
        lambda: path_config,
    )
    monkeypatch.setattr(
        model_cfg_module,
        "load_model_env",
        lambda: model_config,
    )
    monkeypatch.setattr(
        model_cfg_module,
        "load_openai_env",
        lambda: SimpleNamespace(
            inference_provider="vllm", api_base="http://vllm-router:9000/v1"
        ),
    )
    monkeypatch.setattr(
        model_cfg_module,
        "load_clip_model",
        lambda model_id, cache_folder: calls.append(("clip", model_id)),
    )
    monkeypatch.setattr(
        model_cfg_module,
        "load_docling_models",
        lambda: calls.append(("docling", "docling")),
    )
    monkeypatch.setattr(
        model_cfg_module,
        "load_gliner_model",
        lambda model_id, cache_folder: calls.append(("gliner", model_id)),
    )
    monkeypatch.setattr(
        model_cfg_module,
        "load_hf_model",
        lambda model_id, cache_folder, kw, trust_remote_code=False: calls.append(
            (kw, model_id)
        ),
    )
    monkeypatch.setattr(
        model_cfg_module,
        "load_ollama_model",
        lambda *args, **kwargs: calls.append(("ollama", "called")),
    )
    monkeypatch.setattr(
        model_cfg_module,
        "load_whisper_model",
        lambda model_id: calls.append(("whisper", model_id)),
    )

    model_cfg_module.main()

    assert ("clip", "openai/clip-vit-base-patch32") in calls
    assert ("gliner", "gliner-community/gliner_large-v2.5") in calls
    assert ("sparse", "Qdrant/all_miniLM_L6_v2_with_attentions") in calls
    assert ("rerank", "BAAI/bge-reranker-v2-m3") in calls
    assert ("embedding", "BAAI/bge-m3") in calls
    assert ("text", "Qwen/Qwen3.5-2B") in calls
    assert ("vision", "Qwen/Qwen3.5-2B") in calls
    assert ("whisper", "base") in calls
    assert ("ollama", "called") not in calls
