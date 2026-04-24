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
    )
    model_config.__dataclass_fields__ = {
        "image_embed_model": object(),
        "ner_model": object(),
        "sparse_model": object(),
        "rerank_model": object(),
        "embed_model": object(),
        "text_model": object(),
        "vision_model": object(),
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

    model_cfg_module.main()

    assert ("clip", "openai/clip-vit-base-patch32") in calls
    assert ("gliner", "gliner-community/gliner_large-v2.5") in calls
    assert ("sparse", "Qdrant/all_miniLM_L6_v2_with_attentions") in calls
    assert ("rerank", "BAAI/bge-reranker-v2-m3") in calls
    assert ("embedding", "BAAI/bge-m3") in calls
    assert ("text", "Qwen/Qwen3.5-2B") in calls
    assert ("vision", "Qwen/Qwen3.5-2B") in calls
    assert ("ollama", "called") not in calls


def test_load_models_populates_embed_tokenizer_cache(
    tmp_path: Path, monkeypatch
) -> None:
    """``main()`` must pre-warm the HF cache for the embed tokenizer repo.

    Operators set ``EMBED_TOKENIZER_REPO=BAAI/bge-m3`` so the
    ingestion path can load the tokenizer with
    ``local_files_only=True``. ``load-models`` is the only moment
    we are online, so the snapshot has to be fetched here. The
    contract: ``load_hf_model`` is called with
    ``model_id=<embed_tokenizer_repo>``, ``kw="embed-tokenizer"``,
    and ``cache_folder=path_config.hf_hub_cache`` regardless of the
    provider (ollama/vllm/openai) — the tokenizer ships with the
    worker, not the provider.

    Args:
        tmp_path: Temporary cache root.
        monkeypatch: Pytest monkeypatch fixture.
    """
    calls: list[tuple[str, str]] = []

    path_config = SimpleNamespace(hf_hub_cache=tmp_path / "hf")
    path_config.__dataclass_fields__ = {"hf_hub_cache": object()}

    model_config = SimpleNamespace(
        image_embed_model="openai/clip-vit-base-patch32",
        ner_model="gliner-community/gliner_large-v2.5",
        sparse_model="Qdrant/all_miniLM_L6_v2_with_attentions",
        rerank_model="BAAI/bge-reranker-v2-m3",
        embed_model="bge-m3",
        text_model="gpt-oss:20b",
        vision_model="qwen3.5:9b",
        embed_tokenizer_repo="BAAI/bge-m3",
    )
    model_config.__dataclass_fields__ = {
        "image_embed_model": object(),
        "ner_model": object(),
        "sparse_model": object(),
        "rerank_model": object(),
        "embed_model": object(),
        "text_model": object(),
        "vision_model": object(),
        "embed_tokenizer_repo": object(),
    }

    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.setenv("EMBED_TOKENIZER_REPO", "BAAI/bge-m3")

    monkeypatch.setattr(model_cfg_module, "load_path_env", lambda: path_config)
    monkeypatch.setattr(model_cfg_module, "load_model_env", lambda: model_config)
    monkeypatch.setattr(
        model_cfg_module,
        "load_openai_env",
        lambda: SimpleNamespace(
            inference_provider="ollama", api_base="http://localhost:11434/v1"
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

    captured_hf_calls: list[dict[str, object]] = []

    def _fake_load_hf_model(
        model_id: str,
        cache_folder: Path,
        kw: str,
        trust_remote_code: bool = False,
    ) -> None:
        """Record every ``load_hf_model`` invocation.

        Args:
            model_id: HF repo id.
            cache_folder: HF hub cache root.
            kw: Keyword describing the asset class (e.g. "sparse",
                "rerank", "embed-tokenizer").
            trust_remote_code: Forwarded trust_remote_code flag.
        """
        captured_hf_calls.append(
            {
                "model_id": model_id,
                "cache_folder": cache_folder,
                "kw": kw,
                "trust_remote_code": trust_remote_code,
            }
        )
        calls.append((kw, model_id))

    monkeypatch.setattr(model_cfg_module, "load_hf_model", _fake_load_hf_model)
    monkeypatch.setattr(
        model_cfg_module,
        "load_ollama_model",
        lambda *args, **kwargs: calls.append(("ollama", "called")),
    )

    model_cfg_module.main()

    embed_tokenizer_calls = [
        c for c in captured_hf_calls if c["kw"] == "embed-tokenizer"
    ]
    assert embed_tokenizer_calls, (
        "load_hf_model must be called with kw='embed-tokenizer' when "
        "EMBED_TOKENIZER_REPO is set"
    )
    call = embed_tokenizer_calls[0]
    assert call["model_id"] == "BAAI/bge-m3"
    assert call["cache_folder"] == path_config.hf_hub_cache
