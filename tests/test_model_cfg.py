"""Tests for model configuration loading utilities."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

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
        lambda model_id: calls.append(("local-whisper", model_id)),
    )
    monkeypatch.setattr(
        model_cfg_module,
        "load_silero_vad_model",
        lambda: calls.append(("silero-vad", model_cfg_module.SILERO_VAD_REPO)),
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
    assert ("silero-vad", "snakers4/silero-vad") in calls
    assert ("local-whisper", "base") not in calls
    assert ("ollama", "called") not in calls


def test_load_silero_vad_model_caches_via_torch_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: the preloader must forward to :func:`torch.hub.load`.

    The audio reader's VAD guard calls ``torch.hub.load`` lazily on first
    use. ``load-models`` must pre-populate the torch hub cache with the
    exact same repo reference so operators can then flip
    ``DOCINT_OFFLINE=1`` without the guard silently degrading. The test
    stubs ``torch.hub.load`` on the model_cfg module, invokes the
    preloader, and asserts the exact positional/keyword shape of the call.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces ``torch.hub.load`` with
            a capturing stub; the replacement is undone at teardown.
    """

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class _FakeHub:
        """Captures positional and keyword args handed to ``hub.load``."""

        @staticmethod
        def load(*args: Any, **kwargs: Any) -> tuple[str, tuple[str, ...]]:
            calls.append((args, kwargs))
            return ("fake-model", ("fake-utils",))

    monkeypatch.setattr(model_cfg_module.torch, "hub", _FakeHub)

    model_cfg_module.load_silero_vad_model()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (model_cfg_module.SILERO_VAD_REPO,)
    assert kwargs == {"model": "silero_vad", "trust_repo": True}


def test_load_silero_vad_model_swallows_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preload failures must be logged and swallowed, never re-raised.

    Silero VAD is graceful-degrade: the audio reader falls back to
    RMS-only guarding when VAD is unavailable at runtime, so a failed
    preload is an inconvenience — not a reason to abort the rest of
    ``load-models``. This test forces ``torch.hub.load`` to raise and
    asserts that :func:`load_silero_vad_model` returns cleanly and emits
    a ``warning`` that identifies the repo.

    Args:
        monkeypatch (pytest.MonkeyPatch): Injects a failing ``hub.load``
            and a capturing fake logger.
    """

    class _BoomHub:
        """Raises on every ``load`` call."""

        @staticmethod
        def load(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("no network")

    monkeypatch.setattr(model_cfg_module.torch, "hub", _BoomHub)

    warning_messages: list[str] = []

    class _FakeLogger:
        """Minimal logger stub collecting warnings for assertion."""

        def info(self, message: str, *args: Any) -> None:
            """Discard info-level messages — the failure test only checks warnings."""

        def warning(self, message: str, *args: Any) -> None:
            """Record the formatted warning message for later inspection.

            Args:
                message (str): Loguru-style template string.
                *args (Any): Positional interpolation arguments.
            """

            warning_messages.append(message.format(*args))

    monkeypatch.setattr(model_cfg_module, "logger", _FakeLogger())

    # Must return without raising — this is the primary contract.
    model_cfg_module.load_silero_vad_model()

    assert any(
        "Failed to preload Silero VAD" in msg
        and model_cfg_module.SILERO_VAD_REPO in msg
        and "no network" in msg
        for msg in warning_messages
    ), warning_messages
