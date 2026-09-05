"""Tests for model configuration loading utilities."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from docint.utils import model_cfg as model_cfg_module


def _configs(
    monkeypatch: pytest.MonkeyPatch,
    cache_dir: Path,
    repo_id: str,
) -> None:
    """Stub the two env_cfg loaders ``main()`` reads.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        cache_dir: Value for ``PathConfig.hf_hub_cache``.
        repo_id: Value for ``ModelConfig.embed_tokenizer_repo``.
    """
    monkeypatch.setattr(
        model_cfg_module,
        "load_path_env",
        lambda: SimpleNamespace(hf_hub_cache=cache_dir),
    )
    monkeypatch.setattr(
        model_cfg_module,
        "load_model_env",
        lambda: SimpleNamespace(embed_tokenizer_repo=repo_id),
    )


def test_main_downloads_only_the_tokenizer_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cache miss fetches the tokenizer files, never the weights.

    The bge-m3 repo carries ~2.5 GB of weights docint never loads —
    every model call is remote — so the download must be restricted to
    ``TOKENIZER_PATTERNS``.

    Args:
        tmp_path: Temporary cache root.
        monkeypatch: Pytest monkeypatch fixture.
    """
    downloads: list[dict[str, object]] = []
    cache_dir = tmp_path / "hf"
    _configs(monkeypatch, cache_dir, "BAAI/bge-m3")
    monkeypatch.setattr(model_cfg_module, "resolve_hf_cache_path", lambda cache_dir, repo_id: None)
    monkeypatch.setattr(
        model_cfg_module,
        "snapshot_download",
        lambda **kwargs: downloads.append(kwargs),
    )

    model_cfg_module.main()

    assert len(downloads) == 1
    call = downloads[0]
    assert call["repo_id"] == "BAAI/bge-m3"
    assert call["cache_dir"] == cache_dir
    patterns = model_cfg_module.TOKENIZER_PATTERNS
    assert call["allow_patterns"] == list(patterns)
    assert "tokenizer*" in patterns
    assert not any(p.endswith(".safetensors") or p == "*" for p in patterns)


def test_main_skips_a_cached_tokenizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A resolved snapshot short-circuits the download.

    Args:
        tmp_path: Temporary cache root.
        monkeypatch: Pytest monkeypatch fixture.
    """
    downloads: list[dict[str, object]] = []
    snapshot = tmp_path / "hf" / "models--BAAI--bge-m3" / "snapshots" / "abc123"
    _configs(monkeypatch, tmp_path / "hf", "BAAI/bge-m3")
    monkeypatch.setattr(model_cfg_module, "resolve_hf_cache_path", lambda cache_dir, repo_id: snapshot)
    monkeypatch.setattr(
        model_cfg_module,
        "snapshot_download",
        lambda **kwargs: downloads.append(kwargs),
    )

    model_cfg_module.main()

    assert downloads == []


def test_main_is_a_noop_without_a_tokenizer_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty repo id means the provider tokenizes server-side.

    ``load_model_env`` empties ``embed_tokenizer_repo`` for the openai
    provider, so there is nothing to cache and nothing to resolve.

    Args:
        tmp_path: Temporary cache root.
        monkeypatch: Pytest monkeypatch fixture.
    """
    downloads: list[dict[str, object]] = []
    _configs(monkeypatch, tmp_path / "hf", "")

    def _fail(**kwargs: object) -> None:
        """Fail the test if the cache is consulted.

        Args:
            **kwargs: Ignored.
        """
        raise AssertionError("resolve_hf_cache_path must not be called for an empty repo id")

    monkeypatch.setattr(model_cfg_module, "resolve_hf_cache_path", _fail)
    monkeypatch.setattr(
        model_cfg_module,
        "snapshot_download",
        lambda **kwargs: downloads.append(kwargs),
    )

    model_cfg_module.main()

    assert downloads == []


def test_load_embed_tokenizer_passes_the_cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The download lands in the configured hub cache root.

    The Dockerfile builds the image with ``HF_HUB_CACHE=/app/hf-hub``
    and the runtime reads from the same root, so the cache directory
    must come from ``PathConfig`` rather than the HF default.

    Args:
        tmp_path: Temporary cache root.
        monkeypatch: Pytest monkeypatch fixture.
    """
    downloads: list[dict[str, object]] = []
    monkeypatch.setattr(model_cfg_module, "resolve_hf_cache_path", lambda cache_dir, repo_id: None)
    monkeypatch.setattr(
        model_cfg_module,
        "snapshot_download",
        lambda **kwargs: downloads.append(kwargs),
    )

    model_cfg_module.load_embed_tokenizer(repo_id="BAAI/bge-m3", cache_folder=tmp_path / "hub")

    assert downloads[0]["cache_dir"] == tmp_path / "hub"
