"""Tests for model configuration loading utilities."""

from pathlib import Path

from docint.utils.model_cfg import load_llama_cpp_model


def test_load_llama_cpp_model_stores_vlm_assets_under_model_directory(
    tmp_path: Path, monkeypatch
) -> None:
    """Vision model assets should be placed under ``<model-file>/``.

    Args:
        tmp_path: Temporary cache root.
        monkeypatch: Pytest monkeypatch fixture.
    """
    cache_dir = tmp_path / "llama.cpp"
    vision_file = "Qwen3.5-9B-Q4_K_M.gguf"
    mmproj_file = "mmproj-F16.gguf"
    vision_dir = cache_dir / vision_file

    downloaded_to: list[Path] = []

    def _fake_hf_hub_download(
        repo_id: str,
        filename: str,
        cache_dir: Path,
        local_dir: Path,
        local_dir_use_symlinks: bool,
    ) -> str:
        del repo_id, cache_dir, local_dir_use_symlinks
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"gguf")
        downloaded_to.append(destination)
        return str(destination)

    monkeypatch.setattr(
        "docint.utils.model_cfg.resolve_hf_cache_path", lambda **_: None
    )
    monkeypatch.setattr("docint.utils.model_cfg.hf_hub_download", _fake_hf_hub_download)

    load_llama_cpp_model(
        cache_dir=cache_dir,
        model_id=vision_file,
        repo_id="unsloth/Qwen3.5-9B-GGUF",
        kw="vision",
        destination_dir=vision_dir,
    )
    load_llama_cpp_model(
        cache_dir=cache_dir,
        model_id=mmproj_file,
        repo_id="unsloth/Qwen3.5-9B-GGUF",
        kw="vision_mmproj",
        destination_dir=vision_dir,
    )

    assert vision_dir.is_dir()
    assert (vision_dir / vision_file).is_file()
    assert (vision_dir / mmproj_file).is_file()
    assert all(path.parent == vision_dir for path in downloaded_to)
