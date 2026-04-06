"""Assertions for the external vLLM deployment configuration."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_docint_compose_uses_external_vllm_profiles() -> None:
    """Docint should use provider-agnostic profiles without bundled vLLM services."""

    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert 'profiles: ["cpu"]' in compose
    assert 'profiles: ["cuda"]' in compose
    assert "\n  vllm-router:\n" not in compose
    assert "vllm-chat-cuda:" not in compose
    assert "vllm-embed-cuda:" not in compose
    assert "vllm-audio-cuda:" not in compose
    assert "vllm-rerank-cuda:" not in compose
    assert "Dockerfile.vllm" not in compose
    assert "nginx.vllm.conf" not in compose


def test_docint_compose_joins_shared_INFERENCE_NET() -> None:
    """Docint should expose the backend on the shared external proxy network."""

    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "inference-net:" in compose
    assert "name: ${INFERENCE_NET:-inference-net}" in compose
    assert "external: true" in compose
    assert "- docint-backend" in compose


def test_bundled_vllm_artifacts_are_removed_from_docint() -> None:
    """Bundled vLLM assets should be removed from the Docint repository."""

    assert not (REPO_ROOT / "Dockerfile.vllm").exists()
    assert not (REPO_ROOT / "nginx.vllm.conf").exists()
    assert not (REPO_ROOT / "templates" / "docint-vllm").exists()
