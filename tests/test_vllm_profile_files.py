"""Assertions for the external vLLM deployment configuration."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = REPO_ROOT / "templates" / "docint-vllm"


def test_docint_compose_uses_external_vllm_profiles() -> None:
    """Docint should use external vLLM profiles without bundled services."""

    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert 'profiles: ["cpu-vllm"]' in compose
    assert 'profiles: ["cuda-vllm"]' in compose
    assert "OPENAI_API_BASE: ${OPENAI_API_BASE:?" in compose
    assert "vllm-router:" not in compose
    assert "vllm-chat-cuda:" not in compose
    assert "vllm-embed-cuda:" not in compose
    assert "vllm-audio-cuda:" not in compose
    assert "vllm-rerank-cuda:" not in compose
    assert "Dockerfile.vllm" not in compose
    assert "nginx.vllm.conf" not in compose


def test_bundled_vllm_artifacts_are_removed_from_repo_root() -> None:
    """Bundled root-level vLLM deployment artifacts should be gone."""

    assert not (REPO_ROOT / "Dockerfile.vllm").exists()
    assert not (REPO_ROOT / "nginx.vllm.conf").exists()


def test_standalone_vllm_template_contains_required_files() -> None:
    """The standalone vLLM app scaffold should exist in the template path."""

    required_files = {
        ".env.example",
        "docker-compose.yml",
        "Dockerfile.vllm",
        "nginx.vllm.conf",
        "README.md",
    }

    assert TEMPLATE_ROOT.exists()
    assert required_files.issubset({path.name for path in TEMPLATE_ROOT.iterdir()})


def test_standalone_vllm_template_router_exposes_required_routes() -> None:
    """The standalone vLLM router should expose sparse and audio routes."""

    config = (TEMPLATE_ROOT / "nginx.vllm.conf").read_text(encoding="utf-8")

    assert "location = /pooling" in config
    assert "location = /tokenize" in config
    assert "location = /v1/audio/transcriptions" in config
    assert "location = /v1/audio/translations" in config
