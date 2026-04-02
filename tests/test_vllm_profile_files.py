"""Assertions for the bundled vLLM profile configuration files."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_nginx_vllm_routes_include_sparse_and_audio_endpoints() -> None:
    """The vLLM router should expose sparse pooling and audio proxy routes."""

    config = (REPO_ROOT / "nginx.vllm.conf").read_text(encoding="utf-8")

    assert "location = /pooling" in config
    assert "location = /tokenize" in config
    assert "location = /v1/audio/transcriptions" in config
    assert "location = /v1/audio/translations" in config


def test_docker_compose_vllm_profile_includes_audio_service() -> None:
    """The CUDA vLLM profile should include the routed audio service."""

    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "vllm-audio-cuda:" in compose
    # Router waits for all vLLM services to pass their health check.
    assert "vllm-chat-cuda:\n        condition: service_healthy" in compose
    assert "vllm-audio-cuda:\n        condition: service_healthy" in compose
    assert "USE_DEVICE: cpu" in compose
    assert '--hf-overrides \'{"architectures":["BgeM3EmbeddingModel"]}\'' in compose
