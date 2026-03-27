"""Assertions for the bundled vLLM profile configuration files."""

from pathlib import Path


def test_nginx_vllm_routes_include_sparse_and_audio_endpoints() -> None:
    """The vLLM router should expose sparse pooling and audio proxy routes."""

    config = Path("/Users/himarc/dev/nos-tromo/docint/nginx.vllm.conf").read_text()

    assert "location = /pooling" in config
    assert "location = /tokenize" in config
    assert "location = /v1/audio/transcriptions" in config
    assert "location = /v1/audio/translations" in config


def test_docker_compose_vllm_profile_includes_audio_service() -> None:
    """The CUDA vLLM profile should include the routed audio service."""

    compose = Path("/Users/himarc/dev/nos-tromo/docint/docker-compose.yml").read_text()

    assert "vllm-audio-cuda:" in compose
    assert 'depends_on:\n      - vllm-chat-cuda\n      - vllm-embed-cuda\n      - vllm-audio-cuda' in compose
    assert '--hf-overrides \'{"architectures":["BgeM3EmbeddingModel"]}\'' in compose
