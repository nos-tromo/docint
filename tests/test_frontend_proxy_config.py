"""Assertions for the frontend nginx proxy configuration."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_frontend_image_uses_template_for_upload_limit() -> None:
    """The frontend image should render nginx config from a runtime template."""
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.frontend").read_text(encoding="utf-8")

    assert "ENV DOCINT_CLIENT_MAX_BODY_SIZE=1g" in dockerfile
    assert "default.conf.template" in dockerfile
    assert "default.conf /etc/nginx/conf.d/default.conf" not in dockerfile


def test_frontend_compose_exposes_upload_limit_override() -> None:
    """Compose should let operators raise the nginx upload ceiling via .env."""
    compose = (REPO_ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")

    assert "DOCINT_CLIENT_MAX_BODY_SIZE: ${DOCINT_CLIENT_MAX_BODY_SIZE:-1g}" in compose


def test_ingest_proxy_uses_configurable_request_limit() -> None:
    """The ingest nginx location should use the configurable multipart limit."""
    nginx_conf = (REPO_ROOT / "frontend" / "nginx" / "default.conf").read_text(encoding="utf-8")

    assert "client_max_body_size ${DOCINT_CLIENT_MAX_BODY_SIZE};" in nginx_conf
    assert "client_max_body_size 200m;" not in nginx_conf