"""Assertions for the frontend nginx proxy configuration."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _nginx_api_route_tokens() -> list[str]:
    """Return the backend-API ``location`` regex alternation tokens.

    Parses the ``location ~ ^/(a|b|c)(/|$)`` line in the frontend nginx
    config and returns ``[a, b, c]`` so membership checks are independent of
    the token order within the alternation.

    Returns:
        list[str]: The pipe-separated route tokens of the backend-API location.
    """
    nginx_conf = (REPO_ROOT / "frontend" / "nginx" / "default.conf").read_text(encoding="utf-8")
    match = re.search(r"location ~ \^/\(([^)]*)\)", nginx_conf)
    assert match is not None, "backend-API location alternation not found in nginx config"
    return match.group(1).split("|")


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


def test_backend_and_frontend_share_upload_limit_env() -> None:
    """Both services must read the same DOCINT_CLIENT_MAX_BODY_SIZE.

    nginx (frontend) *enforces* the ceiling; the backend only *advertises* it
    via GET /config so the SPA can size upload batches to stay under it. If the
    backend didn't get the var, /config would report the 1g default while nginx
    enforced a raised limit (or vice-versa) and batches would be mis-sized.
    """
    compose = (REPO_ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")

    # The env line must appear twice: once under backend, once under frontend.
    occurrences = compose.count("DOCINT_CLIENT_MAX_BODY_SIZE: ${DOCINT_CLIENT_MAX_BODY_SIZE:-1g}")
    assert occurrences == 2, f"expected the shared upload-limit env in both services, found {occurrences}"


def test_ingest_proxy_uses_configurable_request_limit() -> None:
    """The ingest nginx location should use the configurable multipart limit."""
    nginx_conf = (REPO_ROOT / "frontend" / "nginx" / "default.conf").read_text(encoding="utf-8")

    assert "client_max_body_size ${DOCINT_CLIENT_MAX_BODY_SIZE};" in nginx_conf
    assert "client_max_body_size 200m;" not in nginx_conf


def test_frontend_nginx_proxies_config_endpoint() -> None:
    """The SPA's /config fetch must reach the backend, not the SPA fallback."""
    nginx_conf = (REPO_ROOT / "frontend" / "nginx" / "default.conf").read_text(encoding="utf-8")

    assert "query|summarize|agent|config|version" in nginx_conf


def test_frontend_nginx_proxies_version_endpoint() -> None:
    """The SPA's /version fetch must reach the backend, not the SPA fallback."""
    nginx_conf = (REPO_ROOT / "frontend" / "nginx" / "default.conf").read_text(encoding="utf-8")

    assert "config|version|docs" in nginx_conf


def test_frontend_nginx_proxies_translate_endpoint() -> None:
    """The SPA's /translate fetch must reach the backend, not the SPA fallback.

    Order-independent guard: asserts ``translate`` is one of the backend-API
    location's alternation tokens, so dropping it from the nginx allowlist (the
    "prod serves index.html" failure the dual proxy exists to prevent) fails
    the suite regardless of where in the alternation it sits.
    """
    assert "translate" in _nginx_api_route_tokens()


def _vite_api_prefixes() -> list[str]:
    """Return the Vite dev server's ``API_PREFIXES`` allowlist entries.

    Order-independent: parses the ``API_PREFIXES = [...]`` array literal in
    ``vite.config.ts`` (each entry is proxied under the ``/docint/`` base) so
    membership checks don't depend on where an entry sits in the list.

    Returns:
        list[str]: The API path segments proxied to the backend.
    """
    vite_conf = (REPO_ROOT / "frontend" / "vite.config.ts").read_text(encoding="utf-8")
    match = re.search(r"API_PREFIXES\s*=\s*\[([^\]]*)\]", vite_conf)
    assert match is not None, "API_PREFIXES array not found in vite.config.ts"
    return [tok.strip().strip("'\"") for tok in match.group(1).split(",") if tok.strip()]


def test_frontend_vite_proxies_translate_endpoint() -> None:
    """The Vite dev server must proxy /translate to the backend, not 404.

    The dev-side half of the dual-proxy allowlist: a missing ``translate`` entry
    in the Vite ``API_PREFIXES`` allowlist makes the dev server serve the SPA
    fallback instead of reaching FastAPI. Asserts on the entry's presence
    (order-independent).
    """
    assert "translate" in _vite_api_prefixes()
