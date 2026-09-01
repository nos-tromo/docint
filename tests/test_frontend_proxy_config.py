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
    assert "config" in _nginx_api_route_tokens()


def test_frontend_nginx_proxies_version_endpoint() -> None:
    """The SPA's /version fetch must reach the backend, not the SPA fallback."""
    assert "version" in _nginx_api_route_tokens()


def test_frontend_nginx_proxies_whoami_endpoint() -> None:
    """The header's /whoami fetch must reach the backend, not the SPA fallback."""
    assert "whoami" in _nginx_api_route_tokens()


def test_frontend_nginx_proxies_translate_endpoint() -> None:
    """The SPA's /translate fetch must reach the backend, not the SPA fallback.

    Order-independent guard: asserts ``translate`` is one of the backend-API
    location's alternation tokens, so dropping it from the nginx allowlist (the
    "prod serves index.html" failure the dual proxy exists to prevent) fails
    the suite regardless of where in the alternation it sits.
    """
    assert "translate" in _nginx_api_route_tokens()


DEV_PROXY_TS = REPO_ROOT / "frontend" / "src" / "lib" / "devProxy.ts"


def _vite_api_prefixes() -> list[str]:
    """Return the Vite dev server's ``API_PREFIXES`` allowlist entries.

    Order-independent: parses the ``API_PREFIXES = [...]`` array literal so
    membership checks don't depend on where an entry sits in the list. The list
    lives in ``frontend/src/lib/devProxy.ts`` rather than in ``vite.config.ts``
    (which imports it) so the dev proxy's rules can be unit-tested from vitest
    as well as asserted here.

    Returns:
        list[str]: The API path segments proxied to the backend.
    """
    dev_proxy = DEV_PROXY_TS.read_text(encoding="utf-8")
    match = re.search(r"API_PREFIXES\s*=\s*\[([^\]]*)\]", dev_proxy)
    assert match is not None, f"API_PREFIXES array not found in {DEV_PROXY_TS.name}"
    return [tok.strip().strip("'\"") for tok in match.group(1).split(",") if tok.strip()]


def test_frontend_vite_proxies_whoami_endpoint() -> None:
    """The Vite dev server must proxy /whoami to the backend, not 404."""
    assert "whoami" in _vite_api_prefixes()


def test_frontend_vite_proxies_translate_endpoint() -> None:
    """The Vite dev server must proxy /translate to the backend, not 404.

    The dev-side half of the dual-proxy allowlist: a missing ``translate`` entry
    in the Vite ``API_PREFIXES`` allowlist makes the dev server serve the SPA
    fallback instead of reaching FastAPI. Asserts on the entry's presence
    (order-independent).
    """
    assert "translate" in _vite_api_prefixes()


def _nginx_conf() -> str:
    """Return the rendered frontend nginx config text.

    Returns:
        str: Contents of ``frontend/nginx/default.conf``.
    """
    return (REPO_ROOT / "frontend" / "nginx" / "default.conf").read_text(encoding="utf-8")


def test_preview_route_is_framable_by_the_app_itself() -> None:
    """The in-page preview dialog frames ``/sources/preview`` same-origin.

    The shared security headers say ``X-Frame-Options: DENY`` and
    ``frame-ancestors 'none'``, which forbid rendering in *any* frame — even
    the app's own dialog. Observed live: PDF/JSON previews showed the
    browser's blocked-page icon while a new tab worked, because those headers
    constrain framing but not top-level navigation. The preview route needs
    its own location with same-origin framing; everything else keeps DENY.
    """
    conf = _nginx_conf()
    match = re.search(r"location = /sources/preview \{([^}]*)\}", conf)
    assert match is not None, "dedicated /sources/preview location not found"
    block = match.group(1)
    assert "proxy_pass" in block
    assert "security-headers-framable.conf" in block


def test_framable_headers_allow_only_same_origin_framing() -> None:
    """Same-origin framing only — other sites still cannot embed documents."""
    headers = (REPO_ROOT / "frontend" / "nginx" / "security-headers-framable.conf").read_text(encoding="utf-8")

    assert 'X-Frame-Options "SAMEORIGIN"' in headers
    csp_lines = [line for line in headers.splitlines() if "Content-Security-Policy" in line]
    assert len(csp_lines) == 1
    assert "frame-ancestors 'self'" in csp_lines[0]
    assert "frame-ancestors 'none'" not in csp_lines[0]


def test_framable_headers_keep_the_rest_of_the_policy() -> None:
    """Relaxing framing must not silently drop the other protections."""
    headers = (REPO_ROOT / "frontend" / "nginx" / "security-headers-framable.conf").read_text(encoding="utf-8")

    assert 'X-Content-Type-Options "nosniff"' in headers
    assert "Referrer-Policy" in headers
    assert "Permissions-Policy" in headers


def test_default_security_headers_still_deny_framing() -> None:
    """Only the preview route relaxes; the app and API keep DENY."""
    headers = (REPO_ROOT / "frontend" / "nginx" / "security-headers.conf").read_text(encoding="utf-8")

    assert 'X-Frame-Options "DENY"' in headers
    assert "frame-ancestors 'none'" in headers


def test_frontend_image_ships_the_framable_headers() -> None:
    """The framable variant must reach the container, or nginx fails to boot."""
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.frontend").read_text(encoding="utf-8")

    assert "security-headers-framable.conf" in dockerfile


def _ingest_location_block() -> str:
    """Return the body of the ingest ``location`` block in the nginx config.

    Returns:
        str: The directives between the ingest location's braces.
    """
    nginx_conf = (REPO_ROOT / "frontend" / "nginx" / "default.conf").read_text(encoding="utf-8")
    match = re.search(r"location ~ \^/ingest\(/\|\$\) \{(.*?)\n    \}", nginx_conf, re.DOTALL)
    assert match is not None, "ingest location block not found in nginx config"
    return match.group(1)


def test_ingest_proxy_streams_request_body() -> None:
    """Uploads must stream through nginx unbuffered.

    Without ``proxy_request_buffering off`` nginx spools the whole request
    body to /tmp before forwarding — and the frontend container's /tmp is a
    16m tmpfs, far below client_max_body_size, so any upload batch larger
    than the tmpfs 500s on "no space left on device".
    """
    assert "proxy_request_buffering off;" in _ingest_location_block()


def _api_location_block() -> str:
    """Return the directives inside the backend-API ``location`` block."""
    nginx_conf = (REPO_ROOT / "frontend" / "nginx" / "default.conf").read_text(encoding="utf-8")
    match = re.search(r"location ~ \^/\([^)]*\)\(/\|\$\) \{(.*?)\n    \}", nginx_conf, re.DOTALL)
    assert match is not None, "backend-API location block not found in nginx config"
    return match.group(1)


def test_api_proxy_uses_configurable_json_body_limit() -> None:
    """JSON API bodies must have their own configurable ceiling.

    A full batch add is single-digit MB; without this the location inherits
    nginx's 1m default and 413s before FastAPI sees it.
    """
    assert "client_max_body_size ${DOCINT_API_MAX_BODY_SIZE};" in _api_location_block()


def test_api_proxy_streams_request_body() -> None:
    """Large JSON bodies must stream through nginx unbuffered.

    As at the ingest location: a buffered body spools to /tmp, which is a 16m
    tmpfs here — well under the API body ceiling.
    """
    block = _api_location_block()

    assert "proxy_request_buffering off;" in block
    assert "proxy_http_version 1.1;" in block


def test_frontend_image_defaults_api_body_limit() -> None:
    """The image must default DOCINT_API_MAX_BODY_SIZE.

    envsubst renders an undefined variable empty, leaving `client_max_body_size
    ;` and stopping nginx from booting.
    """
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.frontend").read_text(encoding="utf-8")

    assert "ENV DOCINT_API_MAX_BODY_SIZE=64m" in dockerfile


def test_frontend_compose_exposes_api_body_limit_override() -> None:
    """Only the frontend needs the JSON-API ceiling, unlike the upload one.

    DOCINT_CLIENT_MAX_BODY_SIZE is on both services because the backend
    advertises those bytes via GET /config; this one is enforcement-only.
    """
    compose = (REPO_ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")

    occurrences = compose.count("DOCINT_API_MAX_BODY_SIZE: ${DOCINT_API_MAX_BODY_SIZE:-64m}")
    assert occurrences == 1, f"expected the JSON-API body limit on the frontend only, found {occurrences}"


def test_backend_tmp_is_disk_backed_volume() -> None:
    """The backend's upload spool (/tmp) must be a disk volume, not a RAM tmpfs.

    Starlette spools every multipart body to /tmp before it lands in
    QDRANT_SRC_DIR; a tmpfs sized for DOCINT_CLIENT_MAX_BODY_SIZE (worse:
    concurrent uploads) is host RAM. Mirrors vllm-service's media-tmp volumes.
    """
    compose = (REPO_ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")

    assert "- media-tmp:/tmp" in compose
    assert "/tmp:size=2g" not in compose


def test_ingest_page_and_api_are_split_by_method() -> None:
    """``/ingest`` is both an SPA route and an endpoint, so both proxies split it.

    The SPA's ingest screen and the CLI/batch ``POST /ingest`` share one path.
    Both proxy layers matched the whole prefix to the backend, so opening or
    reloading ``/docint/ingest`` was answered by FastAPI (405) instead of the
    app -- the screen was reachable only by clicking through from another
    route. Only the bare path is ambiguous; everything under ``/ingest/``
    (upload, finalize, jobs, the SSE stream) stays API-only.

    Guards the production half here and the wiring of the dev half; the dev
    rule's own behavior is unit-tested in ``devProxy.test.ts``.
    """
    nginx_conf = (REPO_ROOT / "frontend" / "nginx" / "default.conf").read_text(encoding="utf-8")

    assert 'map "$request_method:$uri" $ingest_spa_page' in nginx_conf
    assert '"GET:/ingest"' in nginx_conf
    assert "error_page 418 = @spa_shell;" in nginx_conf
    assert "location @spa_shell {" in nginx_conf

    vite_conf = (REPO_ROOT / "frontend" / "vite.config.ts").read_text(encoding="utf-8")
    assert "spaShellBypass" in vite_conf, "the dev proxy lost its ingest-page carve-out"


def test_ingest_subpaths_still_reach_the_backend() -> None:
    """The carve-out must not swallow the ingest API it sits in front of.

    ``/ingest/upload``, ``/ingest/finalize`` and the ``/ingest/jobs*`` stream
    are API-only. The nginx map keys the page on the *exact* bare path, so a
    key with a trailing segment would hand an upload to the SPA shell.
    """
    nginx_conf = (REPO_ROOT / "frontend" / "nginx" / "default.conf").read_text(encoding="utf-8")

    match = re.search(r"map \"\$request_method:\$uri\" \$ingest_spa_page \{(.*?)\}", nginx_conf, re.S)
    assert match is not None, "ingest page map not found"
    keys = re.findall(r'"([^"]+)"', match.group(1))
    assert keys, "ingest page map has no keys"
    for key in keys:
        _, _, path = key.partition(":")
        assert path.rstrip("/") == "/ingest", f"{key} would divert an API path to the SPA"
