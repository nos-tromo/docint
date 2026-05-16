"""Tests for principal configuration and the request principal resolver."""

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from docint.core.auth.principal import resolve_principal
from docint.utils.env_cfg import PrincipalConfig, load_principal_env


def test_load_principal_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env vars set, the header name defaults and there is no fallback.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)

    cfg = load_principal_env()

    assert isinstance(cfg, PrincipalConfig)
    assert cfg.header_name == "X-Auth-User"
    assert cfg.default_identity is None


def test_load_principal_env_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit env values override the header name and set a fallback identity.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_AUTH_HEADER", "X-Remote-User")
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")

    cfg = load_principal_env()

    assert cfg.header_name == "X-Remote-User"
    assert cfg.default_identity == "operator"


def test_load_principal_env_blank_identity_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank/whitespace ``DOCINT_DEFAULT_IDENTITY`` normalises to ``None``.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "   ")

    cfg = load_principal_env()

    assert cfg.default_identity is None


def _make_request(headers: dict[str, str] | None = None) -> Request:
    """Build a minimal Starlette ``Request`` with the given headers.

    Args:
        headers (dict[str, str] | None): Header name/value pairs.

    Returns:
        Request: A request object whose ``.headers`` reflects ``headers``.
    """
    raw_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": raw_headers,
    }
    return Request(scope)


def test_resolve_principal_returns_header_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A present trusted header is returned verbatim as the principal.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)

    request = _make_request({"X-Auth-User": "alice"})

    assert resolve_principal(request) == "alice"


def test_resolve_principal_falls_back_to_default_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the header is absent the configured default identity is used.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")

    request = _make_request({})

    assert resolve_principal(request) == "operator"


def test_resolve_principal_fails_closed_without_header_or_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No header and no configured fallback must raise HTTP 401.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)

    request = _make_request({})

    with pytest.raises(HTTPException) as excinfo:
        resolve_principal(request)
    assert excinfo.value.status_code == 401


def test_resolve_principal_honours_custom_header_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured non-default header name is the one consulted.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_AUTH_HEADER", "X-Remote-User")
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)

    request = _make_request({"X-Remote-User": "bob"})

    assert resolve_principal(request) == "bob"
