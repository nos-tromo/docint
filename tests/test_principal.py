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
    raw_headers = [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in (headers or {}).items()]
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": raw_headers,
        "query_string": b"",
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

    assert resolve_principal(request).name == "alice"


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

    assert resolve_principal(request).name == "operator"


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

    assert resolve_principal(request).name == "bob"


def test_load_principal_env_groups_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env vars set, groups config uses the contract defaults.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("DOCINT_GROUPS_HEADER", raising=False)
    monkeypatch.delenv("DOCINT_ADMIN_GROUP", raising=False)
    monkeypatch.delenv("DOCINT_DEFAULT_GROUPS", raising=False)

    cfg = load_principal_env()

    assert cfg.groups_header == "X-Auth-Groups"
    assert cfg.admin_group == "admins"
    assert cfg.default_groups is None


def test_load_principal_env_groups_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit env values override the groups header, admin group, and dev groups.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_GROUPS_HEADER", "X-Remote-Groups")
    monkeypatch.setenv("DOCINT_ADMIN_GROUP", "operators")
    monkeypatch.setenv("DOCINT_DEFAULT_GROUPS", "operators,users")

    cfg = load_principal_env()

    assert cfg.groups_header == "X-Remote-Groups"
    assert cfg.admin_group == "operators"
    assert cfg.default_groups == "operators,users"


def test_load_principal_env_blank_default_groups_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank/whitespace ``DOCINT_DEFAULT_GROUPS`` normalises to ``None``.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_GROUPS", "   ")

    cfg = load_principal_env()

    assert cfg.default_groups is None


def test_resolve_principal_parses_groups_and_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Groups parse from the comma-separated header; admin flag derives from them.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("DOCINT_GROUPS_HEADER", raising=False)
    monkeypatch.delenv("DOCINT_ADMIN_GROUP", raising=False)

    principal = resolve_principal(_make_request({"X-Auth-User": "root", "X-Auth-Groups": "admins, users"}))

    assert principal.groups == frozenset({"admins", "users"})
    assert principal.is_admin is True


def test_resolve_principal_fails_closed_on_missing_or_blank_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing or whitespace-only groups header means no groups and not admin.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("DOCINT_DEFAULT_GROUPS", raising=False)

    for headers in ({"X-Auth-User": "alice"}, {"X-Auth-User": "alice", "X-Auth-Groups": " , ,"}):
        principal = resolve_principal(_make_request(headers))
        assert principal.groups == frozenset()
        assert principal.is_admin is False


def test_resolve_principal_dev_default_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """DOCINT_DEFAULT_GROUPS applies only when the groups header is absent.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    monkeypatch.setenv("DOCINT_DEFAULT_GROUPS", "admins")

    assert resolve_principal(_make_request({})).is_admin is True
    assert resolve_principal(_make_request({"X-Auth-User": "alice", "X-Auth-Groups": "users"})).is_admin is False


def test_effective_owner_rules() -> None:
    """Only admins may act as another owner; everyone else acts as themselves."""
    from docint.core.auth.principal import Principal

    admin = Principal(name="root", groups=frozenset({"admins"}), is_admin=True, requested_owner="alice")
    assert admin.effective_owner == "alice"
    assert (
        Principal(name="root", groups=frozenset({"admins"}), is_admin=True, requested_owner=None).effective_owner
        == "root"
    )
    non_admin = Principal(name="bob", groups=frozenset(), is_admin=False, requested_owner="alice")
    assert non_admin.effective_owner == "bob"


def test_resolve_principal_reads_owner_query_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``owner`` query param lands in ``requested_owner`` (blank -> None).

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    request = _make_request({"X-Auth-User": "root"})
    request.scope["query_string"] = b"owner=alice"
    assert resolve_principal(request).requested_owner == "alice"
