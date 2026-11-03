"""Tests for principal configuration and the request principal resolver."""

import pytest
from _pytest.logging import LogCaptureFixture
from fastapi import HTTPException
from starlette.requests import Request

from docint.core.auth.principal import EDGE_SECRET_HEADER, check_principal_config, resolve_principal
from docint.utils.env_cfg import EDGE_SECRET_MIN_LENGTH, PrincipalConfig, load_principal_env


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


def _make_request(
    headers: dict[str, str] | None = None,
    *,
    method: str = "GET",
    path: str = "/",
    query_string: bytes = b"",
) -> Request:
    """Build a minimal Starlette ``Request`` with the given headers.

    Args:
        headers (dict[str, str] | None): Header name/value pairs.
        method (str): The request method.
        path (str): The request path, without a query string.
        query_string (bytes): The raw query string.

    Returns:
        Request: A request object whose ``.headers`` reflects ``headers``.
    """
    raw_headers = [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in (headers or {}).items()]
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": raw_headers,
        "query_string": query_string,
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


# --- Edge secret: identity headers are believed only from the gateway -------

# Invented values; the wrong one is as long as the right one so only content differs.
_EDGE_SECRET = "test-edge-secret-0123456789abcdefghij"
_WRONG_SECRET = "test-edge-secret-WRONG-0123456789abcd"
_SHORT_SECRET = "short-test-secret"


def test_load_principal_env_edge_secret_unset_or_blank_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset and whitespace-only both leave the check off.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("DOCINT_EDGE_SECRET", raising=False)
    assert load_principal_env().edge_secret is None

    monkeypatch.setenv("DOCINT_EDGE_SECRET", "   ")
    assert load_principal_env().edge_secret is None


def test_load_principal_env_reads_and_strips_edge_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Surrounding whitespace is dropped, as HTTP drops it from the header the gateway sends.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_EDGE_SECRET", f"  {_EDGE_SECRET}\n")

    assert load_principal_env().edge_secret == _EDGE_SECRET


def test_load_principal_env_accepts_a_secret_of_the_minimum_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exactly the minimum length is accepted.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    secret = "k" * EDGE_SECRET_MIN_LENGTH
    monkeypatch.setenv("DOCINT_EDGE_SECRET", secret)

    assert load_principal_env().edge_secret == secret


def test_load_principal_env_rejects_a_short_secret_without_echoing_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A secret below the minimum is a configuration error whose message omits the value.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_EDGE_SECRET", _SHORT_SECRET)

    with pytest.raises(ValueError, match="DOCINT_EDGE_SECRET") as excinfo:
        load_principal_env()
    assert _SHORT_SECRET not in str(excinfo.value)


def test_principal_config_repr_hides_the_edge_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Logging the config object can never print the secret.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_EDGE_SECRET", _EDGE_SECRET)

    assert _EDGE_SECRET not in repr(load_principal_env())


def test_resolve_principal_without_a_secret_trusts_headers_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset: every caller's identity headers are believed, whatever X-Edge-Secret says.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)
    monkeypatch.delenv("DOCINT_DEFAULT_GROUPS", raising=False)
    claimed = {"X-Auth-User": "root", "X-Auth-Groups": "admins", "X-Auth-Name": "Root Example"}

    for headers in (claimed, {**claimed, EDGE_SECRET_HEADER: _WRONG_SECRET}):
        principal = resolve_principal(_make_request(headers))
        assert principal.name == "root"
        assert principal.is_admin is True
        assert principal.display_name == "Root Example"


def test_resolve_principal_with_the_secret_trusts_user_groups_and_display_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request carrying the configured secret is resolved from its headers.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_EDGE_SECRET", _EDGE_SECRET)
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)
    headers = {
        "X-Auth-User": "alice",
        "X-Auth-Groups": "admins, users",
        "X-Auth-Name": "Alice Example",
        EDGE_SECRET_HEADER: _EDGE_SECRET,
    }

    principal = resolve_principal(_make_request(headers, query_string=b"owner=bob"))

    assert principal.name == "alice"
    assert principal.groups == frozenset({"admins", "users"})
    assert principal.is_admin is True
    assert principal.display_name == "Alice Example"
    assert principal.effective_owner == "bob"


@pytest.mark.parametrize("secret_header", [None, _WRONG_SECRET, ""], ids=["missing", "wrong", "empty"])
def test_resolve_principal_untrusted_request_fails_closed_without_a_default(
    monkeypatch: pytest.MonkeyPatch, secret_header: str | None
) -> None:
    """Without the right secret the user header counts for nothing: 401 with no default identity.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        secret_header (str | None): The X-Edge-Secret value sent, ``None`` for none.
    """
    monkeypatch.setenv("DOCINT_EDGE_SECRET", _EDGE_SECRET)
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)
    headers = {"X-Auth-User": "alice"}
    if secret_header is not None:
        headers[EDGE_SECRET_HEADER] = secret_header

    with pytest.raises(HTTPException) as excinfo:
        resolve_principal(_make_request(headers))
    assert excinfo.value.status_code == 401


def test_resolve_principal_untrusted_request_ignores_the_defaults_under_the_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the edge secret set, the dev fallbacks do not authenticate an untrusted request.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_EDGE_SECRET", _EDGE_SECRET)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    monkeypatch.setenv("DOCINT_DEFAULT_GROUPS", "users")
    headers = {
        "X-Auth-User": "alice",
        "X-Auth-Groups": "editors",
        "X-Auth-Name": "Alice Example",
        EDGE_SECRET_HEADER: _WRONG_SECRET,
    }

    with pytest.raises(HTTPException) as exc_info:
        resolve_principal(_make_request(headers))

    assert exc_info.value.status_code == 401


# (edge secret, default identity, X-Edge-Secret sent, X-Auth-User sent) -> name, or 401.
_IDENTITY_MATRIX = [
    pytest.param(None, None, None, "alice", "alice", id="no-secret-no-default-header"),
    pytest.param(None, None, None, None, 401, id="no-secret-no-default-anonymous"),
    pytest.param(None, "operator", None, "alice", "alice", id="no-secret-default-header"),
    pytest.param(None, "operator", None, None, "operator", id="no-secret-default-anonymous"),
    pytest.param(_EDGE_SECRET, None, _EDGE_SECRET, "alice", "alice", id="secret-no-default-trusted"),
    pytest.param(_EDGE_SECRET, None, _WRONG_SECRET, "alice", 401, id="secret-no-default-forged"),
    pytest.param(_EDGE_SECRET, None, None, "alice", 401, id="secret-no-default-unproven"),
    pytest.param(_EDGE_SECRET, "operator", _EDGE_SECRET, "alice", "alice", id="secret-default-trusted"),
    pytest.param(_EDGE_SECRET, "operator", _WRONG_SECRET, "alice", 401, id="secret-default-forged"),
    pytest.param(_EDGE_SECRET, "operator", None, "alice", 401, id="secret-default-unproven"),
    pytest.param(_EDGE_SECRET, "operator", _EDGE_SECRET, None, 401, id="secret-default-trusted-anonymous"),
    pytest.param(_EDGE_SECRET, "operator", None, None, 401, id="secret-default-anonymous"),
]


@pytest.mark.parametrize(("secret", "default", "sent_secret", "user", "expected"), _IDENTITY_MATRIX)
def test_default_identity_is_honoured_only_without_an_edge_secret(
    monkeypatch: pytest.MonkeyPatch,
    secret: str | None,
    default: str | None,
    sent_secret: str | None,
    user: str | None,
    expected: str | int,
) -> None:
    """Every combination of edge secret and default identity, forged headers included.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        secret (str | None): ``DOCINT_EDGE_SECRET``.
        default (str | None): ``DOCINT_DEFAULT_IDENTITY``.
        sent_secret (str | None): The X-Edge-Secret value sent, ``None`` for none.
        user (str | None): The X-Auth-User value sent, ``None`` for none.
        expected (str | int): The principal's name, or the refusal's status.
    """
    for name, value in (("DOCINT_EDGE_SECRET", secret), ("DOCINT_DEFAULT_IDENTITY", default)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    headers: dict[str, str] = {}
    if sent_secret is not None:
        headers[EDGE_SECRET_HEADER] = sent_secret
    if user is not None:
        headers["X-Auth-User"] = user

    if isinstance(expected, int):
        with pytest.raises(HTTPException) as exc_info:
            resolve_principal(_make_request(headers))
        assert exc_info.value.status_code == expected
    else:
        principal = resolve_principal(_make_request(headers))
        assert principal.name == expected


def test_default_groups_never_apply_under_the_edge_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trusted request without a groups header gets none, not the dev default groups.

    Otherwise a default admin group left set beside the secret would make every
    gateway user without a groups header an admin.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_EDGE_SECRET", _EDGE_SECRET)
    monkeypatch.setenv("DOCINT_DEFAULT_GROUPS", "admins")

    principal = resolve_principal(_make_request({"X-Auth-User": "alice", EDGE_SECRET_HEADER: _EDGE_SECRET}))

    assert principal.groups == frozenset()
    assert principal.is_admin is False

    monkeypatch.delenv("DOCINT_EDGE_SECRET")
    assert resolve_principal(_make_request({"X-Auth-User": "alice"})).is_admin is True


@pytest.mark.parametrize("secret_header", [None, _WRONG_SECRET], ids=["missing", "wrong"])
def test_untrusted_admin_claim_never_gains_admin(monkeypatch: pytest.MonkeyPatch, secret_header: str | None) -> None:
    """A forged admin group header grants nothing, and ``?owner=`` then has no effect.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        secret_header (str | None): The X-Edge-Secret value sent, ``None`` for none.
    """
    monkeypatch.setenv("DOCINT_EDGE_SECRET", _EDGE_SECRET)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    monkeypatch.delenv("DOCINT_DEFAULT_GROUPS", raising=False)
    headers = {"X-Auth-User": "root", "X-Auth-Groups": "admins"}
    if secret_header is not None:
        headers[EDGE_SECRET_HEADER] = secret_header

    # Neither the forged admin claim nor the default identity authenticates it.
    with pytest.raises(HTTPException) as exc_info:
        resolve_principal(_make_request(headers, query_string=b"owner=alice"))

    assert exc_info.value.status_code == 401


@pytest.mark.parametrize(
    ("secret_header", "reason"),
    [(None, "X-Edge-Secret missing"), (_WRONG_SECRET, "X-Edge-Secret does not match")],
    ids=["missing", "wrong"],
)
def test_ignored_identity_headers_are_logged_without_any_value(
    monkeypatch: pytest.MonkeyPatch,
    loguru_caplog: LogCaptureFixture,
    secret_header: str | None,
    reason: str,
) -> None:
    """The warning names method, path and reason; never a header value or the query.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Loguru records bridged into caplog.
        secret_header (str | None): The X-Edge-Secret value sent, ``None`` for none.
        reason (str): The reason the warning must give.
    """
    monkeypatch.setenv("DOCINT_EDGE_SECRET", _EDGE_SECRET)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    headers = {"X-Auth-User": "claimed-user", "X-Auth-Groups": "claimed-group", "X-Auth-Name": "Claimed Name"}
    if secret_header is not None:
        headers[EDGE_SECRET_HEADER] = secret_header

    with pytest.raises(HTTPException):
        resolve_principal(
            _make_request(headers, method="POST", path="/collections/select", query_string=b"owner=target-owner")
        )

    messages = [record.getMessage() for record in loguru_caplog.records]
    assert messages == [f"Ignored identity headers on POST '/collections/select': {reason}"]
    for value in ("claimed-user", "claimed-group", "Claimed Name", "target-owner", _WRONG_SECRET, _EDGE_SECRET):
        assert value not in loguru_caplog.text


def test_untrusted_request_without_identity_headers_is_not_logged(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """Nothing was ignored, so there is nothing to warn about.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Loguru records bridged into caplog.
    """
    monkeypatch.setenv("DOCINT_EDGE_SECRET", _EDGE_SECRET)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")

    with pytest.raises(HTTPException):
        resolve_principal(_make_request({}))
    assert loguru_caplog.records == []


def test_check_principal_config_warns_in_the_production_shape(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """No edge secret and no default identity: one warning that names the fix.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Loguru records bridged into caplog.
    """
    monkeypatch.delenv("DOCINT_EDGE_SECRET", raising=False)
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)

    check_principal_config()

    messages = [record.getMessage() for record in loguru_caplog.records]
    assert len(messages) == 1
    assert "trusted from any caller" in messages[0]
    assert "DOCINT_EDGE_SECRET" in messages[0]


@pytest.mark.parametrize(
    ("name", "value"),
    [("DOCINT_EDGE_SECRET", _EDGE_SECRET), ("DOCINT_DEFAULT_IDENTITY", "operator")],
    ids=["edge-secret", "default-identity"],
)
def test_check_principal_config_is_quiet_otherwise(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture, name: str, value: str
) -> None:
    """Either setting silences the warning, and the secret itself is never logged.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Loguru records bridged into caplog.
        name (str): The variable that is set.
        value (str): Its value.
    """
    monkeypatch.delenv("DOCINT_EDGE_SECRET", raising=False)
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)
    monkeypatch.setenv(name, value)

    check_principal_config()

    assert loguru_caplog.records == []


@pytest.mark.parametrize(
    ("name", "value"),
    [("DOCINT_DEFAULT_IDENTITY", "operator"), ("DOCINT_DEFAULT_GROUPS", "admins")],
    ids=["default-identity", "default-groups"],
)
def test_check_principal_config_warns_that_the_defaults_are_ignored_under_the_secret(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture, name: str, value: str
) -> None:
    """A dev fallback beside the edge secret: one warning naming the variables, never their values.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Loguru records bridged into caplog.
        name (str): The fallback that is set.
        value (str): Its value.
    """
    monkeypatch.setenv("DOCINT_EDGE_SECRET", _EDGE_SECRET)
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)
    monkeypatch.delenv("DOCINT_DEFAULT_GROUPS", raising=False)
    monkeypatch.setenv(name, value)

    check_principal_config()

    messages = [record.getMessage() for record in loguru_caplog.records]
    assert len(messages) == 1
    for variable in ("DOCINT_DEFAULT_IDENTITY", "DOCINT_EDGE_SECRET"):
        assert variable in messages[0]
    for secret in (value, _EDGE_SECRET):
        assert secret not in loguru_caplog.text


def test_check_principal_config_rejects_a_short_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """The startup check raises on a secret too short to accept.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_EDGE_SECRET", _SHORT_SECRET)

    with pytest.raises(ValueError, match="DOCINT_EDGE_SECRET"):
        check_principal_config()
