"""Tests for principal configuration and the request principal resolver."""

import pytest

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
