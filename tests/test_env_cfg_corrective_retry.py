"""Tests for ``load_corrective_retry_env``.

The knob gates the one-shot re-retrieval that runs when response validation
flags a weak answer as mismatched, so an operator must be able to turn it off
without disabling validation itself.
"""

from __future__ import annotations

import pytest

from docint.utils.env_cfg import load_corrective_retry_env


@pytest.fixture(autouse=True)
def _clear_corrective_retry_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ``CORRECTIVE_RETRY_ENABLED`` so a developer ``.env`` cannot leak in.

    ``env_cfg`` runs ``load_dotenv()`` at import time, so without this the
    default-value assertions would pass or fail depending on the local ``.env``.
    """
    monkeypatch.delenv("CORRECTIVE_RETRY_ENABLED", raising=False)


def test_corrective_retry_defaults_to_enabled() -> None:
    """With nothing set, the corrective retry is on."""
    assert load_corrective_retry_env().enabled is True


def test_corrective_retry_default_is_overridable_by_caller() -> None:
    """The caller-supplied default applies when the env var is unset."""
    assert load_corrective_retry_env(default_enabled=False).enabled is False


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "Yes"])
def test_corrective_retry_truthy_values_enable(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Truthy spellings enable the retry even against a False default.

    Args:
        value: The environment-variable spelling under test.
        monkeypatch: Pytest environment patcher.
    """
    monkeypatch.setenv("CORRECTIVE_RETRY_ENABLED", value)

    assert load_corrective_retry_env(default_enabled=False).enabled is True


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", ""])
def test_corrective_retry_falsy_values_disable(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anything not in the truthy set disables the retry.

    Args:
        value: The environment-variable spelling under test.
        monkeypatch: Pytest environment patcher.
    """
    monkeypatch.setenv("CORRECTIVE_RETRY_ENABLED", value)

    assert load_corrective_retry_env().enabled is False
