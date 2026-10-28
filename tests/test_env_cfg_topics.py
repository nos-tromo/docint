"""Tests for the topics configuration."""

from pathlib import Path

import pytest
from _pytest.logging import LogCaptureFixture

from docint.utils.env_cfg import (
    TopicsClientConfig,
    TopicsConfig,
    load_path_env,
    load_topics_client_env,
    load_topics_concurrency,
    load_topics_env,
)


def test_topics_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every knob has its documented default.

    The suite keeps ``TOPICS_ON_INGEST`` off (``tests/conftest.py``), so it is
    cleared here to read the real default.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.delenv("TOPICS_ON_INGEST", raising=False)
    assert load_topics_env() == TopicsConfig(
        enabled=True,
        on_ingest=True,
        align_on_ingest=True,
        labels=True,
        fit_max_points=100000,
        min_text_chars=40,
        seed=42,
        min_topic_size=0,
        max_topics=0,
        admit_percentile=5.0,
        merge_min_similarity=0.7,
    )


@pytest.mark.parametrize(("raw", "expected"), [("no", False), ("off", False), ("0", False), ("yes", True)])
def test_topics_bools_follow_the_true_one_yes_rule(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    """A boolean is true only for true, 1 or yes.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        raw: The raw variable value.
        expected: The parsed flag.

    Returns:
        None.
    """
    monkeypatch.setenv("TOPICS_ENABLED", raw)
    assert load_topics_env().enabled is expected


@pytest.mark.parametrize(("raw", "expected"), [("false", False), ("0", False), ("true", True), ("YES", True)])
def test_topics_on_ingest_follows_the_true_one_yes_rule(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
) -> None:
    """``TOPICS_ON_INGEST`` reads like every other topics flag.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        raw: The raw variable value.
        expected: The parsed flag.

    Returns:
        None.
    """
    monkeypatch.setenv("TOPICS_ON_INGEST", raw)
    assert load_topics_env().on_ingest is expected


@pytest.mark.parametrize(
    ("name", "raw", "field", "expected"),
    [
        ("TOPICS_SEED", "abc", "seed", 42),
        ("TOPICS_SEED", "4294967296", "seed", 42),
        ("TOPICS_ADMIT_PERCENTILE", "150", "admit_percentile", 5.0),
        ("TOPICS_ADMIT_PERCENTILE", "nan", "admit_percentile", 5.0),
        ("TOPICS_MERGE_MIN_SIMILARITY", "0", "merge_min_similarity", 0.7),
        ("TOPICS_MERGE_MIN_SIMILARITY", "1.5", "merge_min_similarity", 0.7),
        ("TOPICS_FIT_MAX_POINTS", "0", "fit_max_points", 100000),
    ],
)
def test_unusable_topics_numbers_fall_back_with_a_warning(
    monkeypatch: pytest.MonkeyPatch,
    loguru_caplog: LogCaptureFixture,
    name: str,
    raw: str,
    field: str,
    expected: float,
) -> None:
    """An out-of-range or unparseable number falls back and names the variable.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        loguru_caplog: Captured loguru warnings.
        name: The variable under test.
        raw: Its unusable value.
        field: The config field it feeds.
        expected: The default it falls back to.

    Returns:
        None.
    """
    monkeypatch.setenv(name, raw)
    assert getattr(load_topics_env(), field) == expected
    warnings = [line for line in loguru_caplog.text.splitlines() if name in line]
    assert len(warnings) == 1


def test_topics_range_edges_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero and 100 are valid percentiles, and 1 is a valid similarity.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("TOPICS_ADMIT_PERCENTILE", "0")
    monkeypatch.setenv("TOPICS_MERGE_MIN_SIMILARITY", "1")
    cfg = load_topics_env()
    assert cfg.admit_percentile == 0.0
    assert cfg.merge_min_similarity == 1.0
    monkeypatch.setenv("TOPICS_ADMIT_PERCENTILE", "100")
    assert load_topics_env().admit_percentile == 100.0


def test_topics_min_topic_size_one_is_raised_to_two(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """The service refuses a size of 1, so it is raised to 2; a negative becomes auto.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        loguru_caplog: Captured loguru warnings.

    Returns:
        None.
    """
    monkeypatch.setenv("TOPICS_MIN_TOPIC_SIZE", "1")
    assert load_topics_env().min_topic_size == 2
    assert "TOPICS_MIN_TOPIC_SIZE" in loguru_caplog.text
    monkeypatch.setenv("TOPICS_MIN_TOPIC_SIZE", "-3")
    assert load_topics_env().min_topic_size == 0


def test_topics_client_defaults_do_not_inherit_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """The endpoint knobs ignore the OPENAI_* settings.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_BASE", "x")
    assert load_topics_client_env() == TopicsClientConfig("http://vllm-router:4000", None, 3600.0)


def test_topics_client_strips_a_trailing_slash_and_a_blank_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trailing slash is dropped and a blank key counts as unset.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("TOPICS_API_BASE", "http://topics-only:8000/")
    monkeypatch.setenv("TOPICS_API_KEY", "   ")
    monkeypatch.setenv("TOPICS_TIMEOUT", "12")
    assert load_topics_client_env() == TopicsClientConfig("http://topics-only:8000", None, 12.0)


def test_topics_concurrency_clamps_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrency is at least 1 and defaults on garbage.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    assert load_topics_concurrency() == 1
    monkeypatch.setenv("DOCINT_TOPICS_CONCURRENCY", "0")
    assert load_topics_concurrency() == 1
    monkeypatch.setenv("DOCINT_TOPICS_CONCURRENCY", "abc")
    assert load_topics_concurrency() == 1
    monkeypatch.setenv("DOCINT_TOPICS_CONCURRENCY", "3")
    assert load_topics_concurrency() == 3


def test_topics_dir_defaults_under_docint_home_and_honours_the_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """TOPICS_DIR defaults beside the other docint directories.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Per-test temporary directory.

    Returns:
        None.
    """
    monkeypatch.delenv("TOPICS_DIR")
    cfg = load_path_env()
    assert cfg.topics == cfg.docint_home_dir / "topics"
    monkeypatch.setenv("TOPICS_DIR", str(tmp_path / "t"))
    assert load_path_env().topics == tmp_path / "t"


@pytest.mark.parametrize("other", ["QDRANT_SRC_DIR", "PIPELINE_ARTIFACTS_DIR"])
def test_topics_dir_inside_a_walked_directory_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, other: str
) -> None:
    """TOPICS_DIR may not lie inside, or equal, a directory docint walks.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Per-test temporary directory.
        other: The walked directory's variable.

    Returns:
        None.
    """
    monkeypatch.setenv(other, str(tmp_path / "walked"))
    monkeypatch.setenv("TOPICS_DIR", str(tmp_path / "walked" / "topics"))
    with pytest.raises(ValueError, match=rf"TOPICS_DIR.*{other}"):
        load_path_env()
    monkeypatch.setenv("TOPICS_DIR", str(tmp_path / "walked"))
    with pytest.raises(ValueError, match=rf"TOPICS_DIR.*{other}"):
        load_path_env()


def test_a_sibling_sharing_a_name_prefix_is_accepted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A directory whose name merely starts like the walked one is outside it.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Per-test temporary directory.

    Returns:
        None.
    """
    monkeypatch.setenv("QDRANT_SRC_DIR", str(tmp_path / "src"))
    monkeypatch.setenv("TOPICS_DIR", str(tmp_path / "src-topics"))
    assert load_path_env().topics == tmp_path / "src-topics"
