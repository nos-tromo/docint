"""Tests for the ``topics`` CLI entry point.

Every fixture is synthetic: invented run ids, labels and collection names.
"""

from __future__ import annotations

import types
from typing import Any, ClassVar

import pytest

import docint.cli.topics as cli
from docint.core.topics.service import TopicsOutcome, TopicsRefused
from docint.utils.topics_client import TopicsUnreachable

OVERVIEW = {
    "run_id": "20260102-030405-abcdef",
    "coverage": {"total": 100, "assigned": 90, "unassigned": 10, "pending": 0},
    "topics": [
        {
            "id": 3,
            "size": 60,
            "label": "Gardening",
            "keywords": [["soil", 1.0], ["seed", 0.9], ["leaf", 0.8], ["root", 0.7]],
        },
        {"id": 1, "size": 30, "label": "Sailing", "keywords": [["sail", 1.0], ["mast", 0.9], ["tide", 0.8]]},
    ],
}


class _Service:
    """A topics service whose ``run`` is scripted."""

    def __init__(self, error: Exception | None = None) -> None:
        """Script the run.

        Args:
            error: Raised by ``run`` when set.
        """
        self.error = error
        self.calls: list[tuple[Any, ...]] = []

    def run(self, action: str, physical: str, *, progress: Any, **kwargs: Any) -> TopicsOutcome:
        """Report a tick, then finish or raise.

        Args:
            action: The action.
            physical: Physical collection name.
            progress: Receives ``(stage, done, total)``.
            **kwargs: Unused.

        Returns:
            The outcome.

        Raises:
            Exception: The scripted error.
        """
        self.calls.append((action, physical))
        progress("aligning", 1, 2)
        if self.error is not None:
            raise self.error
        return TopicsOutcome(action, "run-1", 4, 1, 12, False)


class _FakeRAG:
    """Stand-in for :class:`~docint.core.rag.RAG` that touches no Qdrant."""

    service = _Service()
    overview: dict[str, Any] | None = OVERVIEW
    instances: ClassVar[list[_FakeRAG]] = []

    def __init__(self, *, qdrant_collection: str) -> None:
        """Record the collection.

        Args:
            qdrant_collection: The collection.
        """
        self.collection = qdrant_collection
        self.unloaded = False
        self.qdrant_client = types.SimpleNamespace(collection_exists=lambda collection_name: True)
        _FakeRAG.instances.append(self)

    def topics_service(self) -> _Service:
        """Return the scripted service.

        Returns:
            The service.
        """
        return _FakeRAG.service

    def topics_overview(self, physical: str) -> dict[str, Any] | None:
        """Return the scripted overview.

        Args:
            physical: Physical collection name.

        Returns:
            The overview.
        """
        return _FakeRAG.overview

    def collection_created_at(self, physical: str) -> str | None:
        """Return no creation time.

        Args:
            physical: Physical collection name.

        Returns:
            ``None``.
        """
        return None

    def unload_models(self) -> None:
        """Record that the CLI released its models."""
        self.unloaded = True


@pytest.fixture(autouse=True)
def _stub_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the RAG class for the fake and reset its script.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    monkeypatch.setenv("TOPICS_ENABLED", "true")
    _FakeRAG.service = _Service()
    _FakeRAG.overview = OVERVIEW
    _FakeRAG.instances = []
    monkeypatch.setattr(cli, "RAG", _FakeRAG)


def test_parse_args_takes_an_op_and_an_optional_collection() -> None:
    """The op is a required choice; the collection may be left to the prompt."""
    args = cli.parse_args(["build", "mydocs"])
    assert (args.op, args.collection) == ("build", "mydocs")
    assert cli.parse_args(["status"]).collection is None
    with pytest.raises(SystemExit):
        cli.parse_args(["other"])


def test_status_prints_the_overview(capsys: pytest.CaptureFixture[str]) -> None:
    """Status prints one summary line and one line per topic.

    Args:
        capsys: The capture fixture.
    """
    assert cli.run("status", "mydocs") == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("run=20260102-030405-abcdef topics=2 assigned=90/100")
    assert lines[1] == "T3  60  Gardening — soil, seed, leaf"
    assert lines[2] == "T1  30  Sailing — sail, mast, tide"
    assert all(rag.unloaded for rag in _FakeRAG.instances)


def test_status_without_a_run(capsys: pytest.CaptureFixture[str]) -> None:
    """With no run status says so.

    Args:
        capsys: The capture fixture.
    """
    _FakeRAG.overview = None

    assert cli.run("status", "mydocs") == 0
    assert capsys.readouterr().out.strip() == "No topics yet."


def test_build_runs_the_service_and_prints_the_outcome(capsys: pytest.CaptureFixture[str]) -> None:
    """A build runs on the physical collection and prints the outcome line.

    Args:
        capsys: The capture fixture.
    """
    assert cli.run("build", "mydocs") == 0

    assert _FakeRAG.service.calls == [("build", "mydocs")]
    assert capsys.readouterr().out.strip() == "build: run=run-1 topics=4 added=1 assigned=12"


def test_a_refusal_exits_2_with_its_message() -> None:
    """A refusal exits 2."""
    _FakeRAG.service = _Service(error=TopicsRefused("no_run"))

    assert cli.run("align", "mydocs") == 2


def test_an_endpoint_fault_exits_1() -> None:
    """An endpoint fault exits 1."""
    _FakeRAG.service = _Service(error=TopicsUnreachable("down"))

    assert cli.run("build", "mydocs") == 1


def test_disabled_topics_exit_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """A switched-off feature exits 2 before touching Qdrant.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    monkeypatch.setenv("TOPICS_ENABLED", "false")

    assert cli.run("status", "mydocs") == 2
    assert _FakeRAG.instances == []


def test_main_exits_with_the_run_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` raises SystemExit with the operation's code.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    monkeypatch.setattr(cli, "init_logger", lambda: None)
    _FakeRAG.service = _Service(error=TopicsUnreachable("down"))

    with pytest.raises(SystemExit) as exc:
        cli.main(["build", "mydocs"])
    assert exc.value.code == 1
