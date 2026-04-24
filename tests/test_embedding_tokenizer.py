"""RED tests for the offline HF tokenizer-backed token counter builder.

These tests pin the contract of ``docint.utils.embedding_tokenizer``
before the implementation is written:

- ``build_embedding_token_counter(repo_id, cache_dir)`` returns a
  callable that encodes text with ``add_special_tokens=True`` and
  returns the full list of input ids, so ``len(counter(text))`` is the
  authoritative token count that matches what the embedding provider
  will see (BOS/EOS included).
- When the HF snapshot is missing from the local cache the builder
  emits a loguru warning and returns ``None`` rather than raising, so
  callers can fall back gracefully to the character-ratio estimator.
- When ``AutoTokenizer.from_pretrained`` raises (corrupt cache,
  missing ``tokenizer.json``), the builder also emits a warning and
  returns ``None``.

The tests MUST fail on ``HEAD`` (the module does not yet exist) — that
red state is the TDD signal for the implementer to satisfy the
contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger


@pytest.fixture
def loguru_caplog(caplog: LogCaptureFixture) -> Iterable[LogCaptureFixture]:
    """Bridge loguru records into ``caplog`` for the duration of a test.

    Loguru bypasses ``logging`` so the stdlib ``caplog`` fixture does
    not see its records out of the box. This fixture adds a temporary
    loguru sink that re-emits every record through the stdlib handler
    ``caplog`` attaches to the root logger, then tears the sink down on
    exit. Mirrors the ``loguru_caplog`` fixture in
    ``tests/test_json_reader_nextext.py``.

    Args:
        caplog: The standard pytest log-capture fixture.

    Yields:
        The same ``caplog`` fixture, now populated with loguru-sourced
        records at WARNING level and above.
    """
    handler_id = logger.add(
        caplog.handler,
        level="WARNING",
        format="{message}",
    )
    caplog.set_level(logging.WARNING)
    try:
        yield caplog
    finally:
        logger.remove(handler_id)


class _StubTokenizer:
    """Deterministic tokenizer stub returning a fixed id list.

    Mirrors the surface of the ``AutoTokenizer`` instance the builder
    is expected to call: ``.encode(text, add_special_tokens=True)``
    returns a list of integers whose length equals the token count.
    """

    def __init__(
        self,
        ids_with_special: list[int] | None = None,
        ids_without_special: list[int] | None = None,
    ) -> None:
        """Initialise the stub with the id lists it will return.

        Args:
            ids_with_special: List returned when
                ``add_special_tokens=True``.
            ids_without_special: List returned when
                ``add_special_tokens=False``. Defaults to the same list
                as ``ids_with_special`` minus two special tokens.
        """
        self.ids_with_special = ids_with_special or [1, 2, 3, 4]
        self.ids_without_special = (
            ids_without_special
            if ids_without_special is not None
            else self.ids_with_special[1:-1]
        )
        self.calls: list[dict[str, Any]] = []

    def encode(
        self, text: str, add_special_tokens: bool = True
    ) -> list[int]:
        """Record the call and return the configured id list.

        Args:
            text: Input string (ignored; stub returns a fixed list).
            add_special_tokens: Whether to include BOS/EOS.

        Returns:
            The configured id list for the requested mode.
        """
        self.calls.append(
            {"text": text, "add_special_tokens": add_special_tokens}
        )
        if add_special_tokens:
            return list(self.ids_with_special)
        return list(self.ids_without_special)


def test_build_token_counter_returns_callable_when_tokenizer_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: builder returns a callable that reports token counts.

    When ``resolve_hf_cache_path`` finds a snapshot and
    ``AutoTokenizer.from_pretrained(..., local_files_only=True)``
    succeeds, the builder must return a callable whose output is the
    list of encoded ids. ``len(counter(text))`` must equal the length
    of the list the tokenizer returned.
    """
    from docint.utils import embedding_tokenizer

    stub = _StubTokenizer(ids_with_special=[1, 2, 3, 4])

    monkeypatch.setattr(
        embedding_tokenizer,
        "resolve_hf_cache_path",
        lambda cache_dir, repo_id: Path("/tmp/fake-cache/bge-m3"),
    )
    monkeypatch.setattr(
        embedding_tokenizer.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: stub),
    )

    counter = embedding_tokenizer.build_embedding_token_counter(
        "BAAI/bge-m3", Path("/tmp/cache")
    )

    assert callable(counter)
    # mypy-friendly: tell the type checker counter is not None here.
    assert counter is not None
    assert counter("hi") == [1, 2, 3, 4]
    assert len(counter("hi")) == 4


def test_build_token_counter_returns_none_when_cache_missing(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """Builder returns ``None`` and warns when the HF snapshot is absent.

    ``resolve_hf_cache_path`` returning ``None`` means the local HF
    cache has no snapshot for the repo. The builder must emit a
    loguru warning naming the missing cache and return ``None`` so
    callers fall back to the char-ratio estimator.
    """
    from docint.utils import embedding_tokenizer

    monkeypatch.setattr(
        embedding_tokenizer,
        "resolve_hf_cache_path",
        lambda cache_dir, repo_id: None,
    )

    result = embedding_tokenizer.build_embedding_token_counter(
        "BAAI/bge-m3", Path("/tmp/cache")
    )

    assert result is None
    combined = "\n".join(str(r.msg) for r in loguru_caplog.records)
    import re

    assert re.search(r"cache.*not found", combined, re.IGNORECASE), (
        f"expected 'cache ... not found' warning, got: {combined!r}"
    )


def test_build_token_counter_returns_none_when_autotokenizer_raises(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """Builder returns ``None`` and warns when ``from_pretrained`` raises.

    Even if the snapshot directory exists, a corrupt or incomplete
    cache can make ``AutoTokenizer.from_pretrained`` raise
    ``OSError``. The builder must surface the failure as a loguru
    warning and return ``None`` so the ingestion path falls back to
    the char-ratio estimator instead of crashing.
    """
    from docint.utils import embedding_tokenizer

    monkeypatch.setattr(
        embedding_tokenizer,
        "resolve_hf_cache_path",
        lambda cache_dir, repo_id: Path("/tmp/fake-cache/bge-m3"),
    )

    def _raise(*args: Any, **kwargs: Any) -> Any:
        """Raise ``OSError`` to simulate a corrupt cache.

        Args:
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            OSError: Always.
        """
        raise OSError("no tokenizer.json")

    monkeypatch.setattr(
        embedding_tokenizer.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: _raise(*args, **kwargs)),
    )

    result = embedding_tokenizer.build_embedding_token_counter(
        "BAAI/bge-m3", Path("/tmp/cache")
    )

    assert result is None
    assert any(
        record.levelno >= logging.WARNING for record in loguru_caplog.records
    ), "expected at least one warning record"


def test_token_counter_includes_special_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The built counter must pass ``add_special_tokens=True`` to the tokenizer.

    The count must match what the embedding provider will see on the
    wire, including BOS/EOS. If the builder forgot to request special
    tokens the count would under-report by two for bge-m3 and admit
    payloads that overflow the provider's context window.
    """
    from docint.utils import embedding_tokenizer

    stub = _StubTokenizer(
        ids_with_special=[0, 1, 2, 3, 99],
        ids_without_special=[1, 2, 3],
    )
    monkeypatch.setattr(
        embedding_tokenizer,
        "resolve_hf_cache_path",
        lambda cache_dir, repo_id: Path("/tmp/fake-cache/bge-m3"),
    )
    monkeypatch.setattr(
        embedding_tokenizer.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: stub),
    )

    counter = embedding_tokenizer.build_embedding_token_counter(
        "BAAI/bge-m3", Path("/tmp/cache")
    )

    assert counter is not None
    ids = counter("any text")

    assert len(ids) == 5, (
        "token counter must return the add_special_tokens=True length"
    )
    assert stub.calls, "tokenizer.encode was never called"
    assert stub.calls[-1]["add_special_tokens"] is True, (
        f"expected add_special_tokens=True, got {stub.calls[-1]!r}"
    )
