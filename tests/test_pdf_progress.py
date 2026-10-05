"""Tests for routing WeasyPrint's render progress to the thread that asked for it."""

from __future__ import annotations

import importlib.util
import logging
import threading
from pathlib import Path

import pytest

from docint.core.state import pdf_progress, report_render

_WEASYPRINT = logging.getLogger("weasyprint.progress")


def _require_weasyprint() -> None:
    """Skip the test when WeasyPrint's native libraries cannot load."""
    html_cls, error = report_render._load_weasyprint()
    if html_cls is None:
        pytest.skip(f"WeasyPrint is unavailable: {error}")


def _collect() -> tuple[list[tuple[str, int | None]], pdf_progress.ProgressCallback]:
    """Return a list and a callback appending every report to it."""
    seen: list[tuple[str, int | None]] = []
    return seen, lambda stage, page: seen.append((stage, page))


def test_a_render_reports_its_stages_and_first_pass_pages() -> None:
    """Pages count up once; a repagination pass re-lays pages without rewinding the count."""
    seen, callback = _collect()
    with pdf_progress.progress_scope(callback):
        _WEASYPRINT.info("Step 1 - Fetching and parsing HTML - %s", "HTML string")
        _WEASYPRINT.info("Step 3 - Applying CSS")
        _WEASYPRINT.info("Step 5 - Creating layout - Page %d", 1)
        _WEASYPRINT.info("Step 5 - Creating layout - Page %d", 2)
        _WEASYPRINT.info("Step 5 - Creating layout - Repagination #%d", 1)
        _WEASYPRINT.info("Step 5 - Creating layout - Page %d", 1)
        _WEASYPRINT.info("Step 5 - Creating layout - Page %d (up-to-date)", 2)
        _WEASYPRINT.info("Step 6 - Creating PDF")

    assert seen == [
        ("preparing", None),
        ("preparing", None),
        ("layout", 1),
        ("layout", 2),
        ("finishing", None),
        ("finishing", None),
    ]


def test_a_message_it_does_not_know_is_ignored() -> None:
    """A WeasyPrint release adding a step must not break a render."""
    seen, callback = _collect()
    with pdf_progress.progress_scope(callback):
        _WEASYPRINT.info("Step 8 - Something new - %s", "x")

    assert seen == []


def test_another_threads_render_never_reaches_the_callback() -> None:
    """A synchronous export on another worker thread must not feed a job's progress."""
    seen, callback = _collect()
    with pdf_progress.progress_scope(callback):
        other = threading.Thread(target=lambda: _WEASYPRINT.info("Step 5 - Creating layout - Page %d", 9))
        other.start()
        other.join()

    assert seen == []


def test_the_scope_ends_with_the_render_even_when_it_fails() -> None:
    """A worker thread is reused, so a failed render must not leave its callback behind."""
    seen, callback = _collect()
    with pytest.raises(RuntimeError), pdf_progress.progress_scope(callback):
        raise RuntimeError("render failed")
    _WEASYPRINT.info("Step 5 - Creating layout - Page %d", 1)

    assert seen == []


def test_each_record_reaches_the_callback_once() -> None:
    """Opening scope after scope must not stack handlers and double every report."""
    for _ in range(3):
        seen, callback = _collect()
        with pdf_progress.progress_scope(callback):
            _WEASYPRINT.info("Step 6 - Creating PDF")

    assert seen == [("finishing", None)]


def test_a_callback_exception_stops_the_render(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling a job raises from its progress callback, and that must end the render mid-layout."""

    class _Cancelled(Exception):
        pass

    class _FakeHTML:
        def __init__(self, string: str) -> None:
            self.string = string

        def write_pdf(self) -> bytes:
            _WEASYPRINT.info("Step 5 - Creating layout - Page %d", 1)
            raise AssertionError("the render went on after its callback raised")

    def cancel(stage: str, page: int | None) -> None:
        raise _Cancelled

    monkeypatch.setattr(report_render, "_load_weasyprint", lambda: (_FakeHTML, None))

    with pytest.raises(_Cancelled):
        report_render.html_to_pdf("<html></html>", progress=cancel)


def test_a_real_render_reports_its_pages() -> None:
    """Through the real engine, a render reports a laid-out page and then finishes."""
    _require_weasyprint()
    seen, callback = _collect()

    report_render.html_to_pdf("<html><body><p>one</p></body></html>", progress=callback)

    assert ("layout", 1) in seen
    assert seen[-1] == ("finishing", None)


def test_every_message_read_is_still_in_weasyprints_source() -> None:
    """A WeasyPrint upgrade that rewords a step would silently stop the progress; this fails instead."""
    spec = importlib.util.find_spec("weasyprint")
    assert spec is not None and spec.origin is not None
    source = "".join(path.read_text(encoding="utf-8") for path in Path(spec.origin).parent.rglob("*.py"))

    for message in pdf_progress.WEASYPRINT_MESSAGES:
        assert f"'{message}'" in source, message
