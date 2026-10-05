"""WeasyPrint's render progress, routed to the thread that asked for it.

WeasyPrint says how far a render has got only through its
``weasyprint.progress`` logger ("Step 5 - Creating layout - Page 12"). That
logger is not bridged into docint's log — only uvicorn's are — so its INFO
records used to be dropped. One handler, installed on first use, now hands each
record to the callback the *current thread* registered for its render, and to
nobody else: renders run on worker threads, and a synchronous export on another
thread must never feed a background job's progress.

Messages are matched against WeasyPrint's own format strings
(:data:`WEASYPRINT_MESSAGES`), which a test pins to its installed source.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from typing_extensions import override

#: ``callback(stage, page)``: ``stage`` is ``"preparing"``, ``"layout"`` (with
#: the page just laid out) or ``"finishing"``; ``page`` is ``None`` outside layout.
ProgressCallback = Callable[[str, int | None], None]

_PREPARING: frozenset[str] = frozenset(
    {
        "Step 1 - Fetching and parsing HTML - %s",
        "Step 2 - Fetching and parsing CSS - %s",
        "Step 3 - Applying CSS",
        "Step 4 - Creating formatting structure",
    }
)
_PAGE = "Step 5 - Creating layout - Page %d"
_PAGE_REUSED = "Step 5 - Creating layout - Page %d (up-to-date)"
_REPAGINATION = "Step 5 - Creating layout - Repagination #%d"
_FINISHING: frozenset[str] = frozenset({"Step 6 - Creating PDF", "Step 7 - Adding PDF metadata"})

#: Every WeasyPrint progress message this module reads.
WEASYPRINT_MESSAGES: frozenset[str] = _PREPARING | {_PAGE, _PAGE_REUSED, _REPAGINATION} | _FINISHING

_LOGGER_NAME = "weasyprint.progress"
_HANDLER_NAME = "docint-pdf-progress"
_install_lock = threading.Lock()
_local = threading.local()


@dataclass
class _Scope:
    """One thread's render: who to tell, and whether layout has started over."""

    callback: ProgressCallback
    repaginating: bool = False


class _Dispatch(logging.Handler):
    """Hand a progress record to the callback of the thread that logged it."""

    @override
    def emit(self, record: logging.LogRecord) -> None:
        """Report one step of the current thread's render, if it asked to be told.

        A callback's exception propagates on purpose: it is how a cancelled
        job stops its render mid-layout. Only reading the record is guarded.

        Args:
            record (logging.LogRecord): A ``weasyprint.progress`` record.
        """
        scope: _Scope | None = getattr(_local, "scope", None)
        if scope is None:
            return
        message = record.msg
        if message in _PREPARING:
            scope.callback("preparing", None)
        elif message == _PAGE and not scope.repaginating:
            args = record.args
            page = args[0] if isinstance(args, tuple) and args and isinstance(args[0], int) else None
            scope.callback("layout", page)
        elif message == _REPAGINATION:
            # A second pass re-lays pages already counted; their numbers would run backwards.
            scope.repaginating = True
            scope.callback("finishing", None)
        elif message in _FINISHING:
            scope.callback("finishing", None)


def _install() -> None:
    """Attach the dispatching handler to WeasyPrint's progress logger, once per process."""
    with _install_lock:
        logger = logging.getLogger(_LOGGER_NAME)
        if any(handler.get_name() == _HANDLER_NAME for handler in logger.handlers):
            return
        handler = _Dispatch()
        handler.set_name(_HANDLER_NAME)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False


@contextmanager
def progress_scope(callback: ProgressCallback | None) -> Iterator[None]:
    """Route this thread's WeasyPrint progress to ``callback`` while the block runs.

    Args:
        callback (ProgressCallback | None): Told each step of the render;
            ``None`` makes the scope a no-op.

    Yields:
        None: Control, for the render to run in.
    """
    if callback is None:
        yield
        return
    _install()
    previous = getattr(_local, "scope", None)
    _local.scope = _Scope(callback)
    try:
        yield
    finally:
        _local.scope = previous
