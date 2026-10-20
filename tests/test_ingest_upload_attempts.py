"""Upload attempts at one file must never corrupt each other's staged bytes.

The SPA retries a batch whose request stalled, but a stall is exactly when the
abort may never reach the server, so the first attempt's handler can still be
writing when the retry's starts. These tests drive ``ingest_upload``'s stream
by hand to interleave two attempts deterministically.
"""

import io
import json
import os
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import UploadFile
from starlette.requests import Request

import docint.core.api as api_module
from docint.utils.hashing import compute_file_hash

MiB = 1024 * 1024


@pytest.fixture
def anyio_backend() -> str:
    """Run the async tests on asyncio only.

    Returns:
        str: The anyio backend name.
    """
    return "asyncio"


@pytest.fixture
def staging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point uploads at a temporary sources root, with no auth or preprocessing.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): The temporary path fixture.

    Returns:
        Path: The sources root; collection ``c1`` stages under ``<root>/c1``.
    """
    root = tmp_path / "sources"
    root.mkdir()
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: root)
    monkeypatch.setattr(api_module, "resolve_principal", lambda request: object())
    monkeypatch.setattr(api_module, "_register_collection", lambda principal, name: name)
    monkeypatch.setenv("INGEST_PREPROCESS_ON_UPLOAD", "false")
    return root


async def _attempt(filename: str, data: bytes) -> AsyncGenerator[str, None]:
    """Start one upload attempt of ``data`` and return its unread SSE stream.

    Args:
        filename (str): The upload's name.
        data (bytes): The file's bytes.

    Returns:
        AsyncGenerator[str, None]: The attempt's event stream, not yet started.
    """
    upload = UploadFile(io.BytesIO(data), filename=filename)
    response = await api_module.ingest_upload(Request({"type": "http", "headers": []}), "c1", [upload])
    return cast(AsyncGenerator[str, None], response.body_iterator)


def _event(frame: str) -> tuple[str, dict[str, Any]]:
    """Split one SSE frame into its event name and data.

    Args:
        frame (str): A frame as ``_format_sse`` writes it.

    Returns:
        tuple[str, dict[str, Any]]: The event name and its decoded data.
    """
    event_line, data_line = frame.strip().split("\n", 1)
    return event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))


@pytest.mark.anyio
async def test_overlapping_attempts_never_truncate_each_other(staging: Path, tmp_path: Path) -> None:
    """A retry writing the same file must not cut the first attempt's commit short.

    Args:
        staging (Path): The sources root.
        tmp_path (Path): The temporary path fixture, for a reference copy.
    """
    data = bytes(range(256)) * (8 * 1024)  # 2 MiB: two read chunks
    first = await _attempt("a.bin", data)
    retry = await _attempt("a.bin", data)

    assert _event(await anext(first))[0] == "start"
    await anext(first)
    await anext(first)  # both chunks written: the whole file is in hand
    assert _event(await anext(retry))[0] == "start"
    await anext(retry)  # the retry has written its first chunk
    event, saved = _event(await anext(first))  # the first attempt commits
    await retry.aclose()  # the retry is cut off mid-file

    reference = tmp_path / "reference.bin"
    reference.write_bytes(data)
    batch = staging / "c1"
    assert event == "file_saved"
    assert (batch / "a.bin").read_bytes() == data
    assert saved["file_hash"] == compute_file_hash(reference)
    assert not list(batch.glob("*.part"))


@pytest.mark.anyio
async def test_a_cut_off_attempt_leaves_nothing_behind(staging: Path) -> None:
    """An attempt stopped mid-file leaves neither a partial file nor the final one.

    Args:
        staging (Path): The sources root.
    """
    attempt = await _attempt("a.bin", bytes(2 * MiB))
    await anext(attempt)
    await anext(attempt)  # the first chunk is written
    await attempt.aclose()

    batch = staging / "c1"
    assert not (batch / "a.bin").exists()
    assert not list(batch.glob("*.part"))


@pytest.mark.anyio
async def test_an_upload_clears_partials_an_earlier_process_left(staging: Path) -> None:
    """A partial file older than the running process belongs to no attempt still running.

    Args:
        staging (Path): The sources root.
    """
    batch = staging / "c1"
    batch.mkdir()
    stale = batch / "old.bin.0123456789ab.part"
    stale.write_bytes(b"cut off")
    a_day_ago = time.time() - 86_400
    os.utime(stale, (a_day_ago, a_day_ago))
    live = batch / "b.bin.ba9876543210.part"
    live.write_bytes(b"still arriving")

    attempt = await _attempt("a.bin", b"whole")
    frames = [frame async for frame in attempt]

    assert _event(frames[-1])[0] == "upload_complete"
    assert (batch / "a.bin").read_bytes() == b"whole"
    assert not stale.exists()
    assert live.exists()


@pytest.mark.anyio
async def test_a_long_file_name_still_fits_its_temp_file(staging: Path) -> None:
    """The attempt's token must not push a long but valid name past the 255-byte limit.

    Args:
        staging (Path): The sources root.
    """
    name = "a" * 246 + ".pdf"  # 250 bytes: valid, but not with the token appended
    attempt = await _attempt(name, b"%PDF-1.4")
    frames = [frame async for frame in attempt]

    batch = staging / "c1"
    assert _event(frames[-1])[0] == "upload_complete"
    assert (batch / name).read_bytes() == b"%PDF-1.4"
    assert not list(batch.glob("*.part"))


@pytest.mark.anyio
async def test_a_failing_cleanup_never_hides_the_save_error(staging: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Removing the temp file must not turn a reported save failure into a crashed stream.

    Args:
        staging (Path): The sources root.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """

    def fail_hash(path: Path) -> str:
        """Fail the way an unreadable disk would.

        Args:
            path (Path): The file to hash (ignored).

        Raises:
            OSError: Always.
        """
        raise OSError("read failed")

    def fail_unlink(self: Path, missing_ok: bool = False) -> None:
        """Fail the way a read-only volume would.

        Args:
            self (Path): The path to remove (ignored).
            missing_ok (bool): Ignored.

        Raises:
            PermissionError: Always.
        """
        raise PermissionError("read-only")

    monkeypatch.setattr(api_module, "compute_file_hash", fail_hash)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    attempt = await _attempt("a.bin", b"x")
    frames = [frame async for frame in attempt]

    event, data = _event(frames[-1])
    assert event == "error"
    assert data["code"] == "save_failed"
