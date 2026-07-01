"""Tests for the Nextext remote media-processing client."""

import io
import zipfile
from pathlib import Path

import httpx

from docint.utils.env_cfg import NextextConfig
from docint.utils.nextext_client import NextextClient


def _cfg() -> NextextConfig:
    return NextextConfig(
        api_base="http://nextext.test",
        api_key=None,
        timeout=5.0,
        poll_interval=0.0,
        poll_max_seconds=5.0,
        enabled=True,
        keyframes_per_minute=4,
        keyframes_max=20,
        keyframe_dedup_cosine=0.95,
    )


def _keyframes_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("frame_0.jpg", b"\xff\xd8\xff0")
        zf.writestr("frame_1.jpg", b"\xff\xd8\xff1")
    return buf.getvalue()


def _handler(request: httpx.Request) -> httpx.Response:
    if request.method == "POST" and request.url.path == "/jobs":
        return httpx.Response(201, json={"job_id": "J1", "status": "queued"})
    if request.url.path == "/jobs/J1" and request.method == "GET":
        return httpx.Response(200, json={"status": "completed"})
    if request.url.path == "/jobs/J1/artifacts/docint.jsonl":
        return httpx.Response(200, content=b'{"text":"hi","start_seconds":0,"end_seconds":1}\n')
    if request.url.path == "/jobs/J1/artifacts/keyframes.zip":
        return httpx.Response(200, content=_keyframes_zip())
    return httpx.Response(404)


def test_process_media_returns_transcript_and_keyframes(tmp_path: Path) -> None:
    """Test successful media processing returns transcript and keyframes."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fakevideo")
    client = httpx.Client(base_url="http://nextext.test", transport=httpx.MockTransport(_handler))
    result = NextextClient(_cfg(), client=client).process_media(media)
    assert result.status == "completed"
    assert result.transcript_jsonl is not None and b"hi" in result.transcript_jsonl
    assert len(result.keyframes) == 2


def test_process_media_failsoft_on_job_failure(tmp_path: Path) -> None:
    """Test that job failure is handled gracefully with no artifacts returned."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(201, json={"job_id": "J2", "status": "queued"})
        if request.url.path == "/jobs/J2":
            return httpx.Response(200, json={"status": "failed"})
        return httpx.Response(404)

    client = httpx.Client(base_url="http://nextext.test", transport=httpx.MockTransport(handler))
    result = NextextClient(_cfg(), client=client).process_media(media)
    assert result.status == "failed"
    assert result.transcript_jsonl is None
    assert result.keyframes == []


def test_process_media_disabled_no_network_call(tmp_path: Path) -> None:
    """Test that disabled config prevents any network calls."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")

    def error_handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("Network call should not be made when disabled")

    cfg = NextextConfig(
        api_base="http://nextext.test",
        api_key=None,
        timeout=5.0,
        poll_interval=0.0,
        poll_max_seconds=5.0,
        enabled=False,
        keyframes_per_minute=4,
        keyframes_max=20,
        keyframe_dedup_cosine=0.95,
    )

    client = httpx.Client(base_url="http://nextext.test", transport=httpx.MockTransport(error_handler))
    result = NextextClient(cfg, client=client).process_media(media)
    assert result.status == "disabled"
    assert result.transcript_jsonl is None
    assert result.keyframes == []


def test_process_media_poll_error_status(tmp_path: Path) -> None:
    """Test that HTTP errors during polling return poll_error status."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/jobs":
            return httpx.Response(201, json={"job_id": "J9", "status": "queued"})
        if request.url.path == "/jobs/J9" and request.method == "GET":
            return httpx.Response(503)
        return httpx.Response(404)

    client = httpx.Client(base_url="http://nextext.test", transport=httpx.MockTransport(handler))
    result = NextextClient(_cfg(), client=client).process_media(media)
    assert result.status == "poll_error"
    assert result.transcript_jsonl is None
    assert result.keyframes == []
