"""Tests for the Nextext remote media-processing client."""

import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

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
        nextext_max_concurrency=4,
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
        nextext_max_concurrency=4,
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


def test_default_client_sends_identity_header() -> None:
    """The self-built HTTP client carries the trusted identity header.

    Nextext resolves a per-request principal from this header and rejects
    header-less callers 401 unless its server-side default identity is set.
    """
    client = NextextClient(_cfg())
    assert client._client.headers["X-Auth-User"] == "docint"


def test_default_client_identity_header_configurable() -> None:
    """Custom auth_header/identity land on the self-built HTTP client."""
    cfg = NextextConfig(
        api_base="http://nextext.test",
        api_key="sekret",
        timeout=5.0,
        poll_interval=0.0,
        poll_max_seconds=5.0,
        enabled=True,
        keyframes_per_minute=4,
        keyframes_max=20,
        keyframe_dedup_cosine=0.95,
        nextext_max_concurrency=4,
        auth_header="X-Custom-User",
        identity="svc-docint",
    )
    client = NextextClient(cfg)
    assert client._client.headers["X-Custom-User"] == "svc-docint"
    assert client._client.headers["Authorization"] == "Bearer sekret"


def test_default_client_empty_identity_sends_no_header() -> None:
    """An empty identity suppresses the trusted header entirely."""
    cfg = NextextConfig(
        api_base="http://nextext.test",
        api_key=None,
        timeout=5.0,
        poll_interval=0.0,
        poll_max_seconds=5.0,
        enabled=True,
        keyframes_per_minute=4,
        keyframes_max=20,
        keyframe_dedup_cosine=0.95,
        nextext_max_concurrency=4,
        identity="",
    )
    client = NextextClient(cfg)
    assert "X-Auth-User" not in client._client.headers


def _options_from_multipart(request: httpx.Request) -> dict[str, object]:
    """Return the decoded ``options`` JSON of a multipart submission.

    Args:
        request (httpx.Request): The captured ``POST /jobs`` request.

    Returns:
        dict[str, object]: The parsed ``options`` form field.
    """
    body = request.content.decode("utf-8", "replace")
    part = body.split('name="options"', 1)[1]
    # Skip the part's own header block; its value follows the blank line.
    value = part.split("\r\n\r\n", 1)[1].split("\r\n--", 1)[0]
    return json.loads(value)


def test_options_payload_requests_keyframes_explicitly() -> None:
    """The options payload pins every key docint forwards to Nextext.

    Two of them are opt-in switches pointing opposite ways. Keyframe extraction
    defaults to off, so a payload carrying only the rate knobs samples nothing
    and the artifact 404s — indistinguishable from an audio-only clip.
    Captioning defaults to on, so omitting ``visual_context`` buys a vision
    request per frame for prose docint never downloads. The comparison is exact
    so a silently dropped key fails here too.
    """
    payload = json.loads(NextextClient(_cfg())._options_payload())
    assert payload == {"keyframes": True, "visual_context": False, "keyframes_per_minute": 4, "keyframes_max": 20}


def test_submit_multipart_carries_keyframes_option(tmp_path: Path) -> None:
    """The submitted multipart body really carries the job options.

    Pins the outgoing wire contract, not just the payload builder: the form
    field name and its JSON encoding are what Nextext parses.
    """
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fakevideo")
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/jobs":
            seen.update(_options_from_multipart(request))
        return _handler(request)

    client = httpx.Client(base_url="http://nextext.test", transport=httpx.MockTransport(handler))
    result = NextextClient(_cfg(), client=client).process_media(media)
    assert result.status == "completed"
    assert seen == {"keyframes": True, "visual_context": False, "keyframes_per_minute": 4, "keyframes_max": 20}


def test_rejected_options_names_the_required_nextext_version(
    tmp_path: Path, loguru_caplog: pytest.LogCaptureFixture
) -> None:
    """A 422 on submission says which Nextext version this build needs.

    Nextext's ``JobOptions`` forbids unknown fields, so an option this client
    sends to a server that predates it fails the whole submission — the clip is
    skipped with no transcript either, not just without frames. The bare status
    code does not say that, and an operator seeing it for every video in a batch
    should not have to work back to a version floor.
    """
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": [{"type": "extra_forbidden"}]})

    client = httpx.Client(base_url="http://nextext.test", transport=httpx.MockTransport(handler))
    result = NextextClient(_cfg(), client=client).process_media(media)

    assert result.status == "error"
    assert result.transcript_jsonl is None
    assert result.keyframes == []
    assert "v1.9.0" in loguru_caplog.text
    assert "422" in loguru_caplog.text
