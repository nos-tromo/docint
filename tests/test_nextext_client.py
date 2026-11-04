"""Tests for the Nextext remote media-processing client."""

import io
import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from docint.utils.audit_forward import forwarding
from docint.utils.audit_runtime import parse_purpose_header
from docint.utils.env_cfg import NextextConfig
from docint.utils.nextext_client import NextextClient, NextextResult


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


def _keyframes_zip(manifest: object | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("frame_0.jpg", b"\xff\xd8\xff0")
        zf.writestr("frame_1.jpg", b"\xff\xd8\xff1")
        if manifest is not None:
            zf.writestr("manifest.json", json.dumps(manifest))
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
    assert [frame.jpeg for frame in result.keyframes] == [b"\xff\xd8\xff0", b"\xff\xd8\xff1"]
    assert [frame.time_sec for frame in result.keyframes] == [None, None]


def test_keyframes_carry_the_manifest_times(tmp_path: Path) -> None:
    """A manifest pairs each frame with the second it was sampled from."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fakevideo")
    manifest = {
        "frames": [
            {"file": "frame_1.jpg", "index": 1, "time_sec": 12.5},
            {"file": "frame_0.jpg", "index": 0, "time_sec": 0.0},
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/jobs/J1/artifacts/keyframes.zip":
            return httpx.Response(200, content=_keyframes_zip(manifest))
        return _handler(request)

    client = httpx.Client(base_url="http://nextext.test", transport=httpx.MockTransport(handler))
    result = NextextClient(_cfg(), client=client).process_media(media)
    assert [(frame.index, frame.time_sec) for frame in result.keyframes] == [(0, 0.0), (1, 12.5)]
    assert [frame.jpeg for frame in result.keyframes] == [b"\xff\xd8\xff0", b"\xff\xd8\xff1"]


def test_keyframes_ignore_a_manifest_naming_unknown_files(tmp_path: Path) -> None:
    """A frame the manifest does not name keeps its position and no time."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fakevideo")
    manifest = {"frames": [{"file": "other.jpg", "index": 0, "time_sec": 3.0}]}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/jobs/J1/artifacts/keyframes.zip":
            return httpx.Response(200, content=_keyframes_zip(manifest))
        return _handler(request)

    client = httpx.Client(base_url="http://nextext.test", transport=httpx.MockTransport(handler))
    result = NextextClient(_cfg(), client=client).process_media(media)
    assert [frame.time_sec for frame in result.keyframes] == [None, None]


def test_keyframes_survive_a_corrupt_manifest(tmp_path: Path) -> None:
    """Unreadable JSON degrades to untimed frames rather than losing the clip."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fakevideo")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/jobs/J1/artifacts/keyframes.zip":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("frame_0.jpg", b"\xff\xd8\xff0")
                zf.writestr("manifest.json", "{not json")
            return httpx.Response(200, content=buf.getvalue())
        return _handler(request)

    client = httpx.Client(base_url="http://nextext.test", transport=httpx.MockTransport(handler))
    result = NextextClient(_cfg(), client=client).process_media(media)
    assert [(frame.index, frame.time_sec) for frame in result.keyframes] == [(0, None)]


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


#: Every request one clip makes: the submission, a poll, and both artifacts.
_CALLS = [
    ("POST", "/jobs"),
    ("GET", "/jobs/J1"),
    ("GET", "/jobs/J1/artifacts/docint.jsonl"),
    ("GET", "/jobs/J1/artifacts/keyframes.zip"),
]


def _recorded(
    tmp_path: Path, handler: Callable[[httpx.Request], httpx.Response] = _handler
) -> tuple[NextextResult, list[httpx.Request]]:
    """Run one clip through a recording transport.

    Args:
        tmp_path (Path): Where the synthetic clip is written.
        handler (Callable[[httpx.Request], httpx.Response]): The transport's handler.

    Returns:
        tuple[NextextResult, list[httpx.Request]]: The result and every request sent.
    """
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fakevideo")
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.Client(base_url="http://nextext.test", transport=httpx.MockTransport(recording))
    return NextextClient(_cfg(), client=client).process_media(media), seen


def test_every_call_states_docints_own_purpose_when_none_was_stated(tmp_path: Path) -> None:
    """Nextext audits a poll as a query and a download as a disclosure: each states why docint asks."""
    result, seen = _recorded(tmp_path)

    assert result.status == "completed"
    assert [(r.method, r.url.path) for r in seen] == _CALLS
    for request in seen:
        assert parse_purpose_header(request.headers["X-Audit-Purpose"]) == (None, "docint ingest")
        assert "X-Request-ID" not in request.headers


def test_every_call_states_the_purpose_of_the_work_that_triggered_it(tmp_path: Path) -> None:
    """The ingest's stated purpose and its request id go along on all of a clip's calls."""
    with forwarding(case_ref="case-0001", reason="Review of the flagged postings", request_id="req-0001"):
        result, seen = _recorded(tmp_path)

    assert result.status == "completed"
    assert [(r.method, r.url.path) for r in seen] == _CALLS
    for request in seen:
        assert parse_purpose_header(request.headers["X-Audit-Purpose"]) == (
            "case-0001",
            "Review of the flagged postings",
        )
        assert request.headers["X-Request-ID"] == "req-0001"


@pytest.mark.parametrize(
    ("refused", "status"),
    [("/jobs", "error"), ("/jobs/J1", "poll_error"), ("/jobs/J1/artifacts/docint.jsonl", "error")],
)
def test_a_refusal_for_want_of_a_purpose_is_the_usual_ingest_error(tmp_path: Path, refused: str, status: str) -> None:
    """A 428 from Nextext skips the clip the way any other refusal at that step does.

    Args:
        tmp_path (Path): Where the synthetic clip is written.
        refused (str): The path Nextext answers 428.
        status (str): The result status that step's failure has always had.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == refused:
            return httpx.Response(428, json={"detail": "A stated purpose is required for this request."})
        return _handler(request)

    result, _ = _recorded(tmp_path, handler)

    assert result.status == status
    assert result.transcript_jsonl is None
    assert result.keyframes == []
