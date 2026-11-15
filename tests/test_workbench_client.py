"""Tests for the workbench MCP client behind the chat agent's ``run_python`` tool.

The door is mocked from the contract alone (``httpx.MockTransport``): revision
``2026-07-28``, its headers and ``_meta``, the result shape and the reason
tokens. Every name, code and value here is invented.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, ClassVar

import anyio
import httpx
import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger

import docint.utils.workbench_client as wb
from docint import __version__
from docint.utils.audit_forward import forwarding
from docint.utils.audit_runtime import parse_purpose_header
from docint.utils.env_cfg import WorkbenchClientConfig
from docint.utils.workbench_client import (
    MAX_IMAGE_BYTES,
    META_CLIENT_CAPS,
    META_CLIENT_INFO,
    META_PROTOCOL,
    META_SERVER_INFO,
    PNG_SIGNATURE,
    Discovery,
    FailureKind,
    InlineFile,
    OutputFile,
    RunFailure,
    RunResult,
    WorkbenchClient,
    session_pseudonym,
    user_pseudonym,
)

KEY = "k" * 32
CFG = WorkbenchClientConfig(api_base="http://workbench.test", timeout=75.0, api_key=KEY)
JOB = "0" * 32
CODE = "import json\nprint(len(json.load(open('passages.json'))))"
DATA = (InlineFile(name="passages.json", content="[]"),)
CHART = PNG_SIGNATURE + b"\x00\x00\x00\rIHDR" + b"\x00" * 17
CSV = "author,posts\nuser-a,12\nuser-b,7\n"
SERVER_META = {META_SERVER_INFO: {"name": "workbench", "version": "0.2.0"}}
DISCOVERED = Discovery(cap_s=60, server_version="0.2.0")


@dataclass(frozen=True)
class RpcError:
    """A JSON-RPC error the door answers with."""

    code: int
    message: str
    status: int = 500


def _discover_result() -> dict[str, Any]:
    return {
        "supportedVersions": ["2026-07-28", "2025-11-25", "2025-06-18"],
        "capabilities": {"tools": {"listChanged": False}},
        "instructions": "workbench runs Python on the data sent with each call. Results come back inline.",
        "ttlMs": 3600000,
        "cacheScope": "private",
        "resultType": "complete",
        "_meta": SERVER_META,
    }


def _tools_result(time_limit: dict[str, Any] | None = None) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "data": {"type": "array"},
            "time_limit_s": time_limit if time_limit is not None else {"type": "integer", "minimum": 5, "maximum": 60},
        },
        "required": ["code"],
    }
    return {
        "tools": [{"name": "run_python", "description": "Run Python.", "inputSchema": schema}],
        "ttlMs": 300000,
        "cacheScope": "private",
        "resultType": "complete",
        "_meta": SERVER_META,
    }


def _call_result() -> dict[str, Any]:
    structured = {
        "stdout": "2 authors",
        "stderr": "",
        "truncated": False,
        "job_id": JOB,
        "files": [
            {"name": "chart.png", "mime": "image/png", "bytes": len(CHART), "returned": True},
            {"name": "counts.csv", "mime": "text/csv", "bytes": len(CSV), "returned": True},
        ],
        "hint": None,
        "reason": None,
    }
    return {
        "resultType": "complete",
        "isError": False,
        "content": [
            {"type": "text", "text": json.dumps(structured)},
            {
                "type": "image",
                "data": base64.b64encode(CHART).decode(),
                "mimeType": "image/png",
                "annotations": {"audience": ["user"]},
            },
            {
                "type": "resource",
                "resource": {"uri": f"workbench://jobs/{JOB}/out/counts.csv", "mimeType": "text/csv", "text": CSV},
            },
        ],
        "structuredContent": structured,
        "_meta": SERVER_META,
    }


def _refusal(token: str, job_id: str | None = None) -> dict[str, Any]:
    return {
        "resultType": "complete",
        "isError": True,
        "content": [{"type": "text", "text": "busy, retry later"}],
        "structuredContent": {"reason": token, "job_id": job_id},
        "_meta": SERVER_META,
    }


class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        """Start at an arbitrary instant."""
        self.now = 1_000.0

    def __call__(self) -> float:
        """Return the current instant.

        Returns:
            float: Seconds on the fake scale.
        """
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward.

        Args:
            seconds (float): How far.
        """
        self.now += seconds


class FakeDoor:
    """workbench's MCP door as the contract describes it: it answers by ``Mcp-Method`` and records every request."""

    def __init__(self) -> None:
        """Answer every method as a healthy door does, without delay."""
        self.requests: list[tuple[httpx.Request, dict[str, Any]]] = []
        self.answers: dict[str, object] = {
            "server/discover": _discover_result(),
            "tools/list": _tools_result(),
            "tools/call": _call_result(),
        }
        self.slow: dict[str, float] = {}

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        """Record the request and answer it.

        Args:
            request (httpx.Request): The request the client sent.

        Returns:
            httpx.Response: The scripted answer for its method.

        Raises:
            Exception: The scripted transport error, when one is set.
        """
        body = json.loads(request.content)
        self.requests.append((request, body))
        method = request.headers["mcp-method"]
        if method in self.slow:
            await anyio.sleep(self.slow[method])
        answer = self.answers[method]
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, httpx.Response):
            return answer
        if isinstance(answer, RpcError):
            error = {"code": answer.code, "message": answer.message}
            return httpx.Response(answer.status, json={"jsonrpc": "2.0", "id": body["id"], "error": error})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": answer})

    def methods(self) -> list[str]:
        """Return the method of every request so far, in order.

        Returns:
            list[str]: The methods.
        """
        return [body["method"] for _, body in self.requests]


def _client(door: FakeDoor, clock: FakeClock | None = None) -> WorkbenchClient:
    return WorkbenchClient(CFG, transport=httpx.MockTransport(door), clock=clock or FakeClock())


def _run(client: WorkbenchClient, **overrides: Any) -> RunResult | RunFailure:
    arguments: dict[str, Any] = {
        "code": CODE,
        "data": DATA,
        "time_limit_s": 23,
        "owner": "user-a",
        "session_id": "session-1",
        "timeout_s": 40.0,
    }
    arguments.update(overrides)
    return client.run_python(**arguments)


def _levels(caplog: LogCaptureFixture) -> list[str]:
    # httpx's own stdlib records (one INFO line per request) land in caplog too.
    return [record.levelname for record in caplog.records if record.name == wb.__name__]


@pytest.fixture
def log_lines() -> Iterator[list[str]]:
    """Capture every loguru line, DEBUG included, with its traceback if any.

    Yields:
        list[str]: The lines written while the test runs.
    """
    lines: list[str] = []

    def sink(message: object) -> None:
        lines.append(str(message))

    handler_id = logger.add(sink, level="DEBUG", format="{message}")
    try:
        yield lines
    finally:
        logger.remove(handler_id)


@pytest.fixture
def fresh_process_client() -> Iterator[None]:
    """Forget the process-wide client before and after the test."""
    wb.workbench_client.cache_clear()
    try:
        yield
    finally:
        wb.workbench_client.cache_clear()


# --- discovery ---------------------------------------------------------------


def test_discovery_reads_the_cap_and_the_version() -> None:
    """Discovery asks server/discover, then tools/list, and keeps the cap and the version."""
    door = FakeDoor()
    assert WorkbenchClient(CFG, transport=httpx.MockTransport(door)).discover() == DISCOVERED
    assert door.methods() == ["server/discover", "tools/list"]


@pytest.mark.parametrize("change", ["no_modern_revision", "no_run_python", "no_maximum", "not_json_rpc"])
def test_discovery_offers_nothing_without_the_modern_revision_and_a_capped_run_python(change: str) -> None:
    """Without 2026-07-28, a listed run_python and its cap, there is no tool to offer."""
    door = FakeDoor()
    if change == "no_modern_revision":
        door.answers["server/discover"] = {**_discover_result(), "supportedVersions": ["2025-11-25", "2025-06-18"]}
    elif change == "no_run_python":
        listed = _tools_result()
        listed["tools"][0]["name"] = "list_files"
        door.answers["tools/list"] = listed
    elif change == "no_maximum":
        door.answers["tools/list"] = _tools_result(time_limit={"type": "integer", "minimum": 5})
    else:
        door.answers["server/discover"] = httpx.Response(404, json={"detail": "not found"})
    assert _client(door).discover() is None


def test_each_result_is_cached_for_its_ttl() -> None:
    """tools/list is asked again after its 300 s, server/discover after its hour."""
    door = FakeDoor()
    clock = FakeClock()
    client = _client(door, clock)
    client.discover()
    clock.advance(299)
    client.discover()
    assert door.methods() == ["server/discover", "tools/list"]
    clock.advance(2)
    assert client.discover() == DISCOVERED
    assert door.methods() == ["server/discover", "tools/list", "tools/list"]
    clock.advance(3600)
    assert client.discover() == DISCOVERED
    assert door.methods()[-2:] == ["server/discover", "tools/list"]
    assert len(door.methods()) == 5


def test_concurrent_turns_share_one_discovery() -> None:
    """Turns that ask at once wait for the one discovery in progress."""
    door = FakeDoor()
    door.slow = {"server/discover": 0.2, "tools/list": 0.2}
    client = _client(door)
    barrier = threading.Barrier(8)

    def turn(_: int) -> Discovery | None:
        barrier.wait()
        return client.discover()

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(turn, range(8)))
    assert door.methods() == ["server/discover", "tools/list"]
    assert results == [DISCOVERED] * 8


def test_a_failed_discovery_is_tried_again_at_most_once_a_minute() -> None:
    """After a failure, turns within the next 60 s get no tool and send nothing."""
    door = FakeDoor()
    door.answers["server/discover"] = httpx.Response(503, text="unavailable")
    clock = FakeClock()
    client = _client(door, clock)
    assert client.discover() is None
    clock.advance(59)
    assert client.discover() is None
    assert door.methods() == ["server/discover"]
    clock.advance(2)
    door.answers["server/discover"] = _discover_result()
    assert client.discover() == DISCOVERED
    assert door.methods() == ["server/discover", "server/discover", "tools/list"]


def test_discovery_gives_up_after_five_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both requests together get DISCOVERY_TIMEOUT_S; a slow door offers no tool."""
    monkeypatch.setattr(wb, "DISCOVERY_TIMEOUT_S", 0.2)
    door = FakeDoor()
    door.slow = {"server/discover": 1.0}
    started = time.monotonic()
    assert _client(door).discover() is None
    assert time.monotonic() - started < 0.9


# --- requests ----------------------------------------------------------------


def test_every_request_states_the_modern_revision() -> None:
    """Each POST to /mcp carries the key, the content types, the revision, its method and the client's _meta."""
    door = FakeDoor()
    client = _client(door)
    client.discover()
    _run(client)
    assert len(door.requests) == 3
    for request, body in door.requests:
        assert (request.method, str(request.url)) == ("POST", "http://workbench.test/mcp")
        assert request.headers["authorization"] == f"Bearer {KEY}"
        assert request.headers["content-type"] == "application/json"
        assert request.headers["accept"] == "application/json, text/event-stream"
        assert request.headers["mcp-protocol-version"] == "2026-07-28"
        assert request.headers["mcp-method"] == body["method"]
        assert body["jsonrpc"] == "2.0"
        assert body["params"]["_meta"] == {
            META_PROTOCOL: "2026-07-28",
            META_CLIENT_INFO: {"name": "docint", "version": __version__},
            META_CLIENT_CAPS: {},
        }
    assert [("mcp-name" in r.headers) for r, _ in door.requests] == [False, False, True]
    assert door.requests[-1][0].headers["mcp-name"] == "run_python"
    assert "mcp-session-id" not in door.requests[-1][0].headers


def test_the_user_and_the_session_reach_workbench_as_keyed_pseudonyms() -> None:
    """Only tools/call names a user and a session, and only as HMACs under the shared key."""
    door = FakeDoor()
    client = _client(door)
    client.discover()
    _run(client, owner="user-a", session_id="session-1")
    call, _ = door.requests[-1]
    assert (
        call.headers["x-user-id"] == hmac.new(KEY.encode(), b"workbench-user/v1\x00user-a", hashlib.sha256).hexdigest()
    )
    assert (
        call.headers["x-session-id"]
        == hmac.new(KEY.encode(), b"workbench-session/v1\x00session-1", hashlib.sha256).hexdigest()
    )
    assert call.headers["x-user-id"] == user_pseudonym(KEY, "user-a")
    assert call.headers["x-session-id"] == session_pseudonym(KEY, "session-1")
    # Values, not str(headers): the name of httpx's own user-agent header starts with "user-a".
    assert not [value for value in call.headers.values() if "user-a" in value or "session-1" in value]
    for request, body in door.requests:
        assert "user-a" not in request.content.decode() and "session-1" not in request.content.decode()
        if body["method"] != "tools/call":
            assert not {"x-user-id", "x-session-id", "x-audit-purpose", "x-request-id"} & set(request.headers)


def test_a_new_key_gives_new_pseudonyms() -> None:
    """A rotated key gives every user a new pseudonym, each 64 hex digits."""
    old, new = user_pseudonym(KEY, "user-a"), user_pseudonym("j" * 32, "user-a")
    assert old != new
    for value in (old, new, session_pseudonym(KEY, "session-1")):
        assert len(value) == 64 and set(value) <= set("0123456789abcdef")
    assert user_pseudonym(KEY, "session-1") != session_pseudonym(KEY, "session-1")


def test_the_call_states_the_turns_purpose_else_docint_chat() -> None:
    """The bound purpose and request id go out on the call; unbound, the reason is docint chat."""
    door = FakeDoor()
    client = _client(door)
    with forwarding(case_ref="case-0001", reason="review", request_id="req-0001"):
        _run(client)
    stated, _ = door.requests[-1]
    assert parse_purpose_header(stated.headers["x-audit-purpose"]) == ("case-0001", "review")
    assert stated.headers["x-request-id"] == "req-0001"
    _run(client)
    unbound, _ = door.requests[-1]
    assert parse_purpose_header(unbound.headers["x-audit-purpose"]) == (None, "docint chat")
    assert "x-request-id" not in unbound.headers


def test_the_call_sends_its_code_data_and_time_limit() -> None:
    """tools/call names run_python and sends the code, the inline files and the time limit as given."""
    door = FakeDoor()
    _run(_client(door), time_limit_s=23)
    _, body = door.requests[-1]
    assert body["method"] == "tools/call"
    assert body["params"] == {
        "name": "run_python",
        "arguments": {"code": CODE, "data": [{"name": "passages.json", "content": "[]"}], "time_limit_s": 23},
        "_meta": {
            META_PROTOCOL: "2026-07-28",
            META_CLIENT_INFO: {"name": "docint", "version": __version__},
            META_CLIENT_CAPS: {},
        },
    }


def test_one_deadline_bounds_the_whole_call() -> None:
    """Past timeout_s the call is cancelled and answers unavailable, with nothing answered."""
    door = FakeDoor()
    door.slow = {"tools/call": 2.0}
    started = time.monotonic()
    result = _run(_client(door), timeout_s=0.3)
    assert time.monotonic() - started < 1.5
    assert result == RunFailure(kind="unavailable", token="timeout", job_id=None, answered=False)


def test_the_phase_timeouts_stay_as_guards() -> None:
    """Connect, write and pool keep their own short limits inside the call's deadline."""
    assert _client(FakeDoor())._timeout(40.0) == httpx.Timeout(40.0, connect=3.0, write=10.0, pool=3.0)


# --- answers -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("content_type", "body", "token"),
    [
        ("text/event-stream", b"data: {}\n\n", "not_json"),
        ("application/json", b"x" * (4 * 1024 * 1024 + 1), "answer_too_large"),
        ("application/json", b"not json", "not_json_rpc"),
        ("application/json", b'{"detail": "not found"}', "not_json_rpc"),
        ("application/json", b'{"jsonrpc": "2.0", "id": "other", "result": {}}', "not_json_rpc"),
    ],
    ids=["sse", "past_4_mib", "not_json", "not_json_rpc", "another_requests_answer"],
)
def test_an_answer_that_is_not_one_small_json_rpc_message_is_unavailable(
    content_type: str, body: bytes, token: str
) -> None:
    """SSE, an answer past 4 MiB, and anything but one JSON-RPC message are unavailable, unanswered."""
    door = FakeDoor()
    door.answers["tools/call"] = httpx.Response(200, headers={"content-type": content_type}, content=body)
    assert _run(_client(door)) == RunFailure(kind="unavailable", token=token, job_id=None, answered=False)


@pytest.mark.parametrize(
    ("token", "kind"),
    [
        ("too_large", "too_much_data"),
        ("queue_full", "no_capacity"),
        ("wait_timeout", "no_capacity"),
        ("executor_busy", "no_capacity"),
        ("not_completed", "code_failed"),
        ("refused", "unavailable"),
        ("purpose_required", "unavailable"),
        ("invalid_arguments", "unavailable"),
        ("something_new", "unavailable"),
    ],
)
def test_a_refused_run_is_mapped_by_its_token_never_its_text(token: str, kind: FailureKind) -> None:
    """Every refusal reads "busy, retry later"; only structuredContent.reason decides its kind."""
    door = FakeDoor()
    door.answers["tools/call"] = _refusal(token)
    assert _run(_client(door)) == RunFailure(kind=kind, token=token, job_id=None, answered=True)


def test_a_run_that_could_not_complete_keeps_its_job() -> None:
    """A refusal naming the job it ran keeps the job, for the record of what was handed over."""
    door = FakeDoor()
    door.answers["tools/call"] = _refusal("not_completed", job_id=JOB)
    assert _run(_client(door)) == RunFailure(kind="code_failed", token="not_completed", job_id=JOB, answered=True)


def test_a_json_rpc_error_is_unavailable() -> None:
    """A JSON-RPC error is unavailable, named by its code; the door did answer."""
    door = FakeDoor()
    door.answers["tools/call"] = RpcError(-32603, "internal error")
    assert _run(_client(door)) == RunFailure(kind="unavailable", token="jsonrpc:-32603", job_id=None, answered=True)


@pytest.mark.parametrize(
    "image",
    [
        "not base64!",
        base64.b64encode(b"GIF89a" + b"\x00" * 16).decode(),
        base64.b64encode(PNG_SIGNATURE + b"\x00" * MAX_IMAGE_BYTES).decode(),
    ],
    ids=["not_base64", "gif", "past_512_kib"],
)
def test_an_image_must_be_a_png_within_512_kib(image: str) -> None:
    """An image that is not base64, not a PNG, or past 512 KiB makes the answer unusable."""
    door = FakeDoor()
    answer = _call_result()
    answer["content"][1]["data"] = image
    door.answers["tools/call"] = answer
    assert _run(_client(door)) == RunFailure(kind="unavailable", token="bad_image", job_id=JOB, answered=True)


def test_a_png_of_exactly_512_kib_is_kept() -> None:
    """The image cap is inclusive."""
    door = FakeDoor()
    png = PNG_SIGNATURE + b"\x00" * (MAX_IMAGE_BYTES - len(PNG_SIGNATURE))
    answer = _call_result()
    answer["content"][1]["data"] = base64.b64encode(png).decode()
    door.answers["tools/call"] = answer
    result = _run(_client(door))
    assert isinstance(result, RunResult)
    assert result.images == (("chart.png", png),)


def test_a_successful_run_returns_its_outputs() -> None:
    """The job, the streams, the file list, the charts by name and the texts come back as sent."""
    door = FakeDoor()
    assert _run(_client(door)) == RunResult(
        job_id=JOB,
        stdout="2 authors",
        stderr="",
        truncated=False,
        files=(
            OutputFile(name="chart.png", mime="image/png", size=len(CHART), returned=True),
            OutputFile(name="counts.csv", mime="text/csv", size=len(CSV), returned=True),
        ),
        images=(("chart.png", CHART),),
        texts=(("counts.csv", "text/csv", CSV),),
        hint=None,
        server_version="0.2.0",
    )


def test_images_pair_with_the_returned_pngs_in_order() -> None:
    """The n-th image is the n-th returned .png in the file list; a count that disagrees is unusable."""
    second = PNG_SIGNATURE + b"second"
    answer = _call_result()
    answer["structuredContent"]["files"] = [
        {"name": "a.png", "mime": "image/png", "bytes": len(CHART), "returned": True},
        {"name": "dropped.png", "mime": "image/png", "bytes": 900_000, "returned": False},
        {"name": "notes.txt", "mime": "text/plain", "bytes": 5, "returned": True},
        {"name": "b.png", "mime": "image/png", "bytes": len(second), "returned": True},
    ]
    image = copy.deepcopy(answer["content"][1])
    image["data"] = base64.b64encode(second).decode()
    answer["content"].insert(2, image)
    door = FakeDoor()
    door.answers["tools/call"] = answer
    result = _run(_client(door))
    assert isinstance(result, RunResult)
    assert result.images == (("a.png", CHART), ("b.png", second))

    del answer["content"][2]
    assert _run(_client(door)) == RunFailure(kind="unavailable", token="bad_result", job_id=JOB, answered=True)


def test_a_resource_is_named_by_the_last_segment_of_its_uri() -> None:
    """A percent-encoded file name comes back decoded."""
    answer = _call_result()
    answer["content"][2]["resource"]["uri"] = f"workbench://jobs/{JOB}/out/posts%20per%20author.csv"
    door = FakeDoor()
    door.answers["tools/call"] = answer
    result = _run(_client(door))
    assert isinstance(result, RunResult)
    assert result.texts == (("posts per author.csv", "text/csv", CSV),)


def test_the_version_falls_back_to_the_discovery() -> None:
    """A result without serverInfo takes the version discovery read."""
    answer = _call_result()
    del answer["_meta"]
    door = FakeDoor()
    door.answers["server/discover"] = {**_discover_result(), "_meta": {META_SERVER_INFO: {"version": "0.2.1"}}}
    door.answers["tools/call"] = answer
    client = _client(door)
    client.discover()
    result = _run(client)
    assert isinstance(result, RunResult)
    assert result.server_version == "0.2.1"


def test_an_unreachable_call_or_a_refused_argument_drops_the_discovery() -> None:
    """No answer, a timeout, or invalid_arguments: the next turn discovers again. A busy door keeps it."""
    door = FakeDoor()
    client = _client(door)
    assert client.discover() == DISCOVERED

    door.answers["tools/call"] = httpx.ConnectError("connection refused")
    assert _run(client) == RunFailure(kind="unavailable", token="unreachable", job_id=None, answered=False)
    assert client.discover() == DISCOVERED
    assert door.methods()[-2:] == ["server/discover", "tools/list"]

    door.answers["tools/call"] = _refusal("invalid_arguments")
    assert _run(client) == RunFailure(kind="unavailable", token="invalid_arguments", answered=True)
    assert client.discover() == DISCOVERED
    assert door.methods()[-2:] == ["server/discover", "tools/list"]

    door.answers["tools/call"] = _call_result()
    door.slow = {"tools/call": 1.0}
    assert _run(client, timeout_s=0.1) == RunFailure(kind="unavailable", token="timeout", answered=False)
    assert client.discover() == DISCOVERED
    assert door.methods()[-2:] == ["server/discover", "tools/list"]

    door.slow = {}
    door.answers["tools/call"] = _refusal("queue_full")
    assert _run(client) == RunFailure(kind="no_capacity", token="queue_full", answered=True)
    assert client.discover() == DISCOVERED
    assert door.methods()[-1] == "tools/call"
    assert door.methods().count("server/discover") == 4


# --- logging and hygiene -----------------------------------------------------


def test_one_warning_per_change_of_state(loguru_caplog_info: LogCaptureFixture) -> None:
    """Repeated failures warn once; recovery is one INFO line; the next failure warns again."""
    door = FakeDoor()
    door.answers["server/discover"] = httpx.Response(503, text="unavailable")
    clock = FakeClock()
    client = _client(door, clock)
    for _ in range(3):
        assert client.discover() is None
        clock.advance(61)
    assert len(door.requests) == 3
    assert _levels(loguru_caplog_info) == ["WARNING"]

    door.answers["server/discover"] = _discover_result()
    assert client.discover() == DISCOVERED
    clock.advance(4000)
    assert client.discover() == DISCOVERED
    assert _levels(loguru_caplog_info) == ["WARNING", "INFO"]

    door.answers["tools/list"] = httpx.Response(503, text="unavailable")
    clock.advance(4000)
    assert client.discover() is None
    assert _levels(loguru_caplog_info) == ["WARNING", "INFO", "WARNING"]


def test_the_log_holds_no_code_data_or_output(log_lines: list[str]) -> None:
    """Nothing the user's evidence, the model's code or workbench's output holds reaches a log line."""
    secret = "Hafenfest"
    code, data = f"print('{secret}')", (InlineFile(name="passages.json", content=json.dumps([secret])),)
    success = _call_result()
    success["structuredContent"].update(stdout=f"{secret} 12", stderr=secret, hint=secret)
    success["content"][2]["resource"]["text"] = secret
    bad_image = _call_result()
    bad_image["content"][1]["data"] = secret
    answers: list[object] = [
        success,
        _refusal("not_completed"),
        {**_refusal(secret), "content": [{"type": "text", "text": secret}]},
        {**_refusal(f"{secret.lower()}, retry later"), "content": [{"type": "text", "text": secret}]},
        RpcError(-32602, secret, status=400),
        httpx.Response(502, text=secret),
        httpx.Response(200, json={"detail": secret}),
        bad_image,
        httpx.ConnectError(secret),
    ]
    door = FakeDoor()
    door.answers["server/discover"] = httpx.Response(503, text=secret)
    client = _client(door)
    client.discover()
    for answer in answers:
        door.answers["tools/call"] = answer
        _run(client, code=code, data=data)
    assert len(door.requests) == 1 + len(answers)
    assert log_lines
    assert not [line for line in log_lines if secret in line or secret.lower() in line]


class _SpyClient(httpx.AsyncClient):
    """An ``AsyncClient`` that remembers every instance built."""

    built: ClassVar[list[httpx.AsyncClient]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Build the client and remember it.

        Args:
            *args (Any): Passed through.
            **kwargs (Any): Passed through.
        """
        super().__init__(*args, **kwargs)
        _SpyClient.built.append(self)


def test_the_client_takes_no_proxy_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A proxy set in the environment never carries the inline data: the door is reached directly."""
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.setattr(_SpyClient, "built", [])
    monkeypatch.setattr(httpx, "AsyncClient", _SpyClient)
    door = FakeDoor()
    client = _client(door)
    assert client.discover() == DISCOVERED
    assert isinstance(_run(client), RunResult)
    assert len(_SpyClient.built) == 2
    assert all(built.trust_env is False for built in _SpyClient.built)


@pytest.mark.usefixtures("fresh_process_client")
def test_without_a_key_there_is_no_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """The process-wide client exists only while a key and a base are set, and is built once."""
    monkeypatch.setenv("WORKBENCH_API_BASE", "http://workbench.test")
    monkeypatch.delenv("WORKBENCH_API_KEY", raising=False)
    assert wb.workbench_client() is None

    wb.workbench_client.cache_clear()
    monkeypatch.setenv("WORKBENCH_API_KEY", KEY)
    client = wb.workbench_client()
    assert isinstance(client, WorkbenchClient)
    assert wb.workbench_client() is client
