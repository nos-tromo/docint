"""Tests for the topic-model fit client."""

import json
import re
from collections.abc import Callable
from typing import Any

import httpx
import numpy as np
import pytest
from _pytest.logging import LogCaptureFixture

from docint.utils import topics_client
from docint.utils.env_cfg import TopicsClientConfig
from docint.utils.topics_client import (
    FitParams,
    TopicsBadResponse,
    TopicsBusy,
    TopicsClient,
    TopicsEndpointError,
    TopicsRejected,
    TopicsTooLarge,
    TopicsUnreachable,
    encode_texts,
    max_fit_points,
    texts_limit,
)

PARAMS = FitParams(seed=7, min_topic_size=10, nr_topics=0, top_n_words=10, min_df=2, stop_words=("the", "und"))
TEXTS = ["Die Größe", "zwei", "drei"]


def _vectors(n: int = 3, dim: int = 4) -> np.ndarray:
    """Build a small deterministic float64 matrix.

    Args:
        n: Rows.
        dim: Columns.

    Returns:
        The matrix.
    """
    return np.arange(n * dim, dtype="float64").reshape(n, dim) / 10.0


def _ok_body(**overrides: Any) -> dict[str, Any]:
    """Build a valid response body for ``TEXTS``.

    Args:
        **overrides: Fields to replace.

    Returns:
        The body.
    """
    body: dict[str, Any] = {
        "labels": [0, 0, -1],
        "outliers": 1,
        "topics": [{"id": 0, "size": 2, "keywords": [["alpha", 0.5], ["beta", 0.25]]}],
        "versions": {"bertopic": "1"},
        "runtime": {"device": "cpu"},
        "timings": {"fit_s": 1.5},
    }
    body.update(overrides)
    return body


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    base: str = "http://topics-only:8000",
    key: str | None = None,
) -> TopicsClient:
    """Build a client over a mock transport.

    Args:
        handler: The request handler.
        base: The API base.
        key: The Bearer key.

    Returns:
        The client.
    """
    return TopicsClient(TopicsClientConfig(base, key, 5.0), transport=httpx.MockTransport(handler))


def _parts(request: httpx.Request) -> dict[str, tuple[str, bytes]]:
    """Split a multipart request into ``{name: (filename, body)}``.

    Args:
        request: The captured request.

    Returns:
        The parts by field name.
    """
    boundary = re.search(rb"boundary=([^;]+)", request.headers["content-type"].encode()).group(1)  # type: ignore[union-attr]
    parts: dict[str, tuple[str, bytes]] = {}
    for chunk in request.content.split(b"--" + boundary)[1:-1]:
        head, _, body = chunk.partition(b"\r\n\r\n")
        match = re.search(rb'name="([^"]+)"; filename="([^"]+)"', head)
        assert match, "every part must be a file part"
        parts[match.group(1).decode()] = (match.group(2).decode(), body[:-2])
    return parts


def test_fit_posts_three_file_parts_in_the_contract_shape() -> None:
    """The request carries meta, texts and vectors as file parts in the frozen shape.

    Returns:
        None.
    """
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_ok_body())

    vectors = _vectors()
    _client(handler).fit(TEXTS, vectors, PARAMS)
    (request,) = seen
    assert request.url.path == "/topics/fit"
    parts = _parts(request)
    assert {name: filename for name, (filename, _) in parts.items()} == {
        "meta": "meta.json",
        "texts": "texts.json",
        "vectors": "vectors.f32",
    }
    assert json.loads(parts["meta"][1]) == {
        "n": 3,
        "dim": 4,
        "seed": 7,
        "min_topic_size": 10,
        "nr_topics": 0,
        "top_n_words": 10,
        "min_df": 2,
        "stop_words": ["the", "und"],
    }
    assert "größe".encode() in parts["texts"][1].lower()
    assert json.loads(parts["texts"][1]) == TEXTS
    assert parts["vectors"][1] == vectors.astype("<f4").tobytes()


@pytest.mark.parametrize("key", [None, "secret"])
def test_fit_sends_a_bearer_header_only_when_a_key_is_set(key: str | None) -> None:
    """The Authorization header exists only with a key.

    Args:
        key: The configured key.

    Returns:
        None.
    """
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_ok_body())

    _client(handler, key=key).fit(TEXTS, _vectors(), PARAMS)
    assert seen[0].headers.get("authorization") == (f"Bearer {key}" if key else None)


def test_fit_parses_a_valid_response() -> None:
    """A valid body becomes tuples and copied dictionaries.

    Returns:
        None.
    """
    result = _client(lambda request: httpx.Response(200, json=_ok_body())).fit(TEXTS, _vectors(), PARAMS)
    assert result.labels == (0, 0, -1)
    assert result.outliers == 1
    assert result.topics[0].id == 0
    assert result.topics[0].size == 2
    assert result.topics[0].keywords == (("alpha", 0.5), ("beta", 0.25))
    assert result.versions == {"bertopic": "1"}
    assert result.runtime == {"device": "cpu"}
    assert result.timings == {"fit_s": 1.5}


@pytest.mark.parametrize(("header", "expected"), [("12", 12.0), (None, 30.0), ("soon", 30.0), ("-4", 30.0)])
def test_a_busy_service_raises_topics_busy_with_its_retry_after(header: str | None, expected: float) -> None:
    """A 503 is busy, carrying the advised wait or the default.

    Args:
        header: The Retry-After value.
        expected: The wait the error carries.

    Returns:
        None.
    """
    headers = {"Retry-After": header} if header is not None else {}
    client = _client(lambda request: httpx.Response(503, headers=headers, json={"detail": "busy"}))
    with pytest.raises(TopicsBusy) as info:
        client.fit(TEXTS, _vectors(), PARAMS)
    assert info.value.retry_after == expected
    assert info.value.code == "topics_busy"


def test_an_unreachable_endpoint_names_the_url_and_the_knob() -> None:
    """A connection failure names the credential-free URL and TOPICS_API_BASE.

    Returns:
        None.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    client = _client(handler, base="http://u:p@h:1")
    with pytest.raises(TopicsUnreachable) as info:
        client.fit(TEXTS, _vectors(), PARAMS)
    assert client.fit_url in str(info.value)
    assert client.fit_url == "http://h:1/topics/fit"
    assert "TOPICS_API_BASE" in str(info.value)
    assert "u:p" not in str(info.value)


@pytest.mark.parametrize(
    ("detail", "needle"),
    [
        ("texts holds 3 items", "texts holds 3 items"),
        ([{"loc": ["body", "meta"], "msg": "field required"}], "body.meta: field required"),
    ],
)
def test_a_422_detail_is_reported_in_both_forms(detail: Any, needle: str) -> None:
    """A 422 detail is shown whether it is a string or FastAPI's list.

    Args:
        detail: The detail as sent.
        needle: Text the message must contain.

    Returns:
        None.
    """
    client = _client(lambda request: httpx.Response(422, json={"detail": detail}))
    with pytest.raises(TopicsRejected) as info:
        client.fit(TEXTS, _vectors(), PARAMS)
    assert needle in str(info.value)
    assert info.value.status == 422


def test_a_422_detail_is_cut_to_300_characters() -> None:
    """A very long detail is truncated.

    Returns:
        None.
    """
    client = _client(lambda request: httpx.Response(422, json={"detail": "x" * 5000}))
    with pytest.raises(TopicsRejected) as info:
        client.fit(TEXTS, _vectors(), PARAMS)
    assert "x" * 300 in str(info.value)
    assert "x" * 301 not in str(info.value)


def test_a_413_raises_too_large_naming_topics_fit_max_points() -> None:
    """A 413 is too-large and points at TOPICS_FIT_MAX_POINTS.

    Returns:
        None.
    """
    client = _client(lambda request: httpx.Response(413, json={"detail": "n too big"}))
    with pytest.raises(TopicsTooLarge) as info:
        client.fit(TEXTS, _vectors(), PARAMS)
    assert "TOPICS_FIT_MAX_POINTS" in str(info.value)
    assert info.value.code == "topics_too_large"
    assert isinstance(info.value, TopicsRejected)


@pytest.mark.parametrize(
    ("status", "knob"), [(401, "TOPICS_API_KEY"), (403, "TOPICS_API_KEY"), (404, "TOPICS_API_BASE")]
)
def test_auth_and_routing_statuses_name_their_setting(status: int, knob: str) -> None:
    """401, 403 and 404 each name the setting to change.

    Args:
        status: The HTTP status.
        knob: The variable the message must name.

    Returns:
        None.
    """
    client = _client(lambda request: httpx.Response(status))
    with pytest.raises(TopicsRejected) as info:
        client.fit(TEXTS, _vectors(), PARAMS)
    assert knob in str(info.value)
    assert info.value.status == status


def test_a_server_error_is_a_plain_endpoint_error() -> None:
    """A 500 raises the base error with the default code.

    Returns:
        None.
    """
    client = _client(lambda request: httpx.Response(500, json={"detail": "topic-model fit failed"}))
    with pytest.raises(TopicsEndpointError) as info:
        client.fit(TEXTS, _vectors(), PARAMS)
    assert info.value.code == "topics_failed"


def test_upload_timeout_names_the_network_and_timeout_settings() -> None:
    """An HTTP upload deadline is an unreachable service, with the same remedy as a transport timeout."""
    client = _client(lambda request: httpx.Response(408))
    with pytest.raises(TopicsUnreachable) as caught:
        client.fit(TEXTS, _vectors(), PARAMS)
    assert "TOPICS_API_BASE" in str(caught.value)
    assert "TOPICS_TIMEOUT" in str(caught.value)


def test_router_auth_failure_names_the_key_without_echoing_it() -> None:
    """The router's no-db auth error names configuration settings without echoing credentials."""
    body = {"error": {"message": "No connected db. Invalid token synthetic-secret", "type": "no_db_connection"}}
    client = _client(lambda request: httpx.Response(400, json=body), key="synthetic-secret")
    with pytest.raises(TopicsRejected) as caught:
        client.fit(TEXTS, _vectors(), PARAMS)
    assert "TOPICS_API_KEY" in str(caught.value)
    assert "TOPICS_API_BASE" in str(caught.value)
    assert "synthetic-secret" not in str(caught.value)


@pytest.mark.parametrize("prefix", ["bad sample ", "x" * 295])
def test_router_message_is_bounded_and_redacts_the_configured_key(prefix: str) -> None:
    """Router error.message details are useful diagnostics but never echo the configured key.

    Args:
        prefix: Text before the key, possibly placing it across the truncation boundary.
    """
    body = {"error": {"message": prefix + "synthetic-secret " + "x" * 500}}
    client = _client(lambda request: httpx.Response(422, json=body), key="synthetic-secret")
    with pytest.raises(TopicsRejected) as caught:
        client.fit(TEXTS, _vectors(), PARAMS)
    assert prefix in str(caught.value)
    assert "synthetic" not in str(caught.value)
    assert len(str(caught.value)) < 450


@pytest.mark.parametrize("base", ["topics-only:8000", "http://topics-only:bad", "http://", "ftp://topics-only"])
def test_invalid_base_is_a_configuration_error(base: str) -> None:
    """A malformed endpoint is a coded configuration fault before any request.

    Args:
        base: The malformed base URL.
    """
    with pytest.raises(TopicsEndpointError, match="TOPICS_API_BASE"):
        _client(lambda request: pytest.fail("request sent"), base=base).fit(TEXTS, _vectors(), PARAMS)


@pytest.mark.parametrize("error", [httpx.DecodingError, httpx.TooManyRedirects])
def test_other_http_errors_are_coded_endpoint_errors(error: type[httpx.HTTPError]) -> None:
    """HTTP-layer failures cannot escape as uncoded job errors.

    Args:
        error: HTTP exception to raise from the mock transport.
    """

    def failing(request: httpx.Request) -> httpx.Response:
        """Raise the scripted HTTP failure.

        Args:
            request: The attempted request.

        Raises:
            httpx.HTTPError: The scripted failure.
        """
        raise error("synthetic failure")

    with pytest.raises(TopicsEndpointError):
        _client(failing).fit(TEXTS, _vectors(), PARAMS)


@pytest.mark.parametrize(
    "body",
    [
        _ok_body(labels=[0, 0]),
        _ok_body(labels=[0, 3, -1]),
        _ok_body(topics=[{"id": 0, "size": 3, "keywords": []}]),
        _ok_body(outliers=2),
        _ok_body(topics=[{"id": 0, "size": 2, "keywords": [["alpha", 1e999]]}]),
        _ok_body(
            labels=[0, 1, -1],
            topics=[{"id": 0, "size": 1, "keywords": []}, {"id": 0, "size": 1, "keywords": []}],
        ),
        {"labels": "nope"},
    ],
    ids=["label-count", "unlisted-topic", "size", "outliers", "non-finite-score", "duplicate-ids", "shape"],
)
def test_a_response_that_breaks_the_contract_is_rejected(body: dict[str, Any]) -> None:
    """Each inconsistent body raises TopicsBadResponse.

    Args:
        body: The response body.

    Returns:
        None.
    """
    client = _client(lambda request: httpx.Response(200, content=json.dumps(body).replace("Infinity", "1e999")))
    with pytest.raises(TopicsBadResponse):
        client.fit(TEXTS, _vectors(), PARAMS)


def _with(**changes: Any) -> FitParams:
    """Return ``PARAMS`` with some fields replaced.

    Args:
        **changes: Fields to replace.

    Returns:
        The new params.
    """
    return FitParams(**{**PARAMS.__dict__, **changes})


@pytest.mark.parametrize(
    "case",
    [
        "nr_topics",
        "min_df",
        "min_topic_size",
        "top_n_words",
        "seed",
        "long_stop_word",
        "many_stop_words",
        "dim",
        "nan",
        "length_mismatch",
        "texts_limit",
    ],
)
def test_invalid_params_are_refused_before_any_request(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Out-of-bounds inputs raise ValueError and send nothing.

    Args:
        case: Which input is invalid.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_ok_body())

    params = PARAMS
    vectors = _vectors()
    texts = list(TEXTS)
    if case == "nr_topics":
        params = _with(nr_topics=1)
    elif case == "min_df":
        params = _with(min_df=0)
    elif case == "min_topic_size":
        params = _with(min_topic_size=1)
    elif case == "top_n_words":
        params = _with(top_n_words=31)
    elif case == "seed":
        params = _with(seed=-1)
    elif case == "long_stop_word":
        params = _with(stop_words=("x" * 65,))
    elif case == "many_stop_words":
        params = _with(stop_words=("w",) * 10_001)
    elif case == "dim":
        vectors = np.zeros((3, 8193), dtype="float32")
    elif case == "nan":
        vectors[1, 1] = np.nan
    elif case == "length_mismatch":
        texts = texts[:2]
    else:
        monkeypatch.setattr(topics_client, "texts_limit", lambda n: 4)
    with pytest.raises(ValueError):
        _client(handler).fit(texts, vectors, params)
    assert seen == []


@pytest.mark.parametrize(("dim", "expected"), [(1024, 150000), (4096, 37500), (8192, 18750), (384, 150000)])
def test_max_fit_points_caps_by_dimension(dim: int, expected: int) -> None:
    """Wider vectors mean fewer points.

    Args:
        dim: Vector dimension.
        expected: The point cap.

    Returns:
        None.
    """
    assert max_fit_points(dim) == expected


def test_texts_limit_and_encoding() -> None:
    """The limit follows the service formula and the encoding keeps non-ASCII raw.

    Returns:
        None.
    """
    assert texts_limit(1) == 16384 + 65536
    assert texts_limit(10**9) == 536870912
    assert encode_texts(["ö", "a"]) == '["ö","a"]'.encode()


def test_a_fit_logs_counts_only(loguru_caplog_info: LogCaptureFixture) -> None:
    """One INFO line carries counts, with no text and no keyword.

    Args:
        loguru_caplog_info: Captured loguru INFO records.

    Returns:
        None.
    """
    _client(lambda request: httpx.Response(200, json=_ok_body())).fit(TEXTS, _vectors(), PARAMS)
    lines = [line for line in loguru_caplog_info.messages if "Topics fit" in line]
    assert lines == ["Topics fit | n=3 dim=4 topics=1 outliers=1 fit_s=1.50"]
