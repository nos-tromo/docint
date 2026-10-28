"""HTTP client for the remote topic-model fit service.

The full vllm-service stack exposes ``POST /topics/fit`` as a router
pass-through; the topics-only shape serves it directly. The request is three
file parts (``meta.json``, ``texts.json``, ``vectors.f32``) and the response
assigns each input a topic id (``-1`` for an outlier) and lists each topic's
size and keywords. Nothing is embedded or stored on the service side.

Every failure is a :class:`TopicsEndpointError` whose message names the
endpoint (without any credentials in its URL) and the setting to change, so a
misconfiguration is not mistaken for a model fault.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar
from urllib.parse import urlsplit

import httpx
import numpy as np
from loguru import logger
from pydantic import BaseModel, ValidationError

from docint.utils.env_cfg import TopicsClientConfig, load_topics_client_env

FIT_PATH = "/topics/fit"
SERVICE_MAX_POINTS = 150_000
#: Fewest points the service fits; it also needs twice the minimum topic size.
SERVICE_MIN_POINTS = 8
VALUES_PER_POINT = 1024
MAX_DIM = 8192
MAX_STOP_WORDS = 10_000
MAX_STOP_WORD_CHARS = 64
META_MAX_BYTES = 1_048_576
DEFAULT_RETRY_AFTER_S = 30.0

_TEXTS_BYTES_PER_POINT = 16_384
_TEXTS_BYTES_SLACK = 65_536
_TEXTS_MAX_BYTES = 536_870_912
_DETAIL_MAX_CHARS = 300
_UNREACHABLE_REMEDY = "check TOPICS_API_BASE (and TOPICS_TIMEOUT for a timeout)"
#: Most seconds a fit waits to connect. ``TOPICS_TIMEOUT`` bounds the whole
#: fit, which can take an hour; an address that drops packets would otherwise
#: hold every ingest's topics stage for as long before it fails.
CONNECT_TIMEOUT_S = 30.0


def texts_limit(n: int) -> int:
    """Return the most bytes the service accepts for ``n`` texts.

    Args:
        n: Number of texts.

    Returns:
        The byte limit of the ``texts`` part.
    """
    return min(n * _TEXTS_BYTES_PER_POINT + _TEXTS_BYTES_SLACK, _TEXTS_MAX_BYTES)


def max_fit_points(dim: int) -> int:
    """Return the most points a default-configured service fits at ``dim`` dimensions.

    Args:
        dim: Vector dimension.

    Returns:
        The point cap: wider vectors mean fewer points.
    """
    return min(SERVICE_MAX_POINTS, SERVICE_MAX_POINTS * VALUES_PER_POINT // dim)


def encode_texts(texts: Sequence[str]) -> bytes:
    """Encode texts as the UTF-8 JSON array the ``texts`` part carries.

    Args:
        texts: The texts, one per vector.

    Returns:
        The JSON bytes, with non-ASCII characters left unescaped.
    """
    return json.dumps(list(texts), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class FitParams:
    """The settings of one fit.

    Attributes:
        seed: Random seed, 0 <= seed < 2**32.
        min_topic_size: Smallest topic, at least 2.
        nr_topics: Reduction target (counts the outlier class), 0 or at least 2.
        top_n_words: Keywords kept per topic, 1 to 30.
        min_df: Fewest texts a word must occur in to be a keyword, at least 1.
        stop_words: Words never used as keywords.
    """

    seed: int
    min_topic_size: int
    nr_topics: int
    top_n_words: int
    min_df: int
    stop_words: tuple[str, ...]


@dataclass(frozen=True)
class FitTopic:
    """One fitted topic.

    Attributes:
        id: The topic id, 0 or more.
        size: Number of inputs labelled with it.
        keywords: ``(word, score)`` pairs, best first.
    """

    id: int
    size: int
    keywords: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class FitResult:
    """A validated fit response.

    Attributes:
        labels: Topic id per input, ``-1`` for an outlier.
        topics: The topics, in the order the service sent them.
        outliers: Number of ``-1`` labels.
        versions: Library versions the service reported.
        runtime: Runtime details the service reported.
        timings: Timings in seconds the service reported.
    """

    labels: tuple[int, ...]
    topics: tuple[FitTopic, ...]
    outliers: int
    versions: dict[str, str]
    runtime: dict[str, Any]
    timings: dict[str, float]


class TopicsEndpointError(RuntimeError):
    """The fit service could not produce a result; the message names the endpoint and the knob."""

    code: ClassVar[str] = "topics_failed"


class TopicsUnreachable(TopicsEndpointError):
    """The service could not be reached, or did not answer in time."""

    code = "topics_unreachable"


class TopicsBusy(TopicsEndpointError):
    """The service is running another fit.

    Attributes:
        retry_after: Seconds the service asked to wait.
    """

    code = "topics_busy"

    def __init__(self, message: str, retry_after: float = DEFAULT_RETRY_AFTER_S) -> None:
        """Store the message and the advised wait.

        Args:
            message: The human-readable message.
            retry_after: Seconds to wait before trying again.
        """
        super().__init__(message)
        self.retry_after = retry_after


class TopicsRejected(TopicsEndpointError):
    """The service refused the request.

    Attributes:
        status: The HTTP status.
    """

    code = "topics_rejected"

    def __init__(self, message: str, status: int = 0) -> None:
        """Store the message and the status.

        Args:
            message: The human-readable message.
            status: The HTTP status that carried the refusal.
        """
        super().__init__(message)
        self.status = status


class TopicsTooLarge(TopicsRejected):
    """The request exceeded the service's size limits."""

    code = "topics_too_large"


class TopicsBadResponse(TopicsEndpointError):
    """The service answered 200 with a body that breaks the contract."""

    code = "topics_bad_response"


class _TopicModel(BaseModel):
    """Wire form of one topic."""

    id: int
    size: int
    keywords: list[tuple[str, float]]


class _FitModel(BaseModel):
    """Wire form of the response."""

    labels: list[int]
    outliers: int
    topics: list[_TopicModel]
    versions: dict[str, str]
    runtime: dict[str, Any]
    timings: dict[str, float]


def _validate_params(params: FitParams) -> None:
    """Refuse settings the service would reject, before anything is sent.

    Args:
        params: The fit settings.

    Raises:
        ValueError: When a setting is outside the service's bounds.
    """
    problems: list[str] = []
    if not 0 <= params.seed < 2**32:
        problems.append("seed must be in [0, 2**32)")
    if params.min_topic_size < 2:
        problems.append("min_topic_size must be at least 2")
    if params.nr_topics == 1 or params.nr_topics < 0:
        problems.append("nr_topics must be 0 or at least 2")
    if not 1 <= params.top_n_words <= 30:
        problems.append("top_n_words must be 1 to 30")
    if params.min_df < 1:
        problems.append("min_df must be at least 1")
    if len(params.stop_words) > MAX_STOP_WORDS:
        problems.append(f"at most {MAX_STOP_WORDS} stop words")
    if any(len(word) > MAX_STOP_WORD_CHARS for word in params.stop_words):
        problems.append(f"a stop word is over {MAX_STOP_WORD_CHARS} characters")
    if problems:
        raise ValueError("invalid topic fit settings: " + "; ".join(problems))


def _detail_of(response: httpx.Response, *, api_key: str | None = None) -> str:
    """Render a FastAPI detail or router error message, cut short.

    Args:
        response: The error response.
        api_key: Configured key to redact before the detail is truncated.

    Returns:
        The detail text, or an empty string when there is none.
    """
    try:
        body = response.json()
        detail = body.get("detail")
        if detail is None and isinstance(body.get("error"), dict):
            detail = body["error"].get("message")
    except (ValueError, AttributeError):
        return ""
    if isinstance(detail, list):
        parts = []
        for item in detail:
            if isinstance(item, dict):
                loc = ".".join(str(p) for p in item.get("loc", ()))
                parts.append(f"{loc}: {item.get('msg', '')}".strip(": "))
            else:
                parts.append(str(item))
        detail = "; ".join(parts)
    text = str(detail or "")
    if api_key:
        text = text.replace(api_key, "[redacted]")
    return text[:_DETAIL_MAX_CHARS]


def _retry_after_of(response: httpx.Response) -> float:
    """Read ``Retry-After`` as seconds, falling back to the default.

    Args:
        response: The 503 response.

    Returns:
        The advised wait.
    """
    try:
        value = float(response.headers.get("Retry-After", ""))
    except ValueError:
        return DEFAULT_RETRY_AFTER_S
    return value if math.isfinite(value) and value >= 0 else DEFAULT_RETRY_AFTER_S


class TopicsClient:
    """Client for ``POST /topics/fit``."""

    def __init__(
        self, config: TopicsClientConfig | None = None, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        """Store the endpoint settings.

        Args:
            config: Endpoint settings; read from the environment when omitted.
            transport: Optional transport, for tests.
        """
        self._config = config or load_topics_client_env()
        self._transport = transport

    @property
    def fit_url(self) -> str:
        """The fit URL: the base plus :data:`FIT_PATH`, with any userinfo removed."""
        try:
            parts = urlsplit(self._config.api_base)
            host = parts.hostname or ""
            port = parts.port
            if parts.scheme not in {"http", "https"} or not host:
                raise ValueError("not an HTTP endpoint")
        except ValueError:
            raise TopicsEndpointError("Invalid TOPICS_API_BASE; set an HTTP or HTTPS service base URL") from None
        if ":" in host:
            host = f"[{host}]"
        if port:
            host = f"{host}:{port}"
        return f"{parts.scheme}://{host}{parts.path.rstrip('/')}{FIT_PATH}"

    def _error_for(self, response: httpx.Response) -> TopicsEndpointError:
        """Map a non-200 response to the error that names its remedy.

        Args:
            response: The error response.

        Returns:
            The exception to raise.
        """
        status = response.status_code
        where = f"{self.fit_url} answered {status}"
        detail = _detail_of(response, api_key=self._config.api_key)
        suffix = f": {detail}" if detail else ""
        if status == 408:
            return TopicsUnreachable(f"{where}: the upload timed out; {_UNREACHABLE_REMEDY}")
        if status == 503:
            retry_after = _retry_after_of(response)
            return TopicsBusy(f"{where}: another fit is running; retry in {retry_after:g}s", retry_after)
        if status == 413:
            remedy = "lower TOPICS_FIT_MAX_POINTS, or raise the service's TOPICS_MAX_POINTS"
            return TopicsTooLarge(f"{where}{suffix}; {remedy}", status)
        if status in (401, 403):
            remedy = "set TOPICS_API_KEY: the router requires it, the topics-only shape does not"
            return TopicsRejected(f"{where}; {remedy}", status)
        if status == 404:
            remedy = "TOPICS_API_BASE must name the router or the topics-only service, with no path"
            return TopicsRejected(f"{where}; {remedy}", status)
        if status == 400:
            try:
                error = response.json().get("error")
            except (ValueError, AttributeError):
                error = None
            if isinstance(error, dict) and error.get("type") == "no_db_connection":
                remedy = "check TOPICS_API_KEY and TOPICS_API_BASE: the router could not authenticate the key"
                return TopicsRejected(f"{where}; {remedy}", status)
        if status in (400, 422):
            return TopicsRejected(f"{where}{suffix}", status)
        return TopicsEndpointError(f"{where}{suffix}")

    def _parse(self, response: httpx.Response, n: int) -> FitResult:
        """Validate a 200 response against the contract.

        Args:
            response: The response.
            n: Number of inputs sent.

        Returns:
            The validated result.

        Raises:
            TopicsBadResponse: When the body is malformed or inconsistent.
        """
        where = f"{self.fit_url} answered 200 with a body that breaks the contract"
        try:
            body = _FitModel.model_validate_json(response.content)
        except ValidationError:
            raise TopicsBadResponse(f"{where}: not the expected shape") from None
        ids = [topic.id for topic in body.topics]
        if len(body.labels) != n:
            raise TopicsBadResponse(f"{where}: {len(body.labels)} labels for {n} inputs")
        if len(set(ids)) != len(ids) or any(i < 0 for i in ids):
            raise TopicsBadResponse(f"{where}: topic ids are not unique and non-negative")
        counts: dict[int, int] = {}
        for label in body.labels:
            if label != -1 and label not in ids:
                raise TopicsBadResponse(f"{where}: a label names an unlisted topic")
            counts[label] = counts.get(label, 0) + 1
        if any(topic.size != counts.get(topic.id, 0) for topic in body.topics):
            raise TopicsBadResponse(f"{where}: a topic size disagrees with the labels")
        if body.outliers != counts.get(-1, 0):
            raise TopicsBadResponse(f"{where}: outliers disagree with the labels")
        if any(not math.isfinite(score) for topic in body.topics for _, score in topic.keywords):
            raise TopicsBadResponse(f"{where}: a keyword score is not finite")
        return FitResult(
            labels=tuple(body.labels),
            topics=tuple(FitTopic(t.id, t.size, tuple((w, s) for w, s in t.keywords)) for t in body.topics),
            outliers=body.outliers,
            versions=dict(body.versions),
            runtime=dict(body.runtime),
            timings=dict(body.timings),
        )

    def fit(self, texts: Sequence[str], vectors: np.ndarray, params: FitParams) -> FitResult:
        """Fit a topic model to ``vectors`` and return each input's topic.

        Args:
            texts: One text per vector, used for the keywords.
            vectors: ``n x dim`` array of finite floats.
            params: The fit settings.

        Returns:
            The validated result.

        Raises:
            ValueError: When the inputs or settings are outside the service's bounds;
                nothing is sent.
            TopicsUnreachable: When the service cannot be reached or times out.
            TopicsBusy: When the service is running another fit.
            TopicsTooLarge: When the service refuses the request's size.
            TopicsRejected: When the service refuses the request otherwise.
            TopicsBadResponse: When a 200 body breaks the contract.
            TopicsEndpointError: On any other service failure.
        """
        _validate_params(params)
        matrix = np.ascontiguousarray(vectors, dtype="<f4")
        if matrix.ndim != 2 or matrix.shape[0] < 1:
            raise ValueError("vectors must be a non-empty 2-D array")
        n, dim = matrix.shape
        if n != len(texts):
            raise ValueError(f"{len(texts)} texts for {n} vectors")
        if not 1 <= dim <= MAX_DIM:
            raise ValueError(f"vector dimension {dim} is outside 1 to {MAX_DIM}")
        if not np.isfinite(matrix).all():
            raise ValueError("vectors must be finite")
        texts_bytes = encode_texts(texts)
        if len(texts_bytes) > texts_limit(n):
            raise ValueError(f"texts hold {len(texts_bytes)} bytes, over the limit of {texts_limit(n)}")
        meta = {
            "n": n,
            "dim": dim,
            "seed": params.seed,
            "min_topic_size": params.min_topic_size,
            "nr_topics": params.nr_topics,
            "top_n_words": params.top_n_words,
            "min_df": params.min_df,
            "stop_words": list(params.stop_words),
        }
        meta_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(meta_bytes) > META_MAX_BYTES:
            raise ValueError("fit settings exceed the service's meta limit")
        files = {
            "meta": ("meta.json", meta_bytes, "application/json"),
            "texts": ("texts.json", texts_bytes, "application/json"),
            "vectors": ("vectors.f32", matrix.tobytes(), "application/octet-stream"),
        }
        headers = {"Authorization": f"Bearer {self._config.api_key}"} if self._config.api_key else {}
        try:
            timeout = httpx.Timeout(self._config.timeout, connect=min(self._config.timeout, CONNECT_TIMEOUT_S))
            with httpx.Client(timeout=timeout, headers=headers, transport=self._transport) as client:
                response = client.post(self.fit_url, files=files)
        except httpx.HTTPError as exc:
            raise TopicsUnreachable(
                f"{self.fit_url} could not be reached ({type(exc).__name__}); {_UNREACHABLE_REMEDY}"
            ) from None
        if response.status_code != 200:
            raise self._error_for(response)
        result = self._parse(response, n)
        logger.info(
            "Topics fit | n={} dim={} topics={} outliers={} fit_s={:.2f}",
            n,
            dim,
            len(result.topics),
            result.outliers,
            result.timings.get("fit_s", 0.0),
        )
        return result
