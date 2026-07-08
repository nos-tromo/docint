"""RED tests for the ``ModelConfig.embed_tokenizer_repo`` contract.

These tests pin the provider-aware defaults and env override for the
new ``embed_tokenizer_repo`` field on :class:`ModelConfig`:

- ``INFERENCE_PROVIDER=ollama`` -> default ``"BAAI/bge-m3"``.
- ``INFERENCE_PROVIDER=vllm``   -> default ``"BAAI/bge-m3"``.
- ``INFERENCE_PROVIDER=openai`` -> default ``""`` (empty, no local
  tokenizer needed because the provider does tokenization).
- Operators always win: ``EMBED_TOKENIZER_REPO`` overrides the default
  regardless of provider.

The tests MUST fail on ``HEAD`` because ``ModelConfig`` does not yet
carry an ``embed_tokenizer_repo`` field.
"""

from __future__ import annotations

import pytest

from docint.utils.env_cfg import (
    load_embedding_env,
    load_frontend_env,
    load_model_env,
    load_resolution_env,
    load_retrieval_env,
    parse_nginx_size,
)


def test_model_config_embed_tokenizer_repo_default_for_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama default must be ``BAAI/bge-m3`` when no env override is set.

    The Ollama profile loads bge-m3 as the embed model; the matching
    tokenizer repo ships separately so the worker can count tokens
    offline before sending a request to the provider.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.delenv("EMBED_TOKENIZER_REPO", raising=False)

    cfg = load_model_env()

    assert cfg.embed_tokenizer_repo == "BAAI/bge-m3"


def test_model_config_embed_tokenizer_repo_default_for_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VLLM default must also be ``BAAI/bge-m3``.

    The vLLM profile likewise uses bge-m3 for embeddings, so the
    worker-side tokenizer repo must match.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    monkeypatch.delenv("EMBED_TOKENIZER_REPO", raising=False)

    cfg = load_model_env()

    assert cfg.embed_tokenizer_repo == "BAAI/bge-m3"


def test_model_config_embed_tokenizer_repo_empty_for_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI default must be the empty string.

    OpenAI-compatible APIs do tokenization server-side, so the worker
    does not need a local tokenizer snapshot. The empty string signals
    to :class:`RAG` that it should skip the counter build and fall
    back to the char-ratio estimator.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "openai")
    monkeypatch.delenv("EMBED_TOKENIZER_REPO", raising=False)

    cfg = load_model_env()

    assert cfg.embed_tokenizer_repo == ""


def test_model_config_embed_tokenizer_repo_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``EMBED_TOKENIZER_REPO`` overrides the provider default.

    Operators may switch the embed model (and thus the tokenizer
    repo) to e.g. ``Alibaba-NLP/gte-large`` without code changes.
    The env var must take precedence over the provider-aware default.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.setenv("EMBED_TOKENIZER_REPO", "Alibaba-NLP/gte-large")

    cfg = load_model_env()

    assert cfg.embed_tokenizer_repo == "Alibaba-NLP/gte-large"


# ---------------------------------------------------------------------------
# EmbeddingConfig envelope (timeout / batch size / max retries) RED tests
# ---------------------------------------------------------------------------
#
# These tests pin the new ``timeout_seconds``, ``batch_size`` and
# ``max_retries`` fields on :class:`EmbeddingConfig`. They exist because
# the CPU-ollama profile was hitting a 15-minute ingestion timeout: the
# chat and embed clients shared the same 300 s timeout * (1 + 2 retries)
# envelope via ``OpenAIConfig``, and a 100-input batch on bge-m3 on CPU
# legitimately exceeds 300 s. The fix splits the embed envelope out so
# each provider gets sensible defaults (long timeout + small batch for
# slow CPU ollama; short timeout + big batch for fast OpenAI).
#
# Provider-aware defaults:
#   ollama : timeout=1800 s, batch=16,  retries=1
#   vllm   : timeout=600  s, batch=64,  retries=1
#   openai : timeout=60   s, batch=100, retries=2
#
# Validation: ``timeout_seconds > 0``, ``batch_size in [1, 1024]``,
# ``max_retries in [0, 10]``. Operator env vars always win.


def test_embedding_config_default_timeout_for_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama default embed timeout must be 1800 s (30 minutes).

    CPU ollama legitimately takes minutes per 16-input bge-m3 batch; the
    default must be generous enough that a single batch does not trip
    the httpx client's timeout before the model finishes encoding.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.delenv("EMBED_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("EMBED_BATCH_SIZE", raising=False)
    monkeypatch.delenv("EMBED_MAX_RETRIES", raising=False)

    cfg = load_embedding_env()

    assert cfg.timeout_seconds == 1800.0


def test_embedding_config_default_timeout_for_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VLLM default embed timeout must be 600 s (10 minutes).

    vLLM is substantially faster than CPU ollama but slower than hosted
    OpenAI; 10 minutes gives headroom for large batches on a warmed
    server without letting a wedged request hang forever.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    monkeypatch.delenv("EMBED_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("EMBED_BATCH_SIZE", raising=False)
    monkeypatch.delenv("EMBED_MAX_RETRIES", raising=False)

    cfg = load_embedding_env()

    assert cfg.timeout_seconds == 600.0


def test_embedding_config_default_timeout_for_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI default embed timeout must be 60 s.

    Hosted OpenAI responds to embed requests in sub-second latency even
    for 100-input batches; a 60 s envelope surfaces real outages quickly
    rather than masking them with a long wait.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "openai")
    monkeypatch.delenv("EMBED_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("EMBED_BATCH_SIZE", raising=False)
    monkeypatch.delenv("EMBED_MAX_RETRIES", raising=False)

    cfg = load_embedding_env()

    assert cfg.timeout_seconds == 60.0


def test_embedding_config_default_batch_size_for_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama default embed batch size must be 16.

    CPU ollama throughput per batch falls off a cliff past ~16 inputs;
    smaller batches keep wall-clock-per-batch inside the timeout.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.delenv("EMBED_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("EMBED_BATCH_SIZE", raising=False)
    monkeypatch.delenv("EMBED_MAX_RETRIES", raising=False)

    cfg = load_embedding_env()

    assert cfg.batch_size == 16


def test_embedding_config_default_batch_size_for_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VLLM default embed batch size must be 64.

    vLLM handles larger batches efficiently via continuous batching;
    64 is a reasonable middle ground between throughput and memory use.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    monkeypatch.delenv("EMBED_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("EMBED_BATCH_SIZE", raising=False)
    monkeypatch.delenv("EMBED_MAX_RETRIES", raising=False)

    cfg = load_embedding_env()

    assert cfg.batch_size == 64


def test_embedding_config_default_batch_size_for_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI default embed batch size must be 100.

    OpenAI's embed endpoint accepts arrays of up to 2048 inputs; 100 is
    the llama_index default and keeps per-request overhead low without
    tripping provider rate limits.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "openai")
    monkeypatch.delenv("EMBED_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("EMBED_BATCH_SIZE", raising=False)
    monkeypatch.delenv("EMBED_MAX_RETRIES", raising=False)

    cfg = load_embedding_env()

    assert cfg.batch_size == 100


def test_embedding_config_default_max_retries_for_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama default embed retries must be 1.

    Retrying a long CPU embed batch doubles the wait for a flaky
    connection; a single retry is the right tradeoff between resilience
    and wall-clock budget.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.delenv("EMBED_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("EMBED_BATCH_SIZE", raising=False)
    monkeypatch.delenv("EMBED_MAX_RETRIES", raising=False)

    cfg = load_embedding_env()

    assert cfg.max_retries == 1


def test_embedding_config_default_max_retries_for_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI default embed retries must be 2.

    Hosted OpenAI calls are fast and benefit from the same retry
    posture as chat completions; two retries is the llama_index and
    OpenAI SDK default.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "openai")
    monkeypatch.delenv("EMBED_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("EMBED_BATCH_SIZE", raising=False)
    monkeypatch.delenv("EMBED_MAX_RETRIES", raising=False)

    cfg = load_embedding_env()

    assert cfg.max_retries == 2


def test_embedding_config_env_overrides_beat_provider_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator env vars must override the provider-aware defaults.

    ``EMBED_TIMEOUT_SECONDS`` / ``EMBED_BATCH_SIZE`` /
    ``EMBED_MAX_RETRIES`` are the escape hatch operators use to tune an
    individual deployment without patching code; they must take
    precedence over the built-in per-provider defaults.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.setenv("EMBED_TIMEOUT_SECONDS", "42.5")
    monkeypatch.setenv("EMBED_BATCH_SIZE", "7")
    monkeypatch.setenv("EMBED_MAX_RETRIES", "0")

    cfg = load_embedding_env()

    assert cfg.timeout_seconds == 42.5
    assert cfg.batch_size == 7
    assert cfg.max_retries == 0


def test_embedding_config_default_ctx_tokens_for_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama default ``ctx_tokens`` must be 2048.

    Ollama serves ``bge-m3`` with ``num_ctx=2048`` by default and its
    ``/api/show`` response does not publish ``num_ctx`` reliably across
    versions, so probing the live server is unreliable. The embedding
    loader therefore defaults the budget to 2048 on ollama; operators
    who raise the window via a custom Modelfile
    (``ollama create docint-bge-m3 -f deploy/Modelfile.bge-m3``) set
    ``EMBED_CTX_TOKENS=8192`` in ``.env`` to reclaim the full window.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.delenv("EMBED_CTX_TOKENS", raising=False)
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    monkeypatch.delenv("CHAT_MAX_MODEL_LEN", raising=False)

    cfg = load_embedding_env()

    assert cfg.ctx_tokens == 2048


def test_embedding_config_env_var_override_reclaims_8192_on_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator env-var override must take precedence over the 2048 ollama default.

    Pins the escape hatch: when an operator ships a custom Modelfile
    that raises bge-m3's ``num_ctx`` to 8192 (see
    ``deploy/Modelfile.bge-m3``), they set ``EMBED_CTX_TOKENS=8192`` in
    ``.env`` and the loader honours that choice regardless of provider.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.setenv("EMBED_CTX_TOKENS", "8192")

    cfg = load_embedding_env()

    assert cfg.ctx_tokens == 8192


def test_embedding_config_rejects_out_of_range_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Out-of-range env values must raise ``ValueError``.

    Validation bounds: ``timeout_seconds > 0``,
    ``batch_size in [1, 1024]``, ``max_retries in [0, 10]``. The
    loader must reject values outside those ranges so operators get a
    loud failure at boot rather than a silent miscalibration at
    ingestion time.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.delenv("EMBED_BATCH_SIZE", raising=False)
    monkeypatch.delenv("EMBED_MAX_RETRIES", raising=False)

    # timeout_seconds must be strictly positive.
    monkeypatch.setenv("EMBED_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError):
        load_embedding_env()

    monkeypatch.setenv("EMBED_TIMEOUT_SECONDS", "-1")
    with pytest.raises(ValueError):
        load_embedding_env()

    # NaN and ±inf must be rejected: NaN's pathological comparison
    # semantics would silently admit it and inf would disable the HTTP
    # timeout entirely. The guard at ``load_embedding_env`` uses
    # ``math.isfinite`` so both are rejected with a clear diagnostic.
    monkeypatch.setenv("EMBED_TIMEOUT_SECONDS", "nan")
    with pytest.raises(ValueError):
        load_embedding_env()

    monkeypatch.setenv("EMBED_TIMEOUT_SECONDS", "inf")
    with pytest.raises(ValueError):
        load_embedding_env()

    monkeypatch.setenv("EMBED_TIMEOUT_SECONDS", "-inf")
    with pytest.raises(ValueError):
        load_embedding_env()

    monkeypatch.delenv("EMBED_TIMEOUT_SECONDS", raising=False)

    # batch_size must land in [1, 1024].
    monkeypatch.setenv("EMBED_BATCH_SIZE", "0")
    with pytest.raises(ValueError):
        load_embedding_env()

    monkeypatch.setenv("EMBED_BATCH_SIZE", "1025")
    with pytest.raises(ValueError):
        load_embedding_env()

    monkeypatch.delenv("EMBED_BATCH_SIZE", raising=False)

    # max_retries must land in [0, 10].
    monkeypatch.setenv("EMBED_MAX_RETRIES", "-1")
    with pytest.raises(ValueError):
        load_embedding_env()

    monkeypatch.setenv("EMBED_MAX_RETRIES", "11")
    with pytest.raises(ValueError):
        load_embedding_env()


# ---------------------------------------------------------------------------
# RetrievalConfig.parent_context_safety_margin
# ---------------------------------------------------------------------------


def test_parent_context_safety_margin_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default ``parent_context_safety_margin`` is 0.95.

    Reserves 5 % of ``OPENAI_CTX_WINDOW`` for provider-side BOS/EOS and
    char/token-estimator drift so the packer never rides the ceiling.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.delenv("PARENT_CONTEXT_SAFETY_MARGIN", raising=False)

    cfg = load_retrieval_env()

    assert cfg.parent_context_safety_margin == pytest.approx(0.95)


def test_parent_context_safety_margin_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operators can tighten the safety margin via the env var.

    Useful on providers that 4xx close to the ceiling or when the
    char/token estimator undershoots on dense multilingual text.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("PARENT_CONTEXT_SAFETY_MARGIN", "0.85")

    cfg = load_retrieval_env()

    assert cfg.parent_context_safety_margin == pytest.approx(0.85)


def test_parent_context_safety_margin_rejects_out_of_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Out-of-range values fall back to the default without raising.

    A stray typo (``1.5``, ``0``, ``-0.1``) should log a warning and
    keep ingest / query running rather than raise at import time.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    for bad in ("1.5", "0", "-0.1", "not-a-float"):
        monkeypatch.setenv("PARENT_CONTEXT_SAFETY_MARGIN", bad)
        cfg = load_retrieval_env()
        assert cfg.parent_context_safety_margin == pytest.approx(0.95)


def _clear_resolution_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every RES_* override so loader defaults are observable.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    for var in (
        "RES_EMBED_THRESHOLD",
        "RES_LLM_TIEBREAK",
        "RES_CASE_NORMALIZE",
        "RES_VECTOR_K",
    ):
        monkeypatch.delenv(var, raising=False)


def test_resolution_config_defaults_match_chorus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defaults mirror chorus: threshold 0.86, tiebreak+case-norm on, k=5.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    _clear_resolution_env(monkeypatch)

    cfg = load_resolution_env()

    assert cfg.embed_cluster_threshold == pytest.approx(0.86)
    assert cfg.llm_tiebreak_enabled is True
    assert cfg.case_normalize is True
    assert cfg.vector_k == 5


def test_resolution_config_reads_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator env values override every resolution default.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("RES_EMBED_THRESHOLD", "0.91")
    monkeypatch.setenv("RES_LLM_TIEBREAK", "false")
    monkeypatch.setenv("RES_CASE_NORMALIZE", "0")
    monkeypatch.setenv("RES_VECTOR_K", "8")

    cfg = load_resolution_env()

    assert cfg.embed_cluster_threshold == pytest.approx(0.91)
    assert cfg.llm_tiebreak_enabled is False
    assert cfg.case_normalize is False
    assert cfg.vector_k == 8


def test_resolution_config_rejects_threshold_out_of_unit_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cosine threshold outside [0, 1] is a configuration error.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    _clear_resolution_env(monkeypatch)
    for bad in ("1.5", "-0.1", "not-a-float"):
        monkeypatch.setenv("RES_EMBED_THRESHOLD", bad)
        with pytest.raises(ValueError, match="RES_EMBED_THRESHOLD"):
            load_resolution_env()


def test_resolution_config_rejects_non_positive_vector_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``vector_k`` must be a positive integer in range.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    _clear_resolution_env(monkeypatch)
    for bad in ("0", "-3", "not-an-int"):
        monkeypatch.setenv("RES_VECTOR_K", bad)
        with pytest.raises(ValueError, match="RES_VECTOR_K"):
            load_resolution_env()


# ---------------------------------------------------------------------------
# FrontendConfig graph node-count fields
# ---------------------------------------------------------------------------


def _clear_frontend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove frontend env overrides so loader defaults are observable.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    for var in (
        "FRONTEND_COLLECTION_TIMEOUT",
        "NER_GRAPH_TOP_K",
        "NER_GRAPH_MAX_TOP_K",
        "DOCINT_CLIENT_MAX_BODY_SIZE",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1g", 1024**3),
        ("8g", 8 * 1024**3),
        ("512m", 512 * 1024**2),
        ("1024k", 1024 * 1024),
        ("1048576", 1_048_576),  # bare bytes
        ("2G", 2 * 1024**3),  # case-insensitive
        ("  4g  ", 4 * 1024**3),  # whitespace tolerated
    ],
)
def test_parse_nginx_size_valid(value: str, expected: int) -> None:
    """Nginx size notation parses to binary byte counts.

    Args:
        value (str): The nginx-style size string.
        expected (int): The expected byte count.
    """
    assert parse_nginx_size(value, default_bytes=123) == expected


@pytest.mark.parametrize("value", ["", None, "notasize", "1gb", "g", "-5"])
def test_parse_nginx_size_invalid_falls_back(value: str | None) -> None:
    """Malformed sizes fall back to the default so /config never crashes.

    Args:
        value (str | None): An unparseable size string (or None).
    """
    # ``-5`` is a special case: ``int("-5")`` succeeds but is clamped to 0, not
    # the fallback — a negative ceiling would be nonsensical either way.
    if value == "-5":
        assert parse_nginx_size(value, default_bytes=999) == 0
    else:
        assert parse_nginx_size(value, default_bytes=999) == 999


def test_frontend_config_upload_bytes_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an override the upload ceiling defaults to 1 GiB (matches nginx).

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    _clear_frontend_env(monkeypatch)

    cfg = load_frontend_env()

    assert cfg.max_upload_bytes == 1024**3


def test_frontend_config_upload_bytes_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """DOCINT_CLIENT_MAX_BODY_SIZE drives the advertised upload ceiling.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("DOCINT_CLIENT_MAX_BODY_SIZE", "8g")

    cfg = load_frontend_env()

    assert cfg.max_upload_bytes == 8 * 1024**3


def test_frontend_config_graph_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults are 80 nodes with a 500 ceiling.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    _clear_frontend_env(monkeypatch)

    cfg = load_frontend_env()

    assert cfg.graph_top_k == 80
    assert cfg.graph_max_top_k == 500


def test_frontend_config_graph_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operator env vars set both the default and the ceiling.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("NER_GRAPH_TOP_K", "200")
    monkeypatch.setenv("NER_GRAPH_MAX_TOP_K", "1000")

    cfg = load_frontend_env()

    assert cfg.graph_top_k == 200
    assert cfg.graph_max_top_k == 1000


def test_frontend_config_clamps_top_k_to_at_least_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero/negative default is clamped up to 1.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    _clear_frontend_env(monkeypatch)
    monkeypatch.setenv("NER_GRAPH_TOP_K", "0")

    cfg = load_frontend_env()

    assert cfg.graph_top_k == 1


def test_frontend_config_max_never_below_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ceiling is raised to the default when configured below it.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    _clear_frontend_env(monkeypatch)
    monkeypatch.setenv("NER_GRAPH_TOP_K", "300")
    monkeypatch.setenv("NER_GRAPH_MAX_TOP_K", "100")

    cfg = load_frontend_env()

    assert cfg.graph_top_k == 300
    assert cfg.graph_max_top_k == 300
