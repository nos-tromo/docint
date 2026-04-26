"""Centralized environment-variable loaders and configuration dataclasses."""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def set_offline_env() -> None:
    """Log the current offline mode status.

    The actual env vars (HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, etc.) are set at
    module level immediately after ``load_dotenv()`` so they are available before
    ``huggingface_hub`` / ``transformers`` cache their values at import time.
    This function re-applies them (idempotent) and emits a log message.
    """
    if str(os.getenv("DOCINT_OFFLINE", "1")).lower() in {"1", "true", "yes"}:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # Point fastembed at the HF hub cache so it reuses models downloaded
        # by ``model_cfg.py`` instead of trying to fetch them.
        if not os.getenv("FASTEMBED_CACHE_PATH"):
            default_hf_cache = str(Path.home() / ".cache" / "huggingface" / "hub")
            os.environ["FASTEMBED_CACHE_PATH"] = os.getenv(
                "HF_HUB_CACHE", default_hf_cache
            )

        logger.info("Set Hugging Face libraries to offline mode.")
    else:
        logger.info("Hugging Face libraries are in online mode.")


set_offline_env()  # Apply offline settings at module load time


def _apply_device_visibility() -> None:
    """Hide CUDA devices when the backend is configured for CPU-only work.

    When ``USE_DEVICE=cpu`` the process should never initialise a CUDA
    context.  Setting ``CUDA_VISIBLE_DEVICES=""`` before any PyTorch import
    prevents accidental GPU memory allocation that can destabilise co-located
    GPU services (e.g. vLLM workers sharing the same physical GPUs).
    """
    requested = os.getenv("USE_DEVICE", "auto").strip().lower()
    if requested != "cpu":
        return
    # Only override when not already set by the operator.
    if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    logger.info(
        "USE_DEVICE=cpu: set CUDA_VISIBLE_DEVICES='' to prevent GPU context init."
    )


_apply_device_visibility()  # Must run before any torch.cuda call


def resolve_hf_cache_path(
    cache_dir: Path, repo_id: str, filename: str | None = None
) -> Path | None:
    """Resolve a HuggingFace model or file path from the local HF cache.

    HF hub stores downloads under:
        {cache_dir}/models--{org}--{repo}/snapshots/{commit_hash}/

    Args:
        cache_dir (Path): The HF hub cache directory (e.g. ~/.cache/huggingface/hub).
        repo_id (str): The HuggingFace repository ID (e.g. "BAAI/bge-m3").
        filename (str | None): Optional file name within the snapshot directory. If provided,
                  return the path to that specific file; otherwise return the
                  snapshot directory itself.

    Returns:
        Path | None: The resolved Path if found, otherwise None.
    """
    model_dir_name = f"models--{repo_id.replace('/', '--')}"
    model_cache_dir = cache_dir / model_dir_name

    if not model_cache_dir.exists():
        return None

    ref_path = model_cache_dir / "refs" / "main"
    if not ref_path.exists():
        return None

    commit_hash = ref_path.read_text().strip()
    snapshot_path = model_cache_dir / "snapshots" / commit_hash

    if filename:
        file_path = snapshot_path / filename
        return file_path if file_path.exists() else None

    return snapshot_path if snapshot_path.exists() else None


@dataclass(frozen=True)
class EmbeddingConfig:
    """Dataclass for embedding-pipeline context limits and HTTP envelope.

    Separate from :class:`OpenAIConfig` so the chat LLM context window
    (``OPENAI_CTX_WINDOW``) and the embedding context window
    (``EMBED_CTX_TOKENS``) can vary independently — they almost always
    do, since embedding models rarely share a context window with the
    chat LLM served from the same provider. The HTTP envelope
    (``timeout_seconds``, ``batch_size``, ``max_retries``) is also
    carved out here because CPU-served embedding providers (ollama) can
    legitimately take minutes per batch, which would starve the
    lower-latency chat client if both shared the same timeout.
    """

    ctx_tokens: int
    char_token_ratio: float
    ctx_safety_margin: float
    timeout_seconds: float
    batch_size: int
    max_retries: int


def load_embedding_env(
    default_ctx_tokens: int = 8192,
    default_char_token_ratio: float = 3.5,
    default_ctx_safety_margin: float = 0.95,
) -> EmbeddingConfig:
    """Load embedding-pipeline configuration from environment variables.

    Provider-aware defaults:

    - ``openai`` + ``EMBED_MODEL`` starting with ``text-embedding-3-``:
      defaults ``ctx_tokens`` to ``8191`` (OpenAI's documented limit).
    - ``vllm`` with ``CHAT_MAX_MODEL_LEN`` set: uses that value as the
      default (operators keep the vLLM chat and embed servers in sync
      via the same env var).
    - Ollama and everything else: defaults to ``8192``.

    Operators always win — any explicit ``EMBED_CTX_TOKENS`` value
    overrides the provider default.

    Args:
        default_ctx_tokens: Base default when no provider-specific
            hint applies.
        default_char_token_ratio: Characters per token estimator for
            budget checks (see
            :func:`docint.utils.embed_chunking.estimate_tokens`).
        default_ctx_safety_margin: Fraction of ``ctx_tokens`` that
            remains for the user payload after the provider reserves
            BOS/EOS slots. Must lie in ``(0, 1]``.

    Returns:
        EmbeddingConfig: Parsed embedding configuration.
        - ctx_tokens (int): Embedding context window in tokens.
        - char_token_ratio (float): Characters per token estimator.
        - ctx_safety_margin (float): Safety margin fraction.
        - timeout_seconds (float): Per-request HTTP timeout for the
          embed client.
        - batch_size (int): Inputs per embed request.
        - max_retries (int): Retries on transient embed-client
          failures.

    Raises:
        ValueError: When ``EMBED_CTX_TOKENS`` falls outside
            ``[256, 32768]``, ``EMBED_CHAR_TOKEN_RATIO`` is
            non-positive, ``EMBED_CTX_SAFETY_MARGIN`` is outside
            ``(0, 1]``, ``EMBED_TIMEOUT_SECONDS`` is not a finite
            positive float, ``EMBED_BATCH_SIZE`` falls outside
            ``[1, 1024]``, or ``EMBED_MAX_RETRIES`` falls outside
            ``[0, 10]``.
    """
    inference_provider = os.getenv("INFERENCE_PROVIDER", "ollama").strip().lower()
    embed_model = os.getenv("EMBED_MODEL", "").strip()

    provider_default_ctx = default_ctx_tokens
    if inference_provider == "ollama":
        # Ollama's default num_ctx is 2048 regardless of model capacity.
        # Operators who raise it via a Modelfile set EMBED_CTX_TOKENS
        # explicitly. See deploy/Modelfile.bge-m3 and docs/deployment.md.
        provider_default_ctx = 2048
    elif inference_provider == "openai" and embed_model.startswith("text-embedding-3-"):
        provider_default_ctx = 8191
    elif inference_provider == "vllm":
        raw_chat_max_model_len = os.getenv("CHAT_MAX_MODEL_LEN")
        if raw_chat_max_model_len is not None and raw_chat_max_model_len.strip():
            try:
                provider_default_ctx = int(raw_chat_max_model_len)
            except ValueError as exc:
                raise ValueError(
                    f"CHAT_MAX_MODEL_LEN must be an integer, "
                    f"got {raw_chat_max_model_len!r}"
                ) from exc

    raw_ctx_tokens = os.getenv("EMBED_CTX_TOKENS")
    if raw_ctx_tokens is not None and raw_ctx_tokens.strip():
        try:
            ctx_tokens = int(raw_ctx_tokens)
        except ValueError as exc:
            raise ValueError(
                f"EMBED_CTX_TOKENS must be an integer, got {raw_ctx_tokens!r}"
            ) from exc
    else:
        ctx_tokens = provider_default_ctx
    if not (256 <= ctx_tokens <= 32768):
        raise ValueError(
            f"EMBED_CTX_TOKENS={ctx_tokens!r} is out of range — "
            f"must be between 256 and 32768 tokens."
        )

    raw_char_token_ratio = os.getenv("EMBED_CHAR_TOKEN_RATIO")
    if raw_char_token_ratio is not None and raw_char_token_ratio.strip():
        try:
            char_token_ratio = float(raw_char_token_ratio)
        except ValueError as exc:
            raise ValueError(
                f"EMBED_CHAR_TOKEN_RATIO must be a float, got {raw_char_token_ratio!r}"
            ) from exc
    else:
        char_token_ratio = float(default_char_token_ratio)
    if char_token_ratio <= 0:
        raise ValueError(
            f"EMBED_CHAR_TOKEN_RATIO={char_token_ratio!r} is out of range — "
            f"must be positive."
        )

    raw_ctx_safety_margin = os.getenv("EMBED_CTX_SAFETY_MARGIN")
    if raw_ctx_safety_margin is not None and raw_ctx_safety_margin.strip():
        try:
            ctx_safety_margin = float(raw_ctx_safety_margin)
        except ValueError as exc:
            raise ValueError(
                f"EMBED_CTX_SAFETY_MARGIN must be a float, "
                f"got {raw_ctx_safety_margin!r}"
            ) from exc
    else:
        ctx_safety_margin = float(default_ctx_safety_margin)
    if not (0.0 < ctx_safety_margin <= 1.0):
        raise ValueError(
            f"EMBED_CTX_SAFETY_MARGIN={ctx_safety_margin!r} is out of range — "
            f"must be within (0, 1]."
        )

    provider_timeout_default, provider_batch_default, provider_retries_default = (
        _embed_envelope_defaults(inference_provider)
    )

    raw_timeout_seconds = os.getenv("EMBED_TIMEOUT_SECONDS")
    if raw_timeout_seconds is not None and raw_timeout_seconds.strip():
        try:
            timeout_seconds = float(raw_timeout_seconds)
        except ValueError as exc:
            raise ValueError(
                f"EMBED_TIMEOUT_SECONDS must be a float, got {raw_timeout_seconds!r}"
            ) from exc
    else:
        timeout_seconds = float(provider_timeout_default)
    # Guard against NaN and ±inf up front: NaN comparisons are always False, so
    # a ``<= 0`` check would silently admit ``float('nan')`` and ``float('inf')``
    # would disable the HTTP timeout entirely (ingest hangs on a silent provider
    # with the ``worst_case_wait > 3600`` operator warning also skipped).
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError(
            f"EMBED_TIMEOUT_SECONDS={timeout_seconds!r} is out of range — "
            f"must be a finite positive float."
        )

    raw_batch_size = os.getenv("EMBED_BATCH_SIZE")
    if raw_batch_size is not None and raw_batch_size.strip():
        try:
            batch_size = int(raw_batch_size)
        except ValueError as exc:
            raise ValueError(
                f"EMBED_BATCH_SIZE must be an integer, got {raw_batch_size!r}"
            ) from exc
    else:
        batch_size = provider_batch_default
    if not (1 <= batch_size <= 1024):
        raise ValueError(
            f"EMBED_BATCH_SIZE={batch_size!r} is out of range — "
            f"must be between 1 and 1024."
        )

    raw_max_retries = os.getenv("EMBED_MAX_RETRIES")
    if raw_max_retries is not None and raw_max_retries.strip():
        try:
            max_retries = int(raw_max_retries)
        except ValueError as exc:
            raise ValueError(
                f"EMBED_MAX_RETRIES must be an integer, got {raw_max_retries!r}"
            ) from exc
    else:
        max_retries = provider_retries_default
    if not (0 <= max_retries <= 10):
        raise ValueError(
            f"EMBED_MAX_RETRIES={max_retries!r} is out of range — "
            f"must be between 0 and 10."
        )

    return EmbeddingConfig(
        ctx_tokens=ctx_tokens,
        char_token_ratio=char_token_ratio,
        ctx_safety_margin=ctx_safety_margin,
        timeout_seconds=timeout_seconds,
        batch_size=batch_size,
        max_retries=max_retries,
    )


def _embed_envelope_defaults(inference_provider: str) -> tuple[float, int, int]:
    """Return the provider-aware ``(timeout_s, batch_size, max_retries)`` defaults.

    The embedding client runs against a provider whose throughput and
    latency profile differ sharply from the chat client. Defaults are
    tuned so the first batch completes well within the HTTP envelope:

    - ``ollama``: CPU-bound, minutes-per-batch on large corpora →
      long timeout, small batch, minimal retries.
    - ``vllm``: GPU-served on a shared box → moderate timeout, larger
      batch, minimal retries.
    - ``openai`` (and every other remote API): network-latency-bound →
      short timeout, full batch, more retries to absorb transient
      failures.

    Args:
        inference_provider: Value of the ``INFERENCE_PROVIDER`` env var,
            already lower-cased and stripped.

    Returns:
        tuple[float, int, int]: ``(timeout_seconds, batch_size,
        max_retries)`` for the provider.
    """
    if inference_provider == "vllm":
        return 600.0, 64, 1
    if inference_provider == "openai":
        return 60.0, 100, 2
    # Ollama is the default, along with any unknown provider.
    return 1800.0, 16, 1


@dataclass(frozen=True)
class FrontendConfig:
    """Dataclass for frontend configuration."""

    collection_timeout: int


def load_frontend_env(
    default_collection_timeout: int = 120,
) -> FrontendConfig:
    """Loads frontend configuration from environment variables or defaults.

    Args:
        default_collection_timeout (int): Default timeout in seconds for fetching collections from the backend.

    Returns:
        FrontendConfig: Dataclass containing frontend configuration.
        - collection_timeout (int): Timeout in seconds for fetching collections from the backend.
    """
    return FrontendConfig(
        collection_timeout=int(
            os.getenv("FRONTEND_COLLECTION_TIMEOUT", default_collection_timeout)
        )
    )


@dataclass(frozen=True)
class GraphRAGConfig:
    """Dataclass for graph-assisted retrieval configuration."""

    enabled: bool
    neighbor_hops: int
    top_k_nodes: int
    min_edge_weight: int
    max_neighbors: int


def load_graphrag_env(
    default_enabled: bool = True,
    default_neighbor_hops: int = 2,
    default_top_k_nodes: int = 50,
    default_min_edge_weight: int = 3,
    default_max_neighbors: int = 6,
) -> GraphRAGConfig:
    """Load GraphRAG configuration from environment variables.

    Args:
        default_enabled (bool): Whether graph-assisted retrieval is enabled by default.
        default_neighbor_hops (int): Number of graph hops used for query expansion.
        default_top_k_nodes (int): Maximum number of graph nodes kept in memory.
        default_min_edge_weight (int): Minimum edge weight used for graph filtering.
        default_max_neighbors (int): Maximum number of neighbor entities appended to a query.

    Returns:
        GraphRAGConfig: Parsed graph retrieval settings.
    """
    return GraphRAGConfig(
        enabled=str(os.getenv("GRAPHRAG_ENABLED", default_enabled)).lower()
        in {"true", "1", "yes"},
        neighbor_hops=max(
            1, int(os.getenv("GRAPHRAG_NEIGHBOR_HOPS", default_neighbor_hops))
        ),
        top_k_nodes=max(1, int(os.getenv("GRAPHRAG_TOP_K_NODES", default_top_k_nodes))),
        min_edge_weight=max(
            1, int(os.getenv("GRAPHRAG_MIN_EDGE_WEIGHT", default_min_edge_weight))
        ),
        max_neighbors=max(
            1, int(os.getenv("GRAPHRAG_MAX_NEIGHBORS", default_max_neighbors))
        ),
    )


@dataclass(frozen=True)
class HateSpeechConfig:
    """Dataclass for hate-speech detection configuration."""

    enabled: bool
    max_chars: int
    max_workers: int


def load_hate_speech_env(
    default_enabled: bool = False,
    default_max_chars: int = 2048,
    default_max_workers: int = 1,
) -> HateSpeechConfig:
    """Load hate-speech detection settings from environment variables.

    Args:
        default_enabled (bool): Whether hate-speech detection runs during ingestion.
        default_max_chars (int): Maximum characters from each chunk sent to the detector.
        default_max_workers (int): Maximum worker threads for parallel hate-speech detection.

    Returns:
        HateSpeechConfig: Parsed hate-speech detection configuration.
    """
    return HateSpeechConfig(
        enabled=str(os.getenv("ENABLE_HATE_SPEECH_DETECTION", default_enabled)).lower()
        in {"true", "1", "yes"},
        max_chars=max(256, int(os.getenv("HATE_SPEECH_MAX_CHARS", default_max_chars))),
        max_workers=max(
            1, int(os.getenv("HATE_SPEECH_MAX_WORKERS", default_max_workers))
        ),
    )


@dataclass(frozen=True)
class HostConfig:
    """Dataclass for host configuration."""

    backend_host: str
    backend_public_host: str
    qdrant_host: str
    cors_allowed_origins: str


def load_host_env(
    default_backend_host: str = "http://localhost:8000",
    default_qdrant_host: str = "http://localhost:6333",
    default_cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501",
) -> HostConfig:
    """Loads host configuration from environment variables or defaults.

    Args:
        default_backend_host (str): Default backend host URL.
        default_qdrant_host (str): Default Qdrant host URL.
        default_cors_origins (str): Default CORS allowed origins.

    Returns:
        HostConfig: Dataclass containing host configuration.
        - backend_host (str): The backend host URL.
        - backend_public_host (str): The public backend host URL. Required to enable document preview features
            in the Docker environment.
        - qdrant_host (str): The Qdrant host URL.
        - cors_allowed_origins (str): Comma-separated list of allowed CORS origins.
    """
    return HostConfig(
        backend_host=os.getenv("BACKEND_HOST", default_backend_host),
        backend_public_host=os.getenv("BACKEND_PUBLIC_HOST", default_backend_host),
        qdrant_host=os.getenv("QDRANT_HOST", default_qdrant_host),
        cors_allowed_origins=os.getenv("CORS_ALLOWED_ORIGINS", default_cors_origins),
    )


@dataclass(frozen=True)
class ImageIngestionConfig:
    """Configuration for image ingestion and image-vector indexing."""

    enabled: bool
    embedding_enabled: bool
    tagging_enabled: bool
    collection_name: str
    vector_name: str
    cache_by_hash: bool
    fail_on_embedding_error: bool
    fail_on_tagging_error: bool
    retrieve_top_k: int
    tagging_max_image_dimension: int = 1024


def load_image_ingestion_config(
    default_image_ingestion_enabled: bool = True,
    default_image_embedding_enabled: bool = True,
    default_image_tagging_enabled: bool = True,
    default_image_qdrant_collection: str = "{collection}_images",
    default_image_qdrant_vector_name: str = "image-dense",
    default_image_cache_by_hash: bool = True,
    default_fail_on_embedding_error: bool = False,
    default_fail_on_tagging_error: bool = False,
    default_retrieve_top_k: int = 5,
    default_tagging_max_image_dimension: int = 1024,
) -> ImageIngestionConfig:
    """Load image ingestion settings from environment variables.

    Args:
        default_image_ingestion_enabled (bool): Whether image ingestion is enabled by default.
        default_image_embedding_enabled (bool): Whether to generate embeddings for images by default.
        default_image_tagging_enabled (bool): Whether to generate tags for images by default.
        default_image_qdrant_collection (str): Default Qdrant collection name for storing image vectors.
        default_image_qdrant_vector_name (str): Default name of the vector field in Qdrant for image vectors.
        default_image_cache_by_hash (bool): Whether to cache image embeddings by hash to avoid redundant computation. Default is True.
        default_fail_on_embedding_error (bool): Whether to fail the entire ingestion if image embedding fails. Default is False.
        default_fail_on_tagging_error (bool): Whether to fail the entire ingestion if image tagging fails. Default is False.
        default_retrieve_top_k (int): The number of top image matches to retrieve for a text query. Default is 5.
        default_tagging_max_image_dimension (int): Maximum pixel dimension (width or height) for
            images sent to the vision tagging endpoint. Larger images are down-scaled. Default is 1024.

    Returns:
        ImageIngestionConfig: Dataclass containing image ingestion configuration.
        - enabled (bool): Whether image ingestion is enabled.
        - embedding_enabled (bool): Whether to generate embeddings for images.
        - tagging_enabled (bool): Whether to generate tags for images.
        - collection_name (str): The Qdrant collection name for storing image vectors.
        - vector_name (str): The name of the vector field in Qdrant.
        - cache_by_hash (bool): Whether to cache image embeddings by hash to avoid redundant computation.
        - fail_on_embedding_error (bool): Whether to fail the entire ingestion if image embedding fails.
        - fail_on_tagging_error (bool): Whether to fail the entire ingestion if image tagging fails.
        - retrieve_top_k (int): The number of top image matches to retrieve for a text query.
        - tagging_max_image_dimension (int): Maximum pixel dimension (width or height) for
            images sent to the vision tagging endpoint. Larger images are down-scaled.
    """
    return ImageIngestionConfig(
        enabled=str(
            os.getenv("IMAGE_INGESTION_ENABLED", default_image_ingestion_enabled)
        ).lower()
        in {"1", "true", "yes"},
        embedding_enabled=str(
            os.getenv("IMAGE_EMBEDDING_ENABLED", default_image_embedding_enabled)
        ).lower()
        in {"1", "true", "yes"},
        tagging_enabled=str(
            os.getenv("IMAGE_TAGGING_ENABLED", default_image_tagging_enabled)
        ).lower()
        in {"1", "true", "yes"},
        collection_name=os.getenv(
            "IMAGE_QDRANT_COLLECTION", default_image_qdrant_collection
        ),
        vector_name=os.getenv(
            "IMAGE_QDRANT_VECTOR_NAME", default_image_qdrant_vector_name
        ),
        cache_by_hash=str(
            os.getenv("IMAGE_CACHE_BY_HASH", default_image_cache_by_hash)
        ).lower()
        in {"1", "true", "yes"},
        fail_on_embedding_error=str(
            os.getenv("IMAGE_FAIL_ON_EMBED_ERROR", default_fail_on_embedding_error)
        ).lower()
        in {"1", "true", "yes"},
        fail_on_tagging_error=str(
            os.getenv("IMAGE_FAIL_ON_TAG_ERROR", default_fail_on_tagging_error)
        ).lower()
        in {"1", "true", "yes"},
        retrieve_top_k=int(os.getenv("IMAGE_RETRIEVE_TOP_K", default_retrieve_top_k)),
        tagging_max_image_dimension=int(
            os.getenv(
                "IMAGE_TAGGING_MAX_IMAGE_DIM", default_tagging_max_image_dimension
            )
        ),
    )


@dataclass(frozen=True)
class IngestionConfig:
    """Dataclass for ingestion configuration."""

    coarse_chunk_size: int
    docling_accelerator_num_threads: int
    docstore_batch_size: int
    ingest_benchmark_enabled: bool
    ingest_fail_fast: bool
    ingest_manifest_enabled: bool
    ingest_pipeline_overlap_enabled: bool
    streaming_readers_enabled: bool
    ingest_queue_max_size: int
    docstore_max_retries: int
    docstore_retry_backoff_max_seconds: float
    docstore_retry_backoff_seconds: float
    fine_chunk_overlap: int
    fine_chunk_size: int
    hierarchical_chunking_enabled: bool
    ingestion_batch_size: int
    sentence_splitter_chunk_overlap: int
    sentence_splitter_chunk_size: int
    supported_filetypes: list[str]


def load_ingestion_env(
    default_coarse_chunk_size: int = 8192,
    default_docling_accelerator_num_threads: int = 4,
    default_docstore_batch_size: int = 100,
    default_ingest_benchmark_enabled: bool = False,
    default_ingest_fail_fast: bool = False,
    default_ingest_manifest_enabled: bool = True,
    default_ingest_pipeline_overlap_enabled: bool = False,
    default_streaming_readers_enabled: bool = True,
    default_ingest_queue_max_size: int = 4,
    default_docstore_max_retries: int = 3,
    default_docstore_retry_backoff_seconds: float = 0.25,
    default_docstore_retry_backoff_max_seconds: float = 2.0,
    default_fine_chunk_overlap: int = 0,
    default_fine_chunk_size: int = 8192,
    default_hierarchical_chunking_enabled: bool = True,
    default_ingestion_batch_size: int = 50,
    default_sentence_splitter_chunk_overlap: int = 64,
    default_sentence_splitter_chunk_size: int = 1024,
    default_supported_filetypes: list[str] = [
        ".csv",
        ".docx",
        ".gif",
        ".jpeg",
        ".jpg",
        ".json",
        ".jsonl",
        ".md",
        ".ndjson",
        ".parquet",
        ".pdf",
        ".png",
        ".tsv",
        ".txt",
        ".xls",
        ".xlsx",
    ],
) -> IngestionConfig:
    """Loads ingestion configuration from environment variables or defaults.

    Returns:
        IngestionConfig: Dataclass containing ingestion configuration.
        - coarse_chunk_size (int): The coarse chunk size for hierarchical chunking.
        - docling_accelerator_num_threads (int): The default number of threads for Docling accelerator.
        - docstore_batch_size (int): The batch size for document store operations.
        - ingest_benchmark_enabled (bool): Emit ingestion benchmark summary logs
            for throughput and batch diagnostics.
        - ingest_fail_fast (bool): When true, abort ingestion on the first
            persistence failure (current behaviour for CI/strict tests).
            When false (default), the outer ingestion loop logs the
            failure, marks the in-flight file hashes failed in the
            manifest, and continues with the next batch — so one bad
            file does not invalidate the rest of the run.
        - ingest_manifest_enabled (bool): Track in-flight, completed, and
            failed file ingestions in a SQLite manifest for resume
            visibility. Set to ``false`` to disable the manifest writes
            (returns the no-op stub from :class:`NullIngestManifest`).
        - ingest_pipeline_overlap_enabled (bool): When true, run the
            streaming pipeline producer on a background thread so
            enrichment overlaps with persistence. Default false until
            canary measurement confirms throughput gains; flip via
            ``INGEST_PIPELINE_OVERLAP_ENABLED=true``.
        - streaming_readers_enabled (bool): When true, dispatch to each
            reader's ``iter_documents()`` generator directly instead of
            routing through ``SimpleDirectoryReader.load_file()``. Reduces
            peak memory for large CSV/JSONL files. Default false; enable via
            ``STREAMING_READERS_ENABLED=true``.
        - ingest_queue_max_size (int): Maximum number of pre-enriched
            batches buffered between producer and consumer when
            overlap is enabled. Bounds memory under back-pressure.
        - docstore_max_retries (int): Maximum retries for transient docstore
            transport failures (Qdrant vector writes) and SQLite locked-DB
            errors in :class:`SQLiteKVStore`.
        - docstore_retry_backoff_seconds (float): Initial retry backoff in seconds
            for docstore operations.
        - docstore_retry_backoff_max_seconds (float): Maximum retry backoff in
            seconds for docstore/Qdrant operations.
        - fine_chunk_overlap (int): The fine chunk overlap size for hierarchical chunking.
        - fine_chunk_size (int): The fine chunk size for hierarchical chunking.
        - hierarchical_chunking_enabled (bool): Whether hierarchical chunking is enabled.
        - ingestion_batch_size (int): The batch size for ingestion.
        - sentence_splitter_chunk_overlap (int): The chunk overlap size for sentence splitting.
        - sentence_splitter_chunk_size (int): The chunk size for sentence splitting.
        - supported_filetypes (list[str]): List of supported file extensions for ingestion.
    """
    return IngestionConfig(
        coarse_chunk_size=int(
            os.getenv("COARSE_CHUNK_SIZE", default_coarse_chunk_size)
        ),
        docling_accelerator_num_threads=int(
            os.getenv(
                "DOCLING_ACCELERATOR_NUM_THREADS",
                default_docling_accelerator_num_threads,
            )
        ),
        docstore_batch_size=int(
            os.getenv("DOCSTORE_BATCH_SIZE", default_docstore_batch_size)
        ),
        ingest_benchmark_enabled=str(
            os.getenv("INGEST_BENCHMARK_ENABLED", default_ingest_benchmark_enabled)
        ).lower()
        in {"true", "1", "yes"},
        ingest_fail_fast=str(
            os.getenv("INGEST_FAIL_FAST", default_ingest_fail_fast)
        ).lower()
        in {"true", "1", "yes"},
        ingest_manifest_enabled=str(
            os.getenv("INGEST_MANIFEST_ENABLED", default_ingest_manifest_enabled)
        ).lower()
        in {"true", "1", "yes"},
        ingest_pipeline_overlap_enabled=str(
            os.getenv(
                "INGEST_PIPELINE_OVERLAP_ENABLED",
                default_ingest_pipeline_overlap_enabled,
            )
        ).lower()
        in {"true", "1", "yes"},
        streaming_readers_enabled=str(
            os.getenv("STREAMING_READERS_ENABLED", default_streaming_readers_enabled)
        ).lower()
        in {"true", "1", "yes"},
        ingest_queue_max_size=max(
            1,
            int(os.getenv("INGEST_QUEUE_MAX_SIZE", default_ingest_queue_max_size)),
        ),
        docstore_max_retries=max(
            0,
            int(os.getenv("DOCSTORE_MAX_RETRIES", default_docstore_max_retries)),
        ),
        docstore_retry_backoff_seconds=max(
            0.0,
            float(
                os.getenv(
                    "DOCSTORE_RETRY_BACKOFF_SECONDS",
                    default_docstore_retry_backoff_seconds,
                )
            ),
        ),
        docstore_retry_backoff_max_seconds=max(
            0.0,
            float(
                os.getenv(
                    "DOCSTORE_RETRY_BACKOFF_MAX_SECONDS",
                    default_docstore_retry_backoff_max_seconds,
                )
            ),
        ),
        fine_chunk_overlap=int(
            os.getenv("FINE_CHUNK_OVERLAP", default_fine_chunk_overlap)
        ),
        fine_chunk_size=int(os.getenv("FINE_CHUNK_SIZE", default_fine_chunk_size)),
        hierarchical_chunking_enabled=str(
            os.getenv("HIERARCHICAL_CHUNKING_ENABLED", "true")
        ).lower()
        in {"true", "1", "yes"},
        ingestion_batch_size=int(
            os.getenv("INGESTION_BATCH_SIZE", default_ingestion_batch_size)
        ),
        sentence_splitter_chunk_overlap=int(
            os.getenv(
                "SENTENCE_SPLITTER_CHUNK_OVERLAP",
                default_sentence_splitter_chunk_overlap,
            )
        ),
        sentence_splitter_chunk_size=int(
            os.getenv(
                "SENTENCE_SPLITTER_CHUNK_SIZE", default_sentence_splitter_chunk_size
            )
        ),
        supported_filetypes=default_supported_filetypes,
    )


@dataclass(frozen=True)
class ModelConfig:
    """Dataclass for model configuration."""

    embed_model: str
    embed_tokenizer_repo: str
    image_embed_model: str
    ner_model: str
    rerank_model: str
    sparse_model: str
    text_model: str
    vision_model: str


def load_model_env(
    default_embed_model: str = "bge-m3",
    default_image_embed_model: str = "openai/clip-vit-base-patch32",
    default_ner_model: str = "gliner-community/gliner_large-v2.5",
    default_rerank_model: str = "BAAI/bge-reranker-v2-m3",
    default_sparse_model: str = "Qdrant/all_miniLM_L6_v2_with_attentions",
    default_text_model: str = "gpt-oss:20b",
    default_vision_model: str = "qwen3.5:9b",
) -> ModelConfig:
    """Loads model configuration from environment variables or defaults.

    Args:
        default_embed_model(str): Default embedding model identifier for
            Ollama-compatible embeddings.
        default_image_embed_model (str): Default image embedding model identifier.
        default_ner_model (str): Default NER model identifier.
        default_rerank_model (str): Default reranker model identifier.
        default_sparse_model (str): Default sparse model identifier.
        default_text_model_str (str): Default text model identifier.
        default_vision_model_str (str): Default vision model identifier.

    Returns:
        ModelConfig: Dataclass containing model configuration.
        - embed_model (str): The embedding model identifier.
        - embed_tokenizer_repo (str): HF repo id of the tokenizer used
          for offline token counting at ingestion time. Empty when
          tokenization happens on the provider side (e.g. ``openai``).
        - image_embed_model (str): The image embedding model identifier.
        - ner_model (str): The NER model identifier.
        - rerank_model (str): The reranker model identifier.
        - sparse_model (str): The sparse model identifier.
        - text_model (str): The text model identifier.
        - vision_model (str): The vision model identifier.
    """
    inference_provider = os.getenv("INFERENCE_PROVIDER", "ollama").strip().lower()

    default_embed_tokenizer_repo = "BAAI/bge-m3"

    if inference_provider == "vllm":
        default_embed_model = "BAAI/bge-m3"
        default_sparse_model = default_embed_model
        default_text_model = "Qwen/Qwen3.5-2B"
        default_vision_model = "Qwen/Qwen3.5-2B"

    if inference_provider == "openai":
        default_embed_model = "text-embedding-3-small"
        default_text_model = "gpt-4o"
        default_vision_model = "gpt-4o"
        default_embed_tokenizer_repo = ""

    return ModelConfig(
        embed_model=os.getenv("EMBED_MODEL", default_embed_model),
        embed_tokenizer_repo=os.getenv(
            "EMBED_TOKENIZER_REPO", default_embed_tokenizer_repo
        ),
        image_embed_model=os.getenv("IMAGE_EMBED_MODEL", default_image_embed_model),
        ner_model=os.getenv("NER_MODEL", default_ner_model),
        rerank_model=os.getenv("RERANK_MODEL", default_rerank_model),
        sparse_model=os.getenv("SPARSE_MODEL", default_sparse_model),
        text_model=os.getenv("TEXT_MODEL", default_text_model),
        vision_model=os.getenv("VISION_MODEL", default_vision_model),
    )


@dataclass(frozen=True)
class NERConfig:
    """Dataclass for information extraction configuration."""

    enabled: bool
    max_chars: int
    max_workers: int


def load_ner_env(
    default_enabled: bool = True,
    default_max_chars: int = 1024,
    default_max_workers: int = 4,
) -> NERConfig:
    """Loads information extraction configuration from environment variables or defaults.

    Args:
        default_enabled (bool): Default value to enable NER extraction. Set to True to enable by default.
        default_max_chars (int): Default maximum characters for a processed chunk for NER extraction.
        default_max_workers (int): Default maximum worker threads for NER extraction.

    Returns:
        NERConfig: Dataclass containing NER configuration.
        - enabled (bool): Whether to run entity/relation extraction during ingestion.
        - max_chars (int): Maximum characters from each node to send to the extractor.
        - max_workers (int): Maximum number of worker threads for NER extraction.

    Raises:
        ValueError: If an unsupported NER engine is specified.
    """
    return NERConfig(
        enabled=str(os.getenv("NER_ENABLED", default_enabled)).lower()
        in {"true", "1", "yes"},
        max_chars=int(os.getenv("NER_MAX_CHARS", default_max_chars)),
        max_workers=int(os.getenv("NER_MAX_WORKERS", default_max_workers)),
    )


@dataclass(frozen=True)
class OpenAIConfig:
    """Dataclass for OpenAI-compatible API configuration."""

    api_base: str
    api_key: str
    ctx_window: int
    dimensions: int | None
    max_retries: int
    num_output: int
    inference_provider: str
    reuse_client: bool
    seed: int
    temperature: float
    thinking_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"]
    thinking_enabled: bool
    timeout: float
    top_p: float


def load_openai_env(
    default_api_base: str = "http://localhost:11434/v1",
    default_api_key: str = "sk-no-key-required",
    default_ctx_window: int = 4096,
    default_dimensions: int | None = None,
    default_max_retries: int = 2,
    default_num_output: int = 256,
    default_inference_provider: Literal["ollama", "openai", "vllm"] = "ollama",
    default_reuse_client: bool = False,
    default_seed: int = 42,
    default_temperature: float = 0.0,
    default_thinking_effort: Literal[
        "none", "minimal", "low", "medium", "high", "xhigh"
    ] = "medium",
    default_thinking_enabled: bool = False,
    default_timeout: float = 300.0,
    default_top_p: float = 0.1,
) -> OpenAIConfig:
    """Loads OpenAI configuration from environment variables or defaults.

    Args:
        default_api_base (str): Default OpenAI API base URL.
        default_api_key (str): Default OpenAI API key.
        default_ctx_window (int): Default context window size for models that support it.
        default_dimensions (int | None): Optional embedding dimensions override for
            models that support reduced-dimension output.
        default_max_retries (int): Default number of retries.
        default_num_output (int): Default number of tokens reserved for the
            model response by the prompt helper.  Matches the llama_index
            ``LLMMetadata`` default (256).
        default_inference_provider (Literal['ollama', 'openai', 'vllm']): Default inference server type (e.g. "ollama", "openai", "vllm"). Default is "ollama".
        default_reuse_client (bool): Whether to reuse the OpenAI client across calls. Default is False.
        default_seed (int): Default random seed for reproducibility.
        default_temperature (float): Default temperature for text generation.
        default_thinking_effort (Literal['none', 'minimal', 'low', 'medium', 'high', 'xhigh']): Default reasoning effort to request when
            thinking is enabled.
        default_thinking_enabled (bool): Whether OpenAI reasoning/thinking is enabled.
        default_timeout (float): Default timeout in seconds.
        default_top_p (float): Default top_p for nucleus sampling.

    Returns:
        OpenAIConfig: Dataclass containing OpenAI configuration.
        - api_base (str): The OpenAI API base URL.
        - api_key (str): The OpenAI API key.
        - ctx_window (int): The context window size for models that support it.
        - dimensions (int | None): Optional embedding dimensions override for
            embedding models.
        - max_retries (int): The number of retries for API calls.
        - num_output (int): Tokens reserved for the model response.
        - inference_provider (Literal["ollama", "openai", "vllm"]): The inference server type.
        - reuse_client (bool): Whether to reuse the OpenAI client across calls.
        - seed (int): Random seed for reproducibility.
        - temperature (float): Temperature for text generation.
        - thinking_effort (Literal["none", "minimal", "low", "medium", "high", "xhigh"]):
          Reasoning effort requested for OpenAI chat completions when thinking is enabled.
        - thinking_enabled (bool): Whether OpenAI reasoning/thinking is enabled.
        - timeout (float): Timeout in seconds for API calls.
        - top_p (float): Top_p for nucleus sampling.

    Raises:
        ValueError: If an unsupported inference server is specified.
    """
    inference_provider = os.getenv(
        "INFERENCE_PROVIDER", default_inference_provider
    ).lower()
    if inference_provider not in {
        "ollama",
        "openai",
        "vllm",
    }:
        raise ValueError(
            f"Unsupported inference server: {inference_provider}. "
            f"Supported options are: 'ollama', 'openai', 'vllm'."
        )

    raw_dimensions = os.getenv("OPENAI_DIMENSIONS")
    dimensions = (
        default_dimensions
        if raw_dimensions is None or not raw_dimensions.strip()
        else int(raw_dimensions)
    )

    thinking_effort = os.getenv(
        "OPENAI_THINKING_EFFORT", default_thinking_effort
    ).lower()
    allowed_thinking_efforts = {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    }
    if thinking_effort not in allowed_thinking_efforts:
        thinking_effort = default_thinking_effort

    raw_ctx_window = os.getenv("OPENAI_CTX_WINDOW")
    ctx_window = default_ctx_window
    if raw_ctx_window is not None and raw_ctx_window.strip():
        ctx_window = int(raw_ctx_window)
    elif inference_provider == "vllm":
        raw_chat_max_model_len = os.getenv("CHAT_MAX_MODEL_LEN")
        if raw_chat_max_model_len is not None and raw_chat_max_model_len.strip():
            ctx_window = int(raw_chat_max_model_len)
        else:
            ctx_window = max(default_ctx_window, 8192)

    raw_num_output = os.getenv("OPENAI_NUM_OUTPUT")
    num_output = (
        int(raw_num_output)
        if raw_num_output is not None and raw_num_output.strip()
        else default_num_output
    )

    return OpenAIConfig(
        api_base=os.getenv("OPENAI_API_BASE", default_api_base),
        api_key=os.getenv("OPENAI_API_KEY", default_api_key),
        ctx_window=ctx_window,
        dimensions=dimensions,
        max_retries=int(os.getenv("OPENAI_MAX_RETRIES", default_max_retries)),
        num_output=num_output,
        inference_provider=inference_provider,
        reuse_client=str(os.getenv("OPENAI_REUSE_CLIENT", default_reuse_client)).lower()
        in {"true", "1", "yes"},
        seed=int(os.getenv("OPENAI_SEED", default_seed)),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", default_temperature)),
        thinking_effort=cast(
            Literal["none", "minimal", "low", "medium", "high", "xhigh"],
            thinking_effort,
        ),
        thinking_enabled=str(
            os.getenv("OPENAI_ENABLE_THINKING", default_thinking_enabled)
        ).lower()
        in {"true", "1", "yes"},
        timeout=float(os.getenv("OPENAI_TIMEOUT", default_timeout)),
        top_p=float(os.getenv("OPENAI_TOP_P", default_top_p)),
    )


@dataclass(frozen=True)
class PathConfig:
    """Dataclass for path configuration."""

    artifacts: Path
    data: Path
    docint_home_dir: Path
    logs: Path
    queries: Path
    results: Path
    prompts: Path
    qdrant_sources: Path
    hf_hub_cache: Path


def load_path_env() -> PathConfig:
    """Loads path configuration from environment variables or defaults.

    Returns:
        PathConfig: Dataclass containing path configuration.
        - artifacts (Path): Root directory for pipeline processing artifacts.
        - docint_home_dir (Path): The root home directory for docint, used as the base for other default paths. Defaults to ~/docint.
        - data (Path): Path to the data directory.
        - logs (Path): Path to the logs file.
        - queries (Path): Path to the queries file.
        - results (Path): Path to the results directory.
        - prompts (Path): Path to the prompts directory.
        - qdrant_sources (Path): Path to the Qdrant sources directory.
        - hf_hub_cache (Path): Path to the Hugging Face Hub cache directory.
    """
    home_dir: Path = Path.home()
    docint_home_dir: Path = home_dir / "docint"
    default_data_dir: Path = docint_home_dir / "data"
    default_query_dir: Path = docint_home_dir / "queries.txt"
    default_results_dir: Path = docint_home_dir / "results"
    default_model_cache: Path = home_dir / ".cache"
    default_hf_hub_cache: Path = default_model_cache / "huggingface" / "hub"

    utils_dir: Path = Path(__file__).parent.resolve()
    default_prompts_dir: Path = utils_dir / "prompts"
    project_root: Path = utils_dir.parents[1]
    default_log_dir = project_root / ".logs" / "docint.log"

    default_qdrant_sources: Path = docint_home_dir / "qdrant_sources"
    default_artifacts_dir: Path = docint_home_dir / "artifacts"

    return PathConfig(
        artifacts=Path(
            os.getenv("PIPELINE_ARTIFACTS_DIR", default_artifacts_dir)
        ).expanduser(),
        data=Path(os.getenv("DATA_PATH", default_data_dir)).expanduser(),
        docint_home_dir=docint_home_dir,
        logs=Path(os.getenv("LOG_PATH", default_log_dir)).expanduser(),
        queries=Path(os.getenv("QUERIES_PATH", default_query_dir)).expanduser(),
        results=Path(os.getenv("RESULTS_PATH", default_results_dir)).expanduser(),
        prompts=default_prompts_dir,
        qdrant_sources=Path(
            os.getenv("QDRANT_SRC_DIR", default_qdrant_sources)
        ).expanduser(),
        hf_hub_cache=Path(os.getenv("HF_HUB_CACHE", default_hf_hub_cache)).expanduser(),
    )


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the document processing pipeline."""

    text_coverage_threshold: float
    pipeline_version: str
    artifacts_dir: str
    max_retries: int
    force_reprocess: bool
    max_workers: int
    enable_vision_ocr: bool
    vision_ocr_timeout: float
    vision_ocr_max_retries: int
    vision_ocr_max_image_dimension: int
    vision_ocr_max_tokens: int


def load_pipeline_config(
    default_text_coverage_threshold: float = 0.01,
    default_pipeline_version: str = "1.0.0",
    default_artifacts_dir: str | None = None,
    default_max_retries: int = 2,
    default_force_reprocess: bool = False,
    default_max_workers: int = 4,
    default_enable_vision_ocr: bool = True,
    default_vision_ocr_timeout: float = 60.0,
    default_vision_ocr_max_retries: int = 1,
    default_vision_ocr_max_image_dimension: int = 1024,
    default_vision_ocr_max_tokens: int = 4096,
) -> PipelineConfig:
    """Build a ``PipelineConfig`` from environment variables with sensible defaults.

    Args:
        default_text_coverage_threshold (float): Default characters-per-area threshold for OCR classification.
        default_pipeline_version (str): Default pipeline version string.
        default_artifacts_dir (str | None): Default root directory for artifacts. If None, uses the value from ``load_path_env().artifacts``.
        default_max_retries (int): Default maximum retry attempts per page stage.
        default_force_reprocess (bool): Default flag to force reprocessing of pages.
        default_max_workers (int): Default maximum number of parallel workers for document processing.
        default_enable_vision_ocr (bool): Default flag to enable vision OCR fallback for scanned pages.
        default_vision_ocr_timeout (float): Default per-request timeout in seconds for vision OCR API calls.
        default_vision_ocr_max_retries (int): Default maximum retries for a single vision OCR API call.
        default_vision_ocr_max_image_dimension (int): Default maximum pixel dimension for images sent to the vision OCR endpoint.
        default_vision_ocr_max_tokens (int): Default maximum number of tokens the vision LLM may generate per OCR request.

    Returns:
        PipelineConfig: A fully-initialised ``PipelineConfig``.
        - text_coverage_threshold (float): Characters-per-area threshold for OCR classification.
        - pipeline_version (str): Semver string identifying the pipeline logic version.
        - artifacts_dir (str): Root directory for artifact output.
        - max_retries (int): Maximum retry attempts per stage on a given page.
        - force_reprocess (bool): When True, ignore existing artifacts and reprocess.
        - max_workers (int): Maximum parallel workers for document-level processing.
        - enable_vision_ocr (bool): When True, use the vision LLM as a fallback OCR engine for scanned pages that have no extractable text layer.
        - vision_ocr_timeout (float): Per-request timeout in seconds for vision OCR API calls (separate from the global ``OPENAI_TIMEOUT``).
        - vision_ocr_max_retries (int): Maximum retries for a single vision OCR API call.
        - vision_ocr_max_image_dimension (int): Maximum pixel dimension (width or height) for images sent to the vision OCR endpoint.  Larger renders are down-scaled proportionally before encoding.
        - vision_ocr_max_tokens (int): Maximum number of tokens the vision LLM may generate per OCR request.  Keeps response time bounded.
    """
    pipeline_version = os.getenv("PIPELINE_VERSION", default_pipeline_version).strip()
    if not pipeline_version:
        pipeline_version = default_pipeline_version

    return PipelineConfig(
        text_coverage_threshold=float(
            os.getenv(
                "PIPELINE_TEXT_COVERAGE_THRESHOLD", default_text_coverage_threshold
            )
        ),
        pipeline_version=pipeline_version,
        artifacts_dir=str(load_path_env().artifacts),
        max_retries=int(os.getenv("PIPELINE_MAX_RETRIES", default_max_retries)),
        force_reprocess=os.getenv("PIPELINE_FORCE_REPROCESS", "false").lower()
        in {"true", "1", "yes"},
        max_workers=int(os.getenv("PIPELINE_MAX_WORKERS", default_max_workers)),
        enable_vision_ocr=os.getenv("PIPELINE_ENABLE_VISION_OCR", "true").lower()
        in {"true", "1", "yes"},
        vision_ocr_timeout=float(
            os.getenv("PIPELINE_VISION_OCR_TIMEOUT", default_vision_ocr_timeout)
        ),
        vision_ocr_max_retries=int(
            os.getenv("PIPELINE_VISION_OCR_MAX_RETRIES", default_vision_ocr_max_retries)
        ),
        vision_ocr_max_image_dimension=int(
            os.getenv(
                "PIPELINE_VISION_OCR_MAX_IMAGE_DIM",
                default_vision_ocr_max_image_dimension,
            )
        ),
        vision_ocr_max_tokens=int(
            os.getenv("PIPELINE_VISION_OCR_MAX_TOKENS", default_vision_ocr_max_tokens)
        ),
    )


@dataclass(frozen=True)
class ResponseValidationConfig:
    """Dataclass for response validation configuration."""

    enabled: bool


def load_response_validation_env(
    default_enabled: bool = True,
) -> ResponseValidationConfig:
    """Load response-validation configuration from environment variables.

    Args:
        default_enabled (bool): Whether response validation is enabled by default.

    Returns:
        ResponseValidationConfig: Parsed response-validation settings.
        - enabled (bool): Whether to run response validation on generated answers.
    """
    return ResponseValidationConfig(
        enabled=str(os.getenv("RESPONSE_VALIDATION_ENABLED", default_enabled)).lower()
        in {"true", "1", "yes"}
    )


@dataclass(frozen=True)
class RetrievalConfig:
    """Dataclass for RAG (Retrieval-Augmented Generation) configuration."""

    rerank_use_fp16: bool
    retrieve_top_k: int
    chat_response_mode: Literal["auto", "compact", "refine"]
    vector_store_query_mode: Literal["auto", "default", "sparse", "hybrid", "mmr"]
    hybrid_alpha: float
    sparse_top_k: int
    hybrid_top_k: int
    parent_context_enabled: bool
    parent_context_safety_margin: float


def load_retrieval_env(
    default_rerank_use_fp16: bool = False,
    default_retrieve_top_k: int = 20,
    default_chat_response_mode: Literal["auto", "compact", "refine"] = "auto",
    default_vector_store_query_mode: Literal[
        "auto", "default", "sparse", "hybrid", "mmr"
    ] = "auto",
    default_hybrid_alpha: float = 0.5,
    default_sparse_top_k: int = 20,
    default_hybrid_top_k: int = 20,
    default_parent_context_enabled: bool = True,
    default_parent_context_safety_margin: float = 0.95,
) -> RetrievalConfig:
    """Loads retrieval configuration from environment variables or defaults.

    Args:
        default_rerank_use_fp16 (bool): Default flag to use FP16 for reranker model. Default is False.
        default_retrieve_top_k (int): Default number of top documents to retrieve.
        default_chat_response_mode (Literal["auto", "compact", "refine"]): Default response synthesizer mode for chat/query answers. Default is "auto".
        default_vector_store_query_mode (Literal["auto", "default", "sparse", "hybrid", "mmr"]): Default retrieval mode to use for vector store queries. Default is "auto".
        default_hybrid_alpha (float): Default dense-vs-sparse fusion weight for hybrid search. Value should be in [0.0, 1.0]. Default is 0.5.
        default_sparse_top_k (int): Default candidate depth for sparse retrieval in hybrid/sparse modes. Default is 20.
        default_hybrid_top_k (int): Default final candidate depth after dense/sparse fusion. Default is 20.
        default_parent_context_enabled (bool): Default flag to enable hierarchical parent context retrieval. Default is True.
        default_parent_context_safety_margin (float): Default fraction of
            ``OPENAI_CTX_WINDOW`` the chat-budget packer may consume when
            deciding whether to emit a full parent or a windowed slice.
            Reserves headroom for the provider's own BOS/EOS and rough
            tokenizer-estimate drift. Must fall in ``(0, 1]``. Default 0.95.

    Returns:
        RetrievalConfig: Dataclass containing retrieval configuration.
        - rerank_use_fp16 (bool): Whether to use FP16 for the reranker model.
        - retrieve_top_k (int): The number of top documents to retrieve for RAG
        - chat_response_mode (Literal["auto", "compact", "refine"]): The
          response synthesizer mode for chat/query answers.
                - vector_store_query_mode (Literal["auto", "default", "sparse", "hybrid", "mmr"]):
                    Retrieval mode to use for vector store queries.
                - hybrid_alpha (float): Dense-vs-sparse fusion weight for hybrid search.
                - sparse_top_k (int): Candidate depth for sparse retrieval in hybrid/sparse modes.
                - hybrid_top_k (int): Final candidate depth after dense/sparse fusion.
                - parent_context_enabled (bool): Whether fine-grained matches should expand
                    to their hierarchical parent context when available.
                - parent_context_safety_margin (float): Fraction of
                    ``OPENAI_CTX_WINDOW`` the parent-context packer may
                    consume before windowing oversize parents.
    """
    raw_mode = (
        str(os.getenv("CHAT_RESPONSE_MODE", default_chat_response_mode)).strip().lower()
    )
    chat_response_mode: Literal["auto", "compact", "refine"] = "auto"
    if raw_mode in {"compact", "refine"}:
        chat_response_mode = cast(Literal["compact", "refine"], raw_mode)
    elif raw_mode == "auto":
        chat_response_mode = "auto"

    raw_query_mode = (
        str(os.getenv("RETRIEVAL_VECTOR_QUERY_MODE", default_vector_store_query_mode))
        .strip()
        .lower()
    )
    vector_store_query_mode: Literal["auto", "default", "sparse", "hybrid", "mmr"] = (
        "auto"
    )
    if raw_query_mode in {"default", "sparse", "hybrid", "mmr"}:
        vector_store_query_mode = cast(
            Literal["default", "sparse", "hybrid", "mmr"],
            raw_query_mode,
        )
    elif raw_query_mode == "auto":
        vector_store_query_mode = "auto"

    return RetrievalConfig(
        rerank_use_fp16=str(
            os.getenv("RERANK_USE_FP16", default_rerank_use_fp16)
        ).lower()
        in {"true", "1", "yes"},
        retrieve_top_k=int(os.getenv("RETRIEVE_TOP_K", default_retrieve_top_k)),
        chat_response_mode=chat_response_mode,
        vector_store_query_mode=vector_store_query_mode,
        hybrid_alpha=min(
            1.0,
            max(
                0.0,
                float(os.getenv("RETRIEVAL_HYBRID_ALPHA", default_hybrid_alpha)),
            ),
        ),
        sparse_top_k=max(
            1,
            int(os.getenv("RETRIEVAL_SPARSE_TOP_K", default_sparse_top_k)),
        ),
        hybrid_top_k=max(
            1,
            int(os.getenv("RETRIEVAL_HYBRID_TOP_K", default_hybrid_top_k)),
        ),
        parent_context_enabled=str(
            os.getenv(
                "PARENT_CONTEXT_RETRIEVAL_ENABLED",
                default_parent_context_enabled,
            )
        ).lower()
        in {"true", "1", "yes"},
        # Warn-and-fallback (not raise) — query-time tuning knob. A stray
        # out-of-range value should keep the app running with the safe
        # default rather than blocking startup. See
        # :func:`_parse_parent_context_safety_margin`.
        parent_context_safety_margin=_parse_parent_context_safety_margin(
            default=default_parent_context_safety_margin,
        ),
    )


def _parse_parent_context_safety_margin(*, default: float) -> float:
    """Parse ``PARENT_CONTEXT_SAFETY_MARGIN`` and clamp it to ``(0, 1]``.

    The query-time parent-context packer reserves
    ``(1 - safety_margin)`` of the chat context window for provider-side
    BOS/EOS tokens and char/token-estimator drift. Values outside
    ``(0, 1]`` would either disable the guard (``>= 1`` leaves no
    headroom, defeating the purpose) or starve the prompt (``<= 0``
    yields zero usable tokens and blocks all queries). The fallback
    ``default`` is used when the env var is unset, malformed, or out of
    range — we log the override rather than raising so a stray operator
    typo doesn't brick ingest.

    Args:
        default: The value returned when the env var is absent,
            malformed, or out of range.

    Returns:
        A float in ``(0, 1]``.
    """
    raw = os.getenv("PARENT_CONTEXT_SAFETY_MARGIN")
    if raw is None or not raw.strip():
        return default
    try:
        parsed = float(raw)
    except ValueError:
        logger.warning(
            "PARENT_CONTEXT_SAFETY_MARGIN={!r} is not a float — using default {}",
            raw,
            default,
        )
        return default
    if not (0.0 < parsed <= 1.0):
        logger.warning(
            "PARENT_CONTEXT_SAFETY_MARGIN={!r} is out of range (0, 1] — using default {}",
            raw,
            default,
        )
        return default
    return parsed


@dataclass(frozen=True)
class RuntimeConfig:
    """Dataclass for local runtime device preferences."""

    use_device: str


def load_runtime_env(default_use_device: str = "auto") -> RuntimeConfig:
    """Load local runtime settings from environment variables.

    Args:
        default_use_device (str): Preferred device for local auxiliary models.
            Supported values are ``auto``, ``cpu``, ``mps``, ``cuda``, and
            ``cuda:<index>``.

    Returns:
        RuntimeConfig: Parsed runtime configuration.
    """
    normalized_default = default_use_device.strip().lower() or "auto"
    requested_device = str(os.getenv("USE_DEVICE", normalized_default)).strip().lower()
    if not requested_device:
        requested_device = normalized_default

    is_supported = requested_device in {
        "auto",
        "cpu",
        "cuda",
        "mps",
    } or requested_device.startswith("cuda:")
    if not is_supported:
        requested_device = normalized_default

    return RuntimeConfig(use_device=requested_device)


@dataclass(frozen=True)
class SessionConfig:
    """Dataclass for session configuration."""

    session_store: str


def load_session_env(
    default_session_store: str | None = None,
) -> SessionConfig:
    """Loads session configuration from environment variables or defaults.

    Args:
        default_session_store (str): Default session store configuration (e.g. database URL or file path).
            Default is ``sqlite:///{Path.home() / "docint" / "sessions.sqlite3"}``.

    Returns:
        SessionConfig: Dataclass containing session configuration.
        - session_store (str): The session store configuration. Default is
          ``sqlite:///{Path.home() / "docint" / "sessions.sqlite3"}`` unless
          ``SESSION_STORE`` is explicitly configured.
    """
    session_store_override = os.getenv("SESSION_STORE")
    if session_store_override:
        return SessionConfig(session_store=session_store_override)

    if default_session_store is None:
        default_db_path = Path.home() / "docint" / "sessions.sqlite3"
        db_path = Path(os.getenv("SESSIONS_DB_PATH", default_db_path)).expanduser()
        default_session_store = f"sqlite:///{db_path}"

    return SessionConfig(session_store=default_session_store)


@dataclass(frozen=True)
class SummaryConfig:
    """Dataclass for collection summarization precision settings."""

    coverage_target: float
    max_docs: int
    per_doc_top_k: int
    final_source_cap: int
    social_chunking_enabled: bool
    social_candidate_pool: int
    social_diversity_limit: int


def load_summary_env(
    default_coverage_target: float = 0.70,
    default_max_docs: int = 30,
    default_per_doc_top_k: int = 4,
    default_final_source_cap: int = 24,
    default_social_chunking_enabled: bool = True,
    default_social_candidate_pool: int = 48,
    default_social_diversity_limit: int = 2,
) -> SummaryConfig:
    """Load collection summary precision settings from environment variables.

    Args:
        default_coverage_target (float): Target minimum document coverage ratio.
        default_max_docs (int): Maximum number of documents sampled for summary.
        default_per_doc_top_k (int): Maximum evidence chunks retrieved per document.
        default_final_source_cap (int): Maximum number of merged summary sources.

    Returns:
        SummaryConfig: Parsed summary precision settings.
        - coverage_target (float): Target minimum document coverage ratio for summaries. Value is clamped to [0.0, 1.0].
        - max_docs (int): Maximum number of documents to sample for summarization.
        - per_doc_top_k (int): Maximum number of evidence chunks to retrieve per document.
        - final_source_cap (int): Maximum number of merged sources to include in the final summary answer to keep it concise and focused.
        - social_chunking_enabled (bool): Whether row-heavy social/table
          collections should use chunk/post-level summarization.
        - social_candidate_pool (int): Candidate retrieval depth for social/table
          collection summaries.
        - social_diversity_limit (int): Maximum number of sources retained per
          diversity bucket during social/table collection summaries.
    """
    raw_target = float(os.getenv("SUMMARY_COVERAGE_TARGET", default_coverage_target))
    target = min(1.0, max(0.0, raw_target))
    return SummaryConfig(
        coverage_target=target,
        max_docs=max(1, int(os.getenv("SUMMARY_MAX_DOCS", default_max_docs))),
        per_doc_top_k=max(
            1, int(os.getenv("SUMMARY_PER_DOC_TOP_K", default_per_doc_top_k))
        ),
        final_source_cap=max(
            1, int(os.getenv("SUMMARY_FINAL_SOURCE_CAP", default_final_source_cap))
        ),
        social_chunking_enabled=str(
            os.getenv(
                "SUMMARY_SOCIAL_CHUNKING_ENABLED",
                default_social_chunking_enabled,
            )
        ).lower()
        in {"true", "1", "yes"},
        social_candidate_pool=max(
            1,
            int(
                os.getenv(
                    "SUMMARY_SOCIAL_CANDIDATE_POOL",
                    default_social_candidate_pool,
                )
            ),
        ),
        social_diversity_limit=max(
            1,
            int(
                os.getenv(
                    "SUMMARY_SOCIAL_DIVERSITY_LIMIT",
                    default_social_diversity_limit,
                )
            ),
        ),
    )
