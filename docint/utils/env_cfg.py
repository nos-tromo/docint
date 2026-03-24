"""Centralized TOML-backed application configuration."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from dotenv import load_dotenv
from loguru import logger

Role = Literal["backend", "frontend", "worker"]

DEFAULT_PIPELINE_VERSION = "1.0.0"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.toml"
DEFAULT_PROFILE = "local-ollama"
PROFILE_ENV_VAR = "DOCINT_PROFILE"

_TRUTHY_VALUES = {"1", "true", "yes"}
_VALID_ROLES: set[Role] = {"backend", "frontend", "worker"}
_VALID_PROVIDERS = {"ollama", "openai", "vllm"}
_VALID_THINKING_EFFORTS = {
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
}
_ACTIVE_PROFILE: str | None = None
_ACTIVE_ROLE: Role | None = None


def _is_truthy(value: Any) -> bool:
    """Return whether *value* matches an accepted truthy token."""

    return str(value).strip().lower() in _TRUTHY_VALUES


def _as_table(value: Any, *, label: str) -> dict[str, Any]:
    """Return *value* as a plain dict when it is a TOML table."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a TOML table.")
    return dict(value)


def _read_config_document() -> dict[str, Any]:
    """Read the root ``config.toml`` document."""

    if not DEFAULT_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Config file not found at '{DEFAULT_CONFIG_PATH}'. Expected repo-root config.toml."
        )
    with open(DEFAULT_CONFIG_PATH, "rb") as handle:
        return cast(dict[str, Any], tomllib.load(handle))


def _resolve_profile(document: Mapping[str, Any], profile: str | None) -> str:
    """Resolve the active config profile from env, bootstrap state, or TOML."""

    active_profile = (
        profile
        or os.getenv(PROFILE_ENV_VAR)
        or _ACTIVE_PROFILE
        or str(document.get("active_profile") or "").strip()
        or DEFAULT_PROFILE
    )
    profiles = document.get("profiles")
    if not isinstance(profiles, Mapping) or active_profile not in profiles:
        available = (
            ", ".join(sorted(cast(Mapping[str, Any], profiles).keys()))
            if isinstance(profiles, Mapping)
            else ""
        )
        raise ValueError(
            f"Config profile '{active_profile}' is not defined in '{DEFAULT_CONFIG_PATH}'."
            + (f" Available profiles: {available}." if available else "")
        )
    return active_profile


def _resolve_role(role: Role | None) -> Role:
    """Resolve the active runtime role."""

    resolved = role or _ACTIVE_ROLE or cast(Role, "backend")
    if resolved not in _VALID_ROLES:
        raise ValueError(f"Unsupported config role '{resolved}'.")
    return resolved


def _merge_tables(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge TOML table-like mappings."""

    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_tables(
                cast(Mapping[str, Any], merged[key]),
                cast(Mapping[str, Any], value),
            )
        else:
            merged[key] = value
    return merged


def _default_sections() -> dict[str, dict[str, Any]]:
    """Return fallback section values for omitted TOML fields."""

    home_dir = Path.home()
    docint_home_dir = home_dir / "docint"
    project_root = Path(__file__).resolve().parents[2]
    return {
        "runtime": {
            "docint_offline": True,
            "preload_models": False,
        },
        "frontend": {
            "collection_timeout": 120,
        },
        "graphrag": {
            "enabled": True,
            "neighbor_hops": 2,
            "top_k_nodes": 50,
            "min_edge_weight": 3,
            "max_neighbors": 6,
        },
        "hate_speech": {
            "enabled": False,
            "max_chars": 2048,
            "max_workers": 1,
        },
        "image_ingestion": {
            "enabled": True,
            "embedding_enabled": True,
            "tagging_enabled": True,
            "collection_name": "{collection}_images",
            "vector_name": "image-dense",
            "cache_by_hash": True,
            "fail_on_embedding_error": False,
            "fail_on_tagging_error": False,
            "retrieve_top_k": 5,
            "tagging_max_image_dimension": 1024,
        },
        "ingestion": {
            "coarse_chunk_size": 8192,
            "docling_accelerator_num_threads": 4,
            "docstore_batch_size": 100,
            "ingest_benchmark_enabled": False,
            "docstore_max_retries": 3,
            "docstore_retry_backoff_seconds": 0.25,
            "docstore_retry_backoff_max_seconds": 2.0,
            "fine_chunk_overlap": 0,
            "fine_chunk_size": 8192,
            "hierarchical_chunking_enabled": True,
            "ingestion_batch_size": 50,
            "sentence_splitter_chunk_overlap": 64,
            "sentence_splitter_chunk_size": 1024,
            "supported_filetypes": [
                ".avi",
                ".csv",
                ".docx",
                ".flv",
                ".gif",
                ".jpeg",
                ".jpg",
                ".jsonl",
                ".md",
                ".mkv",
                ".mov",
                ".mpeg",
                ".mpg",
                ".mp3",
                ".mp4",
                ".m4a",
                ".m4v",
                ".ogg",
                ".parquet",
                ".pdf",
                ".png",
                ".tsv",
                ".txt",
                ".wav",
                ".webm",
                ".wmv",
                ".xls",
                ".xlsx",
            ],
        },
        "models": {
            "embed_model": "bge-m3",
            "image_embed_model": "openai/clip-vit-base-patch32",
            "ner_model": "gliner-community/gliner_large-v2.5",
            "rerank_model": "BAAI/bge-reranker-v2-m3",
            "sparse_model": "Qdrant/all_miniLM_L6_v2_with_attentions",
            "text_model": "gpt-oss:20b",
            "vision_model": "qwen3.5:9b",
            "whisper_model": "turbo",
        },
        "inference": {
            "api_base": "http://localhost:11434/v1",
            "ctx_window": 4096,
            "dimensions": None,
            "max_retries": 2,
            "provider": "ollama",
            "reuse_client": False,
            "seed": 42,
            "temperature": 0.0,
            "thinking_effort": "medium",
            "thinking_enabled": False,
            "timeout": 300.0,
            "top_p": 0.1,
        },
        "paths": {
            "artifacts": str(docint_home_dir / "artifacts"),
            "data": str(docint_home_dir / "data"),
            "docint_home_dir": str(docint_home_dir),
            "logs": str(project_root / ".logs" / "docint.log"),
            "queries": str(docint_home_dir / "queries.txt"),
            "results": str(docint_home_dir / "results"),
            "qdrant_sources": str(docint_home_dir / "qdrant_sources"),
            "hf_hub_cache": str(home_dir / ".cache" / "huggingface" / "hub"),
        },
        "hosts": {
            "backend_host": "http://localhost:8000",
            "backend_public_host": "http://localhost:8000",
            "qdrant_host": "http://localhost:6333",
            "cors_allowed_origins": "http://localhost:8501,http://127.0.0.1:8501",
        },
        "ner": {
            "enabled": True,
            "max_chars": 1024,
            "max_workers": 4,
        },
        "pipeline": {
            "text_coverage_threshold": 0.01,
            "pipeline_version": DEFAULT_PIPELINE_VERSION,
            "max_retries": 2,
            "force_reprocess": False,
            "max_workers": 4,
            "enable_vision_ocr": True,
            "vision_ocr_timeout": 60.0,
            "vision_ocr_max_retries": 1,
            "vision_ocr_max_image_dimension": 1024,
            "vision_ocr_max_tokens": 4096,
        },
        "response_validation": {
            "enabled": True,
        },
        "retrieval": {
            "rerank_use_fp16": False,
            "retrieve_top_k": 20,
            "chat_response_mode": "auto",
            "vector_store_query_mode": "auto",
            "hybrid_alpha": 0.5,
            "sparse_top_k": 20,
            "hybrid_top_k": 20,
            "parent_context_enabled": True,
        },
        "session": {
            "session_store": "",
        },
        "summary": {
            "coverage_target": 0.70,
            "max_docs": 30,
            "per_doc_top_k": 4,
            "final_source_cap": 24,
            "social_chunking_enabled": True,
            "social_candidate_pool": 48,
            "social_diversity_limit": 2,
        },
        "whisper": {
            "max_workers": 1,
            "task": "transcribe",
        },
    }


def _resolve_sections(
    document: Mapping[str, Any],
    profile: str,
    role: Role,
) -> dict[str, dict[str, Any]]:
    """Resolve merged shared and role-specific config sections."""

    profiles = _as_table(document.get("profiles"), label="profiles")
    profile_table = _as_table(profiles.get(profile), label=f"profiles.{profile}")
    shared = _as_table(
        profile_table.get("shared", {}), label=f"profiles.{profile}.shared"
    )
    role_table = _as_table(
        profile_table.get(role, {}),
        label=f"profiles.{profile}.{role}",
    )
    defaults = _default_sections()
    merged = _merge_tables(defaults, shared)
    merged = _merge_tables(merged, role_table)
    return {key: _as_table(value, label=key) for key, value in merged.items()}


def _path(value: Any) -> Path:
    """Convert TOML path values into expanded ``Path`` objects."""

    raw_value = str(value)
    expanded = raw_value.replace(
        "${PROJECT_ROOT}", str(DEFAULT_CONFIG_PATH.parent)
    ).replace("${HOME}", str(Path.home()))
    return Path(expanded).expanduser()


def _string(section: Mapping[str, Any], key: str) -> str:
    """Read a required string field."""

    value = section.get(key)
    if value is None:
        raise ValueError(f"Missing config value '{key}'.")
    return str(value)


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration."""

    profile: str
    role: Role
    docint_offline: bool
    preload_models: bool


@dataclass(frozen=True)
class FrontendConfig:
    """Frontend configuration."""

    collection_timeout: int


@dataclass(frozen=True)
class GraphRAGConfig:
    """Graph-assisted retrieval configuration."""

    enabled: bool
    neighbor_hops: int
    top_k_nodes: int
    min_edge_weight: int
    max_neighbors: int


@dataclass(frozen=True)
class HateSpeechConfig:
    """Hate-speech detection configuration."""

    enabled: bool
    max_chars: int
    max_workers: int


@dataclass(frozen=True)
class HostConfig:
    """Host configuration."""

    backend_host: str
    backend_public_host: str
    qdrant_host: str
    cors_allowed_origins: str


@dataclass(frozen=True)
class ImageIngestionConfig:
    """Image-ingestion configuration."""

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


@dataclass(frozen=True)
class IngestionConfig:
    """Ingestion configuration."""

    coarse_chunk_size: int
    docling_accelerator_num_threads: int
    docstore_batch_size: int
    ingest_benchmark_enabled: bool
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


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration."""

    embed_model: str
    image_embed_model: str
    ner_model: str
    rerank_model: str
    sparse_model: str
    text_model: str
    vision_model: str
    whisper_model: str


@dataclass(frozen=True)
class NERConfig:
    """NER configuration."""

    enabled: bool
    max_chars: int
    max_workers: int


@dataclass(frozen=True)
class OpenAIConfig:
    """OpenAI-compatible inference configuration."""

    api_base: str
    api_key: str
    ctx_window: int
    dimensions: int | None
    max_retries: int
    inference_provider: str
    reuse_client: bool
    seed: int
    temperature: float
    thinking_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"]
    thinking_enabled: bool
    timeout: float
    top_p: float


@dataclass(frozen=True)
class PathConfig:
    """Path configuration."""

    artifacts: Path
    data: Path
    docint_home_dir: Path
    logs: Path
    queries: Path
    results: Path
    prompts: Path
    qdrant_sources: Path
    hf_hub_cache: Path


@dataclass(frozen=True)
class PipelineConfig:
    """Document pipeline configuration."""

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


@dataclass(frozen=True)
class ResponseValidationConfig:
    """Response validation configuration."""

    enabled: bool


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval configuration."""

    rerank_use_fp16: bool
    retrieve_top_k: int
    chat_response_mode: Literal["auto", "compact", "refine"]
    vector_store_query_mode: Literal["auto", "default", "sparse", "hybrid", "mmr"]
    hybrid_alpha: float
    sparse_top_k: int
    hybrid_top_k: int
    parent_context_enabled: bool


@dataclass(frozen=True)
class SessionConfig:
    """Session-store configuration."""

    session_store: str


@dataclass(frozen=True)
class SummaryConfig:
    """Collection-summary configuration."""

    coverage_target: float
    max_docs: int
    per_doc_top_k: int
    final_source_cap: int
    social_chunking_enabled: bool
    social_candidate_pool: int
    social_diversity_limit: int


@dataclass(frozen=True)
class WhisperConfig:
    """Whisper configuration."""

    max_workers: int
    task: Literal["transcribe", "translate"]


@dataclass(frozen=True)
class AppConfig:
    """Canonical application configuration."""

    runtime: RuntimeConfig
    frontend: FrontendConfig
    graphrag: GraphRAGConfig
    hate_speech: HateSpeechConfig
    hosts: HostConfig
    image_ingestion: ImageIngestionConfig
    ingestion: IngestionConfig
    inference: OpenAIConfig
    models: ModelConfig
    ner: NERConfig
    paths: PathConfig
    pipeline: PipelineConfig
    response_validation: ResponseValidationConfig
    retrieval: RetrievalConfig
    session: SessionConfig
    summary: SummaryConfig
    whisper: WhisperConfig


def _build_runtime_config(
    section: Mapping[str, Any],
    *,
    profile: str,
    role: Role,
) -> RuntimeConfig:
    """Build the runtime config section."""

    return RuntimeConfig(
        profile=profile,
        role=role,
        docint_offline=_is_truthy(section.get("docint_offline", True)),
        preload_models=_is_truthy(section.get("preload_models", False)),
    )


def _build_frontend_config(section: Mapping[str, Any]) -> FrontendConfig:
    """Build the frontend config section."""

    return FrontendConfig(
        collection_timeout=int(section.get("collection_timeout", 120))
    )


def _build_graphrag_config(section: Mapping[str, Any]) -> GraphRAGConfig:
    """Build the GraphRAG config section."""

    return GraphRAGConfig(
        enabled=_is_truthy(section.get("enabled", True)),
        neighbor_hops=max(1, int(section.get("neighbor_hops", 2))),
        top_k_nodes=max(1, int(section.get("top_k_nodes", 50))),
        min_edge_weight=max(1, int(section.get("min_edge_weight", 3))),
        max_neighbors=max(1, int(section.get("max_neighbors", 6))),
    )


def _build_hate_speech_config(section: Mapping[str, Any]) -> HateSpeechConfig:
    """Build the hate-speech config section."""

    return HateSpeechConfig(
        enabled=_is_truthy(section.get("enabled", False)),
        max_chars=max(256, int(section.get("max_chars", 2048))),
        max_workers=max(1, int(section.get("max_workers", 1))),
    )


def _build_host_config(section: Mapping[str, Any]) -> HostConfig:
    """Build the host config section."""

    backend_host = _string(section, "backend_host")
    backend_public_host = _string(section, "backend_public_host") or backend_host
    return HostConfig(
        backend_host=backend_host,
        backend_public_host=backend_public_host,
        qdrant_host=_string(section, "qdrant_host"),
        cors_allowed_origins=_string(section, "cors_allowed_origins"),
    )


def _build_image_ingestion_config(section: Mapping[str, Any]) -> ImageIngestionConfig:
    """Build the image-ingestion config section."""

    return ImageIngestionConfig(
        enabled=_is_truthy(section.get("enabled", True)),
        embedding_enabled=_is_truthy(section.get("embedding_enabled", True)),
        tagging_enabled=_is_truthy(section.get("tagging_enabled", True)),
        collection_name=_string(section, "collection_name"),
        vector_name=_string(section, "vector_name"),
        cache_by_hash=_is_truthy(section.get("cache_by_hash", True)),
        fail_on_embedding_error=_is_truthy(
            section.get("fail_on_embedding_error", False)
        ),
        fail_on_tagging_error=_is_truthy(section.get("fail_on_tagging_error", False)),
        retrieve_top_k=int(section.get("retrieve_top_k", 5)),
        tagging_max_image_dimension=int(
            section.get("tagging_max_image_dimension", 1024)
        ),
    )


def _build_ingestion_config(section: Mapping[str, Any]) -> IngestionConfig:
    """Build the ingestion config section."""

    return IngestionConfig(
        coarse_chunk_size=int(section.get("coarse_chunk_size", 8192)),
        docling_accelerator_num_threads=int(
            section.get("docling_accelerator_num_threads", 4)
        ),
        docstore_batch_size=int(section.get("docstore_batch_size", 100)),
        ingest_benchmark_enabled=_is_truthy(
            section.get("ingest_benchmark_enabled", False)
        ),
        docstore_max_retries=max(0, int(section.get("docstore_max_retries", 3))),
        docstore_retry_backoff_max_seconds=max(
            0.0, float(section.get("docstore_retry_backoff_max_seconds", 2.0))
        ),
        docstore_retry_backoff_seconds=max(
            0.0, float(section.get("docstore_retry_backoff_seconds", 0.25))
        ),
        fine_chunk_overlap=int(section.get("fine_chunk_overlap", 0)),
        fine_chunk_size=int(section.get("fine_chunk_size", 8192)),
        hierarchical_chunking_enabled=_is_truthy(
            section.get("hierarchical_chunking_enabled", True)
        ),
        ingestion_batch_size=int(section.get("ingestion_batch_size", 50)),
        sentence_splitter_chunk_overlap=int(
            section.get("sentence_splitter_chunk_overlap", 64)
        ),
        sentence_splitter_chunk_size=int(
            section.get("sentence_splitter_chunk_size", 1024)
        ),
        supported_filetypes=[
            str(item) for item in section.get("supported_filetypes", [])
        ],
    )


def _build_model_config(section: Mapping[str, Any]) -> ModelConfig:
    """Build the model config section."""

    return ModelConfig(
        embed_model=_string(section, "embed_model"),
        image_embed_model=_string(section, "image_embed_model"),
        ner_model=_string(section, "ner_model"),
        rerank_model=_string(section, "rerank_model"),
        sparse_model=_string(section, "sparse_model"),
        text_model=_string(section, "text_model"),
        vision_model=_string(section, "vision_model"),
        whisper_model=_string(section, "whisper_model"),
    )


def _build_ner_config(section: Mapping[str, Any]) -> NERConfig:
    """Build the NER config section."""

    return NERConfig(
        enabled=_is_truthy(section.get("enabled", True)),
        max_chars=int(section.get("max_chars", 1024)),
        max_workers=int(section.get("max_workers", 4)),
    )


def _build_inference_config(section: Mapping[str, Any]) -> OpenAIConfig:
    """Build the OpenAI-compatible inference config section."""

    provider = _string(section, "provider").lower()
    if provider not in _VALID_PROVIDERS:
        raise ValueError(
            f"Unsupported inference provider '{provider}'. Expected one of {sorted(_VALID_PROVIDERS)}."
        )
    thinking_effort = _string(section, "thinking_effort").lower()
    if thinking_effort not in _VALID_THINKING_EFFORTS:
        raise ValueError(
            f"Unsupported thinking_effort '{thinking_effort}'. Expected one of {sorted(_VALID_THINKING_EFFORTS)}."
        )
    dimensions_value = section.get("dimensions")
    return OpenAIConfig(
        api_base=_string(section, "api_base"),
        api_key=os.getenv("OPENAI_API_KEY", "sk-no-key-required"),
        ctx_window=int(section.get("ctx_window", 4096)),
        dimensions=(
            None
            if dimensions_value in {None, ""}
            else int(cast(int | str, dimensions_value))
        ),
        max_retries=int(section.get("max_retries", 2)),
        inference_provider=provider,
        reuse_client=_is_truthy(section.get("reuse_client", False)),
        seed=int(section.get("seed", 42)),
        temperature=float(section.get("temperature", 0.0)),
        thinking_effort=cast(
            Literal["none", "minimal", "low", "medium", "high", "xhigh"],
            thinking_effort,
        ),
        thinking_enabled=_is_truthy(section.get("thinking_enabled", False)),
        timeout=float(section.get("timeout", 300.0)),
        top_p=float(section.get("top_p", 0.1)),
    )


def _build_path_config(section: Mapping[str, Any]) -> PathConfig:
    """Build the path config section."""

    prompts_dir = Path(__file__).resolve().parent / "prompts"
    return PathConfig(
        artifacts=_path(section.get("artifacts")),
        data=_path(section.get("data")),
        docint_home_dir=_path(section.get("docint_home_dir")),
        logs=_path(section.get("logs")),
        queries=_path(section.get("queries")),
        results=_path(section.get("results")),
        prompts=prompts_dir,
        qdrant_sources=_path(section.get("qdrant_sources")),
        hf_hub_cache=_path(section.get("hf_hub_cache")),
    )


def _build_pipeline_config(
    section: Mapping[str, Any],
    *,
    artifacts_dir: Path,
    default_pipeline_version: str = DEFAULT_PIPELINE_VERSION,
) -> PipelineConfig:
    """Build the pipeline config section."""

    pipeline_version = (
        _string(section, "pipeline_version").strip() or default_pipeline_version
    )
    return PipelineConfig(
        text_coverage_threshold=float(section.get("text_coverage_threshold", 0.01)),
        pipeline_version=pipeline_version,
        artifacts_dir=str(artifacts_dir),
        max_retries=int(section.get("max_retries", 2)),
        force_reprocess=_is_truthy(section.get("force_reprocess", False)),
        max_workers=int(section.get("max_workers", 4)),
        enable_vision_ocr=_is_truthy(section.get("enable_vision_ocr", True)),
        vision_ocr_timeout=float(section.get("vision_ocr_timeout", 60.0)),
        vision_ocr_max_retries=int(section.get("vision_ocr_max_retries", 1)),
        vision_ocr_max_image_dimension=int(
            section.get("vision_ocr_max_image_dimension", 1024)
        ),
        vision_ocr_max_tokens=int(section.get("vision_ocr_max_tokens", 4096)),
    )


def _build_response_validation_config(
    section: Mapping[str, Any],
) -> ResponseValidationConfig:
    """Build the response-validation config section."""

    return ResponseValidationConfig(enabled=_is_truthy(section.get("enabled", True)))


def _build_retrieval_config(section: Mapping[str, Any]) -> RetrievalConfig:
    """Build the retrieval config section."""

    chat_response_mode = _string(section, "chat_response_mode").lower()
    if chat_response_mode not in {"auto", "compact", "refine"}:
        raise ValueError(f"Unsupported chat_response_mode '{chat_response_mode}'.")
    vector_query_mode = _string(section, "vector_store_query_mode").lower()
    if vector_query_mode not in {"auto", "default", "sparse", "hybrid", "mmr"}:
        raise ValueError(f"Unsupported vector_store_query_mode '{vector_query_mode}'.")
    return RetrievalConfig(
        rerank_use_fp16=_is_truthy(section.get("rerank_use_fp16", False)),
        retrieve_top_k=int(section.get("retrieve_top_k", 20)),
        chat_response_mode=cast(
            Literal["auto", "compact", "refine"],
            chat_response_mode,
        ),
        vector_store_query_mode=cast(
            Literal["auto", "default", "sparse", "hybrid", "mmr"],
            vector_query_mode,
        ),
        hybrid_alpha=min(1.0, max(0.0, float(section.get("hybrid_alpha", 0.5)))),
        sparse_top_k=max(1, int(section.get("sparse_top_k", 20))),
        hybrid_top_k=max(1, int(section.get("hybrid_top_k", 20))),
        parent_context_enabled=_is_truthy(section.get("parent_context_enabled", True)),
    )


def _build_session_config(
    section: Mapping[str, Any],
    *,
    paths: PathConfig,
) -> SessionConfig:
    """Build the session config section."""

    session_store = str(section.get("session_store") or "").strip()
    if not session_store:
        if paths.data == paths.docint_home_dir / "data":
            session_store = f"sqlite:///{paths.docint_home_dir / 'sessions.db'}"
        else:
            session_store = f"sqlite:///{paths.data / 'sessions.db'}"
    return SessionConfig(session_store=session_store)


def _build_summary_config(section: Mapping[str, Any]) -> SummaryConfig:
    """Build the summary config section."""

    return SummaryConfig(
        coverage_target=min(1.0, max(0.0, float(section.get("coverage_target", 0.70)))),
        max_docs=max(1, int(section.get("max_docs", 30))),
        per_doc_top_k=max(1, int(section.get("per_doc_top_k", 4))),
        final_source_cap=max(1, int(section.get("final_source_cap", 24))),
        social_chunking_enabled=_is_truthy(
            section.get("social_chunking_enabled", True)
        ),
        social_candidate_pool=max(1, int(section.get("social_candidate_pool", 48))),
        social_diversity_limit=max(1, int(section.get("social_diversity_limit", 2))),
    )


def _build_whisper_config(section: Mapping[str, Any]) -> WhisperConfig:
    """Build the whisper config section."""

    task = _string(section, "task").strip().lower()
    if task not in {"transcribe", "translate"}:
        raise ValueError(f"Unsupported whisper task '{task}'.")
    return WhisperConfig(
        max_workers=max(1, int(section.get("max_workers", 1))),
        task=cast(Literal["transcribe", "translate"], task),
    )


def _validate_config(config: AppConfig) -> None:
    """Validate cross-field invariants for the resolved application config."""

    api_base = config.inference.api_base.rstrip("/")
    if not api_base.endswith("/v1"):
        raise ValueError(
            "Config value 'inference.api_base' must point at an OpenAI-compatible '/v1' endpoint."
        )

    for label, path in {
        "paths.artifacts": config.paths.artifacts,
        "paths.data": config.paths.data,
        "paths.docint_home_dir": config.paths.docint_home_dir,
        "paths.logs": config.paths.logs,
        "paths.queries": config.paths.queries,
        "paths.results": config.paths.results,
        "paths.qdrant_sources": config.paths.qdrant_sources,
        "paths.hf_hub_cache": config.paths.hf_hub_cache,
    }.items():
        if not path.is_absolute():
            raise ValueError(f"Config value '{label}' must be an absolute path.")

    if not config.hosts.backend_host.strip():
        raise ValueError("Config value 'hosts.backend_host' must not be empty.")
    if not config.hosts.backend_public_host.strip():
        raise ValueError("Config value 'hosts.backend_public_host' must not be empty.")
    if not config.hosts.qdrant_host.strip():
        raise ValueError("Config value 'hosts.qdrant_host' must not be empty.")

    if config.inference.inference_provider == "openai" and api_base.startswith(
        "http://localhost"
    ):
        raise ValueError(
            "OpenAI profiles should not point 'inference.api_base' at localhost."
        )
    if config.inference.inference_provider == "ollama" and "11434" not in api_base:
        raise ValueError(
            "Ollama profiles should point 'inference.api_base' at the Ollama server."
        )
    if config.inference.inference_provider == "vllm" and "vllm" not in api_base:
        raise ValueError(
            "vLLM profiles should point 'inference.api_base' at the vLLM router or server."
        )


def load_config(
    *,
    profile: str | None = None,
    role: Role | None = None,
) -> AppConfig:
    """Load the canonical TOML-backed application config."""

    document = _read_config_document()
    resolved_profile = _resolve_profile(document, profile)
    resolved_role = _resolve_role(role)
    sections = _resolve_sections(document, resolved_profile, resolved_role)
    paths = _build_path_config(sections["paths"])
    config = AppConfig(
        runtime=_build_runtime_config(
            sections["runtime"],
            profile=resolved_profile,
            role=resolved_role,
        ),
        frontend=_build_frontend_config(sections["frontend"]),
        graphrag=_build_graphrag_config(sections["graphrag"]),
        hate_speech=_build_hate_speech_config(sections["hate_speech"]),
        hosts=_build_host_config(sections["hosts"]),
        image_ingestion=_build_image_ingestion_config(sections["image_ingestion"]),
        ingestion=_build_ingestion_config(sections["ingestion"]),
        inference=_build_inference_config(sections["inference"]),
        models=_build_model_config(sections["models"]),
        ner=_build_ner_config(sections["ner"]),
        paths=paths,
        pipeline=_build_pipeline_config(
            sections["pipeline"], artifacts_dir=paths.artifacts
        ),
        response_validation=_build_response_validation_config(
            sections["response_validation"]
        ),
        retrieval=_build_retrieval_config(sections["retrieval"]),
        session=_build_session_config(sections["session"], paths=paths),
        summary=_build_summary_config(sections["summary"]),
        whisper=_build_whisper_config(sections["whisper"]),
    )
    _validate_config(config)
    return config


def _apply_runtime_env(config: AppConfig) -> None:
    """Apply Hugging Face and related runtime env vars from the active config."""

    if config.runtime.docint_offline:
        os.environ["DOCINT_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        if not os.getenv("FASTEMBED_CACHE_PATH"):
            os.environ["FASTEMBED_CACHE_PATH"] = str(config.paths.hf_hub_cache)
        logger.info("Set Hugging Face libraries to offline mode.")
        return

    os.environ["DOCINT_OFFLINE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    logger.info("Hugging Face libraries are in online mode.")


def bootstrap_config(
    *,
    role: Role | None = None,
    profile: str | None = None,
) -> AppConfig:
    """Load dotenv, resolve the active TOML profile, and apply runtime env vars."""

    global _ACTIVE_PROFILE, _ACTIVE_ROLE

    load_dotenv(override=False)
    resolved_role = _resolve_role(role)
    config = load_config(profile=profile, role=resolved_role)
    _ACTIVE_PROFILE = config.runtime.profile
    _ACTIVE_ROLE = config.runtime.role
    _apply_runtime_env(config)
    return config


def set_offline_env() -> None:
    """Backward-compatible runtime bootstrap wrapper."""

    bootstrap_config(role=_ACTIVE_ROLE, profile=_ACTIVE_PROFILE)


def set_online_env() -> None:
    """Force online Hugging Face env vars for the current process."""

    os.environ["DOCINT_OFFLINE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    logger.info("Forced Hugging Face libraries to online mode.")


def resolve_hf_cache_path(
    cache_dir: Path, repo_id: str, filename: str | None = None
) -> Path | None:
    """Resolve a Hugging Face cache path for a repo snapshot."""

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


def _current_role(role: Role | None = None) -> Role:
    """Resolve a wrapper role using bootstrap state when available."""

    return _resolve_role(role or _ACTIVE_ROLE or "backend")


def load_frontend_env(
    default_collection_timeout: int = 120,
    *,
    role: Role | None = None,
) -> FrontendConfig:
    """Compatibility wrapper for frontend config."""

    config = load_config(role=_current_role(role))
    if config.frontend.collection_timeout == 0:
        return FrontendConfig(collection_timeout=default_collection_timeout)
    return config.frontend


def load_graphrag_env(
    default_enabled: bool = True,
    default_neighbor_hops: int = 2,
    default_top_k_nodes: int = 50,
    default_min_edge_weight: int = 3,
    default_max_neighbors: int = 6,
    *,
    role: Role | None = None,
) -> GraphRAGConfig:
    """Compatibility wrapper for GraphRAG config."""

    _ = (
        default_enabled,
        default_neighbor_hops,
        default_top_k_nodes,
        default_min_edge_weight,
        default_max_neighbors,
    )
    return load_config(role=_current_role(role)).graphrag


def load_hate_speech_env(
    default_enabled: bool = False,
    default_max_chars: int = 2048,
    default_max_workers: int = 1,
    *,
    role: Role | None = None,
) -> HateSpeechConfig:
    """Compatibility wrapper for hate-speech config."""

    _ = (default_enabled, default_max_chars, default_max_workers)
    return load_config(role=_current_role(role)).hate_speech


def load_host_env(
    default_backend_host: str = "http://localhost:8000",
    default_qdrant_host: str = "http://localhost:6333",
    default_cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501",
    *,
    role: Role | None = None,
) -> HostConfig:
    """Compatibility wrapper for host config."""

    _ = (default_backend_host, default_qdrant_host, default_cors_origins)
    return load_config(role=_current_role(role)).hosts


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
    *,
    role: Role | None = None,
) -> ImageIngestionConfig:
    """Compatibility wrapper for image-ingestion config."""

    _ = (
        default_image_ingestion_enabled,
        default_image_embedding_enabled,
        default_image_tagging_enabled,
        default_image_qdrant_collection,
        default_image_qdrant_vector_name,
        default_image_cache_by_hash,
        default_fail_on_embedding_error,
        default_fail_on_tagging_error,
        default_retrieve_top_k,
        default_tagging_max_image_dimension,
    )
    return load_config(role=_current_role(role)).image_ingestion


def load_ingestion_env(
    default_coarse_chunk_size: int = 8192,
    default_docling_accelerator_num_threads: int = 4,
    default_docstore_batch_size: int = 100,
    default_ingest_benchmark_enabled: bool = False,
    default_docstore_max_retries: int = 3,
    default_docstore_retry_backoff_seconds: float = 0.25,
    default_docstore_retry_backoff_max_seconds: float = 2.0,
    default_fine_chunk_overlap: int = 0,
    default_fine_chunk_size: int = 8192,
    default_hierarchical_chunking_enabled: bool = True,
    default_ingestion_batch_size: int = 50,
    default_sentence_splitter_chunk_overlap: int = 64,
    default_sentence_splitter_chunk_size: int = 1024,
    default_supported_filetypes: list[str] | None = None,
    *,
    role: Role | None = None,
) -> IngestionConfig:
    """Compatibility wrapper for ingestion config."""

    _ = (
        default_coarse_chunk_size,
        default_docling_accelerator_num_threads,
        default_docstore_batch_size,
        default_ingest_benchmark_enabled,
        default_docstore_max_retries,
        default_docstore_retry_backoff_seconds,
        default_docstore_retry_backoff_max_seconds,
        default_fine_chunk_overlap,
        default_fine_chunk_size,
        default_hierarchical_chunking_enabled,
        default_ingestion_batch_size,
        default_sentence_splitter_chunk_overlap,
        default_sentence_splitter_chunk_size,
        default_supported_filetypes,
    )
    return load_config(role=_current_role(role)).ingestion


def load_model_env(
    default_embed_model: str = "bge-m3",
    default_image_embed_model: str = "openai/clip-vit-base-patch32",
    default_ner_model: str = "gliner-community/gliner_large-v2.5",
    default_rerank_model: str = "BAAI/bge-reranker-v2-m3",
    default_sparse_model: str = "Qdrant/all_miniLM_L6_v2_with_attentions",
    default_text_model: str = "gpt-oss:20b",
    default_vision_model: str = "qwen3.5:9b",
    default_whisper_model: str = "turbo",
    *,
    role: Role | None = None,
) -> ModelConfig:
    """Compatibility wrapper for model config."""

    _ = (
        default_embed_model,
        default_image_embed_model,
        default_ner_model,
        default_rerank_model,
        default_sparse_model,
        default_text_model,
        default_vision_model,
        default_whisper_model,
    )
    return load_config(role=_current_role(role)).models


def load_ner_env(
    default_enabled: bool = True,
    default_max_chars: int = 1024,
    default_max_workers: int = 4,
    *,
    role: Role | None = None,
) -> NERConfig:
    """Compatibility wrapper for NER config."""

    _ = (default_enabled, default_max_chars, default_max_workers)
    return load_config(role=_current_role(role)).ner


def load_openai_env(
    default_api_base: str = "http://localhost:11434/v1",
    default_api_key: str = "sk-no-key-required",
    default_ctx_window: int = 4096,
    default_dimensions: int | None = None,
    default_max_retries: int = 2,
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
    *,
    role: Role | None = None,
) -> OpenAIConfig:
    """Compatibility wrapper for inference config."""

    _ = (
        default_api_base,
        default_api_key,
        default_ctx_window,
        default_dimensions,
        default_max_retries,
        default_inference_provider,
        default_reuse_client,
        default_seed,
        default_temperature,
        default_thinking_effort,
        default_thinking_enabled,
        default_timeout,
        default_top_p,
    )
    return load_config(role=_current_role(role)).inference


def load_path_env(*, role: Role | None = None) -> PathConfig:
    """Compatibility wrapper for path config."""

    return load_config(role=_current_role(role)).paths


def load_pipeline_config(
    default_text_coverage_threshold: float = 0.01,
    default_pipeline_version: str = DEFAULT_PIPELINE_VERSION,
    default_artifacts_dir: str | None = None,
    default_max_retries: int = 2,
    default_force_reprocess: bool = False,
    default_max_workers: int = 4,
    default_enable_vision_ocr: bool = True,
    default_vision_ocr_timeout: float = 60.0,
    default_vision_ocr_max_retries: int = 1,
    default_vision_ocr_max_image_dimension: int = 1024,
    default_vision_ocr_max_tokens: int = 4096,
    *,
    role: Role | None = None,
) -> PipelineConfig:
    """Compatibility wrapper for pipeline config."""

    _ = (
        default_text_coverage_threshold,
        default_max_retries,
        default_force_reprocess,
        default_max_workers,
        default_enable_vision_ocr,
        default_vision_ocr_timeout,
        default_vision_ocr_max_retries,
        default_vision_ocr_max_image_dimension,
        default_vision_ocr_max_tokens,
    )
    resolved_role = _current_role(role)
    document = _read_config_document()
    resolved_profile = _resolve_profile(document, None)
    sections = _resolve_sections(document, resolved_profile, resolved_role)
    paths = _build_path_config(sections["paths"])
    artifacts_dir = (
        Path(default_artifacts_dir).expanduser()
        if default_artifacts_dir is not None
        else paths.artifacts
    )
    return _build_pipeline_config(
        sections["pipeline"],
        artifacts_dir=artifacts_dir,
        default_pipeline_version=default_pipeline_version,
    )


def load_response_validation_env(
    default_enabled: bool = True,
    *,
    role: Role | None = None,
) -> ResponseValidationConfig:
    """Compatibility wrapper for response-validation config."""

    _ = default_enabled
    return load_config(role=_current_role(role)).response_validation


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
    *,
    role: Role | None = None,
) -> RetrievalConfig:
    """Compatibility wrapper for retrieval config."""

    _ = (
        default_rerank_use_fp16,
        default_retrieve_top_k,
        default_chat_response_mode,
        default_vector_store_query_mode,
        default_hybrid_alpha,
        default_sparse_top_k,
        default_hybrid_top_k,
        default_parent_context_enabled,
    )
    return load_config(role=_current_role(role)).retrieval


def load_session_env(
    default_session_store: str | None = None,
    *,
    role: Role | None = None,
) -> SessionConfig:
    """Compatibility wrapper for session config."""

    _ = default_session_store
    return load_config(role=_current_role(role)).session


def load_summary_env(
    default_coverage_target: float = 0.70,
    default_max_docs: int = 30,
    default_per_doc_top_k: int = 4,
    default_final_source_cap: int = 24,
    default_social_chunking_enabled: bool = True,
    default_social_candidate_pool: int = 48,
    default_social_diversity_limit: int = 2,
    *,
    role: Role | None = None,
) -> SummaryConfig:
    """Compatibility wrapper for summary config."""

    _ = (
        default_coverage_target,
        default_max_docs,
        default_per_doc_top_k,
        default_final_source_cap,
        default_social_chunking_enabled,
        default_social_candidate_pool,
        default_social_diversity_limit,
    )
    return load_config(role=_current_role(role)).summary


def load_whisper_env(
    default_max_workers: int = 1,
    default_task: Literal["transcribe", "translate"] = "transcribe",
    *,
    role: Role | None = None,
) -> WhisperConfig:
    """Compatibility wrapper for whisper config."""

    _ = (default_max_workers, default_task)
    return load_config(role=_current_role(role)).whisper


PIPELINE_VERSION = DEFAULT_PIPELINE_VERSION
