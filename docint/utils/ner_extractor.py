"""Named-entity recognition and relation extraction via LLM and GLiNER backends."""

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from loguru import logger

from docint.utils.env_cfg import (
    load_model_env,
    load_path_env,
    resolve_hf_cache_path,
    set_offline_env,
)

_GLINER_OFFLINE_DIR = Path(tempfile.gettempdir()) / "docint-gliner-offline"
_DEFAULT_GLINER_CONTEXT_WINDOW = 768
_GLINER_SPECIAL_TOKEN_RESERVE = 2
_GLINER_RUNTIME_CACHE_LOCK = threading.Lock()
_SENTENCE_RE = re.compile(
    r".+?(?:[.!?]+[\"')\]]*(?=\s+|$)|\n{2,}|$)",
    re.DOTALL,
)


@dataclass(slots=True)
class _GLiNERRuntime:
    """Reusable GLiNER runtime state shared across extractor instances."""

    model: Any
    max_tokens: int
    tokenizer: Any | None
    words_splitter: Any | None
    lock: threading.Lock


_GLINER_RUNTIME_CACHE: dict[tuple[str, bool, str], _GLiNERRuntime] = {}


def _parse_ner_payload(raw: str) -> dict[str, Any]:
    """Parse a raw NER model response into JSON-like payload.

    Args:
        raw (str): The raw response string from the NER model.

    Returns:
        dict[str, Any]: The parsed payload dictionary.
    """
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
    except Exception:
        return {}
    return {}


def _resolve_gliner_device(device: str | None) -> str | None:
    """Resolve the execution device for GLiNER.

    Args:
        device (str | None): Requested device name.

    Returns:
        str | None: Device string to pass to ``model.to()``, or ``None`` to
        keep GLiNER on CPU.
    """
    normalized = (device or "auto").strip().lower()
    if not normalized or normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            return "mps"
        return None

    if normalized == "cpu":
        return None

    if normalized == "mps":
        if (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            return "mps"
        logger.warning(
            "GLiNER requested device '{}' but MPS is unavailable; continuing on CPU.",
            normalized,
        )
        return None

    if normalized == "cuda" or normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            logger.warning(
                "GLiNER requested device '{}' but CUDA is unavailable; continuing on CPU.",
                normalized,
            )
            return None

        if normalized.startswith("cuda:"):
            try:
                device_index = int(normalized.split(":", maxsplit=1)[1])
            except ValueError:
                logger.warning(
                    "GLiNER requested invalid device '{}' and will continue on CPU.",
                    normalized,
                )
                return None
            if device_index < 0 or device_index >= torch.cuda.device_count():
                logger.warning(
                    "GLiNER requested device '{}' but it is unavailable; continuing on CPU.",
                    normalized,
                )
                return None
        return normalized

    logger.warning(
        "GLiNER received unsupported device '{}' and will continue on CPU.",
        normalized,
    )
    return None


def build_llm_ner_extractor(
    model: Any, prompt: str, max_chars: int
) -> Callable[[str], tuple[list[dict], list[dict]]]:
    """Create an NER extractor bound to a model and prompt template.

    Args:
        model (Any): The language model instance with a 'complete' method.
        prompt (str): The prompt template for NER extraction.
        max_chars (int): Maximum characters from input text to send to the model.

    Returns:
        Callable[[str], tuple[list[dict], list[dict]]]: The NER extraction function.
    """

    def _extract(text: str) -> tuple[list[dict], list[dict]]:
        """Extract entities and relations from text using the bound model and prompt.

        Args:
            text (str): The input text to extract entities and relations from.

        Returns:
            tuple[list[dict], list[dict]]: A tuple containing two lists: extracted entities and extracted relations.
        """
        snippet = text[:max_chars]
        prompt_text = prompt.format(text=snippet)

        try:
            resp = model.complete(prompt_text)
            raw = resp.text if hasattr(resp, "text") else str(resp)
        except Exception as exc:  # pragma: no cover - model failures are runtime
            logger.warning("NER extraction request failed: {}", exc)
            return [], []

        payload = _parse_ner_payload(raw) if isinstance(raw, str) else {}
        entities_raw = payload.get("entities") if isinstance(payload, dict) else []
        relations_raw = payload.get("relations") if isinstance(payload, dict) else []

        entities: list[dict] = []
        for ent in entities_raw or []:
            if not isinstance(ent, dict):
                continue
            text_val = str(ent.get("text") or ent.get("name") or "").strip()
            if not text_val:
                continue
            entities.append(
                {
                    "text": text_val,
                    "type": ent.get("type") or ent.get("label"),
                    "score": ent.get("score"),
                }
            )

        relations: list[dict] = []
        for rel in relations_raw or []:
            if not isinstance(rel, dict):
                continue
            head = str(rel.get("head") or rel.get("subject") or "").strip()
            tail = str(rel.get("tail") or rel.get("object") or "").strip()
            if not head or not tail:
                continue
            relations.append(
                {
                    "head": head,
                    "tail": tail,
                    "label": rel.get("label") or rel.get("type"),
                    "score": rel.get("score"),
                }
            )

        return entities, relations

    return _extract


def _get_gliner_class() -> type[Any]:
    """Import GLiNER after offline env vars are applied."""
    set_offline_env()

    from gliner import GLiNER

    return GLiNER


def _get_or_load_gliner_runtime(
    model_id: str,
    cache_dir: Path,
    device: str | None,
) -> _GLiNERRuntime:
    """Load or reuse a GLiNER runtime for the requested model/device pair.

    Args:
        model_id (str): Configured GLiNER model identifier.
        cache_dir (Path): Hugging Face cache directory.
        device (str | None): Preferred execution device.

    Returns:
        _GLiNERRuntime: Cached runtime bundle for inference.
    """
    load_id, local_only = _resolve_gliner_load_target(
        model_id=model_id, cache_dir=cache_dir
    )
    target_device = _resolve_gliner_device(device)
    device_key = target_device or "cpu"
    cache_key = (load_id, local_only, device_key)

    with _GLINER_RUNTIME_CACHE_LOCK:
        runtime = _GLINER_RUNTIME_CACHE.get(cache_key)
        if runtime is not None:
            logger.info("Reusing cached GLiNER runtime: {} on {}", model_id, device_key)
            return runtime

        gliner_class = _get_gliner_class()

        try:
            model = gliner_class.from_pretrained(load_id, local_files_only=local_only)
        except Exception as e:
            logger.error("Failed to load GLiNER model: {}. Error: {}", model_id, e)
            raise

        if target_device is not None:
            model = model.to(target_device)
            logger.info("GLiNER moved to {}", target_device)

        runtime = _GLiNERRuntime(
            model=model,
            max_tokens=_resolve_gliner_context_window(model),
            tokenizer=_get_gliner_tokenizer(model),
            words_splitter=_get_gliner_words_splitter(model),
            lock=threading.Lock(),
        )
        _GLINER_RUNTIME_CACHE[cache_key] = runtime
        return runtime


def _load_gliner_config(model_dir: Path) -> dict[str, Any]:
    """Load the GLiNER config for a local model directory.

    Args:
        model_dir (Path): Directory containing ``gliner_config.json``.

    Returns:
        dict[str, Any]: Parsed GLiNER configuration payload.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = model_dir / "gliner_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No GLiNER config file found in {model_dir}")

    return json.loads(config_path.read_text(encoding="utf-8"))


def _link_or_copy_path(source: Path, destination: Path) -> None:
    """Materialize a file or directory at ``destination`` from ``source``.

    Args:
        source (Path): Existing source path.
        destination (Path): Target path to create.
    """
    if destination.exists():
        return

    try:
        destination.symlink_to(source, target_is_directory=source.is_dir())
    except Exception:
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)


def _resolve_local_gliner_dependency(cache_dir: Path, dependency: str) -> Path:
    """Resolve a GLiNER dependency path without allowing network access.

    Args:
        cache_dir (Path): Hugging Face hub cache directory.
        dependency (str): Repo ID or local filesystem path referenced by GLiNER config.

    Returns:
        Path: Local filesystem path for the dependency.

    Raises:
        FileNotFoundError: If the dependency is unavailable locally.
    """
    dependency_path = Path(dependency).expanduser()
    if dependency_path.exists():
        return dependency_path.resolve()

    resolved = resolve_hf_cache_path(cache_dir=cache_dir, repo_id=dependency)
    if resolved is None:
        raise FileNotFoundError(
            "GLiNER offline load requires a local snapshot for "
            f"'{dependency}', but none was found in {cache_dir}."
        )

    return resolved


def _materialize_offline_gliner_dir(model_dir: Path, config: dict[str, Any]) -> Path:
    """Create a local-only GLiNER directory with patched config references.

    Args:
        model_dir (Path): Original local GLiNER model directory.
        config (dict[str, Any]): GLiNER config payload to write into the offline runtime directory.

    Returns:
        Path: Local runtime directory that contains the patched config and links to the
        original model assets.
    """
    digest = hashlib.sha256(
        f"{model_dir.resolve()}\0{json.dumps(config, sort_keys=True)}".encode("utf-8")
    ).hexdigest()[:16]
    runtime_dir = _GLINER_OFFLINE_DIR / digest
    runtime_dir.mkdir(parents=True, exist_ok=True)

    for item in model_dir.iterdir():
        if item.name == "gliner_config.json":
            continue
        _link_or_copy_path(item, runtime_dir / item.name)

    (runtime_dir / "gliner_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return runtime_dir


def _prepare_local_gliner_model_dir(model_dir: Path, cache_dir: Path) -> Path:
    """Prepare a GLiNER directory for strict offline loading.

    GLiNER configs often refer to the backbone model by Hugging Face repo ID
    (for example ``microsoft/deberta-v3-large``). The upstream loader may hand
    that repo ID back to ``transformers`` even when the GLiNER snapshot itself
    was found locally, which can trigger outbound hub resolution attempts. This
    helper rewrites those config references to local snapshot paths only.

    Args:
        model_dir (Path): Local GLiNER model directory or snapshot path.
        cache_dir (Path): Hugging Face hub cache directory.

    Returns:
        Path: A local model directory that is safe to hand to ``GLiNER.from_pretrained``.
    """
    config = _load_gliner_config(model_dir)
    patched = False

    for field in ("model_name", "labels_encoder", "labels_decoder"):
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            continue

        resolved = _resolve_local_gliner_dependency(
            cache_dir=cache_dir, dependency=value
        )
        resolved_str = str(resolved)
        if value != resolved_str:
            config[field] = resolved_str
            patched = True

    if not patched:
        return model_dir

    runtime_dir = _materialize_offline_gliner_dir(model_dir=model_dir, config=config)
    logger.info("Prepared offline GLiNER runtime directory: {}", runtime_dir)
    return runtime_dir


def _resolve_gliner_load_target(model_id: str, cache_dir: Path) -> tuple[str, bool]:
    """Resolve the load target for GLiNER without allowing accidental hub access.

    Args:
        model_id (str): GLiNER repo ID or local filesystem path.
        cache_dir (Path): Hugging Face hub cache directory.

    Returns:
        tuple[str, bool]: Tuple of ``(load_target, local_only)`` suitable for ``from_pretrained``.

    Raises:
        FileNotFoundError: If offline mode is enabled and the model is not cached
            locally.
    """
    local_model_dir = Path(model_id).expanduser()
    if local_model_dir.exists():
        prepared = _prepare_local_gliner_model_dir(
            model_dir=local_model_dir.resolve(),
            cache_dir=cache_dir,
        )
        return str(prepared), True

    resolved = resolve_hf_cache_path(cache_dir=cache_dir, repo_id=model_id)
    if resolved is not None:
        logger.info("Using local GLiNER model path: {}", resolved)
        prepared = _prepare_local_gliner_model_dir(
            model_dir=resolved, cache_dir=cache_dir
        )
        return str(prepared), True

    if os.getenv("HF_HUB_OFFLINE", "0") == "1":
        raise FileNotFoundError(
            f"GLiNER model '{model_id}' is not available in the local cache {cache_dir}."
        )

    return model_id, False


def _resolve_gliner_context_window(model: Any) -> int:
    """Return the usable GLiNER context window in tokens.

    Args:
        model (Any): Loaded GLiNER model instance.

    Returns:
        int: Maximum number of non-special tokens to send in a single request.

    Raises:
        TypeError, ValueError: If the model config contains an invalid max_len value.
    """
    config = getattr(model, "config", None)
    raw_max_len = getattr(config, "max_len", _DEFAULT_GLINER_CONTEXT_WINDOW)
    try:
        max_len = int(raw_max_len)
    except (TypeError, ValueError):
        max_len = _DEFAULT_GLINER_CONTEXT_WINDOW
    return max(1, max_len - _GLINER_SPECIAL_TOKEN_RESERVE)


def _get_gliner_tokenizer(model: Any) -> Any | None:
    """Return the GLiNER backbone tokenizer when available.

    Args:
        model (Any): Loaded GLiNER model instance.

    Returns:
        Any | None: Tokenizer object or ``None`` when unavailable.
    """
    data_processor = getattr(model, "data_processor", None)
    return getattr(data_processor, "transformer_tokenizer", None)


def _get_gliner_words_splitter(model: Any) -> Any | None:
    """Return GLiNER's words_splitter callable when available.

    The words_splitter is the exact tokenizer GLiNER uses when counting words
    for its internal truncation guard (``len(words) > max_len``).  Using it for
    pre-inference chunking ensures the two counts always agree.

    Args:
        model (Any): Loaded GLiNER model instance.

    Returns:
        Any | None: Callable that yields ``(token, start, end)`` triples, or
        ``None`` when the attribute is unavailable.
    """
    data_processor = getattr(model, "data_processor", None)
    return getattr(data_processor, "words_splitter", None)


def _count_text_tokens(
    text: str,
    tokenizer: Any | None,
    words_splitter: Any | None = None,
) -> int:
    """Count tokens for a text span in the same units GLiNER uses for truncation.

    When ``words_splitter`` is provided it is used as the primary counter because
    GLiNER's internal processor truncates by ``len(words)`` where ``words`` is
    produced by the same splitter.  The BPE tokenizer and whitespace fallbacks are
    retained for deployments where ``words_splitter`` is unavailable.

    Args:
        text (str): Input text span.
        tokenizer (Any | None): BPE tokenizer associated with the loaded GLiNER model.
        words_splitter (Any | None): GLiNER words_splitter callable that yields
            ``(token, start, end)`` triples.  When provided, used as the primary
            counting branch to guarantee budget units match GLiNER's truncation units.

    Returns:
        int: Word count (GLiNER units) when ``words_splitter`` is provided; sub-word
        BPE token count when only ``tokenizer`` is available; whitespace word count as
        final fallback.
    """
    stripped = text.strip()
    if not stripped:
        return 0

    if words_splitter is not None and callable(words_splitter):
        try:
            return sum(1 for _ in words_splitter(stripped))
        except Exception as exc:
            logger.warning(
                "GLiNER words_splitter failed ({}); falling back to BPE tokenizer. "
                "Chunking may be under-counted and truncation warnings may reappear.",
                exc,
            )

    if tokenizer is not None and hasattr(tokenizer, "encode"):
        try:
            return len(
                tokenizer.encode(
                    stripped,
                    add_special_tokens=False,
                    truncation=False,
                )
            )
        except TypeError:
            try:
                return len(tokenizer.encode(stripped, add_special_tokens=False))
            except Exception:
                pass
        except Exception:
            pass

    return len(re.findall(r"\S+", stripped))


def _split_text_into_sentences(text: str) -> list[str]:
    """Split text into sentence-like spans while preserving readable boundaries.

    Args:
        text (str): Raw text to split.

    Returns:
        list[str]: Sentence-like spans. Falls back to the full text when no sentence
        boundary is detected.
    """
    stripped = text.strip()
    if not stripped:
        return []

    sentences = [match.group(0).strip() for match in _SENTENCE_RE.finditer(stripped)]
    sentences = [sentence for sentence in sentences if sentence]
    return sentences or [stripped]


def _split_text_into_words(text: str) -> list[str]:
    """Split text into word-like spans for fallback chunking.

    Args:
        text (str): Raw text to split.

    Returns:
        list[str]: Word-like spans. Punctuation stays attached to its word.
    """
    return re.findall(r"\S+", text)


def _split_oversized_token(
    token: str,
    max_tokens: int,
    tokenizer: Any | None,
    words_splitter: Any | None = None,
) -> list[str]:
    """Split a single oversized token as a last-resort fallback.

    Args:
        token (str): Single token-like span that still exceeds the model budget.
        max_tokens (int): Maximum token budget per chunk.
        tokenizer (Any | None): Tokenizer associated with the loaded GLiNER model.
        words_splitter (Any | None): GLiNER words_splitter callable used for token counting.

    Returns:
        list[str]: Smaller character-based chunks guaranteed to fit the budget.
    """
    pieces: list[str] = []
    start = 0
    token = token.strip()
    while start < len(token):
        end = start + 1
        last_fit = end
        while end <= len(token):
            candidate = token[start:end]
            if _count_text_tokens(candidate, tokenizer, words_splitter) > max_tokens:
                break
            last_fit = end
            end += 1
        if last_fit == start:
            last_fit = min(len(token), start + 1)
        pieces.append(token[start:last_fit])
        start = last_fit
    return pieces


def _pack_text_segments(
    segments: list[str],
    max_tokens: int,
    tokenizer: Any | None,
    words_splitter: Any | None = None,
) -> list[str]:
    """Pack sentence or word segments into GLiNER-sized chunks.

    Args:
        segments (list[str]): Ordered text segments to pack.
        max_tokens (int): Maximum token budget per chunk.
        tokenizer (Any | None): Tokenizer associated with the loaded GLiNER model.
        words_splitter (Any | None): GLiNER words_splitter callable used for token counting.

    Returns:
        list[str]: Packed chunks whose token counts fit the requested budget.
    """
    chunks: list[str] = []
    current = ""

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        segment_tokens = _count_text_tokens(segment, tokenizer, words_splitter)
        if segment_tokens > max_tokens:
            if current:
                chunks.append(current)
                current = ""

            word_segments = _split_text_into_words(segment)
            if len(word_segments) <= 1:
                chunks.extend(
                    _split_oversized_token(
                        token=segment,
                        max_tokens=max_tokens,
                        tokenizer=tokenizer,
                        words_splitter=words_splitter,
                    )
                )
            else:
                chunks.extend(
                    _pack_text_segments(
                        segments=word_segments,
                        max_tokens=max_tokens,
                        tokenizer=tokenizer,
                        words_splitter=words_splitter,
                    )
                )
            continue

        candidate = segment if not current else f"{current} {segment}"
        if (
            current
            and _count_text_tokens(candidate, tokenizer, words_splitter) > max_tokens
        ):
            chunks.append(current)
            current = segment
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def _chunk_text_for_gliner(
    text: str,
    max_tokens: int,
    tokenizer: Any | None,
    words_splitter: Any | None = None,
) -> list[str]:
    """Split text into GLiNER-safe chunks with sentence-first packing.

    Args:
        text (str): Raw input text.
        max_tokens (int): Maximum token budget per chunk.
        tokenizer (Any | None): Tokenizer associated with the loaded GLiNER model.
        words_splitter (Any | None): GLiNER words_splitter callable used for token counting.

    Returns:
        list[str]: Ordered list of chunks suitable for repeated GLiNER inference.
    """
    sentences = _split_text_into_sentences(text)
    return _pack_text_segments(
        segments=sentences,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
        words_splitter=words_splitter,
    )


def build_gliner_ner_extractor(
    labels: list[str] | None = None,
    threshold: float = 0.3,
    device: str | None = None,
) -> Callable[[str], tuple[list[dict], list[dict]]]:
    """Create an NER extractor bound to a GLiNER model.

    Args:
        labels (list[str] | None): The entity labels to extract.
        threshold (float): Confidence threshold.
        device (str | None): Preferred execution device.

    Returns:
        Callable[[str], tuple[list[dict], list[dict]]]: The NER extraction function.
    """
    model_id = load_model_env().ner_model

    # Default labels if none provided - covering general domain
    if not labels:
        labels = [
            "bank_account",  # Bank account numbers
            "date",  # Absolute or relative dates or periods.
            "event",  # Named hurricanes, battles, wars, sports events, etc.
            "fac",  # Buildings, airports, highways, bridges, etc.
            "group",  # Nationalities or religious or political groups.
            "lang",  # Any named language.
            "loc",  # Locations, such as countries, cities, states, regions.
            "mail",  # E-Mail addresses.
            "money",  # Monetary values, including unit.
            "org",  # Companies, agencies, institutions, etc.
            "person",  # People, including fictional.
            "phone",  # Phone numbers.
            "time",  # Times smaller than a day.
            "weapon",  # Named vehicles, weapons, or products.
        ]

    logger.info("Loading GLiNER model: {}", model_id)

    hf_cache = load_path_env().hf_hub_cache
    runtime = _get_or_load_gliner_runtime(
        model_id=model_id,
        cache_dir=hf_cache,
        device=device,
    )

    def _extract(text: str) -> tuple[list[dict], list[dict]]:
        """Extract entities using GLiNER.

        Args:
            text (str): Input text.

        Returns:
            tuple[list[dict], list[dict]]: Entities and relations.
        """
        if not text.strip():
            return [], []

        try:
            preds: list[dict[str, Any]] = []
            # GLiNER and its tokenizer share PyO3-backed state that is not safe
            # for concurrent use across ingestion threads.
            with runtime.lock:
                for chunk in _chunk_text_for_gliner(
                    text=text,
                    max_tokens=runtime.max_tokens,
                    tokenizer=runtime.tokenizer,
                    words_splitter=runtime.words_splitter,
                ):
                    # Suppress the "Asking to truncate to max_length but no maximum
                    # length is provided" warning from the internal DeBERTa tokenizer.
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=".*truncat.*max_length.*no maximum length.*",
                        )
                        preds.extend(
                            runtime.model.predict_entities(
                                chunk, labels, threshold=threshold
                            )
                        )
        except Exception as e:
            logger.warning("GLiNER extraction failed: {}", e)
            return [], []

        entities = []
        for p in preds:
            entities.append(
                {
                    "text": p["text"],
                    "type": p["label"],
                    "score": p["score"],
                }
            )

        # GLiNER is pure NER, leaving relations empty
        relations: list[dict] = []

        return entities, relations

    return _extract
