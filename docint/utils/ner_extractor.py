import hashlib
import json
import os
import shutil
import tempfile
import warnings
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


def _load_gliner_config(model_dir: Path) -> dict[str, Any]:
    """Load the GLiNER config for a local model directory.

    Args:
        model_dir: Directory containing ``gliner_config.json``.

    Returns:
        Parsed GLiNER configuration payload.

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
        source: Existing source path.
        destination: Target path to create.
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
        cache_dir: Hugging Face hub cache directory.
        dependency: Repo ID or local filesystem path referenced by GLiNER config.

    Returns:
        Local filesystem path for the dependency.

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
        model_dir: Original local GLiNER model directory.
        config: GLiNER config payload to write into the offline runtime directory.

    Returns:
        Local runtime directory that contains the patched config and links to the
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
        model_dir: Local GLiNER model directory or snapshot path.
        cache_dir: Hugging Face hub cache directory.

    Returns:
        A local model directory that is safe to hand to ``GLiNER.from_pretrained``.
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
        model_id: GLiNER repo ID or local filesystem path.
        cache_dir: Hugging Face hub cache directory.

    Returns:
        Tuple of ``(load_target, local_only)`` suitable for ``from_pretrained``.

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


def build_gliner_ner_extractor(
    labels: list[str] | None = None,
    threshold: float = 0.3,
) -> Callable[[str], tuple[list[dict], list[dict]]]:
    """Create an NER extractor bound to a GLiNER model.

    Args:
        labels (list[str] | None): The entity labels to extract.
        threshold (float): Confidence threshold.

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
    load_id, local_only = _resolve_gliner_load_target(
        model_id=model_id, cache_dir=hf_cache
    )
    gliner_class = _get_gliner_class()

    # We load initially; moving to device happens if available
    try:
        model = gliner_class.from_pretrained(load_id, local_files_only=local_only)
    except Exception as e:
        logger.error("Failed to load GLiNER model: {}. Error: {}", model_id, e)
        raise

    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info("GLiNER moved to CUDA")
    elif torch.backends.mps.is_available():
        model = model.to("mps")
        logger.info("GLiNER moved to MPS")

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
            # GLiNER predict_entities
            # Suppress the "Asking to truncate to max_length but no maximum
            # length is provided" warning from the internal DeBERTa tokenizer.
            # Input chunks are already size-limited by SentenceSplitter, so
            # truncation is not needed.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*truncat.*max_length.*no maximum length.*",
                )
                preds = model.predict_entities(text, labels, threshold=threshold)
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
