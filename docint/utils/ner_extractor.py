import json
import os
import warnings
from typing import Any, Callable

import torch
from loguru import logger
from gliner import GLiNER

from docint.utils.env_cfg import load_model_env, load_path_env, resolve_hf_cache_path


def _parse_ner_payload(raw: str) -> dict[str, Any]:
    """
    Parse a raw NER model response into JSON-like payload.
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


def build_ner_extractor(
    model: Any, prompt: str, max_chars: int
) -> Callable[[str], tuple[list[dict], list[dict]]]:
    """
    Create an NER extractor bound to a model and prompt template.

    Args:
        model (Any): The language model instance with a 'complete' method.
        prompt (str): The prompt template for NER extraction.
        max_chars (int): Maximum characters from input text to send to the model.

    Returns:
        Callable[[str], tuple[list[dict], list[dict]]]: The NER extraction function.
    """

    def _extract(text: str) -> tuple[list[dict], list[dict]]:
        """
        Extract entities and relations from text using the bound model and prompt.

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


def build_gliner_ner_extractor(
    labels: list[str] | None = None,
    threshold: float = 0.3,
) -> Callable[[str], tuple[list[dict], list[dict]]]:
    """
    Create an NER extractor bound to a GLiNER model.

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
            "person",
            "organization",
            "location",
            "date",
            "event",
        ]

    logger.info("Loading GLiNER model: {}", model_id)

    # Resolve from local HF cache when available to avoid network requests
    hf_cache = load_path_env().hf_hub_cache
    resolved = resolve_hf_cache_path(hf_cache, model_id)
    load_id = str(resolved) if resolved else model_id
    local_only = resolved is not None or os.getenv("HF_HUB_OFFLINE", "0") == "1"

    if resolved:
        logger.info("Using local GLiNER model path: {}", resolved)

    # We load initially; moving to device happens if available
    try:
        model = GLiNER.from_pretrained(load_id, local_files_only=local_only)
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
        """
        Extract entities using GLiNER.

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
