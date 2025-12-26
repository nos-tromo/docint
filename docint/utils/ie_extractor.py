import json
from typing import Any, Callable

from loguru import logger


def _parse_ie_payload(raw: str) -> dict[str, Any]:
    """
    Parse a raw IE model response into JSON-like payload.
    Args:
        raw (str): The raw response string from the IE model.

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


def build_ie_extractor(
    model: Any, prompt: str, max_chars: int
) -> Callable[[str], tuple[list[dict], list[dict]]]:
    """
    Create an IE extractor bound to a model and prompt template.
    
    Args:
        model (Any): The language model instance with a 'complete' method.
        prompt (str): The prompt template for IE extraction.
        max_chars (int): Maximum characters from input text to send to the model.

    Returns:
        Callable[[str], tuple[list[dict], list[dict]]]: The IE extraction function.
    """

    def _extract(text: str) -> tuple[list[dict], list[dict]]:
        snippet = text[:max_chars]
        prompt_text = prompt.format(text=snippet)

        try:
            resp = model.complete(prompt_text)
            raw = resp.text if hasattr(resp, "text") else str(resp)
        except Exception as exc:  # pragma: no cover - model failures are runtime
            logger.warning("IE extraction request failed: {}", exc)
            return [], []

        payload = _parse_ie_payload(raw) if isinstance(raw, str) else {}
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
