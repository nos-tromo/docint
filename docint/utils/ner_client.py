"""HTTP client for the remote GLiNER NER service hosted by vllm-service.

This replaces the in-process GLiNER runtime that docint previously shipped.
Production deploys reach the service at ``http://vllm-router:4000/gliner``
(LiteLLM pass-through, Bearer auth via ``NER_API_KEY``); the ner-only
deployment shape reaches it at ``http://gliner-ner:8000/gliner`` (no auth,
trust inference-net). The choice is operator-driven via env vars; this
module is mode-agnostic.

The factory's return shape matches the deleted ``build_gliner_ner_extractor``
so the ingestion pipeline can swap in place: ``Callable[[str],
tuple[entities, relations]]`` where each entity is
``{"text", "type", "score"}`` and ``relations`` is always ``[]`` (GLiNER is
pure NER).
"""

from collections.abc import Callable
from typing import Any

import httpx
from loguru import logger

from docint.utils.env_cfg import NERClientConfig, load_ner_client_env

DEFAULT_NER_LABELS: list[str] = [
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


def _build_client(cfg: NERClientConfig) -> httpx.Client:
    """Construct the shared ``httpx.Client`` used for NER calls."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    return httpx.Client(
        base_url=cfg.api_base,
        timeout=cfg.timeout,
        headers=headers,
    )


def build_remote_ner_extractor(
    labels: list[str] | None = None,
    cfg: NERClientConfig | None = None,
) -> Callable[[str], tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Create an NER extractor that calls the remote vllm-service GLiNER endpoint.

    Args:
        labels (list[str] | None): Candidate entity labels to extract. Falls
            back to :data:`DEFAULT_NER_LABELS` when ``None`` or empty.
        cfg (NERClientConfig | None): Override client configuration. When
            ``None``, reads from the environment via
            :func:`docint.utils.env_cfg.load_ner_client_env`.

    Returns:
        Callable[[str], tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
        A function that takes raw text and returns ``(entities, relations)``
        where each entity is ``{"text", "type", "score"}`` and ``relations``
        is always ``[]`` (GLiNER is pure NER). On any error (network,
        timeout, non-2xx response, malformed payload) the function logs a
        warning and returns ``([], [])`` — matching the existing fail-safe
        behavior of the ingestion pipeline.
    """
    effective_labels = list(labels) if labels else list(DEFAULT_NER_LABELS)
    effective_cfg = cfg if cfg is not None else load_ner_client_env()
    client = _build_client(effective_cfg)
    logger.info(
        "Remote NER extractor ready: api_base={} auth={} threshold={}",
        effective_cfg.api_base,
        "bearer" if effective_cfg.api_key else "none",
        effective_cfg.threshold,
    )

    def _extract(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run remote NER over ``text``."""
        if not text.strip():
            return [], []

        try:
            response = client.post(
                "/gliner",
                json={
                    "text": text,
                    "labels": effective_labels,
                    "threshold": effective_cfg.threshold,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("Remote NER call failed: {}", exc)
            return [], []

        raw_entities = payload.get("entities") if isinstance(payload, dict) else None
        if not isinstance(raw_entities, list):
            return [], []

        entities: list[dict[str, Any]] = []
        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            entity_text = item.get("text")
            entity_label = item.get("label")
            if not entity_text or not entity_label:
                continue
            entities.append(
                {
                    "text": entity_text,
                    "type": entity_label,
                    "score": item.get("score"),
                }
            )

        return entities, []

    return _extract
