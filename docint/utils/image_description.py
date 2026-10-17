"""Caption-style description of a query, for re-scoring images against it.

A cross-encoder answers "does this passage answer this question", so a
question *about* the imagery ("how is the weather?") scores the right caption
in the same band as an unrelated one. Rephrased as the caption a matching
picture would carry, the same query separates them. The description comes
from the shared chat model behind the LiteLLM router, as translation does, and
fails soft: a missing description only costs an image its second look.
"""

from __future__ import annotations

from functools import lru_cache

from loguru import logger

from docint.utils.env_cfg import load_image_ingestion_config, load_language_env, load_model_env
from docint.utils.llm_sanitize import strip_reasoning
from docint.utils.openai_cfg import OpenAIPipeline
from docint.utils.prompt_loader import load_localized_prompt

_DEFAULT_DESCRIPTION_PROMPT = (
    "Write the caption of a picture that would answer the user's query. Output exactly one line of "
    "concrete visual terms, mentioning only what such a picture would show. Of these kinds of terms, use "
    "only those the query needs and never fill in the others: the people and how many there are, their "
    "clothing, objects, the setting, the sky and the weather, visible text. Describe people by what can "
    "be seen, never by name. Output only that line: no question, no explanation, no quotation marks. "
    "Write it in English."
)


class EmptyDescriptionError(ValueError):
    """The model answered with nothing left to rerank against."""


@lru_cache(maxsize=1)
def _pipeline() -> OpenAIPipeline:
    """Process-wide chat pipeline, built once (thread-safe; call_chat is stateless)."""
    return OpenAIPipeline()


@lru_cache(maxsize=1024)
def _describe_cached(query: str, lang: str, model: str) -> str:
    """Describe ``query`` as a caption (cached).

    Raises on failure, so only successful descriptions are cached: an outage
    costs one turn its rescue, not every later turn asking the same. The call
    gets one attempt within ``IMAGE_RESCORE_TIMEOUT``, read on a miss so the
    deadline stays out of the cache key.

    Args:
        query (str): The query the images are scored for.
        lang (str): Locale of the prompt, and so of the description.
        model (str): Chat model to ask.

    Returns:
        str: The description on one line, reasoning removed.

    Raises:
        EmptyDescriptionError: When nothing is left of the reply.
    """
    system_prompt = load_localized_prompt("image_rescore_description", default=_DEFAULT_DESCRIPTION_PROMPT, lang=lang)
    timeout = load_image_ingestion_config().rescore_timeout
    reply, _reasoning = strip_reasoning(
        _pipeline().call_chat(query, system_prompt=system_prompt, model=model, timeout=timeout, max_retries=0)
    )
    description = " ".join(reply.split())
    if not description:
        raise EmptyDescriptionError
    return description


def describe_for_image_match(query: str) -> str | None:
    """Describe what a picture answering ``query`` would show, in the response language.

    The stored captions are written in ``RESPONSE_LANGUAGE``, so the
    description is too; the locale and the chat model are part of the cache
    key, so switching either asks again. The call runs before the answer's
    first token, so it never inherits the chat client's timeout and retries:
    one attempt within ``IMAGE_RESCORE_TIMEOUT``, and a miss only costs the
    image its second look.

    Args:
        query (str): The query the images are scored for.

    Returns:
        str | None: One line of concrete visual terms, or ``None`` for a blank
        query, a failed call or an empty reply. Fail-soft — never raises.
    """
    if not query or not query.strip():
        return None
    try:
        return _describe_cached(query, load_language_env().code, load_model_env().text_model)
    except Exception as exc:  # fail-soft: the image only loses its second look
        logger.warning("Image re-score description unavailable ({}).", type(exc).__name__)
        return None
