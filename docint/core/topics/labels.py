"""Topic labels written by the chat model.

One request per topic asks for a short label and a one-sentence description.
The reply is constrained by a strict JSON schema while the endpoint accepts it
and is retried unconstrained when it does not. A topic whose label cannot be
written is named by its leading keywords, so every topic always has a name.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

from loguru import logger

from docint.core.ingest.hate_speech import is_structured_output_rejection
from docint.utils.llm_sanitize import strip_reasoning
from docint.utils.prompt_loader import load_localized_prompt

LABEL_CHARS = 80
DESCRIPTION_CHARS = 300
PASSAGES_PER_TOPIC = 5
PASSAGE_CHARS = 800
MAX_IN_FLIGHT = 4

Complete = Callable[[str, dict[str, Any] | None], str]

DEFAULT_TEMPLATE = (
    "You name one topic of a collection of passages.\n\n"
    "Its most characteristic words, strongest first: {keywords}\n\n"
    "Passages from the topic:\n{passages}\n\n"
    'Reply with a JSON object with two keys: "description", one sentence saying what the passages have in '
    'common, and "label", a name for the topic of at most six words. Name what the passages share, such as a '
    "theme, an event, a setting or an argument, never the title or author of the work, document or collection "
    "they come from. Use the passages' own terms and invent nothing."
)


@dataclass(frozen=True)
class LabelRequest:
    """What the model is shown for one topic.

    Attributes:
        topic_id (int): The topic's id.
        keywords (tuple[str, ...]): Its keywords, strongest first.
        passages (tuple[str, ...]): Representative passages.
    """

    topic_id: int
    keywords: tuple[str, ...]
    passages: tuple[str, ...]


@dataclass(frozen=True)
class TopicLabel:
    """A topic's name.

    Attributes:
        label (str): The label.
        description (str): A one-sentence description, empty for a fallback.
        written (bool): Whether the model wrote it, as opposed to the keyword fallback.
    """

    label: str
    description: str
    written: bool


def label_response_format() -> dict[str, Any]:
    """Build the strict JSON schema constraining a label reply.

    Returns:
        dict[str, Any]: The ``response_format`` payload.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "topic_label",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["description", "label"],
                "properties": {"description": {"type": "string"}, "label": {"type": "string"}},
            },
        },
    }


def fallback_label(topic_id: int, keywords: Sequence[str]) -> str:
    """Name a topic without the model.

    Args:
        topic_id (int): The topic's id.
        keywords (Sequence[str]): Its keywords, strongest first.

    Returns:
        str: The leading keywords, or ``T<id>`` when there are none.
    """
    return ", ".join(keywords[:4]) or f"T{topic_id}"


def render_prompt(template: str, request: LabelRequest) -> str:
    """Fill the prompt template for one topic.

    Args:
        template (str): A template with ``{keywords}`` and ``{passages}``.
        request (LabelRequest): The topic's keywords and passages.

    Returns:
        str: The prompt.
    """
    passages = "\n".join(f"{i}. {p[:PASSAGE_CHARS]}" for i, p in enumerate(request.passages, start=1))
    values = {"keywords": ", ".join(request.keywords), "passages": passages}
    return re.sub(r"\{(keywords|passages)\}", lambda match: values[match[1]], template)


def _load_object(text: str) -> object:
    """Decode the reply as JSON, or its first ``{...}`` object.

    Args:
        text (str): The reply without reasoning.

    Returns:
        object: The decoded value, or ``None`` when nothing decodes.
    """
    try:
        return json.loads(text)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
            return value
        except ValueError:
            start = text.find("{", start + 1)
    return None


def parse_label(raw: str) -> tuple[str, str] | None:
    """Read a label and description out of a model reply.

    Args:
        raw (str): The raw reply text.

    Returns:
        tuple[str, str] | None: ``(label, description)`` cut to their limits, or
            ``None`` when either is missing or blank.
    """
    clean, _ = strip_reasoning(raw)
    data = _load_object(clean)
    if not isinstance(data, dict):
        return None
    label, description = data.get("label"), data.get("description")
    if not isinstance(label, str) or not isinstance(description, str):
        return None
    label, description = " ".join(label.split()), " ".join(description.split())
    if not label or not description:
        return None
    return label[:LABEL_CHARS], description[:DESCRIPTION_CHARS]


class TopicLabeller:
    """Writes a label per topic through a chat completion callable."""

    def __init__(self, complete: Complete, *, template: str | None = None, max_in_flight: int = MAX_IN_FLIGHT) -> None:
        """Initialise the labeller.

        Args:
            complete (Complete): ``(prompt, response_format or None) -> reply text``.
            template (str | None): The prompt template; the localized prompt when ``None``.
            max_in_flight (int): Most concurrent calls.
        """
        self._complete = complete
        self._template = (
            template if template is not None else load_localized_prompt("topic_label", default=DEFAULT_TEMPLATE)
        )
        self._max_in_flight = max(1, max_in_flight)
        self._structured = True
        self._lock = threading.Lock()

    def _ask(self, prompt: str, response_format: dict[str, Any] | None) -> tuple[str, bool]:
        """Send one request, constrained while the endpoint still accepts the schema.

        Args:
            prompt (str): The rendered prompt.
            response_format (dict[str, Any] | None): The schema, or ``None``.

        Returns:
            tuple[str, bool]: The reply text and whether it was constrained.
        """
        with self._lock:
            constrained = response_format is not None and self._structured
        if constrained:
            try:
                return self._complete(prompt, response_format), True
            except Exception as exc:
                if not is_structured_output_rejection(exc):
                    raise
                with self._lock:
                    first = self._structured
                    self._structured = False
                if first:
                    logger.warning(
                        "The inference endpoint rejected response_format (HTTP {}); labelling topics without it",
                        getattr(exc, "status_code", "?"),
                    )
        return self._complete(prompt, None), False

    def _label_one(self, request: LabelRequest) -> TopicLabel:
        """Label one topic, falling back to its keywords.

        Args:
            request (LabelRequest): The topic's keywords and passages.

        Returns:
            TopicLabel: The written label, or the fallback.
        """
        fallback = TopicLabel(fallback_label(request.topic_id, request.keywords), "", False)
        try:
            prompt = render_prompt(self._template, request)
            raw, constrained = self._ask(prompt, label_response_format())
            parsed = parse_label(raw)
            if parsed is None and constrained:
                raw, _ = self._ask(prompt, None)
                parsed = parse_label(raw)
        except Exception as exc:
            logger.warning("Labelling topic {} failed: {}", request.topic_id, type(exc).__name__)
            return fallback
        if parsed is None:
            logger.warning("The label reply for topic {} was unparseable", request.topic_id)
            return fallback
        return TopicLabel(parsed[0], parsed[1], True)

    def label_all(
        self,
        requests: Sequence[LabelRequest],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[int, TopicLabel]:
        """Label every topic, whatever fails.

        Args:
            requests (Sequence[LabelRequest]): One request per topic.
            progress (Callable[[int, int], None] | None): Called as ``(done, total)``
                on the calling thread after each finished topic; a raise cancels the rest.

        Returns:
            dict[int, TopicLabel]: One entry per request ``topic_id``.
        """
        total = len(requests)
        out: dict[int, TopicLabel] = {}
        executor = ThreadPoolExecutor(max_workers=self._max_in_flight)
        try:
            futures: dict[Future[TopicLabel], int] = {
                executor.submit(self._label_one, request): request.topic_id for request in requests
            }
            pending = set(futures)
            done = 0
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    out[futures[future]] = future.result()
                    done += 1
                    if progress is not None:
                        progress(done, total)
        except BaseException:
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        executor.shutdown(wait=True)
        written = sum(1 for label in out.values() if label.written)
        logger.info("Topic labels | written={} fallback={}", written, len(out) - written)
        return out
