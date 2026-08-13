"""Query reformulation for the corrective retry.

A retrieval whose answer response validation flagged as mismatched gets one
second chance: this agent turns the failed attempt plus the validator's reason
into a fresh retrieval query. It is deliberately fail-soft — returning ``None``
means "no usable reformulation", and every caller treats that as "skip the
retry" rather than as an error, so an outage of the reformulation model can
never downgrade a delivered (if weak) answer into a failure.
"""

from typing import TYPE_CHECKING

from loguru import logger

from docint.utils.prompt_loader import load_localized_prompt

if TYPE_CHECKING:
    from llama_index.core.llms import LLM

DEFAULT_REFORMULATE_RETRIEVAL_PROMPT = (
    "A retrieval attempt failed: the answer generated from the retrieved "
    "evidence was flagged as not matching that evidence. Rewrite the retrieval "
    "query so that a fresh vector search is more likely to surface passages "
    "that actually answer the user's question.\n\n"
    "Rules:\n"
    "- Use the validator's reason to decide what to change: if the evidence was "
    "off-topic, shift the terms; if it was too narrow or too specific, broaden "
    "them; if the failed query drifted from the user's question, steer it back.\n"
    "- Prefer the wording the source documents would plausibly use over the "
    "user's phrasing.\n"
    "- Do not repeat the failed query verbatim.\n"
    "- Do not answer the question; only produce search terms.\n"
    "- Do not invent facts that are absent from the user's question.\n"
    "- Return ONLY the reformulated query, no preamble.\n\n"
    "User's question:\n{user_query}\n\n"
    "Failed retrieval query:\n{failed_query}\n\n"
    "Why the answer was rejected:\n{validation_reason}\n\n"
    "Reformulated retrieval query:\n"
)


class QueryReformulationAgent:
    """Rewrite a retrieval query that produced a rejected answer.

    Holds no conversation state: one call in, one query out, so the same
    instance is safe to build per request alongside the response validator.
    """

    def __init__(self, llm: "LLM | None" = None) -> None:
        """Bind the LLM used for reformulation.

        Args:
            llm (LLM | None): LLM used to rewrite the query. When ``None``,
                :meth:`reformulate` always returns ``None`` and the caller skips
                the retry.
        """
        self.llm = llm
        self._prompt_template = load_localized_prompt(
            "reformulate_retrieval",
            default=DEFAULT_REFORMULATE_RETRIEVAL_PROMPT,
        )

    def reformulate(
        self,
        *,
        user_query: str,
        failed_query: str | None = None,
        validation_reason: str | None = None,
    ) -> str | None:
        """Produce a fresh retrieval query for a rejected answer.

        Args:
            user_query (str): The question the user actually asked. Always the
                anchor — the reformulation targets this, not the failed query.
            failed_query (str | None): The retrieval query that produced the
                rejected answer. Defaults to the user query when absent.
            validation_reason (str | None): The validator's free-text reason for
                rejecting the answer, used to steer what changes.

        Returns:
            str | None: The reformulated query, or ``None`` when no usable
            reformulation could be produced — no LLM bound, the call failed, the
            model returned nothing, or it echoed a query that already failed.
        """
        if self.llm is None or not user_query.strip():
            return None

        prompt = self._prompt_template.format(
            user_query=user_query,
            failed_query=failed_query or user_query,
            validation_reason=validation_reason or "No reason recorded.",
        )

        try:
            response = self.llm.complete(prompt)
        except Exception as exc:
            logger.opt(exception=exc).warning("Query reformulation request failed")
            return None

        candidate = " ".join(str(response.text or "").split())
        if not candidate:
            return None

        # A query that repeats what already failed cannot retrieve anything new,
        # so spending a second generation on it only doubles the latency.
        already_tried = {q.casefold() for q in (failed_query, user_query) if q}
        if candidate.casefold() in already_tried:
            logger.debug("Query reformulation returned an already-tried query; skipping retry")
            return None

        return candidate
