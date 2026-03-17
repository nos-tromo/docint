"""Response agent implementations."""

import json
import re
from typing import TYPE_CHECKING, Any

from loguru import logger

from docint.agents.types import ResponseAgent, RetrievalResult, Turn

if TYPE_CHECKING:
    from llama_index.core.llms import LLM

MAX_VALIDATION_SOURCES = 6
MAX_SOURCE_CHARS = 1200


class PassthroughResponseAgent(ResponseAgent):
    """No-op response post-processor.
    Returns retrieval results unchanged.
    """

    def finalize(self, result: RetrievalResult, turn: Turn) -> RetrievalResult:
        """Return retrieval results unchanged.

        Args:
            result (RetrievalResult): The retrieval result to finalize.
            turn (Turn): The current turn in the conversation.

        Returns:
            RetrievalResult: The unchanged retrieval result.
        """
        _ = turn  # placeholder until custom formatting is needed
        return result


class ResultValidationResponseAgent(ResponseAgent):
    """Validate response quality against retrieved chunks using an LLM."""

    def __init__(self, enabled: bool, llm: "LLM | None" = None) -> None:
        """Initialize the validation agent.

        Args:
            enabled (bool): Whether result validation is enabled.
            llm (LLM | None): LLM used for validation.
        """
        self.enabled = enabled
        self.llm = llm

    def finalize(self, result: RetrievalResult, turn: Turn) -> RetrievalResult:
        """Validate the answer against retrieved sources and set alert metadata.

        Args:
            result (RetrievalResult): The retrieval result to validate.
            turn (Turn): Current conversation turn.

        Returns:
            RetrievalResult: The retrieval result annotated with validation metadata.
        """
        if not self.enabled:
            return result
        if self.llm is None:
            self._set_validation_unavailable(
                result,
                reason="Validation model unavailable.",
            )
            return result
        if not result.answer:
            self._set_validation_unavailable(
                result,
                reason="No answer to validate.",
            )
            return result

        try:
            prompt = self._build_prompt(
                query=turn.user_input,
                answer=result.answer,
                sources=result.sources,
                summary_diagnostics=result.summary_diagnostics,
            )
            response = self.llm.complete(prompt)
            response_text = str(getattr(response, "text", "") or "").strip()
            if not response_text:
                self._set_validation_unavailable(
                    result,
                    reason="Validation model returned empty output.",
                )
                return result
            parsed = self._parse_response(response_text)
            summary_grounded = parsed.get("summary_grounded")
            sources_relevant = parsed.get("sources_relevant")

            if not isinstance(summary_grounded, bool) or not isinstance(
                sources_relevant, bool
            ):
                self._set_validation_unavailable(
                    result,
                    reason="Validation model returned invalid schema.",
                )
                return result

            mismatch = not (summary_grounded and sources_relevant)
            coverage_ratio, coverage_target, coverage_unit = (
                self._extract_coverage_metrics(result.summary_diagnostics)
            )
            if (
                coverage_ratio is not None
                and coverage_target is not None
                and coverage_unit in {"chunks", "posts"}
                and summary_grounded
                and coverage_ratio >= coverage_target
            ):
                # Chunk/post coverage in row-heavy collections can make
                # per-source relevance checks overly strict.
                mismatch = False

            result.validation_checked = True
            result.validation_mismatch = mismatch
            result.validation_reason = (
                str(parsed.get("reason", "Validation mismatch detected"))
                if mismatch
                else None
            )
            return result
        except json.JSONDecodeError:
            self._set_validation_unavailable(
                result,
                reason="Validation model returned non-JSON output.",
            )
            return result
        except Exception as exc:
            logger.warning("Response validation failed: {}", exc)
            self._set_validation_unavailable(
                result,
                reason=f"Validation request failed: {exc}",
            )
            return result

    def _set_validation_unavailable(
        self,
        result: RetrievalResult,
        *,
        reason: str,
    ) -> None:
        """Mark validation as unavailable with a user-facing reason.

        Args:
            result: Retrieval result to annotate.
            reason: Explanation why validation was not completed.
        """
        result.validation_checked = False
        result.validation_mismatch = None
        result.validation_reason = reason

    def _build_prompt(
        self,
        query: str,
        answer: str,
        sources: list[dict[str, Any]],
        summary_diagnostics: dict[str, Any] | None = None,
    ) -> str:
        """Build validation prompt for the secondary LLM check.

        Args:
            query (str): User query.
            answer (str): Generated answer.
            sources (list[dict[str, Any]]): Retrieved chunks/sources.
            summary_diagnostics (dict[str, Any] | None): Optional summary diagnostics for coverage context.

        Returns:
            str: Prompt asking for groundedness/relevance validation.
        """
        sources_text = self._sources_to_text(sources)
        diagnostics_text = self._summary_diagnostics_to_text(summary_diagnostics)
        return (
            "You are a strict response validator for a RAG system.\n"
            "Assess if the answer is faithful to the retrieved sources and if sources fit the query.\n"
            "Return JSON only with this schema:\n"
            "{\n"
            '  "summary_grounded": true|false,\n'
            '  "sources_relevant": true|false,\n'
            '  "reason": "short reason"\n'
            "}\n\n"
            f"Query:\n{query}\n\n"
            f"Answer:\n{answer}\n\n"
            f"{diagnostics_text}"
            f"Retrieved sources:\n{sources_text}\n"
        )

    def _summary_diagnostics_to_text(
        self, summary_diagnostics: dict[str, Any] | None
    ) -> str:
        """Render optional summary diagnostics for validator context.

        Args:
            summary_diagnostics (dict[str, Any] | None): Optional diagnostics payload.

        Returns:
            str: Prompt context block (or empty string).
        """
        if not isinstance(summary_diagnostics, dict):
            return ""
        total_documents = summary_diagnostics.get("total_documents")
        covered_documents = summary_diagnostics.get("covered_documents")
        coverage_ratio = summary_diagnostics.get("coverage_ratio")
        coverage_target = summary_diagnostics.get("coverage_target")
        coverage_unit = summary_diagnostics.get("coverage_unit")
        uncovered = summary_diagnostics.get("uncovered_documents")
        if isinstance(uncovered, list):
            uncovered_text = ", ".join(str(item) for item in uncovered if item)
        else:
            uncovered_text = ""
        return (
            "Summary diagnostics:\n"
            f"- total_documents: {total_documents}\n"
            f"- covered_documents: {covered_documents}\n"
            f"- coverage_ratio: {coverage_ratio}\n"
            f"- coverage_target: {coverage_target}\n"
            f"- coverage_unit: {coverage_unit}\n"
            f"- uncovered_documents: {uncovered_text or '(none)'}\n\n"
        )

    def _extract_coverage_metrics(
        self, summary_diagnostics: dict[str, Any] | None
    ) -> tuple[float | None, float | None, str | None]:
        """Parse optional coverage metrics from summary diagnostics.

        Args:
            summary_diagnostics (dict[str, Any] | None): Optional diagnostics payload.

        Returns:
            tuple[float | None, float | None, str | None]: Parsed coverage ratio,
            target, and coverage unit.
        """
        if not isinstance(summary_diagnostics, dict):
            return (None, None, None)
        coverage_ratio_raw = summary_diagnostics.get("coverage_ratio")
        coverage_target_raw = summary_diagnostics.get("coverage_target")
        coverage_unit_raw = summary_diagnostics.get("coverage_unit")
        if not isinstance(coverage_ratio_raw, (int, float, str)):
            return (None, None, None)
        if not isinstance(coverage_target_raw, (int, float, str)):
            return (None, None, None)
        try:
            coverage_ratio = float(coverage_ratio_raw)
            coverage_target = float(coverage_target_raw)
        except (TypeError, ValueError):
            return (None, None, None)
        coverage_unit = (
            str(coverage_unit_raw).strip().lower()
            if isinstance(coverage_unit_raw, (str, int, float))
            else None
        )
        return (coverage_ratio, coverage_target, coverage_unit)

    def _sources_to_text(self, sources: list[dict[str, Any]]) -> str:
        """Convert source dictionaries to compact text snippets for validation.

        Args:
            sources (list[dict[str, Any]]): Retrieved sources.

        Returns:
            str: Joined source snippets.
        """
        snippets: list[str] = []
        for idx, source in enumerate(sources[:MAX_VALIDATION_SOURCES], start=1):
            text = ""
            for key in ("text", "content", "chunk", "snippet", "node_text"):
                value = source.get(key)
                if value:
                    text = str(value)
                    break
            if not text:
                text = json.dumps(source, ensure_ascii=False)
            snippets.append(f"Source {idx}: {text[:MAX_SOURCE_CHARS]}")
        return "\n\n".join(snippets) if snippets else "(none)"

    def _parse_response(self, text: str) -> dict[str, Any]:
        """Parse JSON payload from the validator model response.

        Args:
            text (str): Raw model output.

        Returns:
            dict[str, Any]: Parsed validation result.
        """
        clean_text = self._extract_json_candidate(text)
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*?\}", clean_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    def _extract_json_candidate(self, text: str) -> str:
        """Extract candidate JSON text from potential markdown fenced output.

        Args:
            text (str): Raw model output.

        Returns:
            str: Candidate JSON payload.
        """
        clean_text = text.strip()
        if clean_text.startswith("```"):
            fenced = clean_text.split("```", maxsplit=2)
            if len(fenced) >= 3:
                clean_text = fenced[1]
                if clean_text.startswith("json"):
                    clean_text = clean_text[4:]
        return clean_text
