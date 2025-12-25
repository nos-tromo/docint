"""Understanding agent implementations."""

import re
from typing import Iterable

from docint.agents.types import IntentAnalysis, Turn, UnderstandingAgent


class SimpleUnderstandingAgent(UnderstandingAgent):
    """Heuristic understanding with intent mapping and lightweight entity extraction."""

    def __init__(
        self,
        default_intent: str = "qa",
        default_confidence: float = 0.6,
        intent_keywords: dict[str, Iterable[str]] | None = None,
    ):
        self.default_intent = default_intent
        self.default_confidence = default_confidence
        self.intent_keywords = intent_keywords or {
            "ie": ["entity", "extract", "ner", "relation"],
            "table": ["table", "row", "column", "csv"],
            "summary": ["summary", "summarize", "overview"],
        }

    def _detect_intent(self, text: str) -> tuple[str, float, str | None]:
        lowered = text.lower()
        for intent, kws in self.intent_keywords.items():
            if any(kw in lowered for kw in kws):
                return intent, 0.8, f"matched keywords for {intent}"
        return self.default_intent, self.default_confidence, None

    def _extract_entities(self, text: str) -> dict[str, str]:
        entities: dict[str, str] = {}
        if text:
            entities["query"] = text
            page_match = re.search(r"page\s*(\d+)", text, flags=re.IGNORECASE)
            if page_match:
                entities["page"] = page_match.group(1)
        return entities

    def analyze(self, turn: Turn) -> IntentAnalysis:
        """Return intent, confidence, entities, and reason."""
        intent, conf, reason = self._detect_intent(turn.user_input)
        entities = self._extract_entities(turn.user_input)
        return IntentAnalysis(
            intent=intent,
            confidence=conf if entities else min(conf, 0.3),
            entities=entities,
            tool=None,
            reason=reason,
        )
