"""Understanding agent implementations."""

import re
from typing import Iterable

from docint.agents.types import IntentAnalysis, Turn, UnderstandingAgent


class SimpleUnderstandingAgent(UnderstandingAgent):
    """
    Heuristic understanding with intent mapping and lightweight entity extraction.
    """

    def __init__(
        self,
        default_intent: str = "qa",
        default_confidence: float = 0.6,
        intent_keywords: dict[str, Iterable[str]] | None = None,
    ):
        """
        Initialize the SimpleUnderstandingAgent.

        Args:
            default_intent (str, optional): The default intent to use when no keywords match. Defaults to "qa".
            default_confidence (float, optional): The default confidence level for the intent. Defaults to 0.6.
            intent_keywords (dict[str, Iterable[str]] | None, optional): A mapping of intents to keywords for detection. Defaults to None.
        """        
        self.default_intent = default_intent
        self.default_confidence = default_confidence
        self.intent_keywords = intent_keywords or {
            "ie": ["entity", "extract", "ner", "relation"],
            "table": ["table", "row", "column", "csv"],
            "summary": ["summary", "summarize", "overview"],
        }

    def _detect_intent(self, text: str) -> tuple[str, float, str | None]:
        """
        Detect intent based on keyword matching.

        Args:
            text (str): The input text to analyze for intent.

        Returns:
            tuple[str, float, str | None]: A tuple containing the detected intent, confidence score, and an optional reason.
        """        
        lowered = text.lower()
        for intent, kws in self.intent_keywords.items():
            if any(kw in lowered for kw in kws):
                return intent, 0.8, f"matched keywords for {intent}"
        return self.default_intent, self.default_confidence, None

    def _extract_entities(self, text: str) -> dict[str, str]:
        """
        Simple entity extraction for 'query' and 'page' entities.

        Args:
            text (str): The input text from which to extract entities.

        Returns:
            dict[str, str]: A dictionary of extracted entities.
        """        
        entities: dict[str, str] = {}
        if text:
            entities["query"] = text
            page_match = re.search(r"page\s*(\d+)", text, flags=re.IGNORECASE)
            if page_match:
                entities["page"] = page_match.group(1)
        return entities

    def analyze(self, turn: Turn) -> IntentAnalysis:
        """
        Return intent, confidence, entities, and reason.
        
        Args:
            turn (Turn): The current turn in the conversation.

        Returns:
            IntentAnalysis: The result of the analysis.
        """
        intent, conf, reason = self._detect_intent(turn.user_input)
        entities = self._extract_entities(turn.user_input)
        return IntentAnalysis(
            intent=intent,
            confidence=conf if entities else min(conf, 0.3),
            entities=entities,
            tool=None,
            reason=reason,
        )
