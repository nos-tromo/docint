"""
Understanding agent implementations.
"""

import json
import re
from typing import Any, Iterable

from llama_index.core.llms import LLM

from docint.agents.context import TurnContext
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
    ) -> None:
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

    def analyze(self, turn: Turn, context: Any | None = None) -> IntentAnalysis:
        """
        Return intent, confidence, entities, and reason.

        Args:
            turn (Turn): The current turn in the conversation.
            context (Any | None, optional): The conversation context.

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


class ContextualUnderstandingAgent(UnderstandingAgent):
    """
    Understanding agent that uses an LLM to determine intent and rewrite queries.
    Replaces keyword matching with semantic reasoning via the LLM.
    """

    def __init__(self, llm: LLM):
        self.llm = llm

    def analyze(self, turn: Turn, context: Any | None = None) -> IntentAnalysis:
        """
        Analyze the turn using LLM reasoning.

        Args:
            turn (Turn): The current turn.
            context (Any | None): The conversation context (TurnContext).

        Returns:
            IntentAnalysis: The analysis result including intent and rewritten query.
        """
        history_str = ""
        if context and isinstance(context, TurnContext) and context.history:
            history_str = self._format_history(context.history)

        prompt = self._build_prompt(turn.user_input, history_str)

        try:
            response = self.llm.complete(prompt)
            result = self._parse_response(response.text)

            return IntentAnalysis(
                intent=result.get("intent", "qa"),
                confidence=0.9,  # LLM decided
                entities={"query": turn.user_input},  # Basic passthrough
                reason=result.get("reason"),
                rewritten_query=result.get("rewritten_query"),
            )
        except Exception:
            # Fallback for robustness
            return IntentAnalysis(
                intent="qa",
                confidence=0.5,
                entities={"query": turn.user_input},
                reason="LLM analysis failed, defaulting to QA",
            )

    def _format_history(self, history: list[dict[str, str]]) -> str:
        """
        Format the conversation history into a string.

        Args:
            history (list[dict[str, str]]): The conversation history as a list of messages

        Returns:
            str: A formatted string representation of the conversation history.
        """
        relevant_history = history[-4:]
        formatted = []
        for msg in relevant_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role in ("user", "assistant"):
                formatted.append(f"{role.capitalize()}: {content}")
        return "\n".join(formatted)

    def _build_prompt(self, query: str, context_str: str) -> str:
        """
        Construct the analysis prompt.

        Args:
            query (str): The user's current query.
            context_str (str): The formatted conversation context.

        Returns:
            str: The complete prompt to send to the LLM for analysis.
        """
        return (
            "You are an expert conversation analyst for a RAG system.\n"
            "Your task is to Analyze the User Query and extract the following JSON:\n"
            "{\n"
            '  "intent": "qa" | "ie" | "table" | "summary",\n'
            '  "rewritten_query": "The fully self-contained query resolving all references using context",\n'
            '  "reason": "Brief explanation of the intent choice"\n'
            "}\n\n"
            "Intents:\n"
            "- 'qa': General questions, searching for information.\n"
            "- 'ie': Request to extract specific entities (people, orgs) or relations.\n"
            "- 'table': Request to look up or extract tabular data/rows.\n"
            "- 'summary': Request to summarize a document or topic.\n\n"
            f"Conversation Context:\n{context_str}\n\n"
            f"User Query: {query}\n\n"
            "Output JSON only:"
        )

    def _parse_response(self, text: str) -> dict[str, Any]:
        """
        Parse the LLM response, handling potential markdown wrapping.

        Args:
            text (str): The raw text response from the LLM.

        Returns:
            dict[str, Any]: The parsed JSON content with intent analysis results.
        """
        clean_text = text.strip()
        # Remove markdown code blocks if present
        if clean_text.startswith("```"):
            clean_text = clean_text.split("```", 2)[1]
            if clean_text.startswith("json"):
                clean_text = clean_text[4:]

        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            # Attempt to find JSON object if mixed with text
            match = re.search(r"\{.*\}", clean_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return {
                "intent": "qa",
                "rewritten_query": None,
                "reason": "Failed to parse JSON",
            }
