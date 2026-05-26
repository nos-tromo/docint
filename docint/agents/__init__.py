"""Agent orchestration package.

Provides lightweight interfaces and building blocks for conversation understanding,
clarification, retrieval, and response generation.
"""

from docint.agents.clarify import SimpleClarificationAgent
from docint.agents.context import TurnContext
from docint.agents.generation import (
    PassthroughResponseAgent,
    ResultValidationResponseAgent,
)
from docint.agents.orchestrator import AgentOrchestrator
from docint.agents.policies import ClarificationConfig, ClarificationPolicy
from docint.agents.retrieval import RAGRetrievalAgent
from docint.agents.tools import ToolRegistry, default_tool_registry
from docint.agents.types import (
    ClarificationRequest,
    IntentAnalysis,
    OrchestratorResult,
    RetrievalRequest,
    RetrievalResult,
    Turn,
)
from docint.agents.understanding import (
    ContextualUnderstandingAgent,
    SimpleUnderstandingAgent,
)

__all__ = [
    "AgentOrchestrator",
    "ClarificationConfig",
    "ClarificationPolicy",
    "ClarificationRequest",
    "ContextualUnderstandingAgent",
    "IntentAnalysis",
    "OrchestratorResult",
    "PassthroughResponseAgent",
    "RAGRetrievalAgent",
    "ResultValidationResponseAgent",
    "RetrievalRequest",
    "RetrievalResult",
    "SimpleClarificationAgent",
    "SimpleUnderstandingAgent",
    "ToolRegistry",
    "Turn",
    "TurnContext",
    "default_tool_registry",
]
