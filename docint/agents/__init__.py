"""Agent orchestration package.

Provides lightweight interfaces and building blocks for conversation understanding,
clarification, retrieval, and response generation.
"""

from docint.agents.types import (
    ClarificationRequest,
    IntentAnalysis,
    OrchestratorResult,
    RetrievalRequest,
    RetrievalResult,
    Turn,
)
from docint.agents.policies import ClarificationConfig, ClarificationPolicy
from docint.agents.orchestrator import AgentOrchestrator
from docint.agents.understanding import SimpleUnderstandingAgent
from docint.agents.clarify import SimpleClarificationAgent
from docint.agents.retrieval import RAGRetrievalAgent
from docint.agents.generation import PassthroughResponseAgent
from docint.agents.context import TurnContext
from docint.agents.tools import ToolRegistry, default_tool_registry

__all__ = [
    "AgentOrchestrator",
    "ClarificationPolicy",
    "ClarificationConfig",
    "ClarificationRequest",
    "IntentAnalysis",
    "OrchestratorResult",
    "PassthroughResponseAgent",
    "RAGRetrievalAgent",
    "RetrievalRequest",
    "RetrievalResult",
    "SimpleClarificationAgent",
    "SimpleUnderstandingAgent",
    "ToolRegistry",
    "default_tool_registry",
    "Turn",
    "TurnContext",
]
