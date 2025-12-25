"""Lightweight registry for agent tools."""

from typing import Any, Callable


class ToolRegistry:
    """Registry of callable tools agents may invoke."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a tool by name."""
        self._tools[name] = fn

    def get(self, name: str) -> Callable[..., Any] | None:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Return registered tool names."""
        return sorted(self._tools.keys())


def default_tool_registry() -> "ToolRegistry":
    """Create a registry with standard tools populated."""
    return ToolRegistry()
