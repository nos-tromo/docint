"""Whether a deployment can run assessment or agent turns at all."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING

from loguru import logger

from docint.utils.env_cfg import load_agent_env, load_assessment_env

if TYPE_CHECKING:
    from docint.agents.turn.probe import ProbeOutcome

#: Below this window the answer step cannot hold its evidence beside a
#: judgement-sized reply, nor an agent turn its tool replies, so the feature switches off rather than degrade.
MIN_CTX_WINDOW = 8192
#: Seconds agent turns stay off after the chat endpoint rejected a request offering tools.
TOOLS_REJECTION_COOLDOWN_S = 600


@cache
def _warn_small_window(ctx_window: int) -> None:
    """Warn once per window size that assessment turns are switched off.

    Args:
        ctx_window (int): The context window that is too small.
    """
    logger.warning(
        "Assessment turns are off: the chat context window is {} tokens, below the {} they need. "
        "Raise OPENAI_CTX_WINDOW (and the context window the chat endpoint serves) to enable them.",
        ctx_window,
        MIN_CTX_WINDOW,
    )


def assessment_available(ctx_window: int) -> bool:
    """Return whether assessment turns may run under this deployment's config.

    Args:
        ctx_window (int): The chat model's resolved context window in tokens.

    Returns:
        bool: ``True`` when the feature is enabled and the window is large enough.
    """
    if not load_assessment_env().enabled:
        return False
    if ctx_window >= MIN_CTX_WINDOW:
        return True
    _warn_small_window(ctx_window)
    return False


@cache
def _warn_small_agent_window(ctx_window: int) -> None:
    """Warn once per window size that agent turns are switched off.

    Args:
        ctx_window (int): The context window that is too small.
    """
    logger.warning(
        "Agent turns are off: the chat context window is {} tokens, below the {} they need. "
        "Raise OPENAI_CTX_WINDOW (and the context window the chat endpoint serves) to enable them.",
        ctx_window,
        MIN_CTX_WINDOW,
    )


def agent_available(ctx_window: int) -> bool:
    """Return whether agent turns may run under this deployment's config.

    Args:
        ctx_window (int): The chat model's resolved context window in tokens.

    Returns:
        bool: ``True`` when the feature is enabled and the window is large enough.
    """
    if not load_agent_env().enabled:
        return False
    if ctx_window >= MIN_CTX_WINDOW:
        return True
    _warn_small_agent_window(ctx_window)
    return False


class ToolsRejection:
    """When the chat endpoint last rejected a request offering tools, remembered for a cooldown.

    Keyed by nothing but time: a process talks to one chat endpoint. While a rejection is
    remembered no turn offers tools, so the endpoint is not sent, turn after turn, a request it
    refuses. A turn's refusal is remembered only once a probe carrying nothing of any turn is
    refused too (see :meth:`confirm`): a request can be refused for its own content, and one
    request must not switch agent turns off for every user.

    Attributes:
        cooldown_s (float): Seconds a rejection keeps agent turns off.
    """

    def __init__(
        self, *, cooldown_s: float = TOOLS_REJECTION_COOLDOWN_S, clock: Callable[[], float] = time.monotonic
    ) -> None:
        """Start with no rejection remembered and no probe running.

        Args:
            cooldown_s (float): Seconds a rejection keeps agent turns off.
            clock (Callable[[], float]): Monotonic clock, injectable for tests.
        """
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._rejected_at: float | None = None
        self._probing = threading.Lock()

    def confirm(self, probe: Callable[[], ProbeOutcome]) -> None:
        """Probe the endpoint after a turn's refusal of tools; remember the rejection only if the probe is refused.

        An accepted probe means the refusal was that request's own; any other outcome shows
        nothing. Each probed refusal logs one warning saying which, with no text of any turn. At
        most one probe runs at a time: a refusal arriving meanwhile neither probes nor is remembered.

        Args:
            probe (Callable[[], ProbeOutcome]): Asks the endpoint, in a request carrying nothing of
                any turn (see :func:`~docint.agents.turn.probe.probe_tools`).
        """
        if not self._probing.acquire(blocking=False):
            return
        try:
            outcome = _probed(probe)
        finally:
            self._probing.release()
        if outcome == "refused":
            self.record()
        elif outcome == "accepted":
            logger.warning(
                "A turn fell back after the chat endpoint rejected its request offering tools, but a probe offering "
                "tools was accepted, so agent turns stay on: that request was rejected for its own content."
            )
        else:
            logger.warning(
                "A turn fell back after the chat endpoint rejected its request offering tools, and a probe could not "
                "tell whether the endpoint accepts tools, so agent turns stay on."
            )

    def record(self) -> None:
        """Keep agent turns off for the cooldown, with one warning naming what the endpoint must accept."""
        self._rejected_at = self._clock()
        logger.warning(
            "The chat endpoint rejects requests offering tools (a probe confirmed it), so agent turns are off for "
            "the next {} minutes and chat turns are answered as plain lookups. "
            'The chat endpoint must accept `tools` with `tool_choice: "auto"`.',
            round(self.cooldown_s / 60),
        )

    @property
    def cached(self) -> bool:
        """Whether a rejection still keeps agent turns off.

        Returns:
            bool: ``True`` within the cooldown of the last rejection.
        """
        rejected_at = self._rejected_at
        return rejected_at is not None and self._clock() - rejected_at < self.cooldown_s


def _probed(probe: Callable[[], ProbeOutcome]) -> ProbeOutcome:
    """Run a probe; one that cannot run, say because its model could not be built, has shown nothing.

    Args:
        probe (Callable[[], ProbeOutcome]): The probe.

    Returns:
        ProbeOutcome: What it found; ``inconclusive`` when it raised.
    """
    try:
        return probe()
    except Exception as exc:
        logger.info("A probe of the chat endpoint's tool support could not run ({})", type(exc).__name__)
        return "inconclusive"


#: This process's tools rejection.
TOOLS_REJECTION = ToolsRejection()


def confirm_tools_rejection(probe: Callable[[], ProbeOutcome]) -> None:
    """Probe the endpoint after a turn's refusal of tools, and remember only a confirmed rejection.

    See :meth:`ToolsRejection.confirm`.

    Args:
        probe (Callable[[], ProbeOutcome]): Asks the endpoint, in a request carrying nothing of any turn.
    """
    TOOLS_REJECTION.confirm(probe)


def tools_rejection_cached() -> bool:
    """Return whether a tools rejection still keeps agent turns off (see :class:`ToolsRejection`).

    Returns:
        bool: ``True`` within the cooldown of the last rejection.
    """
    return TOOLS_REJECTION.cached
