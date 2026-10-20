"""Whether a deployment can run agent turns at all, and whether its chat endpoint accepts tools."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING

from loguru import logger

from docint.utils.env_cfg import load_agent_env

if TYPE_CHECKING:
    from docint.agents.turn.probe import ProbeOutcome

#: Below this window an agent turn has no room for its tool replies, so agent turns switch off rather than degrade.
MIN_CTX_WINDOW = 8192
#: Seconds agent turns stay off after the chat endpoint rejected a request offering tools.
TOOLS_REJECTION_COOLDOWN_S = 600


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
    """When the chat endpoint last rejected a request offering tools, per reasoning flag, remembered for a cooldown.

    Keyed by the reasoning flag and time alone. A process talks to one chat endpoint, but the
    request's reasoning toggle picks the agent client, and a request with reasoning on goes out
    with other settings than one with it off: an endpoint can refuse tools under one alone. While
    a rejection is remembered for a flag, no turn with that flag offers tools, so the endpoint is
    not sent, turn after turn, a request it refuses. A turn's refusal is remembered only once a
    probe carrying nothing of any turn is refused too (see :meth:`confirm`): a request can be
    refused for its own content, and one request must not switch agent turns off for every user.

    Attributes:
        cooldown_s (float): Seconds a rejection keeps agent turns off.
    """

    def __init__(
        self, *, cooldown_s: float = TOOLS_REJECTION_COOLDOWN_S, clock: Callable[[], float] = time.monotonic
    ) -> None:
        """Start with no rejection remembered and no probe running, under either flag.

        Args:
            cooldown_s (float): Seconds a rejection keeps agent turns off.
            clock (Callable[[], float]): Monotonic clock, injectable for tests.
        """
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._rejected_at: dict[bool, float] = {}
        self._probing = {False: threading.Lock(), True: threading.Lock()}

    def confirm(self, probe: Callable[[], ProbeOutcome], *, reasoning: bool) -> None:
        """Probe the endpoint after a turn's refusal of tools; remember the rejection only if the probe is refused.

        The probe must ask with the client of the turn's reasoning flag, under which the rejection
        is remembered. An accepted probe means something in that turn's own request was refused:
        its messages, images or tool schemas. Any other outcome shows nothing. Each probed refusal
        logs one warning saying which, with no text of any turn. At most one probe runs at a time
        per flag: a refusal arriving meanwhile neither probes nor is remembered. While a rejection
        is remembered for the flag, a further refusal tells nothing new: it neither probes nor
        warns, and the cooldown runs on unchanged.

        Args:
            probe (Callable[[], ProbeOutcome]): Asks the endpoint, in a request carrying nothing of
                any turn (see :func:`~docint.agents.turn.probe.probe_tools`).
            reasoning (bool): The reasoning flag the refused turn and the probe ran under.
        """
        reasoning = bool(reasoning)
        if self.cached(reasoning=reasoning):
            return
        lock = self._probing[reasoning]
        if not lock.acquire(blocking=False):
            return
        try:
            outcome = _probed(probe)
        finally:
            lock.release()
        if outcome == "refused":
            self.record(reasoning=reasoning)
        elif outcome == "accepted":
            logger.warning(
                "A turn with reasoning {} fell back after the chat endpoint rejected its request offering tools, but "
                "a probe offering one tool was accepted, so agent turns stay on: something in that turn's own "
                "request, its messages, images or tool schemas, was refused.",
                _mode(reasoning),
            )
        else:
            logger.warning(
                "A turn with reasoning {} fell back after the chat endpoint rejected its request offering tools, and "
                "a probe could not tell whether the endpoint accepts tools, so agent turns stay on.",
                _mode(reasoning),
            )

    def record(self, *, reasoning: bool) -> None:
        """Keep agent turns with this reasoning flag off for the cooldown, with one warning naming the requirement.

        Args:
            reasoning (bool): The reasoning flag the rejection was confirmed under.
        """
        reasoning = bool(reasoning)
        self._rejected_at[reasoning] = self._clock()
        logger.warning(
            "The chat endpoint rejects requests offering tools with reasoning {} (a probe confirmed it), so agent "
            "turns with reasoning {} are off for the next {} minutes and are answered as plain lookups. "
            'The chat endpoint must accept `tools` with `tool_choice: "auto"`.',
            _mode(reasoning),
            _mode(reasoning),
            round(self.cooldown_s / 60),
        )

    def cached(self, *, reasoning: bool) -> bool:
        """Return whether a rejection still keeps agent turns with this reasoning flag off.

        Args:
            reasoning (bool): The turn's reasoning flag.

        Returns:
            bool: ``True`` within the cooldown of the last rejection under that flag.
        """
        rejected_at = self._rejected_at.get(bool(reasoning))
        return rejected_at is not None and self._clock() - rejected_at < self.cooldown_s


def _mode(reasoning: bool) -> str:
    """Name a reasoning flag for a log line.

    Args:
        reasoning (bool): The flag.

    Returns:
        str: ``on`` or ``off``.
    """
    return "on" if reasoning else "off"


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


def confirm_tools_rejection(probe: Callable[[], ProbeOutcome], *, reasoning: bool) -> None:
    """Probe the endpoint after a turn's refusal of tools, and remember only a confirmed rejection.

    See :meth:`ToolsRejection.confirm`.

    Args:
        probe (Callable[[], ProbeOutcome]): Asks the endpoint, in a request carrying nothing of any
            turn, with the client of ``reasoning``.
        reasoning (bool): The reasoning flag the refused turn ran under.
    """
    TOOLS_REJECTION.confirm(probe, reasoning=reasoning)


def tools_rejection_cached(*, reasoning: bool) -> bool:
    """Return whether a tools rejection still keeps agent turns with this reasoning flag off.

    See :class:`ToolsRejection`.

    Args:
        reasoning (bool): The turn's reasoning flag.

    Returns:
        bool: ``True`` within the cooldown of the last rejection under that flag.
    """
    return TOOLS_REJECTION.cached(reasoning=reasoning)
