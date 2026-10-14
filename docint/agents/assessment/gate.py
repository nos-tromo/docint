"""Whether a deployment can run assessment turns at all."""

from functools import cache

from loguru import logger

from docint.utils.env_cfg import load_assessment_env

#: Below this window the answer step cannot hold its evidence beside a
#: judgement-sized reply, so the feature switches off rather than degrade.
MIN_CTX_WINDOW = 8192


@cache
def _warn_small_window(ctx_window: int) -> None:
    """Warn once per window size that assessment turns are switched off.

    Args:
        ctx_window (int): The context window that is too small.
    """
    logger.warning(
        "Assessment turns are off: the chat context window is {} tokens, below the {} they need. "
        "Raise OPENAI_CTX_WINDOW (and the model's num_ctx on Ollama) to enable them.",
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
