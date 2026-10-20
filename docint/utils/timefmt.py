"""Clip offsets as people read them.

Neutral ground for every caller that names a moment in a clip: the extract's frame
captions and the keyframe label a chat model reads (``#3 @ 01:12``) use the one format.
"""

__all__ = ["format_short"]


def format_short(seconds: float | None) -> str:
    """Format seconds as ``mm:ss``, the way a frame caption reads.

    Args:
        seconds (float | None): Offset into a clip.

    Returns:
        str: The stamp, or ``""`` when there is no offset.
    """
    if seconds is None:
        return ""
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"
