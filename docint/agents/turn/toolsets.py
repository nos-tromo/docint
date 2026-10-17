"""The tools an agent turn offers for its retrieval target.

A chat turn carries the evidence it may answer from: ``documents`` (text), ``visual``
(pictures and video frames) or ``all``. The turn offers the tools that fit, so a
documents turn is never invited to call an image tool and a visual turn never searches
text. This module is apart from :mod:`docint.agents.turn.tools` because it needs the
image tools, which import that module.
"""

from __future__ import annotations

from docint.agents.turn.image_tools import KeyframesFn, KeyframesTool, SearchImagesTool
from docint.agents.turn.tools import ReadTool, SearchFn, SearchTool, ToolRegistry


def toolset_for(target: str, *, search: SearchFn, search_images: SearchFn, list_keyframes: KeyframesFn) -> ToolRegistry:
    """Build the tools one turn offers for its retrieval target.

    ``documents`` offers ``search`` and ``read``; ``visual`` offers ``search_images``,
    ``keyframes`` and ``read``; ``all`` offers ``search``, ``search_images``,
    ``keyframes`` and ``read``, in that order. The registry is the turn's own: its tools
    hold no state of the turn, but the callables belong to the request.

    Args:
        target (str): The turn's retrieval target: ``all``, ``documents`` or ``visual``.
        search (SearchFn): Runs one text search through the turn's engine.
        search_images (SearchFn): Finds the pictures that show a description.
        list_keyframes (KeyframesFn): Lists one clip's frames in time order.

    Returns:
        ToolRegistry: A new registry holding the tools the target offers.

    Raises:
        ValueError: When ``target`` is none of the three. The API validates targets, so
            this is a programming error.
    """
    if target == "documents":
        return ToolRegistry([SearchTool(search), ReadTool()])
    if target == "visual":
        return ToolRegistry([SearchImagesTool(search_images), KeyframesTool(list_keyframes), ReadTool()])
    if target == "all":
        return ToolRegistry(
            [SearchTool(search), SearchImagesTool(search_images), KeyframesTool(list_keyframes), ReadTool()]
        )
    raise ValueError(f"Unknown retrieval target {target!r}; expected 'all', 'documents' or 'visual'.")
