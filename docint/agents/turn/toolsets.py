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

    The tools, in the order the model is shown them:

    * ``documents``: ``search``, ``read``;
    * ``visual``: ``search_images``, ``keyframes``, ``read``;
    * ``all``: ``search``, ``search_images``, ``keyframes``, ``read``.

    Build one per turn; the callables are bound to the request, so a registry is never shared.

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
