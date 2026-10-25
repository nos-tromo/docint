"""The tools an agent turn offers for its retrieval target.

A chat turn carries the evidence it may answer from: ``documents`` (text), ``visual``
(pictures and video frames) or ``all``. The turn offers the tools that fit, so a
documents turn is never invited to call an image tool and a visual turn never searches
text. This module is apart from :mod:`docint.agents.turn.tools` because it needs the
image tools, the keyword tool and the topics tool, which import that module.
"""

from __future__ import annotations

from docint.agents.turn.image_tools import KeyframesFn, KeyframesTool, SearchImagesTool
from docint.agents.turn.keyword import KeywordMatcher, KeywordTool
from docint.agents.turn.tools import ReadTool, SearchFn, SearchTool, ToolRegistry
from docint.agents.turn.topics import TopicsSource, TopicsTool


def _keyword_tools(keyword: KeywordMatcher | None) -> list[KeywordTool]:
    """Build the keyword tool a turn is offered, if it has a match for one.

    Args:
        keyword (KeywordMatcher | None): The turn's keyword match.

    Returns:
        list[KeywordTool]: The tool; none when there is no match.
    """
    return [] if keyword is None else [KeywordTool(keyword)]


def _topics_tools(topics: TopicsSource | None) -> list[TopicsTool]:
    """Build the topics tool a turn is offered, if its collection has topics to offer.

    Args:
        topics (TopicsSource | None): The turn's topics.

    Returns:
        list[TopicsTool]: The tool; none when there are no topics.
    """
    return [] if topics is None else [TopicsTool(topics)]


def toolset_for(
    target: str,
    *,
    search: SearchFn,
    search_images: SearchFn,
    list_keyframes: KeyframesFn,
    keyword: KeywordMatcher | None = None,
    topics: TopicsSource | None = None,
) -> ToolRegistry:
    """Build the tools one turn offers for its retrieval target.

    The tools, in the order the model is shown them:

    * ``documents``: ``search``, ``keyword``, ``topics``, ``read``;
    * ``visual``: ``search_images``, ``keyframes``, ``read``;
    * ``all``: ``search``, ``search_images``, ``keyframes``, ``keyword``, ``topics``, ``read``.

    ``keyword`` is offered only when a match is given, ``topics`` only when topics are given,
    and neither to a ``visual`` turn, which answers from pictures. Build one per turn; the
    callables are bound to the request, so a registry is never shared.

    Args:
        target (str): The turn's retrieval target: ``all``, ``documents`` or ``visual``.
        search (SearchFn): Runs one text search through the turn's engine.
        search_images (SearchFn): Finds the pictures that show a description.
        list_keyframes (KeyframesFn): Lists one clip's frames in time order.
        keyword (KeywordMatcher | None): The turn's keyword match, bound to its collection,
            filters and target; ``None`` to offer no keyword tool.
        topics (TopicsSource | None): The turn's topics, bound to its collection and filters;
            ``None`` to offer no topics tool.

    Returns:
        ToolRegistry: A new registry holding the tools the target offers.

    Raises:
        ValueError: When ``target`` is none of the three. The API validates targets, so
            this is a programming error.
    """
    if target == "documents":
        return ToolRegistry([SearchTool(search), *_keyword_tools(keyword), *_topics_tools(topics), ReadTool()])
    if target == "visual":
        return ToolRegistry([SearchImagesTool(search_images), KeyframesTool(list_keyframes), ReadTool()])
    if target == "all":
        return ToolRegistry(
            [
                SearchTool(search),
                SearchImagesTool(search_images),
                KeyframesTool(list_keyframes),
                *_keyword_tools(keyword),
                *_topics_tools(topics),
                ReadTool(),
            ]
        )
    raise ValueError(f"Unknown retrieval target {target!r}; expected 'all', 'documents' or 'visual'.")
