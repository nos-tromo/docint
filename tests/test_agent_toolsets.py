"""The tools a chat turn offers for its retrieval target. Every query, clip and passage here is invented."""

from __future__ import annotations

import re

import pytest
from agent_fakes import ScriptedSearch, programme_passage, tool_call, tool_context, turn_budget
from llama_index.core.schema import NodeWithScore

from docint.agents.turn.budget import schema_tokens
from docint.agents.turn.tools import SearchFn, ToolRegistry
from docint.agents.turn.toolsets import toolset_for
from docint.core.retrieval.visual import RETRIEVAL_TARGETS

CLIP = "harbour_walk.webm"
#: Most words a description may take; every round of every turn is sent all of them.
MAX_DESCRIPTION_WORDS = 60


def _no_frames(key: str) -> list[NodeWithScore]:
    """List a clip's frames in a collection that holds none.

    Args:
        key (str): The clip's file name or content hash.

    Returns:
        list[NodeWithScore]: No frames.
    """
    return []


def _toolset(target: str, *, search: SearchFn | None = None, search_images: SearchFn | None = None) -> ToolRegistry:
    """Build the tools of a turn whose searches find nothing unless given.

    Args:
        target (str): The turn's retrieval target.
        search (SearchFn | None): The text search; one that finds nothing when not given.
        search_images (SearchFn | None): The image search; one that finds nothing when not given.

    Returns:
        ToolRegistry: The tools the target offers.
    """
    return toolset_for(
        target,
        search=search or ScriptedSearch(),
        search_images=search_images or ScriptedSearch(),
        list_keyframes=_no_frames,
    )


def _offered(registry: ToolRegistry) -> list[str]:
    """Name the tools a registry offers, in the order the model is shown them.

    Args:
        registry (ToolRegistry): The registry.

    Returns:
        list[str]: Each tool's name.
    """
    return [schema["function"]["name"] for schema in registry.schemas()]


def test_documents_offers_no_image_tools() -> None:
    """A documents turn answers from text: search and read, and a call to an image tool is refused as unknown."""
    registry = _toolset("documents")

    assert _offered(registry) == ["search", "read"]
    assert registry.names == {"search", "read"}
    refused = registry.call(tool_call("search_images", description="Regenjacken am Hafen"), tool_context())
    assert refused.text == "Error: there is no tool named 'search_images'; the tools you can call are search, read."


def test_visual_offers_only_image_tools_and_read() -> None:
    """A visual turn answers from pictures: the two image tools and read, and a call to the text search is refused."""
    registry = _toolset("visual")

    assert _offered(registry) == ["search_images", "keyframes", "read"]
    assert registry.names == {"search_images", "keyframes", "read"}
    refused = registry.call(tool_call("search", query="Hafenfest Sonntag"), tool_context())
    assert refused.text == (
        "Error: there is no tool named 'search'; the tools you can call are search_images, keyframes, read."
    )


def test_all_offers_every_tool() -> None:
    """The default target offers all four tools, the text search first and read last."""
    registry = _toolset("all")

    assert _offered(registry) == ["search", "search_images", "keyframes", "read"]
    assert registry.names == {"search", "search_images", "keyframes", "read"}


def test_each_tool_runs_the_callable_meant_for_it() -> None:
    """The two searches share one type, so a swapped pair would still type-check: each tool must reach its own."""
    search, search_images = ScriptedSearch([programme_passage()]), ScriptedSearch([])
    listed: list[str] = []

    def list_keyframes(key: str) -> list[NodeWithScore]:
        """Record which clip was asked for.

        Args:
            key (str): The clip's file name or content hash.

        Returns:
            list[NodeWithScore]: No frames.
        """
        listed.append(key)
        return []

    registry = toolset_for("all", search=search, search_images=search_images, list_keyframes=list_keyframes)
    ctx = tool_context()

    registry.call(tool_call("search", query="Hafenfest Sonntag"), ctx)
    registry.call(tool_call("search_images", description="Zwei Personen in Regenjacken"), ctx)
    registry.call(tool_call("keyframes", file=CLIP), ctx)

    assert search.queries == ["Hafenfest Sonntag"]
    assert search_images.queries == ["Zwei Personen in Regenjacken"]
    assert listed == [CLIP]


def test_every_call_builds_a_registry_for_its_own_turn() -> None:
    """The callables are per request, so a registry is never shared between turns."""
    first, second = ScriptedSearch([programme_passage()]), ScriptedSearch([programme_passage()])
    registry_one, registry_two = _toolset("documents", search=first), _toolset("documents", search=second)

    registry_two.call(tool_call("search", query="Stände Sonntag"), tool_context())
    registry_one.call(tool_call("search", query="Hafenfest"), tool_context())

    assert registry_one is not registry_two
    assert (first.queries, second.queries) == (["Hafenfest"], ["Stände Sonntag"])


@pytest.mark.parametrize("target", ["", "everything", "ALL"])
def test_an_unknown_target_is_a_programming_error(target: str) -> None:
    """The API validates targets, so one that reaches the tool set unknown is a bug and raises.

    Args:
        target (str): A target the API never passes on.
    """
    with pytest.raises(ValueError, match="retrieval target"):
        _toolset(target)


@pytest.mark.parametrize("target", RETRIEVAL_TARGETS)
def test_every_target_the_api_accepts_has_a_toolset(target: str) -> None:
    """The tool sets follow the API's vocabulary of targets: none it accepts is refused here.

    Args:
        target (str): A target the API accepts.
    """
    assert _offered(_toolset(target))


def _descriptions() -> dict[str, str]:
    """Read every tool's description as the model is shown it.

    Returns:
        dict[str, str]: Each description, by the tool's name.
    """
    return {schema["function"]["name"]: schema["function"]["description"] for schema in _toolset("all").schemas()}


def test_a_description_never_names_another_tool() -> None:
    """A documents turn offers no image tool, so no description may send the model to one: each stands alone."""
    descriptions = _descriptions()

    naming = {
        name: sorted(other for other in descriptions if other != name and re.search(rf"\b{re.escape(other)}\b", text))
        for name, text in descriptions.items()
    }

    assert {name: others for name, others in naming.items() if others} == {}


def test_a_description_stays_short() -> None:
    """Every round of every turn pays for the descriptions, and the loop must fit an 8k window."""
    words = {name: len(text.split()) for name, text in _descriptions().items()}

    assert {name: count for name, count in words.items() if count > MAX_DESCRIPTION_WORDS} == {}


def test_the_full_toolset_fits_the_schema_allowance_of_an_8k_turn() -> None:
    """All four schemas together cost no more than the tokens the 8k turn budget is sized for."""
    budget = turn_budget()

    assert schema_tokens(_toolset("all").schemas(), budget.char_token_ratio) <= budget.tool_schema_tokens


@pytest.mark.parametrize(
    ("tool", "phrases"),
    [
        ("search", ("documents and transcripts", "what they say", "[n]", "shown before")),
        (
            "search_images",
            (
                "pictures and video frames",
                "who appears",
                "clothing",
                "weather",
                "places",
                "what the picture would show",
                "[n]",
                "shown before",
            ),
        ),
        ("keyframes", ("time order", "specific clip", "passage from one", "[n]")),
        ("read", ("passage [n]", "a reply showed only in part")),
    ],
    ids=["search", "search_images", "keyframes", "read"],
)
def test_a_description_says_what_the_tool_is_for(tool: str, phrases: tuple[str, ...]) -> None:
    """The model picks a tool by its description: the text search by what is said, the image tools by what is seen.

    Args:
        tool (str): The tool's name.
        phrases (tuple[str, ...]): What its description must say, in any case.
    """
    text = _descriptions()[tool].lower()

    assert [phrase for phrase in phrases if phrase not in text] == []
