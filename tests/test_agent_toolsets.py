"""The tools a chat turn offers for its retrieval target. Every query, clip and passage here is invented."""

from __future__ import annotations

import re

import pytest
from agent_fakes import ScriptedSearch, programme_passage, tool_call, tool_context
from llama_index.core.schema import NodeWithScore

from docint.agents.assessment.gate import MIN_CTX_WINDOW
from docint.agents.turn.budget import MIN_ROUND_ROOM, TurnBudget, history_allowance_tokens, schema_tokens
from docint.agents.turn.tools import SearchFn, ToolRegistry, ToolReply
from docint.agents.turn.toolsets import toolset_for
from docint.core.retrieval.visual import RETRIEVAL_TARGETS

CLIP = "harbour_walk.webm"
#: Most words a description may take; every round of every turn is sent all of them.
MAX_DESCRIPTION_WORDS = 60
#: Characters per token the budget is measured with.
RATIO = 3.5
#: The system prompt a plain model is sized for, and a thinking model's, which German text may push up.
PLAIN_SYSTEM_TOKENS = 900
THINKING_SYSTEM_TOKENS = 1200
MESSAGE_TOKENS = 300
REMEDY = "shorten a tool description or raise the context window"


def _no_frames(key: str) -> list[NodeWithScore]:
    """List a clip's frames in a collection that holds none.

    Args:
        key (str): The clip's file name or content hash.

    Returns:
        list[NodeWithScore]: No frames.
    """
    return []


def _toolset(target: str, *, search: SearchFn | None = None, search_images: SearchFn | None = None) -> ToolRegistry:
    """Build the tools of a turn whose searches fail the test if they run, unless given.

    Args:
        target (str): The turn's retrieval target.
        search (SearchFn | None): The text search; one that raises if it runs when not given.
        search_images (SearchFn | None): The image search; one that raises if it runs when not given.

    Returns:
        ToolRegistry: The tools the target offers.
    """
    return toolset_for(
        target,
        search=ScriptedSearch() if search is None else search,
        search_images=ScriptedSearch() if search_images is None else search_images,
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


def _is_refusal_naming_the_offered_tools(reply: ToolReply, registry: ToolRegistry) -> bool:
    """Tell whether a reply is an error that names every tool the registry offers.

    The registry's wording is pinned by its own tests; here only the facts matter.

    Args:
        reply (ToolReply): The reply to a call to a tool the registry does not offer.
        registry (ToolRegistry): The registry that answered.

    Returns:
        bool: ``True`` when the reply opens with ``Error:`` and names each offered tool as a whole word.
    """
    return reply.text.startswith("Error:") and all(re.search(rf"\b{name}\b", reply.text) for name in registry.names)


def test_documents_offers_no_image_tools() -> None:
    """A documents turn answers from text: search and read, and a call to an image tool is refused as unknown."""
    registry = _toolset("documents")

    assert _offered(registry) == ["search", "read"]
    assert registry.names == {"search", "read"}
    refused = registry.call(tool_call("search_images", description="Regenjacken am Hafen"), tool_context())
    assert _is_refusal_naming_the_offered_tools(refused, registry), refused.text


def test_visual_offers_only_image_tools_and_read() -> None:
    """A visual turn answers from pictures: the two image tools and read, and a call to the text search is refused."""
    registry = _toolset("visual")

    assert _offered(registry) == ["search_images", "keyframes", "read"]
    assert registry.names == {"search_images", "keyframes", "read"}
    refused = registry.call(tool_call("search", query="Hafenfest Sonntag"), tool_context())
    assert _is_refusal_naming_the_offered_tools(refused, registry), refused.text


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


def _budget_for(target: str, *, reasoning: bool, system: int) -> TurnBudget:
    """Build the budget of a turn in the smallest window agent turns run in, with a target's real schemas measured.

    Args:
        target (str): The turn's retrieval target; the schemas of the tools it offers are measured.
        reasoning (bool): Whether the model thinks before it answers.
        system (int): Tokens of the system prompt.

    Returns:
        TurnBudget: The budget, its history at the allowance for ``reasoning``.
    """
    return TurnBudget(
        ctx_window=MIN_CTX_WINDOW,
        reasoning=reasoning,
        char_token_ratio=RATIO,
        system_tokens=system,
        tool_schema_tokens=schema_tokens(_toolset(target).schemas(), RATIO),
        history_tokens=history_allowance_tokens(reasoning),
        message_tokens=MESSAGE_TOKENS,
    )


@pytest.mark.parametrize("target", RETRIEVAL_TARGETS)
def test_a_thinking_model_keeps_room_for_tool_results_in_the_smallest_window(target: str) -> None:
    """The tools a target offers leave a thinking model, with a long system prompt, room to run a turn.

    Args:
        target (str): The retrieval target whose schemas are measured.
    """
    budget = _budget_for(target, reasoning=True, system=THINKING_SYSTEM_TOKENS)

    assert budget.available, (
        f"The {target} tools cost {budget.tool_schema_tokens} schema tokens and leave a thinking model "
        f"{budget.round_room} tokens for tool results at {MIN_CTX_WINDOW}, under the {MIN_ROUND_ROOM} a turn needs; "
        f"{REMEDY}."
    )


@pytest.mark.parametrize("target", RETRIEVAL_TARGETS)
def test_a_plain_model_keeps_room_for_three_full_results_in_the_smallest_window(target: str) -> None:
    """The tools a target offers leave a plain model room to read three full tool results before it answers.

    Args:
        target (str): The retrieval target whose schemas are measured.
    """
    budget = _budget_for(target, reasoning=False, system=PLAIN_SYSTEM_TOKENS)
    three_results = 3 * budget.result_chars / budget.char_token_ratio

    assert budget.round_room >= three_results, (
        f"The {target} tools cost {budget.tool_schema_tokens} schema tokens and leave a plain model "
        f"{budget.round_room} tokens for tool results at {MIN_CTX_WINDOW}, under the {three_results:.0f} that three "
        f"full results take; {REMEDY}."
    )


@pytest.mark.parametrize(
    ("tool", "anchors"),
    [
        ("search", ("documents", "transcripts", "[n]")),
        ("search_images", ("who appears", "clothing", "weather", "places", "[n]")),
        ("keyframes", ("specific clip", "[n]")),
        ("read", ("rest of a passage",)),
    ],
    ids=["search", "search_images", "keyframes", "read"],
)
def test_a_description_says_what_the_tool_is_for(tool: str, anchors: tuple[str, ...]) -> None:
    """The model routes by what a description names: text by documents, pictures by what is seen, a clip, a passage.

    Only those anchors are pinned, so a shortening that keeps the routing still passes. ``[n]`` is
    pinned where a description promises numbered passages.

    Args:
        tool (str): The tool's name.
        anchors (tuple[str, ...]): What its description must name, in any case.
    """
    text = _descriptions()[tool].lower()

    assert [anchor for anchor in anchors if anchor not in text] == []
