"""The tools a chat turn offers for its retrieval target. Every query, clip and passage here is invented."""

from __future__ import annotations

import re

import pytest
from agent_fakes import PLANNED_SYSTEM_TOKENS, ScriptedSearch, programme_passage, shown_numbers, tool_call, tool_context
from llama_index.core.schema import NodeWithScore

from docint.agents.assessment.gate import MIN_CTX_WINDOW
from docint.agents.turn.budget import MIN_ROUND_ROOM, TurnBudget, history_allowance_tokens, schema_tokens
from docint.agents.turn.keyword import KeywordMatcher, KeywordResult
from docint.agents.turn.tools import SearchFn, ToolRegistry, ToolReply
from docint.agents.turn.toolsets import toolset_for
from docint.agents.turn.topics import TopicPassages, TopicsListing, TopicsSource
from docint.core.retrieval.visual import RETRIEVAL_TARGETS
from docint.utils.embed_chunking import estimate_tokens
from docint.utils.prompt_loader import load_localized_prompt

CLIP = "harbour_walk.webm"
#: Most words a description may take; every round of every turn is sent all of them.
MAX_DESCRIPTION_WORDS = 60
#: Characters per token the budget is measured with.
RATIO = 3.5
#: The system prompt a thinking model is sized for, past the planned one, as German text may push it up.
THINKING_SYSTEM_TOKENS = 1200
MESSAGE_TOKENS = 300
REMEDY = "shorten a tool description or raise the context window"
#: The fields of a keyword tool on a collection whose every index is ready.
KEYWORD_FIELDS = ("text", "author", "network", "uuid")
#: Most keyword may cost a turn at the tightest budget: its schema with all four fields plus its longest guide.
KEYWORD_TOKENS = 330
#: Most topics may cost a turn: its schema plus its guide line in the locale where it costs most.
TOPICS_TOKENS = 200


def _no_frames(key: str) -> list[NodeWithScore]:
    """List a clip's frames in a collection that holds none.

    Args:
        key (str): The clip's file name or content hash.

    Returns:
        list[NodeWithScore]: No frames.
    """
    return []


def _keyword_matcher(*fields: str) -> KeywordMatcher:
    """Build a keyword matcher that accepts every call and fails the test if it runs.

    Args:
        *fields (str): The fields it offers; the text alone when none is given.

    Returns:
        KeywordMatcher: The matcher.
    """

    def _check(words: str, field: str) -> str | None:
        """Accept the call.

        Args:
            words (str): The words.
            field (str): The field.

        Returns:
            str | None: No reason to refuse.
        """
        return None

    def _run(words: str, field: str) -> KeywordResult:
        """Fail: no test here expects the match to run.

        Args:
            words (str): The words.
            field (str): The field.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("The keyword match ran, which the test did not expect.")

    return KeywordMatcher(fields=fields or ("text",), check=_check, run=_run)


def _topics() -> TopicsSource:
    """Build a topics source that fails the test if it is read.

    Returns:
        TopicsSource: The source.
    """

    def _listing() -> TopicsListing:
        """Fail: no test here expects the topics to be listed.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("The topics were listed, which the test did not expect.")

    def _passages(topic_id: int) -> TopicPassages | None:
        """Fail: no test here expects a topic to be read.

        Args:
            topic_id (int): The topic.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("A topic was read, which the test did not expect.")

    return TopicsSource(listing=_listing, passages=_passages)


def _toolset(
    target: str,
    *,
    search: SearchFn | None = None,
    search_images: SearchFn | None = None,
    keyword: KeywordMatcher | None = None,
    topics: TopicsSource | None = None,
) -> ToolRegistry:
    """Build the tools of a turn whose searches fail the test if they run, unless given.

    Args:
        target (str): The turn's retrieval target.
        search (SearchFn | None): The text search; one that raises if it runs when not given.
        search_images (SearchFn | None): The image search; one that raises if it runs when not given.
        keyword (KeywordMatcher | None): The keyword match the turn offers; none when not given.
        topics (TopicsSource | None): The topics the turn offers; none when not given.

    Returns:
        ToolRegistry: The tools the target offers.
    """
    return toolset_for(
        target,
        search=ScriptedSearch() if search is None else search,
        search_images=ScriptedSearch() if search_images is None else search_images,
        list_keyframes=_no_frames,
        keyword=keyword,
        topics=topics,
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


def test_keyword_is_offered_to_the_turns_that_search_text() -> None:
    """A documents turn gains the keyword check between search and read, an all turn after the image tools."""
    matcher = _keyword_matcher()

    assert _offered(_toolset("documents", keyword=matcher)) == ["search", "keyword", "read"]
    assert _offered(_toolset("all", keyword=matcher)) == ["search", "search_images", "keyframes", "keyword", "read"]
    assert _toolset("all", keyword=matcher).names == {"search", "search_images", "keyframes", "keyword", "read"}


def test_a_visual_turn_is_never_offered_keyword_even_when_given_one() -> None:
    """A word match runs over text, so a visual turn keeps its tools and refuses a call to it as unknown."""
    registry = _toolset("visual", keyword=_keyword_matcher())

    assert _offered(registry) == ["search_images", "keyframes", "read"]
    refused = registry.call(tool_call("keyword", words="Hafenfest"), tool_context())
    assert _is_refusal_naming_the_offered_tools(refused, registry), refused.text


def test_topics_is_offered_to_the_turns_that_search_text_after_keyword() -> None:
    """Topics sits after keyword and before read on a documents or an all turn, and stands alone without keyword."""
    matcher, topics = _keyword_matcher(), _topics()

    assert _offered(_toolset("documents", keyword=matcher, topics=topics)) == ["search", "keyword", "topics", "read"]
    assert _offered(_toolset("all", keyword=matcher, topics=topics)) == [
        "search",
        "search_images",
        "keyframes",
        "keyword",
        "topics",
        "read",
    ]
    assert _offered(_toolset("documents", topics=topics)) == ["search", "topics", "read"]


def test_a_visual_turn_is_never_offered_topics_even_when_given_them() -> None:
    """Topics group text chunks, so a visual turn keeps its tools and refuses a call to it as unknown."""
    registry = _toolset("visual", keyword=_keyword_matcher(), topics=_topics())

    assert _offered(registry) == ["search_images", "keyframes", "read"]
    refused = registry.call(tool_call("topics"), tool_context())
    assert _is_refusal_naming_the_offered_tools(refused, registry), refused.text


@pytest.mark.parametrize(
    ("target", "tools"),
    [
        ("documents", ["search", "read"]),
        ("visual", ["search_images", "keyframes", "read"]),
        ("all", ["search", "search_images", "keyframes", "read"]),
    ],
)
def test_without_a_keyword_match_a_target_offers_the_tools_it_always_did(target: str, tools: list[str]) -> None:
    """A turn given no keyword match, by leaving the argument out, is offered exactly the tools it was before.

    Args:
        target (str): A retrieval target.
        tools (list[str]): The tools it offered before keyword existed, in order.
    """
    registry = toolset_for(target, search=ScriptedSearch(), search_images=ScriptedSearch(), list_keyframes=_no_frames)

    assert _offered(registry) == tools


def test_the_keyword_tool_runs_the_match_it_was_given() -> None:
    """Keyword is wired to the turn's own match, with the fields it offers, not to a stand-in."""
    runs: list[tuple[str, str]] = []

    def _run(words: str, field: str) -> KeywordResult:
        """Record the match and find the programme.

        Args:
            words (str): The words.
            field (str): The field.

        Returns:
            KeywordResult: One passage and its count.
        """
        runs.append((words, field))
        return KeywordResult(nodes=[programme_passage()], matches=1, unchecked=0, keywords=("hafenfest",))

    matcher = KeywordMatcher(fields=("text", "uuid"), check=lambda words, field: None, run=_run)
    registry = _toolset("documents", keyword=matcher)

    reply = registry.call(tool_call("keyword", words="Hafenfest"), tool_context())

    assert runs == [("Hafenfest", "text")]
    assert shown_numbers(reply.text) == [1]
    schema = next(item["function"] for item in registry.schemas() if item["function"]["name"] == "keyword")
    assert schema["parameters"]["properties"]["field"]["enum"] == ["text", "uuid"]


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
    registry = _toolset("all", keyword=_keyword_matcher(), topics=_topics())
    return {schema["function"]["name"]: schema["function"]["description"] for schema in registry.schemas()}


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


def _budget_for(target: str, *, reasoning: bool, system: int, keyword: KeywordMatcher | None = None) -> TurnBudget:
    """Build the budget of a turn in the smallest window agent turns run in, with a target's real schemas measured.

    Args:
        target (str): The turn's retrieval target; the schemas of the tools it offers are measured.
        reasoning (bool): Whether the model thinks before it answers.
        system (int): Tokens of the system prompt.
        keyword (KeywordMatcher | None): The keyword match the turn offers, its schema measured with the
            rest; none when not given.

    Returns:
        TurnBudget: The budget, its history at the allowance for ``reasoning``.
    """
    return TurnBudget(
        ctx_window=MIN_CTX_WINDOW,
        reasoning=reasoning,
        char_token_ratio=RATIO,
        system_tokens=system,
        tool_schema_tokens=schema_tokens(_toolset(target, keyword=keyword).schemas(), RATIO),
        history_tokens=history_allowance_tokens(reasoning),
        message_tokens=MESSAGE_TOKENS,
    )


def _keyword_lines(code: str, fields: tuple[str, ...]) -> str:
    """Write what a turn's system prompt gains when it offers keyword with these fields.

    Args:
        code (str): The locale.
        fields (tuple[str, ...]): The fields it offers, the text first; the fields line is there only when
            that is more than the text.

    Returns:
        str: The keyword line, then the fields line, each after a line break.
    """
    lines = [load_localized_prompt("agent_tools_keyword", default="", required=True, lang=code)]
    if len(fields) > 1:
        fields_line = load_localized_prompt("agent_tools_keyword_fields", default="", required=True, lang=code)
        lines.append(fields_line.format(fields=", ".join(fields)))
    return "".join(f"\n{line}" for line in lines)


def _keyword_guide_tokens(fields: tuple[str, ...]) -> int:
    """Estimate what keyword's lines add to the system prompt, in the locale where they cost most.

    Args:
        fields (tuple[str, ...]): The fields keyword offers, the text first.

    Returns:
        int: Estimated tokens.
    """
    return max(estimate_tokens(_keyword_lines(code, fields), RATIO) for code in ("en", "de"))


def _keyword_schema_tokens(fields: tuple[str, ...]) -> int:
    """Estimate what the keyword tool's schema costs in the prompt when it offers these fields.

    Args:
        fields (tuple[str, ...]): The fields keyword offers, the text first.

    Returns:
        int: Estimated tokens.
    """
    registry = _toolset("all", keyword=_keyword_matcher(*fields))
    return schema_tokens([item for item in registry.schemas() if item["function"]["name"] == "keyword"], RATIO)


def _budget_with_keyword(*, reasoning: bool, fields: tuple[str, ...]) -> TurnBudget:
    """Build the budget of an ``all`` turn that offers keyword, in the smallest window.

    The system prompt is the one planned for the model (see ``PLANNED_SYSTEM_TOKENS`` and
    ``THINKING_SYSTEM_TOKENS``) with keyword's lines added.

    Args:
        reasoning (bool): Whether the model thinks before it answers.
        fields (tuple[str, ...]): The fields keyword offers, the text first.

    Returns:
        TurnBudget: The budget.
    """
    planned = THINKING_SYSTEM_TOKENS if reasoning else PLANNED_SYSTEM_TOKENS
    return _budget_for(
        "all",
        reasoning=reasoning,
        system=planned + _keyword_guide_tokens(fields),
        keyword=_keyword_matcher(*fields),
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
    budget = _budget_for(target, reasoning=False, system=PLANNED_SYSTEM_TOKENS)
    three_results = 3 * budget.result_chars / budget.char_token_ratio

    assert budget.round_room >= three_results, (
        f"The {target} tools cost {budget.tool_schema_tokens} schema tokens and leave a plain model "
        f"{budget.round_room} tokens for tool results at {MIN_CTX_WINDOW}, under the {three_results:.0f} that three "
        f"full results take; {REMEDY}."
    )


def test_keyword_costs_a_turn_no_more_than_its_planned_tokens() -> None:
    """With all four fields, keyword's schema and its lines in the system prompt stay within the tokens planned."""
    schema, guide = _keyword_schema_tokens(KEYWORD_FIELDS), _keyword_guide_tokens(KEYWORD_FIELDS)

    assert schema + guide <= KEYWORD_TOKENS, (
        f"keyword costs {schema} schema tokens and {guide} prompt tokens, over the {KEYWORD_TOKENS} planned; "
        f"shorten the keyword lines or its schema, or plan for more."
    )


def _topics_cost() -> tuple[int, int]:
    """Estimate what the topics tool costs a turn: its schema, and its guide line where it costs most.

    Returns:
        tuple[int, int]: The schema tokens and the guide tokens.
    """
    registry = _toolset("all", topics=_topics())
    schema = schema_tokens([item for item in registry.schemas() if item["function"]["name"] == "topics"], RATIO)
    guide = max(
        estimate_tokens("\n" + load_localized_prompt("agent_tools_topics", default="", required=True, lang=code), RATIO)
        for code in ("en", "de")
    )
    return schema, guide


def test_topics_costs_a_turn_no_more_than_its_planned_tokens() -> None:
    """The topics tool's schema and its guide line stay within the tokens planned, which stay under 300."""
    schema, guide = _topics_cost()

    assert TOPICS_TOKENS <= 300
    assert schema + guide <= TOPICS_TOKENS, (
        f"topics costs {schema} schema tokens and {guide} prompt tokens, over the {TOPICS_TOKENS} planned; "
        f"shorten its line or its schema, or plan for more."
    )


def test_a_thinking_model_keeps_room_for_a_text_only_keyword_in_the_smallest_window() -> None:
    """Keyword on the text alone (its line, no fields line) leaves a thinking model room to run a turn at 8k."""
    budget = _budget_with_keyword(reasoning=True, fields=("text",))

    assert budget.available, (
        f"A text-only keyword costs {_keyword_schema_tokens(('text',))} schema tokens and leaves a thinking model "
        f"{budget.round_room} tokens for tool results at {MIN_CTX_WINDOW}, under the {MIN_ROUND_ROOM} a turn needs; "
        f"{REMEDY}."
    )


def test_a_plain_model_keeps_room_for_three_full_results_with_all_four_keyword_fields_in_the_smallest_window() -> None:
    """With all four fields, keyword still leaves a plain model, the default, room for three full tool results at 8k."""
    budget = _budget_with_keyword(reasoning=False, fields=KEYWORD_FIELDS)
    three_results = 3 * budget.result_chars / budget.char_token_ratio

    assert budget.round_room >= three_results, (
        f"Keyword with all four fields costs {_keyword_schema_tokens(KEYWORD_FIELDS)} schema tokens and leaves a "
        f"plain model {budget.round_room} tokens for tool results at {MIN_CTX_WINDOW}, under the "
        f"{three_results:.0f} that three full results take; {REMEDY}."
    )


def test_a_thinking_model_has_no_room_left_for_keyword_with_all_four_fields_in_the_smallest_window() -> None:
    """With all four fields, keyword leaves a thinking model at 8k under the room a turn needs; on purpose.

    This measures the planned worst case: the system prompt planned for a thinking model, with keyword's
    lines in the locale where they cost most. Thinking is off by default, so the default 8k deployment is
    not here. A thinking model there, on a collection whose every index is ready, is the case
    ``RAG.agent_turn`` handles by offering keyword on the text alone, which still fits (see the text-only
    test above); only a turn that has no room even for that is built without keyword. The fields' notes
    and the tool's limits stay in the schema for routing quality rather than being cut to fit.
    """
    budget = _budget_with_keyword(reasoning=True, fields=KEYWORD_FIELDS)

    assert not budget.available, (
        f"Keyword with all four fields leaves a thinking model {budget.round_room} tokens for tool results at "
        f"{MIN_CTX_WINDOW}, enough for the {MIN_ROUND_ROOM} a turn needs; it no longer has to fall back to the "
        f"text there, so pin that it fits instead."
    )


@pytest.mark.parametrize(
    ("tool", "anchors"),
    [
        ("search", ("documents", "transcripts", "[n]")),
        ("search_images", ("who appears", "clothing", "weather", "places", "[n]")),
        ("keyframes", ("specific clip", "[n]")),
        ("keyword", ("count", "exact", "[n]")),
        ("topics", ("topics", "share", "[n]")),
        ("read", ("rest of a passage",)),
    ],
    ids=["search", "search_images", "keyframes", "keyword", "topics", "read"],
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


def test_the_keyframes_index_is_the_frame_label_not_the_citation_number() -> None:
    """A frame has a citation number [n] and a label #n; a model that sees both must know index takes the label."""
    schema = next(item["function"] for item in _toolset("visual").schemas() if item["function"]["name"] == "keyframes")

    assert re.search(r"\bindex\b[^.;]*#n", schema["description"])
    assert "#n" in schema["parameters"]["properties"]["index"]["description"]
