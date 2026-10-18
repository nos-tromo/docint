"""One agent turn: the model reads the conversation, calls tools, reads every reply, and answers.

The loop follows an SDK agent loop. Each round is one model call, offered the turn's tools with
``tool_choice="auto"``: a round that calls tools has every call answered under its call's id and
goes again, and a round that calls none is the answer, streamed as it comes. A round's text is
provisional (see :func:`~docint.agents.turn.stream.stream_round`); text written before calls
becomes a ``note`` step, which tells the client to withdraw whatever of it was streamed.

The first round is held whole: none of its text is shown before it ends, and text it wrote
before calls becomes its note unseen. Held so, it can be asked again, once per turn, for either
of two causes. With a search note, a first reply that calls no tool is dropped as if never
written, its reasoning too, and the round runs again with the note joined to the user's message.
A first round that breaks off after its first chunk has shown nothing, and runs again as it was
sent: the break-off is not about searching. The second try is no tool round, but a model call
under the first round's rules: the limits are checked before it, counting a note still due; it
waits for a model slot as the first round does, and fails as the first round does, though a
refusal of it no longer means the endpoint takes no tools. It streams as later rounds do; if it
too calls no tool, its reply is the answer. Until a round's calls are in the transcript, nothing
has answered the search note, so an answer round sent before then carries the final note alone:
the search note asks for a search that round cannot make.

Limits make the model answer in one more round, its last: the same transcript plus the final note
as a user message (joined to a user message it would follow: strict chat templates refuse two in
a row), the tools still declared but ``tool_choice="none"``. An endpoint that ignores
``tool_choice`` may answer that round with tool calls and no text: the round is then sent once
more declaring no tools, and that reply is the answer. ``agent.reason`` names the
first cause: the rounds spent (``round_budget``), the time spent (``time_budget``), no room left
for one more full tool reply or for a round's own calls (``context_budget``), or two rounds in a
row that added no passage (``idle``). Limits are checked between rounds, and the time also
bounds every call's I/O; a round is never cut off while it streams, so an answer already streaming
is never abandoned. An answer the endpoint cut at its length limit is kept, as written, and flagged
``answer_cut``, the one reason that replaces the first cause: a cut answer matters more than a
limit or a failure before it.

Calls and room. Every call id gets a reply, answered by
:class:`~docint.agents.turn.calls.RoundCalls`: concurrently on the search slots, numbered in call
order, within the room and the time left. The turn's room for tool replies is the budget's
``round_room`` less the final note. A round's own cost, its assistant message and the least
reply each of its calls needs, must fit the room left; a round that does not is withdrawn
before any of its calls runs, its shown text withdrawn by its note, and the model answers.
Only a call its tool answered is a step in the trail; an error reply, for arguments that will
not do or for a call that did not run or failed, reaches the model alone. A call that could not
be prepared, whose search failed or found no slot, or whose answer raised, marks the turn
``research_failed`` without ending it: evidence may be missing.

Pictures. With a thumbnail fetcher, the pictures a round admits are shown to the model after its
tool replies, and a round refused with them is sent once more without (see
:mod:`~docint.agents.turn.images` and :meth:`AgentLoop._round`).

A pinned scope is answered from the pinned passages alone, in one round that offers no tools (see
:mod:`~docint.agents.turn.scope` and :meth:`AgentLoop._scoped`).

Failures. Model calls run on the loop thread, one model slot each. Before the turn has shown
anything, as when the first round's second try breaks off before its text shows, a failure goes
to the caller, who can still answer another way: no model slot within
``MODEL_SLOT_WAIT_S`` raises :class:`~docint.agents.assessment.bounded.NoSlot`, the first call
refused as sent (HTTP 400 or 422, not an overflow) raises :class:`ToolsUnavailable`, and anything
else is raised as it came. Once the turn has shown a step or text, it ends with an
:class:`AgentResult` unless the answer round itself fails; a result always comes from an answer
round that ran:

* a round offered tools that finds no model slot within ``MODEL_SLOT_WAIT_S`` (never past the
  time left), or whose model call fails, makes the model answer (``research_failed``); text it
  showed before breaking off is withdrawn by a note, as a round ending in calls withdraws its text;
* a round the endpoint finds too long is retried once as the answer, every tool reply cut to
  ``OVERFLOW_REPLY_SHARE`` (``context_budget``); with nothing to cut, or on a second overflow, it
  raises ``ValueError`` naming the context window;
* the answer round waits for a model slot up to the time left, and at least ``MODEL_SLOT_WAIT_S``;
  without one it raises ``NoSlot``, and any other failure of it raises the error as it came: there
  is no answer to keep, and an empty or partial one must not pass as whole. Its resend without
  tools fails the same way, an overflow of it raising ``ValueError`` naming the context window.

Nothing here logs a query, an answer, a tool argument or document text: only counts, tool names,
exception types and reason tokens.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import openai
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock, ToolCallBlock
from llama_index.core.schema import BaseNode, NodeWithScore
from loguru import logger

from docint.agents.assessment.bounded import MODEL_SLOT_POOL, MODEL_SLOT_WAIT_S, SEARCH_SLOT_POOL, NoSlot
from docint.agents.assessment.ledger import EvidenceLedger, is_context_overflow
from docint.agents.assessment.split import SentinelStreamSplitter
from docint.agents.turn.budget import MESSAGE_OVERHEAD_TOKENS, TurnBudget, message_cost
from docint.agents.turn.calls import RoundCalls
from docint.agents.turn.images import TurnImages
from docint.agents.turn.scope import EvidenceDoesNotFit, scope_message
from docint.agents.turn.stream import Provisional, RoundDone, stream_round
from docint.agents.turn.tools import ToolContext, ToolRegistry, nested_too_deeply
from docint.agents.types import AgentAction, AgentProgress
from docint.utils.text_budget import fit_with_marker

if TYPE_CHECKING:
    from llama_index.core.llms import LLM

    from docint.agents.turn.history import ConversationWindow
    from docint.core.retrieval.visual import ThumbnailFetcher

#: Longest note, in characters, that a round's text before its calls leaves in the trail and the transcript.
NOTE_CHARS = 500
#: Share of each tool reply the overflow retry keeps.
OVERFLOW_REPLY_SHARE = 0.5
#: Rounds in a row that add no passage before the model must answer.
IDLE_ROUNDS = 2
#: HTTP statuses with which an endpoint refuses a request as sent, rather than failing to serve it.
_REFUSED_STATUSES = frozenset({400, 422})
#: What a second overflow raises; the API reports a ``ValueError`` naming the context window as ``context_overflow``.
_OVERFLOW_MESSAGE = "The agent turn's prompt exceeds the model's context window."


class ToolsUnavailable(Exception):
    """The endpoint refused the turn's first call, tools declared, before any token.

    Nothing was shown, so the caller answers without tools; the refusal is the ``__cause__``.
    """


@dataclass(frozen=True)
class AgentResult:
    """What an agent turn produced, for normalization and persistence.

    Attributes:
        raw_text (str): The answer exactly as the model wrote it, sentinel included, reasoning
            left out; empty only when the answer round wrote nothing. A result always comes from
            an answer round that ran.
        nodes (list[NodeWithScore]): The turn's evidence numbered in ledger order: every passage
            a tool reply showed, or a scoped turn's pinned passages, stamped with its
            ``citation_index``; one whose picture the model was shown also carries
            ``VISUAL_IMAGES_ATTACHED_KEY``.
        steps (int): Tool rounds whose calls were answered; a round withdrawn for want of room
            is not counted.
        actions (list[dict[str, Any]]): The trail: one ``{action, detail, found}`` per note and
            per call its tool answered, in order; a scoped turn's is its ``seed`` alone.
        reason (str | None): The first cause that cut the turn short or left evidence out: a
            limit, a failed round, a call that could not be prepared, a failed search or missing
            search slot; ``answer_cut``, replacing any of these, when the endpoint cut the answer
            at its length limit; ``None`` when there was none.
        images_attached (int): Pictures the model was shown: those in the transcript when the
            turn ended, none once a refusal took them back.
        reasoning (str | None): Every round's reasoning in order, joined by blank lines; ``None``
            when there was none. Never logged.
    """

    raw_text: str
    nodes: list[NodeWithScore]
    steps: int
    actions: list[dict[str, Any]]
    reason: str | None
    images_attached: int = 0
    reasoning: str | None = None


@dataclass
class _Turn:
    """The state of one running turn.

    Attributes:
        ctx (ToolContext): The tools' evidence, budget, LLM view and memory.
        base (list[ChatMessage]): The system prompt, the history and the user's message.
        schemas (list[dict[str, Any]]): The tools as every round declares them.
        started (float): The clock's reading when the turn began.
        images (TurnImages): The pictures shown to the model; none unless the turn has a fetcher.
        transcript (list[ChatMessage]): The rounds so far: assistant messages and tool replies.
        spent (int): Tokens the transcript costs, and the search note once it is joined.
        steps (int): Tool rounds whose calls were answered; a withdrawn round is not counted.
        idle (int): Rounds in a row that added no passage.
        model_calls (int): Model calls begun.
        shown (bool): Whether anything has been yielded.
        reason (str | None): The first cause that cut the turn short or left evidence out, until
            an answer cut at the length limit replaces it with ``answer_cut``.
        shortened (bool): Whether the tool replies were cut after an overflow.
        actions (list[dict[str, Any]]): The trail so far.
        reasoning (list[str]): Each round's reasoning.
        asked_again (bool): Whether the turn's one re-ask of the first round is taken, by a reply
            that called no tool or by a break-off.
        note_due (bool): Whether the search note waits to join the user's message: it does once the
            first round runs again, and never when a limit makes the model answer first.
        bare_base (list[ChatMessage] | None): The base as it was before the search note joined it,
            for an answer round sent while no round has answered to the note; ``None`` until then.
    """

    ctx: ToolContext
    base: list[ChatMessage]
    schemas: list[dict[str, Any]]
    started: float
    images: TurnImages
    transcript: list[ChatMessage] = field(default_factory=list)
    spent: int = 0
    steps: int = 0
    idle: int = 0
    model_calls: int = 0
    shown: bool = False
    reason: str | None = None
    shortened: bool = False
    actions: list[dict[str, Any]] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    asked_again: bool = False
    note_due: bool = False
    bare_base: list[ChatMessage] | None = None

    def stop(self, reason: str) -> None:
        """Record why the model must answer now, keeping the first cause.

        Args:
            reason (str): The reason token.
        """
        self.reason = self.reason or reason


@dataclass
class _Streamed:
    """How one round's stream went.

    Attributes:
        splitter (SentinelStreamSplitter): Swaps the sentinel for the label in what the round shows.
        done (RoundDone | None): How the round ended; ``None`` when it failed.
        error (Exception | None): Why it failed.
        sent (str): The round's text that went out as provisional text.
        shown (bool): Whether any of its text was yielded.
        began (bool): Whether its first event arrived, for a round held whole its first chunk; a
            round that failed without one was refused as sent or never reached.
    """

    splitter: SentinelStreamSplitter = field(default_factory=SentinelStreamSplitter)
    done: RoundDone | None = None
    error: Exception | None = None
    sent: str = ""
    shown: bool = False
    began: bool = False


def refused_as_sent(exc: BaseException) -> bool:
    """Return whether an endpoint refused a request as sent rather than failed to serve it.

    Args:
        exc (BaseException): How a model call failed.

    Returns:
        bool: ``True`` for an HTTP 400 or 422 that is not a context overflow.
    """
    return (
        isinstance(exc, openai.APIStatusError) and exc.status_code in _REFUSED_STATUSES and not is_context_overflow(exc)
    )


def _raise_if_refused(turn: _Turn, exc: Exception) -> None:
    """Raise :class:`ToolsUnavailable` when the endpoint refused the turn's first call as sent.

    Only before anything was shown: the caller then answers without tools.

    Args:
        turn (_Turn): The turn.
        exc (Exception): How the model call failed.

    Raises:
        ToolsUnavailable: For an HTTP 400 or 422 answer, not an overflow, to the turn's first
            model call with nothing shown yet; the refusal is its ``__cause__``.
    """
    if refused_as_sent(exc) and not turn.shown and turn.model_calls == 1:
        raise ToolsUnavailable("The chat endpoint refused a request offering tools.") from exc


def _echoed(arguments: str) -> str:
    """Return a call's arguments as the transcript echoes them back.

    Chat templates parse the arguments of an earlier call as JSON, so arguments that are not a
    JSON object would fail the next round. They are echoed as an empty object, and so are
    arguments the registry refuses as nested too deeply (the same rule decides both), so no
    endpoint ever re-parses them; the call's error reply says what was wrong.

    Args:
        arguments (str): The arguments as the model sent them.

    Returns:
        str: The arguments when they are a JSON object the registry reads, else ``{}``.
    """
    try:
        value = json.loads(arguments)
    except (ValueError, RecursionError):
        return "{}"
    return arguments if isinstance(value, dict) and not nested_too_deeply(value) else "{}"


def _with_note(messages: Sequence[ChatMessage], note: str) -> list[ChatMessage]:
    """Add the final note to the messages as a user message, never as a second one in a row.

    Strict chat templates refuse two user messages in a row, so a note that would follow one
    joins it: at the end of its last text block, or as a text block of its own after its others
    (a picture, say). The messages given are never changed.

    Args:
        messages (Sequence[ChatMessage]): The prompt so far, ending in a user or a tool message.
        note (str): The final note.

    Returns:
        list[ChatMessage]: The prompt with the note added.
    """
    last = messages[-1]
    if last.role != MessageRole.USER:
        return [*messages, ChatMessage(role=MessageRole.USER, content=note)]
    blocks = list(last.blocks)
    if blocks and isinstance(blocks[-1], TextBlock):
        blocks[-1] = TextBlock(text=f"{blocks[-1].text}\n\n{note}")
    else:
        blocks.append(TextBlock(text=note))
    joined = ChatMessage(role=MessageRole.USER, blocks=blocks, additional_kwargs=dict(last.additional_kwargs))
    return [*messages[:-1], joined]


def _shortened(transcript: Sequence[ChatMessage]) -> list[ChatMessage] | None:
    """Cut every tool reply to ``OVERFLOW_REPLY_SHARE`` of its length, marking what it lost.

    Args:
        transcript (Sequence[ChatMessage]): The rounds so far.

    Returns:
        list[ChatMessage] | None: The transcript with its replies cut; ``None`` when no reply is
        long enough for the cut to shorten it.
    """
    cut = False
    kept: list[ChatMessage] = []
    for message in transcript:
        text = message.content or ""
        if message.role == MessageRole.TOOL:
            shorter, trimmed = fit_with_marker(text, int(len(text) * OVERFLOW_REPLY_SHARE))
            if trimmed and len(shorter) < len(text):
                message = ChatMessage(
                    role=MessageRole.TOOL, content=shorter, additional_kwargs=dict(message.additional_kwargs)
                )
                cut = True
        kept.append(message)
    return kept if cut else None


class AgentLoop:
    """Run one chat turn as an agent loop over the turn's tools."""

    def __init__(
        self,
        *,
        llm: LLM,
        registry: ToolRegistry,
        budget: TurnBudget,
        system_prompt: str,
        final_note: str,
        max_rounds: int,
        time_budget_s: float,
        max_calls_per_round: int,
        char_token_ratio: float,
        llm_view: Callable[[BaseNode], BaseNode],
        clock: Callable[[], float] = time.monotonic,
        model_slots: threading.Semaphore = MODEL_SLOT_POOL,
        search_slots: threading.Semaphore = SEARCH_SLOT_POOL,
        reasoning_open: bool = False,
        scope_note: str = "",
        fetch_thumbnails: ThumbnailFetcher | None = None,
        legend_template: str = "",
        max_images: int = 0,
        search_note: str = "",
    ) -> None:
        """Bind the model, the tools and the limits.

        Args:
            llm (LLM): The chat model every round calls; its ``stream_chat`` must pass the raw
                chunks on ``raw``.
            registry (ToolRegistry): The tools the turn offers; one per turn.
            budget (TurnBudget): The turn's budget, measured on this system prompt, these tools'
                schemas, the window and the message.
            system_prompt (str): The formatted system prompt, the window's summary block in it.
            final_note (str): What the last round tells the model: answer now, no more tools.
            max_rounds (int): Rounds that may call tools; ``0`` allows none.
            time_budget_s (float): Seconds the turn may take before the model must answer.
            max_calls_per_round (int): Calls one round may run; the rest are skipped.
            char_token_ratio (float): Characters per token, the ratio the budget was measured with.
            llm_view (Callable[[BaseNode], BaseNode]): Hides the metadata the model must not see.
            clock (Callable[[], float]): Monotonic clock, injectable for tests.
            model_slots (threading.Semaphore): The model-call slots shared by the process
                (``MODEL_SLOT_POOL``); injectable for tests.
            search_slots (threading.Semaphore): The slots tool I/O runs in, shared by the process
                (``SEARCH_SLOT_POOL``); injectable for tests.
            reasoning_open (bool): Whether thinking was asked for, so a reply may open in reasoning
                the chat template began; pass it only when the request enables thinking.
            scope_note (str): What a scoped turn's message tells the model ahead of the pinned
                passages: answer from them alone; left out when empty.
            fetch_thumbnails (ThumbnailFetcher | None): Fetches the stored thumbnails of the
                pictures a round admits, so the model sees them (see
                :mod:`~docint.agents.turn.images`); ``None`` shows no picture.
            legend_template (str): Tells the model the pictures are its sources, with an
                ``{image_legend}`` slot for the lines tying each to its number.
            max_images (int): The most pictures a turn shows; ``0`` shows none.
            search_note (str): What a first round that called no tool is told when it is asked
                again: look up what the answer relies on first. Empty, such a reply is the answer.

        Raises:
            ValueError: If ``char_token_ratio`` is not positive, or is not the ratio the budget
                was measured with: the transcript is priced against the budget's room.
        """
        if char_token_ratio != budget.char_token_ratio:
            raise ValueError(
                f"char_token_ratio {char_token_ratio} is not the budget's {budget.char_token_ratio}; "
                "the turn's room was measured with the budget's."
            )
        self._llm = llm
        self._registry = registry
        self._budget = budget
        self._system_prompt = system_prompt
        self._final_note = final_note
        self._max_rounds = max_rounds
        self._time_budget_s = time_budget_s
        self._ratio = char_token_ratio
        self._llm_view = llm_view
        self._clock = clock
        self._model_slots = model_slots
        self._reasoning_open = reasoning_open
        self._scope_note = scope_note
        self._fetch_thumbnails = fetch_thumbnails
        self._legend_template = legend_template
        self._max_images = max_images
        self._search_note = search_note
        self._room = budget.round_room - message_cost(final_note, char_token_ratio)
        #: What the pinned passages may take of the room: the scope note and its envelope come first.
        self._scope_room = self._room - message_cost(scope_note, char_token_ratio)
        self._calls = RoundCalls(
            registry=registry,
            result_chars=budget.result_chars,
            max_calls_per_round=max_calls_per_round,
            char_token_ratio=char_token_ratio,
            search_slots=search_slots,
        )

    def _time_left(self, turn: _Turn) -> float:
        """Return the seconds the turn has left.

        Args:
            turn (_Turn): The turn.

        Returns:
            float: The time left; negative once the budget is spent.
        """
        return self._time_budget_s - (self._clock() - turn.started)

    def run(
        self,
        *,
        window: ConversationWindow,
        message: str,
        ledger: EvidenceLedger,
        pinned: Sequence[NodeWithScore] | None = None,
    ) -> Iterator[AgentProgress | str | AgentResult]:
        """Run the turn.

        Args:
            window (ConversationWindow): The conversation before the message; its summary block
                is already in the system prompt, and its recent exchanges are sent as messages.
            message (str): The user's message, verbatim.
            ledger (EvidenceLedger): The turn's evidence, empty, its cap set by the caller.
            pinned (Sequence[NodeWithScore] | None): The passages the user pinned: the turn then
                answers from them alone, in one round that offers no tools (see :meth:`_scoped`).
                ``None`` for a turn without a scope; an empty scope is still one, never widened.

        Yields:
            AgentProgress | str | AgentResult: Steps and answer text as they come, the label
            swapped in for the sentinel, then exactly one result.

        Raises:
            EvidenceDoesNotFit: Before anything is yielded, when the pinned passages cannot all be
                held, shown legibly in the room, or sent without overflowing the context window.
            NoSlot: When a round finds no model slot before anything is yielded (the first round,
                or a later one of a turn that has shown nothing), or the answer round finds none.
            ToolsUnavailable: Before anything is yielded, when the first call is refused as sent.
            ValueError: Naming the context window, when the answer does not fit even after the
                tool replies were cut, or a scoped round overflows once its seed step was shown.
            Exception: The endpoint's error, raised as it came, when it fails before anything is
                yielded or fails the answer.
        """
        ctx = ToolContext(ledger=ledger, budget=self._budget, llm_view=self._llm_view)
        system = ChatMessage(role=MessageRole.SYSTEM, content=self._system_prompt)
        if pinned is not None:
            yield from self._scoped(ctx, [system, *window.messages()], message, pinned)
            return
        turn = _Turn(
            ctx=ctx,
            base=[system, *window.messages(), ChatMessage(role=MessageRole.USER, content=message)],
            schemas=self._registry.schemas(),
            started=self._clock(),
            images=TurnImages(
                char_token_ratio=self._ratio,
                fetch=self._fetch_thumbnails,
                legend_template=self._legend_template,
                max_images=self._max_images,
                round_room=self._budget.round_room,
            ),
        )
        while True:
            limit = self._limit(turn)
            if limit is not None:
                turn.stop(limit)
                break
            if turn.note_due:
                self._join_search_note(turn)
            first = not turn.steps and not turn.asked_again
            streamed = yield from self._round(turn, answer=False, hold=first)
            if streamed is None:
                if not turn.shown:
                    raise NoSlot
                logger.warning("No model slot came free for an agent round; the model answers from what it found")
                turn.stop("research_failed")
                break
            if streamed.done is None:
                if first and streamed.began:
                    # Held whole, it showed nothing: the break-off takes the turn's one re-ask, with no note.
                    logger.warning(
                        "An agent turn's first round broke off ({}) before anything was shown",
                        type(streamed.error).__name__,
                    )
                    turn.asked_again = True
                    continue
                note = self._failed_round(turn, streamed)
                if note is not None:
                    yield note
                break
            if first and not streamed.done.calls and self._search_note and turn.schemas:
                # Dropped unseen; the first round runs again with the search note, unless a limit comes first.
                turn.asked_again = turn.note_due = True
                continue
            if streamed.done.reasoning:
                turn.reasoning.append(streamed.done.reasoning)
            if not streamed.done.calls:
                yield from self._answer(turn, streamed)
                return
            held = len(turn.ctx.ledger)
            withdrawn = yield from self._tool_round(turn, streamed.done)
            if withdrawn:
                break
            self._show_pictures(turn, held)
        yield from self._final_round(turn)

    def _limit(self, turn: _Turn) -> str | None:
        """Say which limit, if any, makes the model answer before the next round.

        The room counts a search note still due, which the next round would carry.

        Args:
            turn (_Turn): The turn.

        Returns:
            str | None: ``round_budget``, ``time_budget``, ``context_budget`` or ``idle``, checked
            in that order; ``None`` while the turn may go on.
        """
        if turn.steps >= self._max_rounds:
            return "round_budget"
        if self._time_left(turn) <= 0:
            return "time_budget"
        due = message_cost(self._search_note, self._ratio) if turn.note_due else 0
        # One full reply, and the envelope of the message that asks for it.
        if self._room - turn.spent - due < self._calls.full_reply + MESSAGE_OVERHEAD_TOKENS:
            return "context_budget"
        if turn.idle >= IDLE_ROUNDS:
            return "idle"
        return None

    def _join_search_note(self, turn: _Turn) -> None:
        """Join the search note to the user's message as the first round runs again, its cost now spent.

        The first reply, held whole and unseen, called no tool and was dropped: it reaches neither
        the transcript nor the trail, and its reasoning went with it. Strict chat templates refuse
        two user messages in a row, so the note joins the user's.

        Args:
            turn (_Turn): The turn, the search note due.
        """
        turn.note_due = False
        turn.bare_base, turn.base = turn.base, _with_note(turn.base, self._search_note)
        turn.spent += message_cost(self._search_note, self._ratio)
        logger.info("An agent turn's first round called no tool; it is asked once more, to search first")

    def _round(
        self, turn: _Turn, *, answer: bool, hold: bool = False
    ) -> Generator[AgentProgress | str, None, _Streamed | None]:
        """Stream a tool round or the forced answer; a round refused with pictures goes once more without them.

        A round whose request carries pictures and fails before its first event is taken as
        refusing them, as ``VisualImagesMixin`` takes it: every picture is taken back (see
        :meth:`~docint.agents.turn.images.TurnImages.drop`) and the round is sent again without
        any, once. Should that fail too, it fails as any round does; a dead endpoint does.

        Args:
            turn (_Turn): The turn.
            answer (bool): Whether this is the answer round: no tool to choose, the final note joined.
            hold (bool): Whether the round is held whole, its text released by nothing but its end.

        Yields:
            AgentProgress | str: Answer text the round releases; it opens with no step.

        Returns:
            _Streamed | None: How the round went; ``None`` when no model slot came free.
        """
        retried = False
        while True:
            carried = turn.images.count
            streamed = yield from self._stream(
                turn,
                self._answer_prompt(turn) if answer else [*turn.base, *turn.transcript],
                tool_choice="none" if answer else "auto",
                wait=self._slot_wait(turn, answer=answer),
                hold=hold,
            )
            if streamed is None or streamed.began or not carried or retried:
                return streamed
            logger.warning(
                "An agent round carrying {} pictures failed before its first event ({}); it is sent again without them",
                carried,
                type(streamed.error).__name__,
            )
            turn.transcript, freed = turn.images.drop(turn.transcript)
            turn.spent -= freed
            retried = True

    def _answer_prompt(self, turn: _Turn) -> list[ChatMessage]:
        """Return what an answer round is sent: the turn so far, the final note joined last.

        While the transcript is empty, no round has answered to the search note, so the base it
        joined is left out: the note asks for a search the answer round cannot make.

        Args:
            turn (_Turn): The turn.

        Returns:
            list[ChatMessage]: The prompt.
        """
        base = turn.bare_base if turn.bare_base is not None and not turn.transcript else turn.base
        return _with_note([*base, *turn.transcript], self._final_note)

    def _slot_wait(self, turn: _Turn, *, answer: bool) -> float:
        """Return the longest a round waits for a model slot.

        Args:
            turn (_Turn): The turn.
            answer (bool): Whether the round is the answer round.

        Returns:
            float: Seconds: a started turn's answer waits the time left, and at least
            ``MODEL_SLOT_WAIT_S``; any other round waits that at most, never past the time left.
        """
        time_left = max(0.0, self._time_left(turn))
        # A started turn owes its answer: it waits the time left, and at least the usual wait.
        return max(MODEL_SLOT_WAIT_S, time_left) if answer and turn.shown else min(MODEL_SLOT_WAIT_S, time_left)

    def _show_pictures(self, turn: _Turn, held: int) -> None:
        """Show the model the pictures a round admitted, in one message after its tool replies.

        Args:
            turn (_Turn): The turn, the round's tool replies in its transcript.
            held (int): How many passages the ledger held before the round's calls.
        """
        admitted = list(enumerate(turn.ctx.ledger.entries()[held:], start=held + 1))
        shown = turn.images.attach(admitted, room=self._room - turn.spent)
        if shown is not None:
            message, cost = shown
            turn.transcript.append(message)
            turn.spent += cost

    def _stream(
        self,
        turn: _Turn,
        messages: list[ChatMessage],
        *,
        tool_choice: str,
        wait: float,
        opening: Callable[[], AgentProgress] | None = None,
        offer_tools: bool = True,
        hold: bool = False,
    ) -> Generator[AgentProgress | str, None, _Streamed | None]:
        """Stream one round in a model slot, yielding what it releases with the label swapped in.

        The slot is held while the round streams and given back when it ends, fails, or the
        caller stops reading. A failure is returned, not raised, for the caller to judge.

        Args:
            turn (_Turn): The turn.
            messages (list[ChatMessage]): What the round is sent.
            tool_choice (str): ``auto`` for a tool round, ``none`` for the answer round.
            wait (float): The longest the round waits for a model slot.
            opening (Callable[[], AgentProgress] | None): Records the step the round opens with,
                yielded when its first event arrives: a round that fails before then shows nothing.
            offer_tools (bool): Whether the round declares the turn's tools; without them it sends
                neither ``tools`` nor ``tool_choice``.
            hold (bool): Whether the round is held whole, releasing no text while it streams.

        Yields:
            AgentProgress | str: The opening step, then the answer text the round releases.

        Returns:
            _Streamed | None: How the round went; ``None`` when no model slot came free.
        """
        if not self._model_slots.acquire(timeout=wait):
            return None
        try:
            turn.model_calls += 1
            streamed = _Streamed()
            try:
                for event in stream_round(
                    self._llm,
                    messages,
                    tools=turn.schemas if offer_tools else [],
                    tool_choice=tool_choice,
                    reasoning_open=self._reasoning_open,
                    hold=hold,
                ):
                    if not streamed.began:
                        streamed.began = True
                        if opening is not None:
                            yield opening()
                    if isinstance(event, Provisional):
                        streamed.sent += event.text
                        text = streamed.splitter.feed(event.text)
                        if text:
                            streamed.shown = turn.shown = True
                            yield text
                    elif isinstance(event, RoundDone):
                        streamed.done = event
            except Exception as exc:
                streamed.done, streamed.error = None, exc
            return streamed
        finally:
            self._model_slots.release()

    def _answer(self, turn: _Turn, streamed: _Streamed) -> Iterator[str | AgentResult]:
        """End the turn with the answer a round wrote: send its rest, then the result.

        An answer the endpoint cut at its length limit is kept, as written, and flagged
        ``answer_cut``, which replaces any earlier cause: a cut answer matters more than a limit
        or a failure before it.

        Args:
            turn (_Turn): The turn.
            streamed (_Streamed): The answer round, ended.

        Yields:
            str | AgentResult: The rest of the answer, then the result.
        """
        done = cast(RoundDone, streamed.done)
        if done.finish_reason == "length":
            logger.warning(
                "An agent turn's answer was cut at the length limit (earlier reason: {})", turn.reason or "none"
            )
            turn.reason = "answer_cut"
        yield from self._answer_tail(turn, streamed)
        yield self._result(turn, done.text)

    def _answer_tail(self, turn: _Turn, streamed: _Streamed) -> Iterator[str]:
        """Send what an answer round wrote and has not released yet, then what the splitter holds.

        Args:
            turn (_Turn): The turn.
            streamed (_Streamed): The answer round, ended.

        Yields:
            str: The rest of the answer.
        """
        if streamed.done is None:
            return
        # What went out provisionally is the start of the round's text, so the rest is its tail.
        unsent = streamed.done.text[len(streamed.sent) :]
        for piece in (streamed.splitter.feed(unsent) if unsent else "", streamed.splitter.flush()):
            if piece:
                turn.shown = True
                yield piece

    def _result(self, turn: _Turn, text: str) -> AgentResult:
        """Build the turn's result.

        Args:
            turn (_Turn): The turn.
            text (str): The answer as the model wrote it.

        Returns:
            AgentResult: The result.
        """
        return AgentResult(
            raw_text=text,
            nodes=turn.images.stamp(turn.ctx.ledger.numbered()),
            steps=turn.steps,
            actions=list(turn.actions),
            reason=turn.reason,
            images_attached=turn.images.count,
            reasoning="\n\n".join(turn.reasoning) or None,
        )

    def _overflowed(self, turn: _Turn, cause: Exception) -> None:
        """Cut the tool replies so the answer round can be retried, once.

        Args:
            turn (_Turn): The turn.
            cause (Exception): The endpoint's overflow error.

        Raises:
            ValueError: Naming the context window, when the replies were cut before or no reply
                is long enough to cut.
        """
        shorter = None if turn.shortened else _shortened(turn.transcript)
        if shorter is None:
            raise ValueError(_OVERFLOW_MESSAGE) from cause
        logger.warning("An agent round overflowed the context window; the model answers from shortened tool replies")
        turn.transcript, turn.shortened = shorter, True
        turn.stop("context_budget")

    def _failed_round(self, turn: _Turn, streamed: _Streamed) -> AgentProgress | None:
        """Judge a round offered tools that failed: retry an overflow, raise before anything was shown, else answer.

        Args:
            turn (_Turn): The turn.
            streamed (_Streamed): The failed round.

        Returns:
            AgentProgress | None: A note withdrawing the text the round showed before it broke
            off, for the caller to yield; ``None`` when it showed none.

        Raises:
            ToolsUnavailable: When the turn's first call was refused as sent.
            ValueError: Naming the context window, when an overflow leaves nothing to cut.
            Exception: The round's error, when nothing was shown yet.
        """
        exc = cast(Exception, streamed.error)
        if streamed.shown:
            # Its text is provisional: a note withdraws it, as for a round that ends in calls.
            logger.warning("An agent round broke off ({}); the model answers anew", type(exc).__name__)
            turn.stop("research_failed")
            return self._step(turn, "note", " ".join(streamed.sent.split())[:NOTE_CHARS], 0, step=turn.steps + 1)
        if is_context_overflow(exc):
            self._overflowed(turn, exc)
            return None
        if not turn.shown:
            _raise_if_refused(turn, exc)
            raise exc
        logger.warning("An agent round failed ({}); the model answers from what it found", type(exc).__name__)
        turn.stop("research_failed")
        return None

    def _final_round(self, turn: _Turn) -> Iterator[AgentProgress | str | AgentResult]:
        """Make the model answer with no tool to choose, and end the turn.

        A reply that holds tool calls and no text is sent once more, declaring no tools (see
        :meth:`_resend_without_tools`).

        Args:
            turn (_Turn): The turn, its reason recorded.

        Yields:
            AgentProgress | str | AgentResult: The answer text, then the result.

        Raises:
            NoSlot: When no model slot comes free for the answer or its resend, whether or not
                anything was shown.
            ToolsUnavailable: When this is the turn's first call and it is refused as sent.
            ValueError: Naming the context window, on an overflow with nothing left to cut, or
                when the resend overflows.
            Exception: The endpoint's error, when the answer or its resend fails.
        """
        logger.info("An agent turn answers after {} tool rounds ({})", turn.steps, turn.reason)
        while True:
            streamed = yield from self._round(turn, answer=True)
            if streamed is None:
                # No answer can be written: a started turn fails as any failed answer round does.
                if turn.shown:
                    logger.warning("No model slot came free for an agent turn's answer")
                raise NoSlot
            if streamed.done is not None:
                break
            exc = cast(Exception, streamed.error)
            if streamed.shown:
                raise exc
            if is_context_overflow(exc):
                self._overflowed(turn, exc)
                continue
            _raise_if_refused(turn, exc)
            raise exc
        if streamed.done.calls and not streamed.done.text.strip():
            streamed = yield from self._resend_without_tools(turn, streamed)
        yield from self._finish(turn, streamed)

    def _resend_without_tools(
        self, turn: _Turn, streamed: _Streamed
    ) -> Generator[AgentProgress | str, None, _Streamed]:
        """Send an answer round that only called tools once more, declaring no tools.

        An endpoint that ignores ``tool_choice`` answers the forced round with calls. The same
        messages go out once more with neither ``tools`` nor ``tool_choice``, and that reply is
        the answer, empty or not. The resend fails as the answer round does: there is no answer
        to keep, and an empty one must not pass as whole.

        Args:
            turn (_Turn): The turn.
            streamed (_Streamed): The answer round, ended in calls with no text.

        Yields:
            AgentProgress | str: Answer text the resend releases.

        Returns:
            _Streamed: The resend, ended.

        Raises:
            NoSlot: When no model slot comes free for it.
            ValueError: Naming the context window, when it overflows before any text.
            Exception: The endpoint's error, as it came, when it fails otherwise.
        """
        done = cast(RoundDone, streamed.done)
        logger.warning(
            "An agent turn's answer round only called tools ({}); it is sent once more with no tools declared",
            len(done.calls),
        )
        if done.reasoning:
            turn.reasoning.append(done.reasoning)
        resent = yield from self._stream(
            turn,
            self._answer_prompt(turn),
            tool_choice="none",
            wait=self._slot_wait(turn, answer=True),
            offer_tools=False,
        )
        if resent is None:
            raise NoSlot
        if resent.done is None:
            exc = cast(Exception, resent.error)
            if not resent.shown and is_context_overflow(exc):
                raise ValueError(_OVERFLOW_MESSAGE) from exc
            raise exc
        return resent

    def _finish(self, turn: _Turn, streamed: _Streamed) -> Iterator[str | AgentResult]:
        """End the turn with an answer round that ran: keep its reasoning, run none of its calls, send the answer.

        Args:
            turn (_Turn): The turn.
            streamed (_Streamed): The answer round, ended.

        Yields:
            str | AgentResult: The rest of the answer, then the result.
        """
        done = cast(RoundDone, streamed.done)
        if done.reasoning:
            turn.reasoning.append(done.reasoning)
        if done.calls:
            logger.warning("An agent turn's answer round asked for {} tool calls anyway; none ran", len(done.calls))
        yield from self._answer(turn, streamed)

    def _scoped(
        self, ctx: ToolContext, history: list[ChatMessage], message: str, pinned: Sequence[NodeWithScore]
    ) -> Iterator[AgentProgress | str | AgentResult]:
        """Answer a pinned scope in one round that offers no tools, from one user message holding the passages.

        The passages enter the ledger and are rendered within the loop's room less the scope
        note and its envelope (see :func:`~docint.agents.turn.scope.scope_message`). The seed
        step goes out with the round's first event, so a round that fails before it has shown
        nothing: an overflow then raises :class:`EvidenceDoesNotFit`, which sends the caller
        down its scoped path, and any other failure is raised as it came, a refusal included,
        since no tools were offered. Once begun, the round fails as the answer round does: an
        overflow before any text raises ``ValueError`` naming the context window, and any other
        failure is raised as it came. A scope outranks the retrieval target: no picture is attached.

        Args:
            ctx (ToolContext): The turn's evidence, budget and LLM view.
            history (list[ChatMessage]): The system prompt and the window's messages.
            message (str): The user's message, verbatim.
            pinned (Sequence[NodeWithScore]): The passages the user pinned.

        Yields:
            AgentProgress | str | AgentResult: The seed step, the answer text, then the result.

        Raises:
            EvidenceDoesNotFit: Before anything is yielded, when the passages cannot all be held
                or shown legibly, or the round overflows the context window.
            NoSlot: When no model slot comes free; nothing was yielded.
            ValueError: Naming the context window, when the round overflows after its seed step
                but before any text.
            Exception: The endpoint's error, as it came, when the round fails.
        """
        text, found = scope_message(
            pinned,
            ledger=ctx.ledger,
            note=self._scope_note,
            message=message,
            llm_view=self._llm_view,
            budget_tokens=self._scope_room,
            char_token_ratio=self._ratio,
        )
        turn = _Turn(
            ctx=ctx,
            base=[*history, ChatMessage(role=MessageRole.USER, content=text)],
            schemas=[],
            started=self._clock(),
            images=TurnImages(char_token_ratio=self._ratio),
        )
        logger.info("An agent turn answers from {} pinned passages", len(ctx.ledger))
        streamed = yield from self._stream(
            turn,
            list(turn.base),
            tool_choice="none",
            wait=min(MODEL_SLOT_WAIT_S, max(0.0, self._time_left(turn))),
            opening=lambda: self._step(turn, "seed", "", found, step=0),
        )
        if streamed is None:
            raise NoSlot
        if streamed.done is None:
            exc = cast(Exception, streamed.error)
            if streamed.shown or not is_context_overflow(exc):
                raise exc
            if not turn.shown:
                raise EvidenceDoesNotFit(len(ctx.ledger)) from exc
            # The seed went out, so the caller cannot fall back: the overflow is reported as the answer round's.
            raise ValueError(_OVERFLOW_MESSAGE) from exc
        yield from self._finish(turn, streamed)

    def _step(self, turn: _Turn, action: str, detail: str, found: int, *, step: int | None = None) -> AgentProgress:
        """Record one step in the trail.

        Args:
            turn (_Turn): The turn.
            action (str): A note, or the name of the tool that answered the call.
            detail (str): The note's text, or the call's main argument.
            found (int): Passages the call admitted or extended.
            step (int | None): The round it belongs to; the last tool round by default.

        Returns:
            AgentProgress: The step, numbered by its round.
        """
        turn.actions.append({"action": action, "detail": detail, "found": found})
        turn.shown = True
        number = turn.steps if step is None else step
        return AgentProgress(step=number, action=cast(AgentAction, action), detail=detail, found=found)

    def _tool_round(self, turn: _Turn, done: RoundDone) -> Generator[AgentProgress, None, bool]:
        """Answer a round's calls and add the round to the transcript, or withdraw it when it does not fit.

        The round's own cost, its assistant message and the least reply each call needs, calls
        past the round's limit included, must fit the room left. A round that does not is
        withdrawn: its calls never reach the transcript, so they need no replies, it is no tool
        round, and the model answers next (``context_budget``).

        Args:
            turn (_Turn): The turn.
            done (RoundDone): The round, ended in calls.

        Yields:
            AgentProgress: The round's note, if it wrote text, then one step per call its tool answered.

        Returns:
            bool: Whether the round was withdrawn.
        """
        number = turn.steps + 1
        note = " ".join(done.text.split())[:NOTE_CHARS]
        if note:
            yield self._step(turn, "note", note, 0, step=number)
        preface = fit_with_marker(done.text, NOTE_CHARS)[0] if done.text else ""
        echoed = [_echoed(call.arguments) for call in done.calls]
        blocks: list[Any] = [TextBlock(text=preface)] if preface else []
        blocks.extend(
            ToolCallBlock(tool_call_id=call.id, tool_name=call.name, tool_kwargs=arguments)
            for call, arguments in zip(done.calls, echoed, strict=True)
        )
        written = preface + "".join(
            json.dumps({"name": call.name, "arguments": arguments}, ensure_ascii=False)
            for call, arguments in zip(done.calls, echoed, strict=True)
        )
        asked = message_cost(written, self._ratio)
        prepared = self._calls.prepare(done.calls, turn.ctx)
        room = self._room - turn.spent - asked
        if room < self._calls.least(prepared):
            logger.warning(
                "An agent round's {} tool calls do not fit the room left; it is withdrawn and the model answers",
                len(done.calls),
            )
            turn.stop("context_budget")
            return True
        turn.steps = number
        answered = self._calls.answer(prepared, turn.ctx, room=room, time_left=lambda: self._time_left(turn))
        if answered.failed:
            # Evidence a failed search would have found is missing; the turn goes on, and says so.
            turn.stop("research_failed")
        turn.transcript.append(ChatMessage(role=MessageRole.ASSISTANT, blocks=blocks))
        turn.spent += asked
        found = 0
        for call, reply in zip(done.calls, answered.replies, strict=True):
            turn.transcript.append(
                ChatMessage(role=MessageRole.TOOL, content=reply.text, additional_kwargs={"tool_call_id": call.id})
            )
            turn.spent += message_cost(reply.text, self._ratio)
            found += reply.found
            if not reply.error:
                yield self._step(turn, reply.action, reply.detail, reply.found)
        turn.idle = 0 if found else turn.idle + 1
        return False
