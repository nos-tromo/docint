import { useEffect, useReducer, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { DownloadButton, NewButton, PageHeader, SendButton } from '@infra/ui'
import { streamQuery } from '@/api/chat'
import { ApiError } from '@/api/client'
import { describeError, streamErrorText } from '@/api/errorMessage'
import { clearScope, setScope } from '@/api/scope'
import { clearSessionTopic } from '@/api/sessions'
import type { AgentStep, ChatFinalEvent } from '@/api/types'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { useSessionHistory } from '@/hooks/useSessions'
import { useStickToBottom } from '@/hooks/useStickToBottom'
import { useReportDedupeKeys } from '@/hooks/useReports'
import { sessionTopicKey, useChatTopicFocus, useTopicFocusRule } from '@/hooks/useTopics'
import { useActiveCollection } from '@/hooks/useActiveCollection'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'
import { draftKey, useChatUiStore } from '@/stores/chatUi'
import { scopeChunkIds, scopeFor, searchKeyFor, useSearchUiStore } from '@/stores/searchUi'
import { focusInScope, toSessionTopic, useTopicFocusStore } from '@/stores/topicFocus'
import { useQueryClient } from '@tanstack/react-query'
import { sessionsKey } from '@/hooks/useSessions'
import { ChatControls } from '@/components/chat/ChatControls'
import { ChatTurn, type ChatTurnData } from '@/components/chat/ChatTurn'
import { ScopeBanner } from '@/components/chat/ScopeBanner'
import { SearchPanel } from '@/components/chat/SearchPanel'
import { TopicFocusChip } from '@/components/chat/TopicFocusChip'
import { cn } from '@/lib/cn'
import { downloadText } from '@/lib/csv'
import { chatTranscriptToText } from '@/lib/exports'
import { useT } from '@/i18n/LanguageContext'

interface State {
  turns: ChatTurnData[]
  inflight: boolean
}
type Action =
  | { type: 'reset' }
  | { type: 'set_turns'; turns: ChatTurnData[] }
  | { type: 'start'; user: string; scopeRequested?: number; unsavedId?: string }
  | { type: 'token'; token: string }
  | { type: 'retry'; query: string }
  | { type: 'agent'; step: AgentStep }
  | { type: 'finalize'; meta: ChatFinalEvent }
  | { type: 'fail'; error?: string; note?: string; agentNote?: string }

function reducer(s: State, a: Action): State {
  switch (a.type) {
    case 'reset':
      return { turns: [], inflight: false }
    case 'set_turns':
      return { ...s, turns: a.turns }
    case 'start':
      return {
        ...s,
        inflight: true,
        turns: [
          ...s.turns,
          {
            user: a.user,
            assistant: '',
            done: false,
            meta: null,
            error: null,
            scopeRequested: a.scopeRequested,
            unsavedId: a.unsavedId
          }
        ]
      }
    // The three cases below fold into the open turn. A session switch resets
    // the transcript while a stream is still being torn down, so that turn
    // can legitimately be gone by the time a late frame lands — fold into
    // nothing rather than into `undefined`.
    case 'token': {
      const last = s.turns[s.turns.length - 1]
      if (!last) return s
      const updated = { ...last, assistant: last.assistant + a.token }
      return { ...s, turns: [...s.turns.slice(0, -1), updated] }
    }
    // The backend rejected the answer it just streamed and is re-answering.
    // Drop the rejected tokens: the replacement streams in behind this frame,
    // and keeping both would splice two answers into one.
    case 'retry': {
      const last = s.turns[s.turns.length - 1]
      if (!last) return s
      const updated = { ...last, assistant: '', retryQuery: a.query }
      return { ...s, turns: [...s.turns.slice(0, -1), updated] }
    }
    // A step of an agent turn; only the final envelope finishes the turn. A note
    // withdraws the text streamed before it, as a retry does, and stays in the trail.
    case 'agent': {
      const last = s.turns[s.turns.length - 1]
      if (!last) return s
      const assistant = a.step.action === 'note' ? '' : last.assistant
      const updated = { ...last, assistant, agentSteps: [...(last.agentSteps ?? []), a.step] }
      return { ...s, turns: [...s.turns.slice(0, -1), updated] }
    }
    case 'finalize': {
      const last = s.turns[s.turns.length - 1]
      if (!last) return { ...s, inflight: false }
      // The envelope's own text wins over the accumulated tokens: when a retry
      // is abandoned mid-stream the backend falls back to reporting the first
      // answer, and the tokens on screen are then a fragment of the attempt
      // that was dropped.
      const finalText = a.meta.answer ?? a.meta.message ?? (a.meta.response || last.assistant)
      const updated = { ...last, assistant: finalText, done: true, meta: a.meta }
      return { ...s, inflight: false, turns: [...s.turns.slice(0, -1), updated] }
    }
    // An answer cut off mid-text carries the caveat in its text, wherever the text goes. A judgement
    // is labelled once its answer is complete, so a turn that showed a step says it may hold one.
    case 'fail': {
      const last = s.turns[s.turns.length - 1]
      if (!last) return { ...s, inflight: false }
      const note = (last.agentSteps?.length ? a.agentNote : undefined) ?? a.note
      const cutOff = !!note && !last.done && last.assistant.trim() !== ''
      const assistant = cutOff ? `*${note}*\n\n${last.assistant}` : last.assistant
      const updated = { ...last, assistant, done: true, error: a.error ?? null }
      return { ...s, inflight: false, turns: [...s.turns.slice(0, -1), updated] }
    }
  }
}

export { reducer as chatReducer }

/** The rail's chevron: left while the panel is open, right while it is shut. */
const RailChevron = ({ open }: { open: boolean }) => (
  <svg
    viewBox="0 0 24 24"
    className={`h-4 w-4 transition-transform duration-300 ${open ? '' : 'rotate-180'}`}
    fill="none"
    stroke="currentColor"
    strokeWidth="1.75"
    aria-hidden="true"
  >
    <path d="M15 6l-6 6 6 6" />
  </svg>
)

export function Chat() {
  const t = useT()
  const params = useParams()
  const navigate = useNavigate()
  const sessionIdParam = params.sessionId ?? null
  const setCurrentSessionId = useUiStore((s) => s.setCurrentSessionId)
  const currentSessionId = useUiStore((s) => s.currentSessionId)
  const selectedCollection = useUiStore((s) => s.selectedCollection)
  const activeReportId = useReportStore((s) => s.activeReportId)
  const reportDedupeKeys = useReportDedupeKeys(activeReportId)
  const filters = useChatFiltersStore()
  const setRetrievalTarget = useChatFiltersStore((s) => s.setRetrievalTarget)
  const { rule: topicRule, pending: topicPending } = useTopicFocusRule()
  const { focus: topicFocus } = useChatTopicFocus()
  const activeCollection = useActiveCollection()
  const clearTopicPick = useTopicFocusStore((s) => s.clear)
  const qc = useQueryClient()
  const history = useSessionHistory(sessionIdParam)
  const [state, dispatch] = useReducer(reducer, { turns: [], inflight: false })
  // `state.turns` is a fresh array on every streamed token, so the transcript
  // follows the growing answer — but only while the user sits at the bottom.
  const transcript = useStickToBottom<HTMLDivElement>(state.turns)
  const abortRef = useRef<AbortController | null>(null)
  const key = draftKey(sessionIdParam)
  const draft = useChatUiStore((s) => s.drafts[key] ?? '')
  const setDraft = useChatUiStore((s) => s.setDraft)
  const clearDraft = useChatUiStore((s) => s.clearDraft)
  const sidePanelOpen = useChatUiStore((s) => s.sidePanelOpen)
  const toggleSidePanel = useChatUiStore((s) => s.toggleSidePanel)
  // The scope follows the *effective* session: a new chat holds its selection
  // under 'new' until the backend mints an id on the first turn.
  const scopeKey = searchKeyFor(currentSessionId)
  const scope = useSearchUiStore((s) => scopeFor(s, scopeKey))
  const adoptScope = useSearchUiStore((s) => s.adoptScope)
  const dropScope = useSearchUiStore((s) => s.clearScope)
  const setScopeMeta = useSearchUiStore((s) => s.setScopeMeta)
  const scopedChunkIds = scopeChunkIds(scope)
  const [scopeError, setScopeError] = useState<string | null>(null)

  const describeScopeFailure = (err: unknown): string => {
    if (err instanceof ApiError && err.status === 422) return t('search.budget_exceeded')
    const described = describeError(err)
    return t(described.key, described.vars)
  }

  const unscope = async () => {
    const sessionId = currentSessionId
    dropScope(scopeKey)
    setScopeError(null)
    if (!sessionId) return
    try {
      await clearScope(sessionId)
    } catch (err) {
      // The local scope is already gone; say so rather than silently leaving
      // the server still restricting answers.
      setScopeError(describeScopeFailure(err))
    }
  }

  // Kept with the session it is about, so it is never shown over another chat.
  const [topicError, setTopicError] = useState<{ sessionId: string; message: string } | null>(null)

  const unfocus = async () => {
    setTopicError(null)
    const sessionId = currentSessionId
    // A chat not started yet holds only the "Ask in chat" pick.
    if (!sessionId) {
      clearTopicPick()
      return
    }
    // Dropped here first, so the chip goes at once. A read already in flight is
    // cancelled, or its answer would put the topic straight back.
    const key = sessionTopicKey(sessionId, activeCollection)
    await qc.cancelQueries({ queryKey: key })
    qc.setQueryData(key, { topic_focus: null })
    try {
      await clearSessionTopic(sessionId)
      qc.setQueryData(key, { topic_focus: null })
    } catch {
      // Still stored with the chat: read it back, so the chip shows what the
      // chat keeps, and say why it is still there.
      void qc.invalidateQueries({ queryKey: key })
      setTopicError({ sessionId, message: t('chat.topic_focus_remove_error') })
    }
  }

  useEffect(() => {
    setCurrentSessionId(sessionIdParam)
  }, [sessionIdParam, setCurrentSessionId])

  useEffect(() => {
    // Both chat routes render the same `Chat` element, so React Router keeps
    // this component mounted when the session changes — including "New
    // session", which just drops the `:sessionId` segment. Nothing else
    // clears the reducer: the history effect only ever writes turns *in*, and
    // it is disabled for a session-less chat. Without this reset the previous
    // transcript stayed on screen and starting a new session looked like a
    // no-op until the user visited another section (which unmounted us).
    abortRef.current?.abort()
    abortRef.current = null
    dispatch({ type: 'reset' })
  }, [sessionIdParam])

  useEffect(() => {
    // Starting a new chat only clears the session id. From `/chat/:sessionId`
    // that drops the URL segment too and the effect above does the reset — but
    // a chat that minted its session on the first turn never had that segment
    // (the id lives only in the store), so nothing the effect above watches
    // changed and the finished transcript stayed on screen, making the click
    // look like it had done nothing. Reset on the *effective* session going
    // away, which covers both, and switching collections besides.
    if (sessionIdParam !== null || currentSessionId !== null) return
    abortRef.current?.abort()
    abortRef.current = null
    dispatch({ type: 'reset' })
    // "Answer from" belongs to one chat: the next one starts from everything.
    setRetrievalTarget('all')
  }, [sessionIdParam, currentSessionId, setRetrievalTarget])

  const onNewChat = () => {
    setCurrentSessionId(null)
    // A new chat starts without a topic; "Ask in chat" is how one gets one.
    clearTopicPick()
    // The effect above only sees a session going away; a chat with no turns, or
    // a stateless one, never had one to lose.
    setRetrievalTarget('all')
    // Required even when the id lived only in the store: from a pinned
    // `/chat/:sessionId` the param effect would otherwise put it straight back.
    navigate('/chat')
  }

  useEffect(() => {
    if (!history.data) return
    const turns: ChatTurnData[] = []
    let pendingUser: string | null = null
    for (const m of history.data.messages) {
      if (m.role === 'user') pendingUser = m.content
      else {
        turns.push({
          user: pendingUser ?? '',
          assistant: m.content,
          done: true,
          meta: m.sources
            ? ({
                sources: m.sources,
                session_id: sessionIdParam ?? '',
                validation_checked: m.validation_checked,
                validation_mismatch: m.validation_mismatch,
                validation_reason: m.validation_reason,
                retried: m.retried,
                retry_query: m.retry_query,
                answer_mode: m.answer_mode,
                agent: m.agent,
                retrieval_target: m.retrieval_target
              } as ChatFinalEvent)
            : null
        })
        pendingUser = null
      }
    }
    dispatch({ type: 'set_turns', turns })
  }, [history.data, sessionIdParam])

  const send = async () => {
    const message = draft.trim()
    // Until the chat's topic and its run are read, whether the turn carries the topic's rule is not
    // known; sent now, it would answer from the whole collection.
    if (!message || state.inflight || topicPending) return
    // Read once: the same list is sent with the question and recorded on the
    // turn, so the guardrail below compares what was asked for against what
    // the server says it did.
    const pinned = scopedChunkIds
    // Imagery carries no topic, so on a visual turn the rule would empty the answer; the chip says so.
    const rule = filters.retrievalTarget === 'visual' ? null : topicRule
    // Only the turn that creates the session carries the topic: the session row
    // does not exist before it, and keeps the topic itself after, so no later
    // turn can bring back one the user removed. Gated on the pick rather than
    // the rule, which is null on a visual turn or for a stale run.
    const sentTopic =
      !currentSessionId && topicFocus && focusInScope(topicFocus, activeCollection) ? topicFocus : null
    dispatch({
      type: 'start',
      user: message,
      scopeRequested: pinned.length || undefined,
      // Not crypto.randomUUID: it is absent outside a secure context (a
      // plain-HTTP dev host), and one turn in flight at a time keeps a
      // millisecond timestamp unique.
      unsavedId: filters.retrievalMode === 'stateless' ? Date.now().toString(36) : undefined
    })
    clearDraft(key)

    const ac = new AbortController()
    abortRef.current = ac
    // Passed with every failure; the reducer heads a cut-off answer with one, the agent note after a step.
    const note = t('chat.answer_interrupted')
    const agentNote = t('chat.assessment_interrupted')
    // Set once the turn has ended either way, so nothing arriving later can fail it.
    let settled = false
    try {
      for await (const ev of streamQuery(
        {
          question: message,
          session_id: currentSessionId ?? undefined,
          // WS2 backend resolves + owner-gates the collection per request; the
          // client selection is the single source of truth (no server-side
          // active collection anymore).
          collection: selectedCollection ?? undefined,
          metadata_filters: [...filters.buildPayload(), ...(rule ? [rule] : [])],
          retrieval_mode: filters.retrievalMode,
          retrieval_target: filters.retrievalTarget,
          reasoning: filters.reasoning,
          // The scope rides along with the question it scopes. An unstarted
          // chat has no session row to pin it to yet, and even on an open one
          // a checkbox ticked a moment ago may still be in flight — either way
          // the request is the only place this turn's scope is certain.
          scope_chunk_ids: pinned.length > 0 ? pinned : undefined,
          topic_focus: sentTopic ? toSessionTopic(sentTopic) : undefined
        },
        ac.signal
      )) {
        // /stream_query emits untyped SSE frames (no `event:` line), so
        // every event arrives as `'message'`. Discriminate by payload
        // shape: a token-only frame carries `{token}` and nothing else
        // metadata-like; the final envelope carries `{response, sources,
        // session_id, ...}`. Requiring no `session_id` on token frames
        // keeps the discriminator safe if a future backend ever folds
        // both into a single frame.
        const data = ev.data as Record<string, unknown> | string
        if (typeof data !== 'object' || data === null) continue
        const isTokenFrame =
          typeof data.token === 'string' &&
          !('session_id' in data) &&
          !('sources' in data) &&
          !('response' in data) &&
          !('answer' in data)
        if (isTokenFrame) {
          dispatch({ type: 'token', token: data.token as string })
          continue
        }
        // Announced before the replacement answer streams, so the rejected
        // one is discarded on arrival rather than swapped out silently at the
        // end. Must be handled before the finalize fallthrough below.
        if (typeof data.retry === 'object' && data.retry !== null) {
          const retryQuery = (data.retry as Record<string, unknown>).query
          if (typeof retryQuery === 'string') dispatch({ type: 'retry', query: retryQuery })
          continue
        }
        // Must be caught before the finalize fallthrough below: an unhandled
        // step frame would finish the turn and re-enable Send mid-research.
        // The final envelope also carries `agent` (the trail), with a session id.
        if (
          typeof data.agent === 'object' &&
          data.agent !== null &&
          !('session_id' in data) &&
          !('response' in data)
        ) {
          dispatch({ type: 'agent', step: data.agent as AgentStep })
          continue
        }
        if (typeof data.error === 'string') {
          // Post-D2 the backend sends a static protocol flag here, not
          // prose — never render the field itself. The validated `code`
          // token selects the copy and is shown for support triage.
          // The fallback is the *reported-failure* copy, not the
          // stream-ended one: an error frame means the backend answered
          // and named its failure. Only the transport path below, which
          // carries no code at all, may claim the stream ended.
          settled = true
          dispatch({
            type: 'fail',
            error: streamErrorText(t, data.code, 'chat.error_backend_failed'),
            note,
            agentNote
          })
          continue
        }
        const final = data as unknown as ChatFinalEvent
        settled = true
        dispatch({ type: 'finalize', meta: final })
        if (!currentSessionId && final.session_id) {
          // The server stored the topic with the session this turn created.
          // Seeded before the chat switches to it, so the chip never blinks
          // out while that copy is read.
          qc.setQueryData(sessionTopicKey(final.session_id, activeCollection), {
            topic_focus: sentTopic ? toSessionTopic(sentTopic) : null
          })
          setCurrentSessionId(final.session_id)
          // The pick is spent. A stream that ends after another "Ask in chat"
          // leaves that newer pick for the chat it was made for.
          if (sentTopic && useTopicFocusStore.getState().focus === sentTopic) clearTopicPick()
          // The backend minted the session id on this turn, so the selection
          // that rode along with the request now has a row to live on. It has
          // already scoped this answer; this round trip moves the local copy
          // onto the new id and reads back what the selection costs, which is
          // what the token meter shows.
          adoptScope(searchKeyFor(null), final.session_id)
          if (pinned.length > 0) {
            try {
              const stored = await setScope(
                final.session_id,
                pinned,
                selectedCollection ?? undefined
              )
              setScopeMeta(final.session_id, {
                usableTokens: stored.usable_tokens,
                missing: stored.missing
              })
            } catch (scopeErr) {
              // Refused (typically over budget): drop the local copy so the
              // banner never claims a scope the server does not hold.
              dropScope(final.session_id)
              setScopeError(describeScopeFailure(scopeErr))
            }
          }
        }
        qc.invalidateQueries({ queryKey: sessionsKey })
      }
      // Closed with neither its envelope nor an error frame, as a backend shut down mid-answer does.
      if (!settled && !ac.signal.aborted) {
        settled = true
        dispatch({ type: 'fail', error: t('chat.error_stream_ended'), note, agentNote })
      }
    } catch (e) {
      // A finished turn stays finished. An aborted request's turn is gone: the
      // session switch that aborted it reset the transcript, and the last turn
      // now belongs to someone else.
      if (settled || ac.signal.aborted) return
      // A pinned session resumed against the wrong collection (409) or a send
      // with no active collection (400) reaches here as a typed ApiError. Give
      // each an actionable message. Anything else is a transport failure — a
      // backend OOM mid-stream surfaces as a generic "network error" TypeError
      // from the reader; the underlying message is not user-visible (see
      // describeError), so just report the stream ended.
      let error: string
      if (e instanceof ApiError && e.status === 409) {
        error = t('chat.error_wrong_collection')
      } else if (e instanceof ApiError && e.status === 400) {
        error = t('chat.error_no_collection')
      } else if (e instanceof ApiError && e.status === 422 && pinned.length > 0) {
        // The scope rode along with the question, so the server is the first
        // to measure a first-turn selection — its refusal arrives here rather
        // than on the scope endpoint.
        error = t('search.budget_exceeded')
      } else if (e instanceof ApiError) {
        const d = describeError(e)
        error = t(d.key, d.vars)
      } else {
        // A pure transport failure (stream aborted mid-read) has no status
        // to report; the actionable message is that the stream ended.
        error = t('chat.error_stream_ended')
      }
      dispatch({ type: 'fail', error, note, agentNote })
    } finally {
      abortRef.current = null
    }
  }

  return (
    <div className="p-8 flex flex-col h-full min-h-0">
      {/* The header spans both columns rather than riding on the chat one.
          Inside the grid its right edge was the transcript's, so opening the
          panel slid every button 20rem to the left — the controls moving
          because something else did. Out here they hold the page's right edge
          and only the panel moves. */}
      <div className="flex items-start justify-between gap-3 mb-4">
        <PageHeader title={t('chat.title')} className="mb-0" />
        {/* The topic filter sits innermost, so its coming and going moves no control; on a row too
            narrow for it, it alone wraps, below the controls (`flex-wrap-reverse`). */}
        <div className="flex min-w-0 flex-wrap-reverse items-center justify-end gap-x-3 gap-y-2">
          <TopicFocusChip onRemove={() => void unfocus()} />
          {/* The metadata filters and the retrieval mode belong here, to the
              chat, not to the search panel: they narrow what any answer
              retrieves against, whether or not the panel is even open. In the
              panel they read as controls over the keyword index, which they
              are not.

              Order within the group is stable-first: Download appears only once
              a transcript exists, so it sits innermost, where its arrival pushes
              nothing sideways. */}
          <div className="flex items-center gap-3">
            {state.turns.length > 0 && (
              <DownloadButton
                label={t('chat.download')}
                onClick={() =>
                  downloadText(
                    `chat_${currentSessionId ?? 'session'}.txt`,
                    chatTranscriptToText(state.turns, t)
                  )
                }
              />
            )}
            {/* The sidebar's copy of this is the one reachable from every
                route; this one is where the wish arrives — at the foot of an
                answer, without crossing the page to the session list. */}
            <NewButton label={t('common.new_chat')} onClick={onNewChat} />
            <ChatControls />
            {/* The panel toggle is last: it is the control nearest the thing it
                opens, and it is the one button here that changes the page's
                shape rather than acting on the chat. It carries one bit of
                state, because that bit has consequences while the panel is
                shut — a live scope tints it, since a chat answering from
                hand-picked chunks behind a closed panel is a trap. */}
            <button
              type="button"
              onClick={toggleSidePanel}
              aria-expanded={sidePanelOpen}
              aria-controls="chat-side-panel"
              aria-label={sidePanelOpen ? t('search.collapse') : t('search.expand')}
              className={cn(
                'flex h-8 w-8 shrink-0 items-center justify-center rounded-md transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-border',
                scopedChunkIds.length > 0
                  ? 'bg-primary/15 text-primary'
                  : 'text-muted-foreground hover:text-foreground focus-visible:text-foreground'
              )}
              data-scoped={scopedChunkIds.length > 0 || undefined}
            >
              <RailChevron open={sidePanelOpen} />
            </button>
          </div>
        </div>
      </div>

      {/* The columns are animated rather than swapped, so collapsing the panel
          reflows the transcript smoothly instead of making it pop. The gap has
          to animate too: at `1fr 0` a static `gap-6` leaves 24px of nothing on
          the right, and the chat would stay narrower than every other route's
          content box — the thing this collapse is supposed to fix. */}
      <div
        className="grid min-h-0 flex-1 transition-[grid-template-columns,column-gap] duration-300 ease-out"
        style={{
          gridTemplateColumns: sidePanelOpen ? '1fr 22rem' : '1fr 0px',
          columnGap: sidePanelOpen ? '1.5rem' : '0px'
        }}
      >
        {/* min-h-0 on the section (grid item) and the messages list (flex
            item) lets them shrink below their content, so the list scrolls
            internally instead of the section outgrowing its row — which made
            the whole page scroll and pushed the composer flush against the
            viewport bottom, past the p-8 padding box. min-w-0 does the same
            for width: a wide answer table scrolls in its own box instead of
            widening the column past the viewport. */}
        <section className="flex flex-col h-full min-h-0 min-w-0">
          <ScopeBanner
            count={scopedChunkIds.length}
            missing={scope.missing}
            onClear={() => void unscope()}
          />
          {scopeError && (
            <p className="mb-3 text-xs text-red-500" role="alert">
              {scopeError}
            </p>
          )}
          {topicError && topicError.sessionId === currentSessionId && (
            <p className="mb-3 text-xs text-red-500" role="alert">
              {topicError.message}
            </p>
          )}
          <div
            ref={transcript.ref}
            onScroll={transcript.onScroll}
            className="flex-1 min-h-0 overflow-auto space-y-6 pr-2"
          >
            {state.turns.map((t, i) => (
              <ChatTurn
                // The transcript is append-only and turns carry no id; the index is
                // already load-bearing here, passed down as turnIdx.
                // eslint-disable-next-line @eslint-react/no-array-index-key
                key={i}
                turn={t}
                sessionId={t.meta?.session_id ?? currentSessionId ?? undefined}
                turnIdx={i}
                reportDedupeKeys={reportDedupeKeys}
              />
            ))}
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault()
              void send()
            }}
            className="mt-4 flex items-end gap-2 pr-2"
          >
            <textarea
              value={draft}
              onChange={(e) => setDraft(key, e.target.value)}
              onKeyDown={(e) => {
                // Enter submits; Shift+Enter inserts a newline (standard
                // chat-composer behavior). IME composition is excluded so
                // Enter while composing doesn't accidentally submit.
                if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault()
                  void send()
                }
              }}
              placeholder={t('chat.ask_placeholder')}
              rows={1}
              className="flex-1 bg-muted border border-border rounded-md px-3 py-2 resize-none max-h-40 leading-6"
            />
            {/* size="md" so the shell's aspect-square lands at 40x40, the
                composer's own height. `busy` replaces what used to be a typed
                '…' standing in for a spinner, and IconButton folds busy into
                disabled, so the in-flight guard is unchanged. */}
            <SendButton
              label={t('chat.send')}
              type="submit"
              variant="primary"
              size="md"
              busy={state.inflight}
              disabled={!draft.trim() || topicPending}
            />
          </form>
          {/* Only once there is something to send: a reopened chat reads its topic every time, and a
              hint flashing past on each would say nothing. */}
          {topicPending && draft.trim() && (
            <p role="status" className="mt-1 pr-2 text-xs text-muted-foreground">
              {t('chat.topic_focus_reading')}
            </p>
          )}
        </section>

        {/* Kept mounted while collapsed so a running search stays warm and the
            typed query survives — the column narrows to nothing and clips it,
            rather than the panel unmounting. */}
        <aside className="h-full min-h-0 overflow-hidden">
          <div id="chat-side-panel" hidden={!sidePanelOpen} className="h-full min-h-0 min-w-0">
            <SearchPanel sessionId={currentSessionId} />
          </div>
        </aside>
      </div>
    </div>
  )
}
