import { useEffect, useReducer, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'
import { Button, PageHeader } from '@infra/ui'
import { streamQuery } from '@/api/chat'
import { ApiError } from '@/api/client'
import { describeError, streamErrorText } from '@/api/errorMessage'
import { clearScope, setScope } from '@/api/scope'
import type { ChatFinalEvent } from '@/api/types'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { useSessionHistory } from '@/hooks/useSessions'
import { useStickToBottom } from '@/hooks/useStickToBottom'
import { useReportDedupeKeys } from '@/hooks/useReports'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'
import { draftKey, useChatUiStore } from '@/stores/chatUi'
import { scopeChunkIds, scopeFor, searchKeyFor, useSearchUiStore } from '@/stores/searchUi'
import { useQueryClient } from '@tanstack/react-query'
import { sessionsKey } from '@/hooks/useSessions'
import { ChatControls } from '@/components/chat/ChatControls'
import { DownloadButton } from '@/components/common/DownloadAction'
import { ChatTurn, type ChatTurnData } from '@/components/chat/ChatTurn'
import { ScopeBanner } from '@/components/chat/ScopeBanner'
import { SearchPanel } from '@/components/chat/SearchPanel'
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
  | { type: 'start'; user: string; scopeRequested?: number }
  | { type: 'token'; token: string }
  | { type: 'finalize'; meta: ChatFinalEvent }
  | { type: 'fail'; error?: string }

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
            scopeRequested: a.scopeRequested
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
    case 'finalize': {
      const last = s.turns[s.turns.length - 1]
      if (!last) return { ...s, inflight: false }
      const finalText = a.meta.answer ?? a.meta.message ?? last.assistant
      const updated = { ...last, assistant: finalText, done: true, meta: a.meta }
      return { ...s, inflight: false, turns: [...s.turns.slice(0, -1), updated] }
    }
    case 'fail': {
      const last = s.turns[s.turns.length - 1]
      if (!last) return { ...s, inflight: false }
      const updated = { ...last, done: true, error: a.error ?? null }
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
  const sessionIdParam = params.sessionId ?? null
  const setCurrentSessionId = useUiStore((s) => s.setCurrentSessionId)
  const currentSessionId = useUiStore((s) => s.currentSessionId)
  const selectedCollection = useUiStore((s) => s.selectedCollection)
  const activeReportId = useReportStore((s) => s.activeReportId)
  const reportDedupeKeys = useReportDedupeKeys(activeReportId)
  const filters = useChatFiltersStore()
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
                validation_reason: m.validation_reason
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
    if (!message || state.inflight) return
    // Read once: the same list is sent with the question and recorded on the
    // turn, so the guardrail below compares what was asked for against what
    // the server says it did.
    const pinned = scopedChunkIds
    dispatch({ type: 'start', user: message, scopeRequested: pinned.length || undefined })
    clearDraft(key)

    const ac = new AbortController()
    abortRef.current = ac
    try {
      for await (const ev of streamQuery(
        {
          question: message,
          session_id: currentSessionId ?? undefined,
          // WS2 backend resolves + owner-gates the collection per request; the
          // client selection is the single source of truth (no server-side
          // active collection anymore).
          collection: selectedCollection ?? undefined,
          metadata_filters: filters.buildPayload(),
          retrieval_mode: filters.retrievalMode,
          // The scope rides along with the question it scopes. An unstarted
          // chat has no session row to pin it to yet, and even on an open one
          // a checkbox ticked a moment ago may still be in flight — either way
          // the request is the only place this turn's scope is certain.
          scope_chunk_ids: pinned.length > 0 ? pinned : undefined
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
        if (typeof data.error === 'string') {
          // Post-D2 the backend sends a static protocol flag here, not
          // prose — never render the field itself. The validated `code`
          // token selects the copy and is shown for support triage.
          // The fallback is the *reported-failure* copy, not the
          // stream-ended one: an error frame means the backend answered
          // and named its failure. Only the transport path below, which
          // carries no code at all, may claim the stream ended.
          dispatch({ type: 'fail', error: streamErrorText(t, data.code, 'chat.error_backend_failed') })
          continue
        }
        const final = data as unknown as ChatFinalEvent
        dispatch({ type: 'finalize', meta: final })
        if (!currentSessionId && final.session_id) {
          setCurrentSessionId(final.session_id)
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
    } catch (e) {
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
      dispatch({ type: 'fail', error })
    } finally {
      abortRef.current = null
    }
  }

  return (
    // The column widths are animated rather than swapped, so collapsing the
    // panel reflows the transcript smoothly instead of making it pop.
    <div
      className="p-8 grid gap-6 h-full transition-[grid-template-columns] duration-300 ease-out"
      style={{ gridTemplateColumns: sidePanelOpen ? '1fr 22rem' : '1fr 2.75rem' }}
    >
      {/* min-h-0 on the section (grid item) and the messages list (flex
          item) lets them shrink below their content, so the list scrolls
          internally instead of the section outgrowing h-full — which made
          the whole page scroll and pushed the composer flush against the
          viewport bottom, past the p-8 padding box. */}
      <section className="flex flex-col h-full min-h-0">
        <div className="flex items-center justify-between mb-4">
          <PageHeader title={t('chat.title')} className="mb-0" />
          {/* The metadata filters and the retrieval mode belong here, to the
              chat, not to the search panel: they narrow what any answer
              retrieves against, whether or not the panel is even open. In the
              panel they read as controls over the keyword index, which they
              are not.

              They are last in the row, so they hold the right edge: Download
              appears only once a transcript exists, and controls placed
              outboard of it would slide sideways the first time an answer
              lands. */}
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
            <ChatControls />
          </div>
        </div>
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
        <div
          ref={transcript.ref}
          onScroll={transcript.onScroll}
          className="flex-1 min-h-0 overflow-auto space-y-6 pr-2"
        >
          {state.turns.map((t, i) => (
            <ChatTurn
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
          className="mt-4 flex items-end gap-2"
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
          <Button
            variant="primary"
            type="submit"
            disabled={state.inflight || !draft.trim()}
          >
            {state.inflight ? '…' : t('chat.send')}
          </Button>
        </form>
      </section>

      {/* The rail is deliberately quieter than the app sidebar's hamburger:
          a slim chevron on the panel's own edge, muted until hovered or
          focused, and nothing else — it opens and shuts the panel, and every
          detail about a search belongs in the panel it opens.

          It does carry one bit of state, because that bit has consequences
          while hidden: a live scope tints the control, since a chat answering
          from hand-picked chunks behind a shut panel is a trap. Padding stays
          put in both states so the tint appears without nudging anything. */}
      <aside className="flex h-full min-h-0 gap-2 overflow-hidden">
        <div className="flex shrink-0 flex-col items-center gap-2">
          <button
            type="button"
            onClick={toggleSidePanel}
            aria-expanded={sidePanelOpen}
            aria-controls="chat-side-panel"
            aria-label={sidePanelOpen ? t('search.collapse') : t('search.expand')}
            // 32px square, like every control in the chat header: both columns
            // start on the same line, so matching the size is what puts this
            // button on that line rather than 4px above it.
            className={cn(
              'flex h-8 w-8 items-center justify-center rounded-md transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-border',
              scopedChunkIds.length > 0
                ? 'bg-primary/15 text-primary'
                : 'text-muted-foreground hover:text-foreground focus-visible:text-foreground'
            )}
            data-scoped={scopedChunkIds.length > 0 || undefined}
          >
            <RailChevron open={sidePanelOpen} />
          </button>
        </div>
        {/* Kept mounted while collapsed so a running search stays warm and
            the typed query survives a collapse. */}
        <div id="chat-side-panel" hidden={!sidePanelOpen} className="h-full min-h-0 min-w-0 flex-1">
          <SearchPanel sessionId={currentSessionId} />
        </div>
      </aside>
    </div>
  )
}
