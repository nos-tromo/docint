import { useEffect, useReducer, useRef } from 'react'
import { useParams } from 'react-router-dom'
import { Button, PageHeader } from '@infra/ui'
import { streamQuery } from '@/api/chat'
import { ApiError } from '@/api/client'
import { describeError, streamErrorText } from '@/api/errorMessage'
import type { ChatFinalEvent } from '@/api/types'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { useSessionHistory } from '@/hooks/useSessions'
import { useReportDedupeKeys } from '@/hooks/useReports'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'
import { draftKey, useChatUiStore } from '@/stores/chatUi'
import { useQueryClient } from '@tanstack/react-query'
import { sessionsKey } from '@/hooks/useSessions'
import { ChatTurn, type ChatTurnData } from '@/components/chat/ChatTurn'
import { FilterBuilder } from '@/components/chat/FilterBuilder'
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
  | { type: 'start'; user: string }
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
          { user: a.user, assistant: '', done: false, meta: null, error: null }
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
  const abortRef = useRef<AbortController | null>(null)
  const key = draftKey(sessionIdParam)
  const draft = useChatUiStore((s) => s.drafts[key] ?? '')
  const setDraft = useChatUiStore((s) => s.setDraft)
  const clearDraft = useChatUiStore((s) => s.clearDraft)

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
    dispatch({ type: 'start', user: message })
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
          query_mode: filters.queryMode,
          retrieval_mode: filters.retrievalMode
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
          dispatch({ type: 'fail', error: streamErrorText(t, data.code, 'chat.error_stream_ended') })
          continue
        }
        const final = data as unknown as ChatFinalEvent
        dispatch({ type: 'finalize', meta: final })
        if (!currentSessionId && final.session_id) {
          setCurrentSessionId(final.session_id)
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
    <div className="p-8 grid grid-cols-[1fr_22rem] gap-6 h-full">
      {/* min-h-0 on the section (grid item) and the messages list (flex
          item) lets them shrink below their content, so the list scrolls
          internally instead of the section outgrowing h-full — which made
          the whole page scroll and pushed the composer flush against the
          viewport bottom, past the p-8 padding box. */}
      <section className="flex flex-col h-full min-h-0">
        <div className="flex items-center justify-between mb-4">
          <PageHeader title={t('chat.title')} className="mb-0" />
          {state.turns.length > 0 && (
            <button
              type="button"
              onClick={() =>
                downloadText(
                  `chat_${currentSessionId ?? 'session'}.txt`,
                  chatTranscriptToText(state.turns, t)
                )
              }
              className="px-3 py-1 rounded-md border border-border text-sm"
            >
              {t('chat.download')}
            </button>
          )}
        </div>
        <div className="flex-1 min-h-0 overflow-auto space-y-6 pr-2">
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

      <aside className="space-y-4">
        <div className="flex flex-col gap-2 text-sm">
          <label className="flex flex-col gap-1">
            <span className="text-xs uppercase text-muted-foreground">{t('chat.query_mode')}</span>
            <select
              value={filters.queryMode}
              onChange={(e) => filters.setQueryMode(e.target.value as typeof filters.queryMode)}
              className="bg-muted border border-border rounded-md px-2 py-1"
            >
              <option value="answer">{t('chat.mode_answer')}</option>
              <option value="entity_occurrence">{t('chat.mode_entity_occurrence')}</option>
              <option value="entity_occurrence_multi">
                {t('chat.mode_entity_occurrence_multi')}
              </option>
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs uppercase text-muted-foreground">{t('chat.retrieval')}</span>
            <select
              value={filters.retrievalMode}
              onChange={(e) =>
                filters.setRetrievalMode(e.target.value as typeof filters.retrievalMode)
              }
              className="bg-muted border border-border rounded-md px-2 py-1"
            >
              <option value="session">{t('chat.retrieval_session')}</option>
              <option value="stateless">{t('chat.retrieval_stateless')}</option>
            </select>
          </label>
        </div>
        <FilterBuilder />
      </aside>
    </div>
  )
}
