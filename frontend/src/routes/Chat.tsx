import { useEffect, useReducer, useRef } from 'react'
import { useParams } from 'react-router-dom'
import { Button } from '@infra/ui'
import { streamQuery } from '@/api/chat'
import type { ChatFinalEvent } from '@/api/types'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { useSessionHistory } from '@/hooks/useSessions'
import { useUiStore } from '@/stores/ui'
import { useQueryClient } from '@tanstack/react-query'
import { sessionsKey } from '@/hooks/useSessions'
import { ChatTurn, type ChatTurnData } from '@/components/chat/ChatTurn'
import { FilterBuilder } from '@/components/chat/FilterBuilder'
import { downloadText } from '@/lib/csv'
import { chatTranscriptToText } from '@/lib/exports'

interface State {
  turns: ChatTurnData[]
  inflight: boolean
  draft: string
}
type Action =
  | { type: 'set_turns'; turns: ChatTurnData[] }
  | { type: 'set_draft'; value: string }
  | { type: 'start'; user: string }
  | { type: 'token'; token: string }
  | { type: 'finalize'; meta: ChatFinalEvent }
  | { type: 'fail'; error?: string }

function reducer(s: State, a: Action): State {
  switch (a.type) {
    case 'set_turns':
      return { ...s, turns: a.turns }
    case 'set_draft':
      return { ...s, draft: a.value }
    case 'start':
      return {
        ...s,
        draft: '',
        inflight: true,
        turns: [
          ...s.turns,
          { user: a.user, assistant: '', done: false, meta: null, error: null }
        ]
      }
    case 'token': {
      const last = s.turns[s.turns.length - 1]
      const updated = { ...last, assistant: last.assistant + a.token }
      return { ...s, turns: [...s.turns.slice(0, -1), updated] }
    }
    case 'finalize': {
      const last = s.turns[s.turns.length - 1]
      const finalText = a.meta.answer ?? a.meta.message ?? last.assistant
      const updated = { ...last, assistant: finalText, done: true, meta: a.meta }
      return { ...s, inflight: false, turns: [...s.turns.slice(0, -1), updated] }
    }
    case 'fail': {
      const last = s.turns[s.turns.length - 1]
      const updated = { ...last, done: true, error: a.error ?? null }
      return { ...s, inflight: false, turns: [...s.turns.slice(0, -1), updated] }
    }
  }
}

export function Chat() {
  const params = useParams()
  const sessionIdParam = params.sessionId ?? null
  const setCurrentSessionId = useUiStore((s) => s.setCurrentSessionId)
  const currentSessionId = useUiStore((s) => s.currentSessionId)
  const filters = useChatFiltersStore()
  const qc = useQueryClient()
  const history = useSessionHistory(sessionIdParam)
  const [state, dispatch] = useReducer(reducer, { turns: [], inflight: false, draft: '' })
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    setCurrentSessionId(sessionIdParam)
  }, [sessionIdParam, setCurrentSessionId])

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
    const message = state.draft.trim()
    if (!message || state.inflight) return
    dispatch({ type: 'start', user: message })

    const ac = new AbortController()
    abortRef.current = ac
    try {
      for await (const ev of streamQuery(
        {
          question: message,
          session_id: currentSessionId ?? undefined,
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
          dispatch({ type: 'fail', error: data.error })
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
      // A backend OOM mid-stream surfaces as a generic "Failed to fetch"
      // / "network error" TypeError from the underlying reader. The
      // transport-level message isn't actionable on its own; flag the
      // most likely cause and keep the raw detail for forensics.
      const detail = e instanceof Error ? e.message : String(e)
      dispatch({
        type: 'fail',
        error:
          'Chat stream ended unexpectedly — the backend may have crashed ' +
          '(out of memory while loading NER/LLM models is the usual cause). ' +
          `Check backend logs and try again. (transport: ${detail})`
      })
    } finally {
      abortRef.current = null
    }
  }

  return (
    <div className="p-8 grid grid-cols-[1fr_22rem] gap-6 h-full">
      <section className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-semibold">Chat</h1>
          {state.turns.length > 0 && (
            <button
              type="button"
              onClick={() =>
                downloadText(
                  `chat_${currentSessionId ?? 'session'}.txt`,
                  chatTranscriptToText(state.turns)
                )
              }
              className="px-3 py-1 rounded-md border border-border text-sm"
            >
              Download
            </button>
          )}
        </div>
        <div className="flex-1 overflow-auto space-y-6 pr-2">
          {state.turns.map((t, i) => (
            <ChatTurn key={i} turn={t} />
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
            value={state.draft}
            onChange={(e) => dispatch({ type: 'set_draft', value: e.target.value })}
            onKeyDown={(e) => {
              // Enter submits; Shift+Enter inserts a newline (standard
              // chat-composer behavior). IME composition is excluded so
              // Enter while composing doesn't accidentally submit.
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault()
                void send()
              }
            }}
            placeholder="Ask something…"
            rows={1}
            className="flex-1 bg-zinc-900 border border-border rounded-md px-3 py-2 resize-none max-h-40 leading-6"
          />
          <Button
            variant="primary"
            type="submit"
            disabled={state.inflight || !state.draft.trim()}
          >
            {state.inflight ? '…' : 'Send'}
          </Button>
        </form>
      </section>

      <aside className="space-y-4">
        <div className="flex flex-col gap-2 text-sm">
          <label className="flex flex-col gap-1">
            <span className="text-xs uppercase text-muted-foreground">Query mode</span>
            <select
              value={filters.queryMode}
              onChange={(e) => filters.setQueryMode(e.target.value as typeof filters.queryMode)}
              className="bg-zinc-900 border border-border rounded-md px-2 py-1"
            >
              <option value="answer">Answer</option>
              <option value="entity_occurrence">Entity occurrence</option>
              <option value="entity_occurrence_multi">Entity occurrence (multi)</option>
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs uppercase text-muted-foreground">Retrieval</span>
            <select
              value={filters.retrievalMode}
              onChange={(e) =>
                filters.setRetrievalMode(e.target.value as typeof filters.retrievalMode)
              }
              className="bg-zinc-900 border border-border rounded-md px-2 py-1"
            >
              <option value="session">Session</option>
              <option value="stateless">Stateless</option>
            </select>
          </label>
        </div>
        <FilterBuilder />
      </aside>
    </div>
  )
}
