import { useEffect, useReducer, useRef } from 'react'
import { useParams } from 'react-router-dom'
import { streamQuery } from '@/api/chat'
import type { ChatFinalEvent } from '@/api/types'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { useSessionHistory } from '@/hooks/useSessions'
import { useUiStore } from '@/stores/ui'
import { useQueryClient } from '@tanstack/react-query'
import { sessionsKey } from '@/hooks/useSessions'
import { ChatTurn, type ChatTurnData } from '@/components/chat/ChatTurn'
import { FilterBuilder } from '@/components/chat/FilterBuilder'

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
  | { type: 'fail' }

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
        turns: [...s.turns, { user: a.user, assistant: '', done: false, meta: null }]
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
      const updated = { ...last, done: true }
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
          meta: m.citations ? ({ sources: m.citations, session_id: sessionIdParam ?? '' } as ChatFinalEvent) : null
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
        if (ev.event === 'token') {
          const tok = (ev.data as { token?: string }).token ?? ''
          dispatch({ type: 'token', token: tok })
        } else if (ev.event === 'done') {
          const final = ev.data as ChatFinalEvent
          dispatch({ type: 'finalize', meta: final })
          if (!currentSessionId && final.session_id) {
            setCurrentSessionId(final.session_id)
          }
          qc.invalidateQueries({ queryKey: sessionsKey })
        }
      }
    } catch {
      dispatch({ type: 'fail' })
    } finally {
      abortRef.current = null
    }
  }

  return (
    <div className="p-8 grid grid-cols-[1fr_22rem] gap-6 h-full">
      <section className="flex flex-col h-full">
        <h1 className="text-2xl font-semibold mb-4">Chat</h1>
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
          className="mt-4 flex gap-2"
        >
          <textarea
            value={state.draft}
            onChange={(e) => dispatch({ type: 'set_draft', value: e.target.value })}
            placeholder="Ask something…"
            rows={2}
            className="flex-1 bg-zinc-900 border border-border rounded-md px-3 py-2"
          />
          <button
            type="submit"
            disabled={state.inflight || !state.draft.trim()}
            className="px-4 py-2 rounded-md bg-zinc-100 text-zinc-900 disabled:opacity-50"
          >
            {state.inflight ? '…' : 'Send'}
          </button>
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
