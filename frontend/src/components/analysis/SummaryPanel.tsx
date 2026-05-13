import { useReducer } from 'react'
import { streamSummary } from '@/api/analysis'
import { Citation } from '@/components/chat/Citation'
import { ValidationBanner } from '@/components/chat/ValidationBanner'
import { CoverageBanner } from './CoverageBanner'
import type { SummaryResponse } from '@/api/types'

interface State {
  text: string
  done: boolean
  busy: boolean
  meta: SummaryResponse | null
  error: string | null
}
type Action =
  | { type: 'start' }
  | { type: 'token'; v: string }
  | { type: 'done'; meta: SummaryResponse }
  | { type: 'fail'; error: string }

function reducer(s: State, a: Action): State {
  switch (a.type) {
    case 'start':
      return { text: '', done: false, busy: true, meta: null, error: null }
    case 'token':
      return { ...s, text: s.text + a.v }
    case 'done':
      return { ...s, busy: false, done: true, meta: a.meta, text: a.meta.summary || s.text }
    case 'fail':
      return { ...s, busy: false, done: true, error: a.error }
  }
}

export function SummaryPanel() {
  const [state, dispatch] = useReducer(reducer, {
    text: '',
    done: false,
    busy: false,
    meta: null,
    error: null
  })

  const generate = async (refresh: boolean) => {
    dispatch({ type: 'start' })
    try {
      for await (const ev of streamSummary(refresh)) {
        if (ev.event === 'token') dispatch({ type: 'token', v: (ev.data as { token: string }).token })
        else if (ev.event === 'done') dispatch({ type: 'done', meta: ev.data as SummaryResponse })
      }
    } catch (e) {
      dispatch({ type: 'fail', error: e instanceof Error ? e.message : String(e) })
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => generate(false)}
          disabled={state.busy}
          className="px-3 py-1 rounded-md bg-zinc-100 text-zinc-900 disabled:opacity-50"
        >
          {state.busy ? 'Generating…' : 'Generate'}
        </button>
        <button
          type="button"
          onClick={() => generate(true)}
          disabled={state.busy}
          className="px-3 py-1 rounded-md border border-border"
        >
          Refresh
        </button>
      </div>

      {state.error && <div className="text-red-400 text-sm">{state.error}</div>}
      {state.text && (
        <div className="rounded-md border border-border bg-zinc-900 p-4 whitespace-pre-wrap text-sm">
          {state.text}
        </div>
      )}
      {state.meta && <ValidationBanner v={state.meta} />}
      {state.meta?.summary_diagnostics && (
        <CoverageBanner d={state.meta.summary_diagnostics} />
      )}
      {state.meta?.sources && state.meta.sources.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs uppercase text-muted-foreground">Sources</div>
          {state.meta.sources.map((s) => (
            <Citation key={s.id} source={s} />
          ))}
        </div>
      )}
    </div>
  )
}
