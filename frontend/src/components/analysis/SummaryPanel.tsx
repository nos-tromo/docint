import { useReducer } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { streamSummary } from '@/api/analysis'
import { Citation } from '@/components/chat/Citation'
import { ValidationBanner } from '@/components/chat/ValidationBanner'
import { downloadText } from '@/lib/csv'
import { summaryToMarkdown } from '@/lib/exports'
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
        // /summarize/stream emits untyped SSE frames; discriminate by
        // payload shape (mirrors Chat).
        const data = ev.data as Record<string, unknown> | string
        if (typeof data !== 'object' || data === null) continue
        if (typeof data.token === 'string') {
          dispatch({ type: 'token', v: data.token })
          continue
        }
        if (typeof data.error === 'string') {
          dispatch({ type: 'fail', error: data.error })
          continue
        }
        dispatch({ type: 'done', meta: data as unknown as SummaryResponse })
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
        {state.text && (
          <button
            type="button"
            onClick={() =>
              downloadText(
                'summary.md',
                summaryToMarkdown(state.meta, state.text),
                'text/markdown;charset=utf-8'
              )
            }
            className="ml-auto px-3 py-1 rounded-md border border-border"
          >
            Download MD
          </button>
        )}
      </div>

      {state.error && <div className="text-red-400 text-sm">{state.error}</div>}
      {state.text && (
        <div className="rounded-md border border-border bg-zinc-900 p-4 text-sm">
          <div className="prose prose-invert prose-sm max-w-none prose-p:my-2 prose-pre:bg-zinc-950 prose-code:before:content-none prose-code:after:content-none">
            <Markdown remarkPlugins={[remarkGfm]}>{state.text}</Markdown>
          </div>
        </div>
      )}
      {state.meta && <ValidationBanner v={state.meta} />}
      {state.meta?.summary_diagnostics && (
        <CoverageBanner d={state.meta.summary_diagnostics} />
      )}
      {state.meta?.sources && state.meta.sources.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs uppercase text-muted-foreground">Sources</div>
          {state.meta.sources.map((s, i) => (
            <Citation
              key={s.id ?? `${s.filename}-${s.page ?? ''}-${s.row ?? ''}-${i}`}
              source={s}
            />
          ))}
        </div>
      )}
    </div>
  )
}
