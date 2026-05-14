import { useReducer, useState } from 'react'
import { streamIngestUpload } from '@/api/ingest'
import { useSelectCollection, useCollections, collectionsKey } from '@/hooks/useCollections'
import { useQueryClient } from '@tanstack/react-query'
import { useUiStore } from '@/stores/ui'
import type { IngestEvent } from '@/api/types'
import { Dropzone } from '@/components/ingest/Dropzone'
import { EventTimeline } from '@/components/ingest/EventTimeline'

interface State {
  collection: string
  files: File[]
  events: IngestEvent[]
  busy: boolean
}
type Action =
  | { type: 'set_collection'; v: string }
  | { type: 'add_files'; v: File[] }
  | { type: 'reset_files' }
  | { type: 'start' }
  | { type: 'event'; v: IngestEvent }
  | { type: 'done' }

// Progress messages like "Extracting entities: 1/2 chunks processed" and
// "Extracting entities: 2/2 chunks processed" share a kind (digits stripped)
// and should update one timeline line in place instead of stacking.
function progressKind(ev: IngestEvent): string | null {
  if (ev.event !== 'ingestion_progress') return null
  const message = (ev.data as { message?: unknown })?.message
  if (typeof message !== 'string') return null
  return message.replace(/\d+/g, '#').trim()
}

function reducer(s: State, a: Action): State {
  switch (a.type) {
    case 'set_collection':
      return { ...s, collection: a.v }
    case 'add_files':
      return { ...s, files: [...s.files, ...a.v] }
    case 'reset_files':
      return { ...s, files: [] }
    case 'start':
      return { ...s, busy: true, events: [] }
    case 'event': {
      const last = s.events[s.events.length - 1]
      const incomingKind = progressKind(a.v)
      const lastKind = last ? progressKind(last) : null
      if (incomingKind && incomingKind === lastKind) {
        return { ...s, events: [...s.events.slice(0, -1), a.v] }
      }
      return { ...s, events: [...s.events, a.v] }
    }
    case 'done':
      return { ...s, busy: false, files: [] }
  }
}

export function Ingest() {
  const [state, dispatch] = useReducer(reducer, {
    collection: '',
    files: [],
    events: [],
    busy: false
  })
  const [error, setError] = useState<string | null>(null)
  const setSelected = useUiStore((s) => s.setSelectedCollection)
  const selectMutation = useSelectCollection()
  const qc = useQueryClient()
  const { data: collections } = useCollections()

  const submit = async () => {
    if (!state.collection || state.files.length === 0) return
    setError(null)
    dispatch({ type: 'start' })
    // Track whether the backend ever produced a terminal event so that
    // we can tell "ingestion finished" from "the SSE stream died". An
    // OOM-killed backend either (a) ends the stream silently, or
    // (b) makes the browser fetch throw with a generic "network error" /
    // "Failed to fetch" — both end up here without a terminal event,
    // and we should surface the same actionable message in either case
    // rather than relaying the raw transport-level message verbatim.
    const truncationMessage =
      'Ingestion was interrupted before completing. The backend may have ' +
      'crashed (out of memory while loading NER/LLM models is the usual ' +
      'cause for CSV/large-text ingests). Check the backend logs and try ' +
      'again with more memory allocated to Docker.'
    let sawTerminal = false
    try {
      for await (const ev of streamIngestUpload(state.collection, state.files)) {
        dispatch({ type: 'event', v: ev as IngestEvent })
        if (ev.event === 'error') {
          const data = ev.data as Record<string, unknown>
          setError(String(data.message ?? 'Ingestion failed'))
          sawTerminal = true
          continue
        }
        if (ev.event === 'ingestion_complete') {
          sawTerminal = true
          await selectMutation.mutateAsync(state.collection)
          setSelected(state.collection)
          await qc.invalidateQueries({ queryKey: collectionsKey })
        }
      }
      if (!sawTerminal) setError(truncationMessage)
    } catch (e) {
      if (sawTerminal) {
        setError(e instanceof Error ? e.message : String(e))
      } else {
        // The fetch/stream threw before any terminal event arrived. The
        // raw browser message ("network error" / "Failed to fetch") is
        // not actionable on its own — the real cause is almost always
        // the backend dying mid-stream. Show the truncation notice and
        // append the underlying message so the cause is still visible.
        const detail = e instanceof Error ? e.message : String(e)
        setError(`${truncationMessage} (transport: ${detail})`)
      }
    } finally {
      dispatch({ type: 'done' })
    }
  }

  return (
    <div className="p-8 max-w-3xl space-y-4">
      <h1 className="text-2xl font-semibold">Ingest</h1>

      <label className="flex flex-col gap-1 text-sm max-w-sm">
        <span className="text-xs uppercase text-muted-foreground">Collection</span>
        <input
          list="existing-collections"
          value={state.collection}
          onChange={(e) => dispatch({ type: 'set_collection', v: e.target.value })}
          placeholder="my-collection"
          className="bg-zinc-900 border border-border rounded-md px-2 py-1 font-sans text-sm"
        />
        <datalist id="existing-collections">
          {collections?.map((c) => (
            <option key={c} value={c} />
          ))}
        </datalist>
      </label>

      <Dropzone disabled={state.busy} onFiles={(v) => dispatch({ type: 'add_files', v })} />

      {state.files.length > 0 && (
        <ul className="text-sm space-y-1">
          {state.files.map((f) => (
            <li key={f.name}>
              {f.name} <span className="text-muted-foreground">({f.size} bytes)</span>
            </li>
          ))}
        </ul>
      )}

      <div className="flex gap-2">
        <button
          type="button"
          onClick={submit}
          disabled={state.busy || !state.collection || state.files.length === 0}
          className="px-4 py-2 rounded-md bg-zinc-100 text-zinc-900 disabled:opacity-50"
        >
          {state.busy ? 'Ingesting…' : 'Ingest'}
        </button>
        {state.files.length > 0 && (
          <button
            type="button"
            onClick={() => dispatch({ type: 'reset_files' })}
            className="px-4 py-2 rounded-md border border-border"
          >
            Clear files
          </button>
        )}
      </div>

      {error && <div className="text-red-400 text-sm">{error}</div>}
      {state.events.length > 0 && (
        <div className="rounded-lg border border-border bg-zinc-900 p-4">
          <EventTimeline events={state.events} />
        </div>
      )}
    </div>
  )
}
