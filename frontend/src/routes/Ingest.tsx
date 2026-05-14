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
  hybrid: boolean
  files: File[]
  events: IngestEvent[]
  busy: boolean
}
type Action =
  | { type: 'set_collection'; v: string }
  | { type: 'set_hybrid'; v: boolean }
  | { type: 'add_files'; v: File[] }
  | { type: 'reset_files' }
  | { type: 'start' }
  | { type: 'event'; v: IngestEvent }
  | { type: 'done' }

function reducer(s: State, a: Action): State {
  switch (a.type) {
    case 'set_collection':
      return { ...s, collection: a.v }
    case 'set_hybrid':
      return { ...s, hybrid: a.v }
    case 'add_files':
      return { ...s, files: [...s.files, ...a.v] }
    case 'reset_files':
      return { ...s, files: [] }
    case 'start':
      return { ...s, busy: true, events: [] }
    case 'event':
      return { ...s, events: [...s.events, a.v] }
    case 'done':
      return { ...s, busy: false, files: [] }
  }
}

export function Ingest() {
  const [state, dispatch] = useReducer(reducer, {
    collection: '',
    hybrid: true,
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
    try {
      for await (const ev of streamIngestUpload(state.collection, state.files, state.hybrid)) {
        dispatch({ type: 'event', v: ev as IngestEvent })
        if (ev.event === 'error') {
          const data = ev.data as Record<string, unknown>
          setError(String(data.message ?? 'Ingestion failed'))
          continue
        }
        if (ev.event === 'ingestion_complete') {
          await selectMutation.mutateAsync(state.collection)
          setSelected(state.collection)
          await qc.invalidateQueries({ queryKey: collectionsKey })
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      dispatch({ type: 'done' })
    }
  }

  return (
    <div className="p-8 max-w-3xl space-y-4">
      <h1 className="text-2xl font-semibold">Ingest</h1>

      <div className="grid grid-cols-2 gap-3">
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-xs uppercase text-muted-foreground">Collection</span>
          <input
            list="existing-collections"
            value={state.collection}
            onChange={(e) => dispatch({ type: 'set_collection', v: e.target.value })}
            placeholder="my-collection"
            className="bg-zinc-900 border border-border rounded-md px-2 py-1"
          />
          <datalist id="existing-collections">
            {collections?.map((c) => (
              <option key={c} value={c} />
            ))}
          </datalist>
        </label>
        <label className="flex items-center gap-2 text-sm mt-5">
          <input
            type="checkbox"
            checked={state.hybrid}
            onChange={(e) => dispatch({ type: 'set_hybrid', v: e.target.checked })}
          />
          Hybrid search
        </label>
      </div>

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
