import { useMemo, useReducer, useState } from 'react'
import { Button } from '@infra/ui'
import { streamIngestUploadBatched } from '@/api/ingest'
import { useSelectCollection, useCollections, collectionsKey } from '@/hooks/useCollections'
import { useConfig } from '@/hooks/useConfig'
import { useQueryClient } from '@tanstack/react-query'
import { useUiStore } from '@/stores/ui'
import type { IngestEvent } from '@/api/types'
import { Dropzone } from '@/components/ingest/Dropzone'
import { IngestionStatus } from '@/components/ingest/IngestionStatus'
import { deriveIngestStatus } from '@/lib/ingestStatus'

/**
 * Per-request upload ceiling assumed only until `/config` loads (or if that
 * fetch fails). Deliberately well under the 1 GiB nginx default so batches
 * never 413 even before the real `max_upload_bytes` is known.
 */
const FALLBACK_UPLOAD_LIMIT_BYTES = 512 * 1024 * 1024

interface State {
  collection: string
  files: File[]
  events: IngestEvent[]
  /**
   * Snapshot of file sizes taken when ingestion starts. `state.files` is
   * cleared in `'done'` for UX reasons, so we keep the sizes around here
   * to power the per-file upload progress bar.
   */
  fileSizes: Record<string, number>
  busy: boolean
}
type Action =
  | { type: 'set_collection'; v: string }
  | { type: 'add_files'; v: File[] }
  | { type: 'reset_files' }
  | { type: 'start'; sizes: Record<string, number> }
  | { type: 'event'; v: IngestEvent }
  | { type: 'done' }

// Progress messages like "Extracting entities: 1/2 chunks processed" and
// "Extracting entities: 2/2 chunks processed" share a kind (digits stripped)
// and should update one event entry in place instead of stacking. The
// derived status snapshot does its own collapsing too, but this keeps
// `state.events` from growing without bound on long-running ingests.
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
      return { ...s, busy: true, events: [], fileSizes: a.sizes }
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
    fileSizes: {},
    busy: false
  })
  const [error, setError] = useState<string | null>(null)
  const [warnings, setWarnings] = useState<string[]>([])
  const setSelected = useUiStore((s) => s.setSelectedCollection)
  const selectMutation = useSelectCollection()
  const qc = useQueryClient()
  const { data: collections } = useCollections()
  const { data: config } = useConfig()

  const status = useMemo(
    () => deriveIngestStatus(state.events, state.fileSizes),
    [state.events, state.fileSizes]
  )

  const submit = async () => {
    if (!state.collection || state.files.length === 0) return
    setError(null)
    setWarnings([])
    // The selection is uploaded as staged batches that each stay under the
    // server's per-request ceiling (`/config` max_upload_bytes; the batched
    // stream applies the safety margin), then ingested in one finalize pass.
    // This lets a multi-GB selection ingest instead of being rejected as one
    // oversized body (nginx 413), and ingestion sees the whole selection at once.
    const limitBytes = config?.max_upload_bytes ?? FALLBACK_UPLOAD_LIMIT_BYTES
    const sizes: Record<string, number> = {}
    for (const f of state.files) sizes[f.name] = f.size
    dispatch({ type: 'start', sizes })
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
      for await (const ev of streamIngestUploadBatched(state.collection, state.files, limitBytes)) {
        dispatch({ type: 'event', v: ev })
        if (ev.event === 'warning') {
          // A batch was skipped (e.g. a lone oversize file → 413, or a transient
          // drop). The upload continues; collect the reason so the user sees why.
          const data = ev.data as Record<string, unknown>
          const msg = typeof data.message === 'string' ? data.message : null
          if (msg) setWarnings((prev) => [...prev, msg])
          continue
        }
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
          // Partial success: some batches committed, others were skipped. Keep
          // the collection selected (its ingested files are usable) but flag the
          // skipped files so the outcome isn't silently reported as a clean run.
          const data = ev.data as Record<string, unknown>
          if (typeof data.failed_message === 'string' && data.failed_message) {
            setError(data.failed_message)
          }
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
          className="bg-zinc-900 border border-border rounded-md px-2 py-1 text-sm"
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
        <Button
          variant="primary"
          onClick={submit}
          disabled={state.busy || !state.collection || state.files.length === 0}
        >
          {state.busy ? 'Ingesting…' : 'Ingest'}
        </Button>
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
      {warnings.length > 0 && (
        <ul className="text-amber-400 text-sm space-y-1" role="alert">
          {warnings.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}
      {status.phase !== 'idle' && <IngestionStatus status={status} />}
    </div>
  )
}
