import type { IngestEvent, IngestJobSnapshot } from '@/api/types'

export type IngestPhase =
  | 'idle'
  | 'uploading'
  | 'processing'
  | 'complete'
  | 'error'

export interface IngestTask {
  /** Stable React key, e.g. 'entities' | 'hate'. */
  key: string
  /** Human-readable label rendered in the UI. */
  label: string
  current: number
  total: number
}

export interface IngestStageInfo {
  label: string
  current: number
  total: number
  currentItem?: string
}

export interface IngestStatus {
  phase: IngestPhase
  collection?: string
  totalFiles: number
  filesSaved: number
  uploadingFile?: string
  uploadingBytes?: number
  uploadingTotalBytes?: number
  stage?: IngestStageInfo
  tasks: IngestTask[]
  /** Number of files fully indexed by the core pipeline. */
  indexed: number
  /** Total chunks observed across "indexed N chunks" messages. */
  totalChunks: number
  startedAt?: number
  /** Terminal error copy — set only for client-composed `error` events (see
   *  the `error` case in `deriveIngestStatus`); left `undefined` for a
   *  backend-composed one so the renderer's catalog fallback is used instead
   *  of untranslated backend prose. */
  errorMessage?: string
  /** Machine-readable error code from a backend-composed `error` event
   *  (jobs.py's closed enum, e.g. `ingestion_failed`). The renderer passes it
   *  through `streamErrorText` so localized catalog copy carries the
   *  regex-validated code token for support triage — the behavior PR #356
   *  established for the old finalize stream. `undefined` on client-composed
   *  errors, which never set a code. */
  errorCode?: string
  /** Non-terminal warning messages accumulated from `warning` events across
   *  both the upload leg and the job leg — e.g. a soft-empty ingest or a
   *  failed post-ingest entity resolution. */
  warnings: string[]
  finishedAt?: number
}

export type ProgressKind = 'stage' | 'indexed' | 'task' | 'unknown'

export interface ParsedProgress {
  kind: ProgressKind
  label?: string
  current?: number
  total?: number
  item?: string
  taskKey?: string
}

const RE_STAGE = /^Core pipeline processing PDF \((\d+)\/(\d+)\): (.+)$/
const RE_INDEXED = /^Core pipeline indexed (\d+) chunks: (.+)$/
const RE_ENTITIES = /^Extracting entities:\s*(\d+)\/(\d+) chunks processed$/
const RE_HATE = /^Detecting hate speech:\s*(\d+)\/(\d+) chunks processed$/

/**
 * Parse a free-form `ingestion_progress.message` payload into a structured
 * progress descriptor.
 *
 * The backend emits human-readable status strings; we recognise the four
 * known formats (`Core pipeline processing PDF`, `Core pipeline indexed`,
 * `Extracting entities`, `Detecting hate speech`) and return `kind: 'unknown'`
 * for anything else so the UI can show a generic "Working…" indicator.
 *
 * Args:
 *   message: Raw status string from the SSE stream.
 *
 * Returns:
 *   A `ParsedProgress` whose `kind` indicates which template matched.
 */
export function parseProgressMessage(message: string): ParsedProgress {
  if (typeof message !== 'string') return { kind: 'unknown' }
  const trimmed = message.trim()

  const stage = trimmed.match(RE_STAGE)
  if (stage) {
    return {
      kind: 'stage',
      label: 'Processing PDFs',
      current: Number(stage[1]),
      total: Number(stage[2]),
      item: stage[3]
    }
  }

  const indexed = trimmed.match(RE_INDEXED)
  if (indexed) {
    return {
      kind: 'indexed',
      current: Number(indexed[1]),
      item: indexed[2]
    }
  }

  const entities = trimmed.match(RE_ENTITIES)
  if (entities) {
    return {
      kind: 'task',
      taskKey: 'entities',
      label: 'Entities',
      current: Number(entities[1]),
      total: Number(entities[2])
    }
  }

  const hate = trimmed.match(RE_HATE)
  if (hate) {
    return {
      kind: 'task',
      taskKey: 'hate',
      label: 'Hate detection',
      current: Number(hate[1]),
      total: Number(hate[2])
    }
  }

  return { kind: 'unknown' }
}

/**
 * SSE event names whose frames are pure counter updates: only the newest
 * carries information, and a long run emits thousands of them. Both job kinds
 * multiplex through one store, so `summary_progress` must be listed alongside
 * `ingestion_progress` — otherwise a 3,000-unit summary build appends one
 * entry per unit and every append re-scans the whole log (`selectHasRunningJob`
 * is O(n)) and re-renders the sidebar.
 */
const COLLAPSIBLE_PROGRESS_EVENTS: ReadonlySet<IngestEvent['event']> = new Set([
  'ingestion_progress',
  'summary_progress'
])

/**
 * Return the "kind" of a progress event — its event name plus its message with
 * digits masked — so consecutive updates of the same counter collapse into one
 * entry.
 *
 * "Extracting entities: 1/9 chunks processed" and ".. 2/9 .." share a kind;
 * the event name is part of the kind so an ingest frame and a summary frame
 * that happen to carry identical prose never collapse into each other.
 *
 * @param ev - The event to classify.
 * @returns The masked kind, or null for non-progress events.
 */
export function progressKind(ev: IngestEvent): string | null {
  if (!COLLAPSIBLE_PROGRESS_EVENTS.has(ev.event)) return null
  const message = (ev.data as { message?: unknown })?.message
  if (typeof message !== 'string') return null
  return `${ev.event}:${message.replace(/\d+/g, '#').trim()}`
}

/**
 * Return whether a progress event is one of the enrichment counters
 * (`Extracting entities` / `Detecting hate speech`), the only frames the
 * backend emits interleaved (both stages run from one pool).
 *
 * @param ev - The event to classify.
 * @returns True for an enrichment counter frame.
 */
function isEnrichmentCounter(ev: IngestEvent): boolean {
  const message = (ev.data as { message?: unknown })?.message
  if (typeof message !== 'string') return false
  return RE_ENTITIES.test(message) || RE_HATE.test(message)
}

/**
 * Append an event to a log, collapsing a repeat of the previous progress kind
 * in place. The interleaving enrichment counters (entities / hate speech, the
 * only frames the backend alternates) additionally collapse into their most
 * recent same-kind entry within the trailing run of counter frames, so an
 * alternating stream stays at one entry per counter. Any other frame — and
 * any non-progress entry — bounds that scan, which keeps appends O(1) and
 * prevents digit-masking from merging distinct per-file frames (e.g.
 * `indexed 12 chunks: report_v1.pdf` / `indexed 30 chunks: report_v2.pdf`).
 * Keeps the log bounded on long ingests.
 *
 * @param events - The existing log.
 * @param next - The event to append.
 * @returns A new log array.
 */
export function appendCollapsedEvent(events: IngestEvent[], next: IngestEvent): IngestEvent[] {
  const nextKind = progressKind(next)
  if (nextKind) {
    const scanAcrossCounters = isEnrichmentCounter(next)
    for (let i = events.length - 1; i >= 0; i -= 1) {
      if (progressKind(events[i]) === nextKind) {
        const out = events.slice()
        out[i] = next
        return out
      }
      if (!scanAcrossCounters || !isEnrichmentCounter(events[i])) break
    }
  }
  return [...events, next]
}

function dataOf(ev: IngestEvent): Record<string, unknown> {
  return (ev.data ?? {}) as Record<string, unknown>
}

function strOf(v: unknown): string | undefined {
  return typeof v === 'string' ? v : undefined
}

function numOf(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined
}

/**
 * Reduce a list of SSE ingest events into a single status snapshot.
 *
 * The reducer is intentionally tolerant: unknown progress messages are
 * ignored, missing fields are treated as undefined, and tasks update
 * in place by `taskKey` so progress bars stay stable across renders.
 *
 * Args:
 *   events: All ingest events seen so far, in arrival order.
 *   fileSizes: Optional map of filename to size in bytes (from `File.size`)
 *     used to display per-file upload bars.
 *
 * Returns:
 *   The derived `IngestStatus` snapshot.
 */
export function deriveIngestStatus(
  events: IngestEvent[],
  fileSizes?: Record<string, number>
): IngestStatus {
  const status: IngestStatus = {
    phase: 'idle',
    totalFiles: 0,
    filesSaved: 0,
    tasks: [],
    indexed: 0,
    totalChunks: 0,
    warnings: []
  }

  for (const ev of events) {
    const d = dataOf(ev)
    switch (ev.event) {
      case 'start': {
        status.phase = 'uploading'
        status.collection = strOf(d.collection) ?? status.collection
        const files = Array.isArray(d.files) ? (d.files as unknown[]) : []
        // Accumulate rather than assign so totalFiles spans every `start` in the
        // stream. A batched upload emits one synthetic `start` listing all files
        // (so this runs once), but should the stream ever carry a `start` per
        // batch, the count must sum, not reset to the last batch. Single-request
        // ingests emit exactly one `start`, so this matches the old behaviour.
        status.totalFiles += files.length
        // Anchor the elapsed timer to the *arrival* time of the start event,
        // stamped once on the event itself (IngestEvent.receivedAt). Reading
        // Date.now() here instead would reset startedAt on every re-derivation
        // — this reducer re-runs for each incoming event — making the timer
        // snap back to zero on every batch.
        if (status.startedAt === undefined) status.startedAt = ev.receivedAt
        break
      }
      case 'upload_progress': {
        const filename = strOf(d.filename)
        const bytes = numOf(d.bytes_written)
        if (filename) {
          status.uploadingFile = filename
          status.uploadingBytes = bytes
          if (fileSizes && filename in fileSizes) {
            status.uploadingTotalBytes = fileSizes[filename]
          } else {
            status.uploadingTotalBytes = undefined
          }
        }
        break
      }
      case 'file_saved': {
        status.filesSaved += 1
        const filename = strOf(d.filename)
        if (filename && status.uploadingFile === filename) {
          status.uploadingBytes = undefined
        }
        break
      }
      case 'ingestion_started': {
        status.phase = 'processing'
        status.collection = strOf(d.collection) ?? status.collection
        status.uploadingFile = undefined
        status.uploadingBytes = undefined
        status.uploadingTotalBytes = undefined
        break
      }
      case 'ingestion_progress': {
        const message = strOf(d.message)
        if (!message) break
        const parsed = parseProgressMessage(message)
        if (parsed.kind === 'stage') {
          status.stage = {
            label: parsed.label ?? 'Processing',
            current: parsed.current ?? 0,
            total: parsed.total ?? 0,
            currentItem: parsed.item
          }
        } else if (parsed.kind === 'indexed') {
          status.indexed += 1
          status.totalChunks += parsed.current ?? 0
          if (status.stage) {
            const stageTotal = status.stage.total || 0
            const next = status.stage.current + 1
            status.stage = {
              ...status.stage,
              current: stageTotal > 0 ? Math.min(next, stageTotal) : next
            }
          }
        } else if (parsed.kind === 'task' && parsed.taskKey) {
          const key = parsed.taskKey
          const incoming: IngestTask = {
            key,
            label: parsed.label ?? key,
            current: parsed.current ?? 0,
            total: parsed.total ?? 0
          }
          const idx = status.tasks.findIndex((t) => t.key === key)
          if (idx === -1) {
            status.tasks = [...status.tasks, incoming]
          } else {
            const next = status.tasks.slice()
            next[idx] = incoming
            status.tasks = next
          }
        }
        // unknown kinds intentionally ignored — UI falls back to "Working…"
        break
      }
      case 'warning': {
        // Every warning is unique information (a soft-empty ingest, a
        // reader-unsupported batch, a failed post-ingest entity resolution)
        // and the run can emit several — accumulate rather than overwrite.
        const message = strOf(d.message)
        if (message) status.warnings = [...status.warnings, message]
        break
      }
      case 'ingestion_complete': {
        status.phase = 'complete'
        status.collection = strOf(d.collection) ?? status.collection
        status.uploadingFile = undefined
        status.uploadingBytes = undefined
        status.uploadingTotalBytes = undefined
        status.finishedAt = ev.receivedAt
        break
      }
      case 'error': {
        status.phase = 'error'
        // Not every `error` event reaching this reducer is client-composed:
        // the upload leg's own errors are rewritten to catalog copy by
        // streamIngestUploadBatched, but job-stream errors (jobs.py) carry
        // the backend's static protocol copy straight through, tagged with a
        // `code` field the client-composed ones never set. Capture `message`
        // only for the client-composed case — never render backend prose —
        // and capture `code` for the backend case, so ErrorBody can render
        // localized catalog copy tagged with the validated code token
        // (via streamErrorText) instead of an undifferentiated fallback.
        status.errorMessage = typeof d.code === 'string' ? undefined : strOf(d.message)
        status.errorCode = typeof d.code === 'string' ? d.code : undefined
        status.finishedAt = ev.receivedAt
        break
      }
    }
  }

  return status
}

/**
 * Format a byte count using binary units (KiB/MiB-style sizing) but with
 * decimal-style "KB"/"MB"/"GB" labels, matching common UI conventions.
 *
 * Args:
 *   n: Non-negative byte count.
 *
 * Returns:
 *   Human-readable string such as `"0 B"`, `"1023 B"`, or `"1.4 MB"`.
 */
export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return '0 B'
  if (n < 1024) return `${Math.trunc(n)} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let value = n / 1024
  let unitIdx = 0
  while (value >= 1024 && unitIdx < units.length - 1) {
    value /= 1024
    unitIdx += 1
  }
  // One decimal place, truncated (not rounded up) so 1.499 MB renders as 1.4 MB.
  const truncated = Math.floor(value * 10) / 10
  return `${truncated.toFixed(1)} ${units[unitIdx]}`
}

/**
 * Format an elapsed duration in milliseconds, scaling with magnitude:
 * `MM:SS` under an hour, `H:MM:SS` under a day, `Nd HH:MM:SS` beyond.
 * The `d` day marker is the DIN 1301 unit symbol, shared across locales,
 * so this module stays free of catalog lookups.
 *
 * Args:
 *   ms: Duration in milliseconds.
 *
 * Returns:
 *   String of the form `"03:42"`, `"1:02:05"`, or `"1d 17:40:37"`.
 */
export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return '00:00'
  const totalSeconds = Math.floor(ms / 1000)
  const days = Math.floor(totalSeconds / 86_400)
  const hours = Math.floor((totalSeconds % 86_400) / 3_600)
  const minutes = Math.floor((totalSeconds % 3_600) / 60)
  const seconds = totalSeconds % 60
  const mm = String(minutes).padStart(2, '0')
  const ss = String(seconds).padStart(2, '0')
  if (days > 0) return `${days}d ${String(hours).padStart(2, '0')}:${mm}:${ss}`
  if (hours > 0) return `${hours}:${mm}:${ss}`
  return `${mm}:${ss}`
}

/**
 * Fill a derived status's timeline from the server-owned job snapshot when
 * the client-side anchor is missing.
 *
 * After a reload/reattach the merged event log holds only replayed job
 * frames — no synthetic upload `start` — so `startedAt` is undefined and the
 * elapsed timer never renders. When filling from the snapshot, `finished_at`
 * is taken from the *same source*: a replayed terminal frame stamps
 * `finishedAt` with its arrival time, so pairing a server start with that
 * client finish would measure start→reload instead of start→finish (and a
 * same-source pair cancels server/client clock skew). The client
 * `finishedAt` is kept only when the snapshot has no `finished_at` yet —
 * i.e. the terminal event arrived live, where its arrival time is accurate.
 * A client-anchored timeline is returned unchanged.
 *
 * Args:
 *   status: Status derived from the merged event log.
 *   snapshot: Server job snapshot, if one is loaded.
 *
 * Returns:
 *   `status`, with `startedAt`/`finishedAt` filled from the snapshot when
 *   the client log carried no `start` frame.
 */
export function withServerTimes(
  status: IngestStatus,
  snapshot?: Pick<IngestJobSnapshot, 'started_at' | 'finished_at'> | null
): IngestStatus {
  if (status.startedAt !== undefined || !snapshot?.started_at) return status
  const startedAt = Date.parse(snapshot.started_at)
  if (Number.isNaN(startedAt)) return status
  const serverFinishedAt = snapshot.finished_at
    ? Date.parse(snapshot.finished_at)
    : NaN
  return {
    ...status,
    startedAt,
    finishedAt: Number.isNaN(serverFinishedAt)
      ? status.finishedAt
      : serverFinishedAt
  }
}
