import type { IngestEvent } from '@/api/types'

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
  errorMessage?: string
  startedAt?: number
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
    totalChunks: 0
  }

  for (const ev of events) {
    const d = dataOf(ev)
    switch (ev.event) {
      case 'start': {
        status.phase = 'uploading'
        status.collection = strOf(d.collection) ?? status.collection
        const files = Array.isArray(d.files) ? (d.files as unknown[]) : []
        status.totalFiles = files.length
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
        status.errorMessage = strOf(d.message) ?? 'Ingestion failed'
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
 * Format an elapsed duration in milliseconds as `MM:SS` (or `MMM:SS` for
 * runs longer than an hour). Hours are intentionally rolled into the
 * minutes column to keep the column width predictable in the status card.
 *
 * Args:
 *   ms: Duration in milliseconds.
 *
 * Returns:
 *   String of the form `"03:42"` or `"62:05"`.
 */
export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return '00:00'
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  const mm = String(minutes).padStart(2, '0')
  const ss = String(seconds).padStart(2, '0')
  return `${mm}:${ss}`
}
