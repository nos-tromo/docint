import type { IngestEvent } from '@/api/types'

const ICON: Record<string, string> = {
  start: '▶',
  upload_progress: '↑',
  file_saved: '✓',
  ingestion_started: '⚙',
  ingestion_progress: '·',
  ingestion_complete: '✓',
  error: '!'
}

export function EventTimeline({ events }: { events: IngestEvent[] }) {
  return (
    <ol className="text-sm space-y-1">
      {events.map((e, i) => (
        <li key={i} className="flex gap-2">
          <span className="text-muted-foreground w-4">{ICON[e.event] ?? '•'}</span>
          <span className="text-muted-foreground w-44 shrink-0">{e.event}</span>
          <span className="truncate">{describe(e)}</span>
        </li>
      ))}
    </ol>
  )
}

function describe(e: IngestEvent): string {
  const d = e.data as Record<string, unknown>
  if (e.event === 'upload_progress') return `${d.filename} (${d.bytes_written} bytes)`
  if (e.event === 'file_saved') return `${d.filename} → ${d.file_hash}`
  if (e.event === 'ingestion_progress') return String(d.message ?? '')
  if (e.event === 'ingestion_complete') return `done · ${d.collection}`
  if (e.event === 'error') return String(d.message ?? d)
  return Object.entries(d).map(([k, v]) => `${k}=${String(v)}`).join(' ')
}
