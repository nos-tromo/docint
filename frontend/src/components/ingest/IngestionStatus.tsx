import { useEffect, useState } from 'react'
import { cn } from '@/lib/cn'
import {
  formatBytes,
  formatDuration,
  type IngestPhase,
  type IngestStatus
} from '@/lib/ingestStatus'

type Tone = 'sky' | 'amber' | 'emerald' | 'red'

interface PhaseTheme {
  border: string
  pill: string
  label: string
  text: string
  pulse: boolean
  tone: Tone
}

const PHASE_THEME: Record<IngestPhase, PhaseTheme> = {
  idle: {
    border: 'border-zinc-800',
    pill: 'bg-zinc-500',
    label: 'text-zinc-300',
    text: 'Idle',
    pulse: false,
    tone: 'sky'
  },
  uploading: {
    border: 'border-sky-700',
    pill: 'bg-sky-400',
    label: 'text-sky-200',
    text: 'Uploading',
    pulse: true,
    tone: 'sky'
  },
  processing: {
    border: 'border-amber-700',
    pill: 'bg-amber-400',
    label: 'text-amber-200',
    text: 'Processing',
    pulse: true,
    tone: 'amber'
  },
  complete: {
    border: 'border-emerald-700',
    pill: 'bg-emerald-400',
    label: 'text-emerald-200',
    text: 'Complete',
    pulse: false,
    tone: 'emerald'
  },
  error: {
    border: 'border-red-700',
    pill: 'bg-red-400',
    label: 'text-red-200',
    text: 'Failed',
    pulse: false,
    tone: 'red'
  }
}

function Bar({
  value,
  max,
  tone
}: {
  value: number
  max: number
  tone: Tone
}) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0
  const fill =
    tone === 'sky'
      ? 'bg-sky-500'
      : tone === 'amber'
        ? 'bg-amber-500'
        : tone === 'emerald'
          ? 'bg-emerald-500'
          : 'bg-red-500'
  return (
    <div className="h-1.5 w-full rounded-full bg-zinc-800 overflow-hidden">
      <div
        className={cn('h-full transition-[width] duration-300 ease-out', fill)}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}

function useElapsedMs(
  phase: IngestPhase,
  startedAt?: number,
  finishedAt?: number
): number {
  // Tick once per second while the ingest is in flight so the elapsed
  // counter updates without re-deriving status on every animation frame.
  const [now, setNow] = useState(() => Date.now())
  const ticking = phase === 'uploading' || phase === 'processing'
  useEffect(() => {
    if (!ticking) return
    const id = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [ticking])
  if (startedAt === undefined) return 0
  const end = finishedAt ?? (ticking ? now : startedAt)
  return Math.max(0, end - startedAt)
}

export function IngestionStatus({ status }: { status: IngestStatus }) {
  const theme = PHASE_THEME[status.phase]
  const elapsedMs = useElapsedMs(status.phase, status.startedAt, status.finishedAt)
  if (status.phase === 'idle') return null

  return (
    <div
      className={cn('rounded-lg border bg-zinc-950 p-4', theme.border)}
      role="status"
      aria-live="polite"
    >
      <Header status={status} theme={theme} elapsedMs={elapsedMs} />

      {status.phase === 'uploading' && <UploadingBody status={status} />}
      {status.phase === 'processing' && <ProcessingBody status={status} />}
      {status.phase === 'complete' && <CompleteBody status={status} />}
      {status.phase === 'error' && <ErrorBody status={status} />}
    </div>
  )
}

function Header({
  status,
  theme,
  elapsedMs
}: {
  status: IngestStatus
  theme: PhaseTheme
  elapsedMs: number
}) {
  const showTimer = status.startedAt !== undefined && status.phase !== 'error'
  const icon =
    status.phase === 'complete' ? '✓' : status.phase === 'error' ? '✗' : null
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="flex items-center gap-2 min-w-0">
        {icon ? (
          <span
            className={cn(
              'text-xs font-medium',
              status.phase === 'complete' ? 'text-emerald-200' : 'text-red-200'
            )}
            aria-hidden="true"
          >
            {icon}
          </span>
        ) : (
          <span
            className={cn(
              'size-2 rounded-full shrink-0',
              theme.pill,
              theme.pulse && 'animate-pulse'
            )}
            aria-hidden="true"
          />
        )}
        <span
          className={cn(
            'text-xs font-medium uppercase tracking-wide',
            theme.label
          )}
        >
          {theme.text}
        </span>
        {status.collection && (
          <>
            <span className="text-muted-foreground text-xs">·</span>
            <span className="text-sm text-foreground truncate">
              {status.collection}
            </span>
          </>
        )}
      </div>
      {showTimer && (
        <span className="tabular-nums text-xs text-muted-foreground shrink-0">
          <span aria-hidden="true">⏱ </span>
          {formatDuration(elapsedMs)}
        </span>
      )}
    </div>
  )
}

function UploadingBody({ status }: { status: IngestStatus }) {
  const { uploadingFile, uploadingBytes, uploadingTotalBytes } = status
  const fileIndex = Math.min(status.filesSaved + 1, Math.max(1, status.totalFiles))
  const bytesText =
    uploadingBytes !== undefined
      ? uploadingTotalBytes
        ? `${formatBytes(uploadingBytes)} / ${formatBytes(uploadingTotalBytes)}`
        : formatBytes(uploadingBytes)
      : undefined
  const barValue =
    uploadingBytes !== undefined && uploadingTotalBytes
      ? uploadingBytes
      : status.filesSaved
  const barMax =
    uploadingBytes !== undefined && uploadingTotalBytes
      ? uploadingTotalBytes
      : Math.max(1, status.totalFiles)
  return (
    <div className="mt-3 space-y-2">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-sm text-foreground">
          {status.totalFiles > 0
            ? `Saving file ${fileIndex} of ${status.totalFiles}`
            : 'Uploading files'}
        </span>
        {bytesText && (
          <span className="tabular-nums text-xs text-muted-foreground">
            {bytesText}
          </span>
        )}
      </div>
      {uploadingFile && (
        <div className="text-xs text-muted-foreground truncate">
          {uploadingFile}
        </div>
      )}
      <Bar value={barValue} max={barMax} tone="sky" />
      {status.totalFiles > 0 && (
        <div className="text-xs text-muted-foreground border-t border-zinc-800 pt-3 mt-3">
          <span className="tabular-nums">{status.filesSaved}</span>
          {' of '}
          <span className="tabular-nums">{status.totalFiles}</span>
          {' files saved'}
        </div>
      )}
    </div>
  )
}

function ProcessingBody({ status }: { status: IngestStatus }) {
  const hasStage = !!status.stage
  const hasTasks = status.tasks.length > 0
  const showWorking = !hasStage && !hasTasks
  return (
    <div className="mt-3 space-y-3">
      {hasStage && status.stage && (
        <div className="space-y-1.5">
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-sm text-foreground">{status.stage.label}</span>
            <span className="tabular-nums text-xs text-muted-foreground">
              {status.stage.current} of {status.stage.total}
            </span>
          </div>
          <Bar
            value={status.stage.current}
            max={status.stage.total || 1}
            tone="amber"
          />
          {status.stage.currentItem && (
            <div className="text-xs text-muted-foreground truncate">
              {status.stage.currentItem}
            </div>
          )}
        </div>
      )}

      {hasTasks && (
        <div className={cn('space-y-2', hasStage && 'border-t border-zinc-800 pt-3')}>
          {status.tasks.map((task) => (
            <div key={task.key} className="space-y-1">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-sm text-foreground">{task.label}</span>
                <span className="tabular-nums text-xs text-muted-foreground">
                  {task.current}/{task.total}
                </span>
              </div>
              <Bar value={task.current} max={task.total || 1} tone="amber" />
            </div>
          ))}
        </div>
      )}

      {showWorking && (
        <div className="text-sm text-muted-foreground">Working…</div>
      )}

      {(status.filesSaved > 0 || status.indexed > 0) && (
        <div className="text-xs text-muted-foreground border-t border-zinc-800 pt-3">
          <span className="tabular-nums">{status.filesSaved}</span>
          {' files saved · '}
          <span className="tabular-nums">{status.indexed}</span>
          {' PDFs indexed'}
        </div>
      )}
    </div>
  )
}

function CompleteBody({ status }: { status: IngestStatus }) {
  const fileCount = Math.max(status.indexed, status.filesSaved, status.totalFiles)
  const parts: string[] = []
  if (fileCount > 0) parts.push(`${fileCount} files indexed`)
  if (status.totalChunks > 0) parts.push(`${status.totalChunks} chunks`)
  const summary = parts.length > 0 ? parts.join(' · ') : 'Ingestion finished'
  return (
    <div className="mt-3 text-sm text-emerald-200 tabular-nums">
      {summary}
    </div>
  )
}

function ErrorBody({ status }: { status: IngestStatus }) {
  return (
    <div className="mt-3 text-sm text-red-200">
      {status.errorMessage ?? 'Ingestion failed.'}
    </div>
  )
}
