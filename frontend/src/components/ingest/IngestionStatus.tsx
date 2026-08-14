import { useEffect, useState } from 'react'
import { cn } from '@/lib/cn'
import {
  formatBytes,
  formatDuration,
  type IngestPhase,
  type IngestStatus
} from '@/lib/ingestStatus'
import { useT } from '@/i18n/LanguageContext'
import type { Strings } from '@/i18n'
import { streamErrorText } from '@/api/errorMessage'

type Tone = 'sky' | 'amber' | 'emerald' | 'red'

interface PhaseTheme {
  border: string
  pill: string
  label: string
  textKey: keyof Strings
  pulse: boolean
  tone: Tone
}

// `status.stage.label` / `task.label` are the literal strings ingestStatus.ts's
// parseProgressMessage() derived from the backend's free-form progress
// messages — not translated there (a pure parsing module with no `useT()`).
// LABEL_KEY-style mapping here, mirroring hateCategoryLabel.ts: an
// unrecognized label (e.g. a future backend stage) falls back to the raw
// string as-is rather than throwing or rendering blank.
const STAGE_LABEL_KEY: Partial<Record<string, keyof Strings>> = {
  'Processing PDFs': 'ingest.stage_processing_pdfs'
}
const TASK_LABEL_KEY: Partial<Record<string, keyof Strings>> = {
  Entities: 'table.col_entities',
  'Hate detection': 'ingest.task_hate_detection'
}

function stageLabel(raw: string, t: (key: keyof Strings) => string): string {
  const key = STAGE_LABEL_KEY[raw]
  return key ? t(key) : raw
}

function taskLabel(raw: string, t: (key: keyof Strings) => string): string {
  const key = TASK_LABEL_KEY[raw]
  return key ? t(key) : raw
}

const PHASE_THEME: Record<IngestPhase, PhaseTheme> = {
  idle: {
    border: 'border-border',
    pill: 'bg-muted-foreground',
    label: 'text-muted-foreground',
    textKey: 'ingest.status_idle',
    pulse: false,
    tone: 'sky'
  },
  uploading: {
    border: 'border-sky-700',
    pill: 'bg-sky-400',
    // text-[var(--status-*-fg)]: text rendered directly on the theme-reactive
    // bg-muted panel below needs its own light/dark pair (see globals.css) —
    // a fixed Tailwind shade like text-sky-200 is only AA on a dark bg.
    label: 'text-[var(--status-sky-fg)]',
    textKey: 'ingest.status_uploading',
    pulse: true,
    tone: 'sky'
  },
  processing: {
    border: 'border-amber-700',
    pill: 'bg-amber-400',
    label: 'text-[var(--status-amber-fg)]',
    textKey: 'ingest.status_processing',
    pulse: true,
    tone: 'amber'
  },
  complete: {
    border: 'border-emerald-700',
    pill: 'bg-emerald-400',
    label: 'text-[var(--status-emerald-fg)]',
    textKey: 'ingest.status_complete',
    pulse: false,
    tone: 'emerald'
  },
  error: {
    border: 'border-red-700',
    pill: 'bg-red-400',
    label: 'text-[var(--status-red-fg)]',
    textKey: 'ingest.status_failed',
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
    <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
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
      className={cn('rounded-lg border bg-muted p-4', theme.border)}
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
  const t = useT()
  const showTimer = status.startedAt !== undefined
  const icon =
    status.phase === 'complete' ? '✓' : status.phase === 'error' ? '✗' : null
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="flex items-center gap-2 min-w-0">
        {icon ? (
          <span
            className={cn(
              'text-xs font-medium',
              status.phase === 'complete'
                ? 'text-[var(--status-emerald-fg)]'
                : 'text-[var(--status-red-fg)]'
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
          {t(theme.textKey)}
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
  const t = useT()
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
            ? t('ingest.saving_file', { current: fileIndex, total: status.totalFiles })
            : t('ingest.uploading_files')}
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
        <div className="text-xs text-muted-foreground border-t border-border pt-3 mt-3 tabular-nums">
          {t('ingest.files_saved_of', { saved: status.filesSaved, total: status.totalFiles })}
        </div>
      )}
    </div>
  )
}

function ProcessingBody({ status }: { status: IngestStatus }) {
  const t = useT()
  const hasStage = !!status.stage
  const hasTasks = status.tasks.length > 0
  const showWorking = !hasStage && !hasTasks
  return (
    <div className="mt-3 space-y-3">
      {hasStage && status.stage && (
        <div className="space-y-1.5">
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-sm text-foreground">{stageLabel(status.stage.label, t)}</span>
            <span className="tabular-nums text-xs text-muted-foreground">
              {t('ingest.stage_progress', { current: status.stage.current, total: status.stage.total })}
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
        <div className={cn('space-y-2', hasStage && 'border-t border-border pt-3')}>
          {status.tasks.map((task) => (
            <div key={task.key} className="space-y-1">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-sm text-foreground">{taskLabel(task.label, t)}</span>
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
        <div className="text-sm text-muted-foreground">{t('ingest.working')}</div>
      )}

      {(status.filesSaved > 0 || status.indexed > 0) && (
        <div className="text-xs text-muted-foreground border-t border-border pt-3 tabular-nums">
          {t('ingest.files_saved_indexed', { saved: status.filesSaved, indexed: status.indexed })}
        </div>
      )}
    </div>
  )
}

// No duration here on purpose: the header timer has been ticking in this exact
// spot for the whole run, so on completion it simply stops. Restating the same
// number one line down puts it in the card twice, and moves the answer away
// from where the eye was already watching for it. The header timer is also the
// wider of the two — it renders on the error phase as well, and whenever it
// has a duration to show, this summary would have had the identical one.
function CompleteBody({ status }: { status: IngestStatus }) {
  const t = useT()
  const fileCount = Math.max(status.indexed, status.filesSaved, status.totalFiles)
  const parts: string[] = []
  if (fileCount > 0) parts.push(t('ingest.files_indexed', { count: fileCount }))
  if (status.totalChunks > 0) parts.push(t('ingest.chunks', { count: status.totalChunks }))
  const summary = parts.length > 0 ? parts.join(' · ') : t('ingest.finished')
  return (
    <div className="mt-3 text-sm text-[var(--status-emerald-fg)] tabular-nums">
      {summary}
    </div>
  )
}

function ErrorBody({ status }: { status: IngestStatus }) {
  const t = useT()
  // errorMessage is client-composed catalog copy (see deriveIngestStatus).
  // A backend-composed job error instead carries a machine-readable
  // errorCode: streamErrorText renders the generic catalog copy tagged with
  // the regex-validated code token — "Ingestion failed. (ingestion_failed)"
  // — matching how chat/summary render stream errors for support triage.
  // With neither set, streamErrorText degrades to the bare catalog copy.
  return (
    <div className="mt-3 text-sm text-[var(--status-red-fg)]">
      {status.errorMessage ?? streamErrorText(t, status.errorCode, 'ingest.failed_default')}
    </div>
  )
}
