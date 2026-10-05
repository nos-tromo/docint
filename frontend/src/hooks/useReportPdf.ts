import { useEffect, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { queueReportPdf, reportPdfHref, reportPdfStatus } from '@/api/reports'
import { selectJobEvents, useIngestJobsStore } from '@/stores/ingestJobs'
import type { IngestEvent, IngestJobSnapshot, ReportPdfStatus, ReportSummary } from '@/api/types'

/** How often the status route is asked while a render is queued or running. */
export const REPORT_PDF_POLL_MS = 3000

/** What a running render is doing, as its progress frames name it. */
export type ReportPdfStage = 'preparing' | 'layout' | 'finishing'

export type ReportPdfPhase = 'idle' | 'queued' | 'running' | 'ready' | 'stale' | 'failed'

export interface ReportPdfView {
  phase: ReportPdfPhase
  /** Set while `running`. */
  stage: ReportPdfStage | null
  /** The page the first layout pass has reached; set only while laying out. */
  page: number | null
}

/** The stored render, as the status route reports it. */
export type ReportPdfStored = NonNullable<ReportPdfStatus['pdf']>

/** Root of every status query for one report, whatever its `updated_at`. */
const reportPdfKey = (id: number) => ['report-pdf', id] as const

const STAGES: ReadonlySet<string> = new Set<ReportPdfStage>(['preparing', 'layout', 'finishing'])

/**
 * Jobs this tab queued or adopted — the only ones it may download unprompted.
 * Module-level so a request outlives the Report tab unmounting; in memory so a
 * reloaded page asked for nothing and downloads nothing on its own.
 */
const requestedJobs = new Set<string>()

/**
 * Jobs whose PDF this tab already downloaded. The job stream replays a job's
 * history on every reconnect, so its completion can be seen more than once.
 */
const downloadedJobs = new Set<string>()

/** Forget which jobs this tab requested and downloaded. For tests. */
export function resetReportPdfJobs(): void {
  requestedJobs.clear()
  downloadedJobs.clear()
}

function isActive(job: IngestJobSnapshot | null | undefined): boolean {
  return job?.status === 'queued' || job?.status === 'running'
}

/**
 * Read one render's frames: whether it has started, and its latest stage and
 * page. The store collapses consecutive progress frames, so the last one seen
 * is the newest.
 */
function readFrames(frames: IngestEvent[]): { started: boolean; stage: ReportPdfStage | null; page: number | null } {
  let started = false
  let stage: ReportPdfStage | null = null
  let page: number | null = null
  for (const frame of frames) {
    if (frame.event === 'report_pdf_started') {
      started = true
    } else if (frame.event === 'report_pdf_progress') {
      started = true
      const data = frame.data ?? {}
      if (typeof data.stage === 'string' && STAGES.has(data.stage)) stage = data.stage as ReportPdfStage
      page = typeof data.page === 'number' && Number.isFinite(data.page) ? data.page : null
    }
  }
  return { started, stage, page }
}

/**
 * Fold what is known about a report's PDF into one state.
 *
 * A render in flight outranks everything; then a current PDF, which can be
 * downloaded whatever happened since; then a failure; then an outdated PDF.
 * A cancelled job counts as no job at all.
 *
 * @param input.job - The newest report-PDF job the status route reports.
 * @param input.pdf - The stored render the status route reports.
 * @param input.frames - The active job's frames from the shared job stream.
 * @param input.pending - True while this tab's own request is not yet
 *   confirmed by the status route.
 * @param input.startFailed - True when this tab's last request was refused.
 * @returns The phase, plus the stage and page while running.
 */
export function deriveReportPdfView(input: {
  job: IngestJobSnapshot | null
  pdf: ReportPdfStatus['pdf']
  frames: IngestEvent[]
  pending: boolean
  startFailed: boolean
}): ReportPdfView {
  const live = readFrames(input.frames)
  const stage = live.stage ?? 'preparing'
  const running: ReportPdfView = { phase: 'running', stage, page: stage === 'layout' ? live.page : null }
  if (input.pending) return running
  if (isActive(input.job)) {
    // A job waiting for the render slot emits no frames, so `queued` is the
    // status route's to say; a started frame only means it moved on since.
    return input.job?.status === 'running' || live.started ? running : { phase: 'queued', stage: null, page: null }
  }
  if (input.pdf?.current) return { phase: 'ready', stage: null, page: null }
  if (input.startFailed || input.job?.status === 'failed') return { phase: 'failed', stage: null, page: null }
  if (input.pdf) return { phase: 'stale', stage: null, page: null }
  return { phase: 'idle', stage: null, page: null }
}

/** Hand the browser a file to download, the way a clicked link would. */
function downloadFile(href: string, filename: string): void {
  const anchor = document.createElement('a')
  anchor.href = href
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
}

/**
 * This tab's own request for a render of one report: kept until the status
 * route has answered once since, or for good when the queue route refused.
 */
interface LocalRequest {
  reportId: number
  /** Null while the queue route has not answered yet. */
  jobId: string | null
  /** When the queue route answered: a status fetched before then cannot know the job. */
  at: number
  /** The queue route refused. */
  failed: boolean
}

/**
 * The PDF export of one report: a background render with progress, and a
 * download once it is ready.
 *
 * Rendering a large report outlasts the gateway's request timeout, so the PDF
 * is a job (`POST /reports/{id}/pdf`). The status route is the authority on
 * which job is the report's and on whether the stored PDF still matches the
 * report; the shared job stream only adds live progress. A job the status
 * route does not report — forgotten by a backend restart, say — is ignored,
 * whatever frames for it the stream left behind.
 *
 * The status is keyed by the report's `updated_at`, so any edit refetches it
 * and `current` stays honest. A render this tab asked for downloads itself
 * once it completes, exactly once; one it did not ask for never does.
 *
 * @param report - The loaded report.
 * @returns The derived state, the stored PDF, whether the status is still
 *   loading, and `start` to queue a render.
 */
export function useReportPdf(report: Pick<ReportSummary, 'id' | 'updated_at'>) {
  const queryClient = useQueryClient()
  const [request, setRequest] = useState<LocalRequest | null>(null)
  const own = request?.reportId === report.id ? request : null

  const statusQuery = useQuery({
    queryKey: [...reportPdfKey(report.id), report.updated_at],
    queryFn: () => reportPdfStatus(report.id),
    staleTime: 0,
    // An edit changes the key; keeping this report's last answer meanwhile
    // stops a running render's status line from blinking out.
    placeholderData: (previous, previousQuery) =>
      previousQuery?.queryKey[1] === report.id ? previous : undefined,
    refetchInterval: (query) => {
      const data = query.state.data
      if (isActive(data?.job)) return REPORT_PDF_POLL_MS
      const unconfirmed =
        own?.jobId != null && data?.job?.job_id !== own.jobId && query.state.dataUpdatedAt < own.at
      return unconfirmed ? REPORT_PDF_POLL_MS : false
    }
  })

  const status = statusQuery.data
  const job = status?.job ?? null
  const stored = status?.pdf ?? null
  // A placeholder answer predates the edit that changed the key, so it cannot
  // vouch that its PDF matches the report as it now stands.
  const pdf = stored && statusQuery.isPlaceholderData ? { ...stored, current: false } : stored

  const answeredSince = own?.jobId != null && statusQuery.dataUpdatedAt >= own.at
  const pending = own !== null && !own.failed && (own.jobId === null || (job?.job_id !== own.jobId && !answeredSince))
  // From the first status fetched after the queue answered, the status route
  // alone says which job is the report's — a later key starts at no data.
  const settledJobId = answeredSince ? own.jobId : null
  useEffect(() => {
    if (settledJobId) setRequest((r) => (r?.jobId === settledJobId ? null : r))
  }, [settledJobId])

  const activeJobId = pending ? (own?.jobId ?? null) : (job?.job_id ?? null)
  const frames = useIngestJobsStore(selectJobEvents(activeJobId))
  const ended = useIngestJobsStore((s) => (activeJobId ? s.terminal[activeJobId] === true : false))

  // The stream says the run ended while the status still describes it as in
  // flight (or does not know it yet): ask again rather than wait for a poll.
  const statusBehind = ended && !(job?.job_id === activeJobId && !isActive(job))
  useEffect(() => {
    if (statusBehind) void queryClient.invalidateQueries({ queryKey: reportPdfKey(report.id) })
  }, [statusBehind, report.id, queryClient])

  const completedJobId = job?.status === 'completed' ? job.job_id : null
  const currentFilename = pdf?.current ? pdf.filename : null
  useEffect(() => {
    if (!completedJobId || !currentFilename) return
    if (!requestedJobs.has(completedJobId) || downloadedJobs.has(completedJobId)) return
    downloadedJobs.add(completedJobId)
    downloadFile(reportPdfHref(report.id), currentFilename)
  }, [completedJobId, currentFilename, report.id])

  const view = deriveReportPdfView({ job, pdf, frames, pending, startFailed: own?.failed ?? false })

  const start = async (): Promise<void> => {
    if (pending) return
    const reportId = report.id
    setRequest({ reportId, jobId: null, at: 0, failed: false })
    let jobId: string
    try {
      jobId = (await queueReportPdf(reportId)).job_id
    } catch (e) {
      console.error('Queueing the report PDF failed', e)
      setRequest({ reportId, jobId: null, at: 0, failed: true })
      return
    }
    requestedJobs.add(jobId)
    setRequest({ reportId, jobId, at: Date.now(), failed: false })
    await queryClient.invalidateQueries({ queryKey: reportPdfKey(reportId) })
  }

  // `loading`: no status yet, so whether a current PDF exists is unknown.
  return { ...view, pdf, loading: statusQuery.isPending, start }
}
