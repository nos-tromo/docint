import { describe, it, expect, beforeEach, afterEach, vi, type MockInstance } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { resetReportPdfJobs, useReportPdf } from './useReportPdf'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import type { IngestEvent, IngestJobSnapshot, ReportPdfStatus } from '@/api/types'

let status: ReportPdfStatus = { job: null, pdf: null }
const reportPdfStatus = vi.fn(async () => status)
const queueReportPdf = vi.fn(async (): Promise<{ job_id: string; adopted: boolean }> => ({
  job_id: 'j1',
  adopted: false
}))

vi.mock('@/api/reports', async () => {
  const actual = await vi.importActual<typeof import('@/api/reports')>('@/api/reports')
  return {
    ...actual,
    reportPdfStatus: () => reportPdfStatus(),
    queueReportPdf: () => queueReportPdf()
  }
})

const REPORT = { id: 1, updated_at: '2026-03-01T10:00:00Z' }

const RENDER = {
  report_id: 1,
  filename: 'report-1-Case Alpha.pdf',
  size: 52_000,
  created_at: '2026-03-01T10:05:00Z',
  report_updated_at: '2026-03-01T10:00:00Z',
  pages: 12
}

function job(jobId: string, jobStatus: IngestJobSnapshot['status']): IngestJobSnapshot {
  return {
    job_id: jobId,
    collection: 'case-files',
    status: jobStatus,
    message: null,
    error: null,
    empty: false,
    resolution: null,
    kind: 'report_pdf',
    target: '1',
    artifact: null,
    created_at: '2026-03-01T10:01:00Z',
    run_started_at: '2026-03-01T10:01:00Z',
    started_at: null,
    finished_at: null,
    duration_ms: null
  }
}

function frame(jobId: string, event: IngestEvent['event'], data: Record<string, unknown> = {}) {
  useIngestJobsStore.getState().appendEvent(jobId, { event, data: { job_id: jobId, ...data } })
}

let client: QueryClient

function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

let click: MockInstance<() => void>

/** The anchors the hook clicked, i.e. the downloads it started. */
function downloads(): HTMLAnchorElement[] {
  return click.mock.contexts as HTMLAnchorElement[]
}

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  status = { job: null, pdf: null }
  reportPdfStatus.mockClear()
  queueReportPdf.mockClear()
  useIngestJobsStore.getState().clear()
  resetReportPdfJobs()
  click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
})

afterEach(() => {
  vi.restoreAllMocks()
  client.clear()
})

describe('useReportPdf — derived state', () => {
  it('is idle with no job and no stored PDF', async () => {
    const { result } = renderHook(() => useReportPdf(REPORT), { wrapper })
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.phase).toBe('idle')
  })

  it('is queued while the job waits for the render slot, which emits no frames', async () => {
    status = { job: job('j1', 'queued'), pdf: null }
    const { result } = renderHook(() => useReportPdf(REPORT), { wrapper })
    await waitFor(() => expect(result.current.phase).toBe('queued'))
  })

  it('is running with the page the layout pass has reached', async () => {
    status = { job: job('j1', 'running'), pdf: null }
    frame('j1', 'report_pdf_started')
    frame('j1', 'report_pdf_progress', { message: 'Laying out page 3', stage: 'layout', page: 3 })
    const { result } = renderHook(() => useReportPdf(REPORT), { wrapper })
    await waitFor(() => expect(result.current.phase).toBe('running'))
    expect(result.current.stage).toBe('layout')
    expect(result.current.page).toBe(3)
  })

  it('takes a started frame over a queued status the poll has not caught up with', async () => {
    status = { job: job('j1', 'queued'), pdf: null }
    frame('j1', 'report_pdf_started')
    const { result } = renderHook(() => useReportPdf(REPORT), { wrapper })
    await waitFor(() => expect(result.current.phase).toBe('running'))
    expect(result.current.stage).toBe('preparing')
    expect(result.current.page).toBeNull()
  })

  it('is ready when the stored PDF matches the report', async () => {
    status = { job: job('j1', 'completed'), pdf: { ...RENDER, current: true } }
    const { result } = renderHook(() => useReportPdf(REPORT), { wrapper })
    await waitFor(() => expect(result.current.phase).toBe('ready'))
    expect(result.current.pdf?.filename).toBe('report-1-Case Alpha.pdf')
  })

  it('is stale when the report changed since the stored PDF', async () => {
    status = { job: null, pdf: { ...RENDER, current: false } }
    const { result } = renderHook(() => useReportPdf(REPORT), { wrapper })
    await waitFor(() => expect(result.current.phase).toBe('stale'))
  })

  it('is failed when the newest job failed', async () => {
    status = { job: job('j1', 'failed'), pdf: { ...RENDER, current: false } }
    const { result } = renderHook(() => useReportPdf(REPORT), { wrapper })
    await waitFor(() => expect(result.current.phase).toBe('failed'))
  })

  it('treats a cancelled job as no job at all', async () => {
    status = { job: job('j1', 'cancelled'), pdf: null }
    const { result } = renderHook(() => useReportPdf(REPORT), { wrapper })
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.phase).toBe('idle')
  })

  it('ignores the frames of a job the status route no longer reports', async () => {
    status = { job: job('j-old', 'running'), pdf: null }
    frame('j-old', 'report_pdf_started')
    frame('j-old', 'report_pdf_progress', { message: 'Laying out page 8', stage: 'layout', page: 8 })
    const { result } = renderHook(() => useReportPdf(REPORT), { wrapper })
    await waitFor(() => expect(result.current.phase).toBe('running'))

    // The backend restarted: its registry is empty, though the stream's last
    // frames for the old job are still in the store and never ended.
    status = { job: null, pdf: null }
    await act(() => client.invalidateQueries())
    await waitFor(() => expect(result.current.phase).toBe('idle'))

    // A new render waits for the slot. It has no frames of its own, and the
    // old job's are not its progress.
    status = { job: job('j-new', 'queued'), pdf: null }
    await act(() => client.invalidateQueries())
    await waitFor(() => expect(result.current.phase).toBe('queued'))
  })

  it('refetches the status whenever the report is edited', async () => {
    const { result, rerender } = renderHook((report) => useReportPdf(report), {
      wrapper,
      initialProps: REPORT
    })
    await waitFor(() => expect(result.current.loading).toBe(false))
    const before = reportPdfStatus.mock.calls.length
    rerender({ ...REPORT, updated_at: '2026-03-01T11:00:00Z' })
    await waitFor(() => expect(reportPdfStatus.mock.calls.length).toBe(before + 1))
  })

  it('leaves its own request to the status route once that has answered', async () => {
    const { result, rerender } = renderHook((report) => useReportPdf(report), {
      wrapper,
      initialProps: REPORT
    })
    await waitFor(() => expect(result.current.loading).toBe(false))
    status = { job: job('j1', 'running'), pdf: null }
    await act(() => result.current.start())
    await waitFor(() => expect(result.current.phase).toBe('running'))

    // The registry evicts the finished job; then the report is edited while
    // the status request for the new key is still out.
    status = { job: null, pdf: null }
    await act(() => client.invalidateQueries())
    await waitFor(() => expect(result.current.phase).toBe('idle'))
    reportPdfStatus.mockImplementationOnce(() => new Promise<ReportPdfStatus>(() => {}))
    rerender({ ...REPORT, updated_at: '2026-03-01T11:00:00Z' })
    expect(result.current.phase).toBe('idle')
  })

  it('reports a refused request as failed', async () => {
    queueReportPdf.mockRejectedValueOnce(new Error('503'))
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const { result } = renderHook(() => useReportPdf(REPORT), { wrapper })
    await waitFor(() => expect(result.current.loading).toBe(false))
    await act(() => result.current.start())
    expect(result.current.phase).toBe('failed')
  })
})

describe('useReportPdf — download when ready', () => {
  /** Queue a render from this tab, then let it run and complete. */
  async function renderAndComplete() {
    const hook = renderHook(() => useReportPdf(REPORT), { wrapper })
    await waitFor(() => expect(hook.result.current.loading).toBe(false))

    status = { job: job('j1', 'running'), pdf: null }
    await act(() => hook.result.current.start())
    expect(queueReportPdf).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(hook.result.current.phase).toBe('running'))
    expect(click).not.toHaveBeenCalled()

    // The terminal frame is what makes the hook ask the status route again.
    status = { job: job('j1', 'completed'), pdf: { ...RENDER, current: true } }
    act(() => {
      frame('j1', 'report_pdf_started')
      frame('j1', 'report_pdf_completed', { empty: false, duration_ms: 4200, artifact: RENDER })
    })
    await waitFor(() => expect(hook.result.current.phase).toBe('ready'))
    return hook
  }

  it('downloads the PDF once the render this tab asked for completes', async () => {
    await renderAndComplete()
    await waitFor(() => expect(click).toHaveBeenCalledTimes(1))
    const [anchor] = downloads()
    expect(anchor.getAttribute('href')).toMatch(/\/reports\/1\/pdf$/)
    expect(anchor.download).toBe('report-1-Case Alpha.pdf')
  })

  it('does not download again when the stream replays the completion', async () => {
    const { unmount } = await renderAndComplete()
    await waitFor(() => expect(click).toHaveBeenCalledTimes(1))

    // A reconnect replays the job's history; a return to the tab remounts the
    // hook, which then finds the same completed job and current PDF.
    act(() => {
      frame('j1', 'report_pdf_started')
      frame('j1', 'report_pdf_completed', { empty: false, duration_ms: 4200, artifact: RENDER })
    })
    unmount()
    const again = renderHook(() => useReportPdf(REPORT), { wrapper })
    await waitFor(() => expect(again.result.current.phase).toBe('ready'))
    expect(click).toHaveBeenCalledTimes(1)
  })

  it('downloads an adopted render too — this tab asked for it', async () => {
    queueReportPdf.mockResolvedValueOnce({ job_id: 'j1', adopted: true })
    await renderAndComplete()
    await waitFor(() => expect(click).toHaveBeenCalledTimes(1))
  })

  it('never downloads a render this tab did not ask for', async () => {
    status = { job: job('j9', 'running'), pdf: null }
    const { result } = renderHook(() => useReportPdf(REPORT), { wrapper })
    await waitFor(() => expect(result.current.phase).toBe('running'))

    status = { job: job('j9', 'completed'), pdf: { ...RENDER, current: true } }
    act(() => {
      frame('j9', 'report_pdf_started')
      frame('j9', 'report_pdf_completed', { empty: false, duration_ms: 3100, artifact: RENDER })
    })
    await waitFor(() => expect(result.current.phase).toBe('ready'))
    expect(click).not.toHaveBeenCalled()
  })

  it('does not download a render that finished after the report changed', async () => {
    const hook = renderHook(() => useReportPdf(REPORT), { wrapper })
    await waitFor(() => expect(hook.result.current.loading).toBe(false))
    status = { job: job('j1', 'running'), pdf: null }
    await act(() => hook.result.current.start())

    status = { job: job('j1', 'completed'), pdf: { ...RENDER, current: false } }
    act(() => frame('j1', 'report_pdf_completed', { empty: false, duration_ms: 4200, artifact: RENDER }))
    await waitFor(() => expect(hook.result.current.phase).toBe('stale'))
    expect(click).not.toHaveBeenCalled()
  })
})
