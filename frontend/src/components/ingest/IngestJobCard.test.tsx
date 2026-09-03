import { render, screen, waitFor } from '@testing-library/react'
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'
import { IngestJobCard } from './IngestJobCard'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import { useIngestRunStore } from '@/stores/ingestRun'

function renderIn(ui: ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

beforeEach(() => {
  useIngestRunStore.getState().reset()
  useIngestJobsStore.getState().clear()
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok: false,
      status: 404,
      json: async () => ({ detail: 'not found' }),
      text: async () => '{"detail":"not found"}'
    }))
  )
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('IngestJobCard', () => {
  it('shows the job its own warnings, not the screen’s', async () => {
    useIngestRunStore.getState().trackJob('job-1', 'mydocs')
    // Handled, so the 404 from the stubbed fetch does not read as interrupted.
    useIngestRunStore.getState().markJobHandled('job-1')
    const { appendEvent } = useIngestJobsStore.getState()
    appendEvent('job-1', {
      event: 'ingestion_started',
      data: { collection: 'mydocs' },
      receivedAt: 1
    })
    appendEvent('job-1', {
      event: 'warning',
      data: { message: 'One file held no extractable text.' },
      receivedAt: 2
    })

    renderIn(<IngestJobCard jobId="job-1" collection="mydocs" />)

    await waitFor(() =>
      expect(screen.getByText('One file held no extractable text.')).toBeInTheDocument()
    )
  })

  it('counts the upload leg that produced this job', async () => {
    useIngestRunStore.getState().trackJob('job-1', 'mydocs', [
      { event: 'start', data: { collection: 'mydocs', files: ['a.txt', 'b.txt'] }, receivedAt: 1 },
      { event: 'file_saved', data: { filename: 'a.txt' }, receivedAt: 2 },
      { event: 'file_saved', data: { filename: 'b.txt' }, receivedAt: 3 }
    ])
    useIngestRunStore.getState().markJobHandled('job-1')
    useIngestJobsStore.getState().appendEvent('job-1', {
      event: 'ingestion_started',
      data: { collection: 'mydocs' },
      receivedAt: 4
    })

    renderIn(<IngestJobCard jobId="job-1" collection="mydocs" />)

    // The upload leg belongs to the job it produced, so its card reports what
    // that run actually saved rather than starting from zero.
    await waitFor(() =>
      expect(screen.getByText('2 files saved · 0 files indexed')).toBeInTheDocument()
    )
  })

  it('names its collection even before a frame carries one', async () => {
    useIngestRunStore.getState().trackJob('job-1', 'quarterly-reports')
    useIngestRunStore.getState().markJobHandled('job-1')
    useIngestJobsStore.getState().appendEvent('job-1', {
      event: 'ingestion_progress',
      data: { message: 'Extracting entities: 1/9 chunks processed' },
      receivedAt: 1
    })

    renderIn(<IngestJobCard jobId="job-1" collection="quarterly-reports" />)

    await waitFor(() => expect(screen.getByText('quarterly-reports')).toBeInTheDocument())
  })
})
