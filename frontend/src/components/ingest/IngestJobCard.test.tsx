import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'
import { IngestJobCard } from './IngestJobCard'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import { useIngestRunStore } from '@/stores/ingestRun'
import { useUiStore } from '@/stores/ui'
import { setOwnerParam } from '@/api/client'
import { deriveIngestStatus } from '@/lib/ingestStatus'

function renderIn(ui: ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

beforeEach(() => {
  useIngestRunStore.getState().reset()
  useIngestJobsStore.getState().clear()
  useUiStore.setState({ selectedCollection: null, selectedOwner: null })
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
  setOwnerParam(null)
})

describe('IngestJobCard', () => {
  it('shows the job its own warnings, inside its own box', async () => {
    useIngestRunStore.getState().trackJob('job-1', 'mydocs', null)
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

    renderIn(<IngestJobCard jobId="job-1" collection="mydocs" owner={null} />)

    await waitFor(() =>
      expect(
        within(screen.getByRole('status')).getByText('One file held no extractable text.')
      ).toBeInTheDocument()
    )
  })

  it('counts the upload leg that produced this job', async () => {
    useIngestRunStore.getState().trackJob(
      'job-1',
      'mydocs',
      null,
      deriveIngestStatus([
        { event: 'start', data: { collection: 'mydocs', files: ['a.txt', 'b.txt'] }, receivedAt: 1 },
        { event: 'file_saved', data: { filename: 'a.txt' }, receivedAt: 2 },
        { event: 'file_saved', data: { filename: 'b.txt' }, receivedAt: 3 }
      ])
    )
    useIngestRunStore.getState().markJobHandled('job-1')
    useIngestJobsStore.getState().appendEvent('job-1', {
      event: 'ingestion_started',
      data: { collection: 'mydocs' },
      receivedAt: 4
    })

    renderIn(<IngestJobCard jobId="job-1" collection="mydocs" owner={null} />)

    // The upload leg belongs to the job it produced, so its card reports what
    // that run actually saved rather than starting from zero.
    await waitFor(() =>
      expect(screen.getByText('2 files saved · 0 files indexed')).toBeInTheDocument()
    )
  })

  it('names its collection even before a frame carries one', async () => {
    useIngestRunStore.getState().trackJob('job-1', 'quarterly-reports', null)
    useIngestRunStore.getState().markJobHandled('job-1')
    useIngestJobsStore.getState().appendEvent('job-1', {
      event: 'ingestion_progress',
      data: { message: 'Extracting entities: 1/9 chunks processed' },
      receivedAt: 1
    })

    renderIn(<IngestJobCard jobId="job-1" collection="quarterly-reports" owner={null} />)

    await waitFor(() => expect(screen.getByText('quarterly-reports')).toBeInTheDocument())
  })
})

describe('IngestJobCard preprocess bars', () => {
  it('keeps reporting the pool the job is still reading through', async () => {
    // The pool does not stop at finalize — the job's own prefetch re-submits
    // the batch — so the bars the upload leg showed must carry on here.
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: unknown) => {
        const url = String(input)
        if (url.includes('/ingest/staged')) {
          return {
            ok: true,
            status: 200,
            json: async () => ({
              collection: 'mydocs',
              files: 3,
              bytes: 9,
              partial: 0,
              entries: [],
              entries_truncated: true,
              preprocess: {
                running: 1,
                queued: 0,
                stages: [{ stage: 'media', done: 1, total: 3, failed: 0 }]
              }
            })
          }
        }
        return {
          ok: false,
          status: 404,
          json: async () => ({ detail: 'not found' }),
          text: async () => '{"detail":"not found"}'
        }
      })
    )
    useIngestRunStore.getState().trackJob('job-1', 'mydocs', null)
    useIngestRunStore.getState().markJobHandled('job-1')
    useIngestJobsStore.getState().appendEvent('job-1', {
      event: 'ingestion_started',
      data: { collection: 'mydocs' },
      receivedAt: 1
    })

    renderIn(<IngestJobCard jobId="job-1" collection="mydocs" owner={null} />)

    expect(await screen.findByText('Transcribing audio and video')).toBeInTheDocument()
    expect(screen.getByText('1/3')).toBeInTheDocument()
  })
})

/** A fetch stand-in answering `answers` by URL substring, and 404 to anything else. */
function stubFetch(urls: string[], answers: Record<string, unknown>) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: unknown) => {
      const u = String(input)
      urls.push(u)
      const hit = Object.keys(answers).find((part) => u.includes(part))
      if (hit) {
        const body = answers[hit]
        return { ok: true, status: 200, json: async () => body, text: async () => JSON.stringify(body) }
      }
      return {
        ok: false,
        status: 404,
        json: async () => ({ detail: 'not found' }),
        text: async () => '{"detail":"not found"}'
      }
    })
  )
}

describe('IngestJobCard owner', () => {
  it('checks, and re-runs, its job under the owner it was queued in', async () => {
    const urls: string[] = []
    stubFetch(urls, { '/ingest/finalize': { job_id: 'job-2' } })
    // The sidebar has since moved on to another user's collections.
    setOwnerParam('bob')
    useIngestRunStore.getState().trackJob('job-1', 'mydocs', 'alice')

    renderIn(<IngestJobCard jobId="job-1" collection="mydocs" owner="alice" />)

    // The backend restarted, so the job is gone and the card offers a re-run.
    fireEvent.click(await screen.findByRole('button', { name: 'Run again' }))

    await waitFor(() =>
      expect(useIngestRunStore.getState().trackedJobs[0]).toEqual({
        job_id: 'job-2',
        collection: 'mydocs',
        owner: 'alice'
      })
    )
    expect(urls.find((u) => u.includes('/ingest/jobs/job-1'))).toContain('owner=alice')
    expect(urls.find((u) => u.includes('/ingest/finalize'))).toContain('owner=alice')
  })

  it('selects the finished run\'s collection under the owner it was queued in', async () => {
    const urls: string[] = []
    stubFetch(urls, { '/collections/select': { ok: true, name: 'hers' } })
    setOwnerParam('bob')
    useIngestRunStore.getState().trackJob('job-1', 'hers', 'alice')
    useIngestJobsStore.getState().appendEvent('job-1', {
      event: 'ingestion_complete',
      data: { job_id: 'job-1', collection: 'hers' },
      receivedAt: 1
    })

    renderIn(<IngestJobCard jobId="job-1" collection="hers" owner="alice" />)

    await waitFor(() =>
      expect(useUiStore.getState()).toMatchObject({ selectedCollection: 'hers', selectedOwner: 'alice' })
    )
    expect(urls.find((u) => u.includes('/collections/select'))).toContain('owner=alice')
  })
})
