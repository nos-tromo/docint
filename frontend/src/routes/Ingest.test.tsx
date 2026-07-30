import { render, screen, waitFor } from '@testing-library/react'
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'
import { Ingest } from './Ingest'
import { useIngestRunStore } from '@/stores/ingestRun'
import { useIngestJobsStore } from '@/stores/ingestJobs'

function jsonRes(body: unknown) {
  return { ok: true, status: 200, json: async () => body, text: async () => JSON.stringify(body) }
}

function renderIn(ui: ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>{ui}</MemoryRouter>
    </QueryClientProvider>
  )
}

let fetchMock: ReturnType<typeof vi.fn>

beforeEach(() => {
  useIngestRunStore.getState().reset()
  useIngestJobsStore.getState().clear()
  fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const u = typeof input === 'string' ? input : input.toString()
    if (u.includes('/collections/select')) return jsonRes({ ok: true, name: 'mydocs' })
    if (u.includes('/collections/list')) return jsonRes([])
    if (u.includes('/config/ingest-defaults')) return jsonRes({ ner: false, hate_speech: false })
    if (u.includes('/ingest/jobs')) return jsonRes({ jobs: [] })
    if (u.includes('/config'))
      return jsonRes({
        graph_top_k: 0,
        graph_max_top_k: 0,
        collection_timeout: 0,
        max_upload_bytes: 1024 * 1024,
        language: 'en'
      })
    return jsonRes(null)
  })
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  vi.restoreAllMocks()
})

/** Count calls whose URL matches a substring, across every fetch so far. */
function callsMatching(pattern: string): number {
  return fetchMock.mock.calls.filter((c) => String(c[0]).includes(pattern)).length
}

describe('Ingest', () => {
  it('restores the collection name from the store on mount', () => {
    useIngestRunStore.getState().setCollection('mydocs')
    renderIn(<Ingest />)
    // The collection input carries a `list` attribute (the existing-collections
    // datalist), which gives it the ARIA "combobox" role rather than "textbox".
    expect(screen.getByRole('combobox', { name: /collection/i })).toHaveValue('mydocs')
  })

  it('renders live progress for the active job', async () => {
    useIngestRunStore.setState({ activeJobId: 'job-1' })
    useIngestJobsStore.getState().appendEvent('job-1', {
      event: 'ingestion_progress',
      data: { job_id: 'job-1', message: 'Extracting entities: 3/9 chunks processed' },
      receivedAt: Date.now()
    })

    renderIn(<Ingest />)
    await waitFor(() => expect(screen.getByText(/3\s*\/\s*9/)).toBeInTheDocument())
  })

  it('offers a re-run when the active job is unknown to the server', async () => {
    useIngestRunStore.setState({ activeJobId: 'ghost' })
    renderIn(<Ingest />)
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /run again|erneut/i })).toBeInTheDocument()
    )
  })
})

describe('Ingest post-ingest side effects', () => {
  it('selects the collection exactly once, even when the terminal event replays', async () => {
    // Regression test for the reconnect-replay case: the SSE stream resets a
    // job's log on `ingestion_started` and re-delivers its whole collapsed
    // history, so a naive effect keyed only on "last event is
    // ingestion_complete" would re-fire the collection-select side effect on
    // every replay.
    useIngestRunStore.setState({ activeJobId: 'job-1', collection: 'mydocs' })
    const { appendEvent } = useIngestJobsStore.getState()
    const complete = () =>
      appendEvent('job-1', {
        event: 'ingestion_complete',
        data: { job_id: 'job-1', collection: 'mydocs' },
        receivedAt: Date.now()
      })
    appendEvent('job-1', {
      event: 'ingestion_started',
      data: { job_id: 'job-1', collection: 'mydocs' },
      receivedAt: Date.now()
    })
    complete()

    renderIn(<Ingest />)
    await waitFor(() => expect(callsMatching('/collections/select')).toBe(1))

    // Simulate a reconnect replay of the same job's history.
    appendEvent('job-1', {
      event: 'ingestion_started',
      data: { job_id: 'job-1', collection: 'mydocs' },
      receivedAt: Date.now()
    })
    complete()

    // Give a wrongly-re-firing effect a chance to make a second call before
    // asserting the count stayed at one.
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(callsMatching('/collections/select')).toBe(1)
  })
})
