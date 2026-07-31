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
/** Job ids the mocked `/ingest/jobs` list currently reports as known. Tests
 *  populate this to mark a job as *not* interrupted; the default (empty) is
 *  "the server has genuinely never/no-longer heard of this job". */
let knownJobIds: Set<string>

function jobSnapshot(id: string) {
  return {
    job_id: id,
    collection: 'mydocs',
    status: 'running' as const,
    message: null,
    error: null,
    empty: false,
    resolution: null,
    created_at: '',
    started_at: null,
    finished_at: null
  }
}

beforeEach(() => {
  useIngestRunStore.getState().reset()
  useIngestJobsStore.getState().clear()
  knownJobIds = new Set()
  fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const u = typeof input === 'string' ? input : input.toString()
    if (u.includes('/collections/select')) return jsonRes({ ok: true, name: 'mydocs' })
    if (u.includes('/collections/list')) return jsonRes([])
    if (u.includes('/config/ingest-defaults')) return jsonRes({ ner: false, hate_speech: false })
    if (u.includes('/ingest/finalize')) return jsonRes({ job_id: 'job-2' })
    if (u.includes('/ingest/jobs')) return jsonRes({ jobs: [...knownJobIds].map(jobSnapshot) })
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
    // Mount with no active job first and let the initial (empty)
    // `/ingest/jobs` fetch settle — mirroring the real timeline: the view is
    // already mounted (its own `/ingest/jobs` snapshot taken well before this
    // moment) when the user's run.start()/rerun sets `activeJobId`. Setting
    // `activeJobId` *before* the first render instead would race this
    // effect's invalidation against the query's own first-mount fetch and
    // the two coalesce into one request — a real react-query behavior, but
    // not the scenario this regression is about.
    renderIn(<Ingest />)
    await waitFor(() => expect(callsMatching('/ingest/jobs')).toBeGreaterThanOrEqual(1))

    useIngestRunStore.setState({ activeJobId: 'job-1' })
    useIngestJobsStore.getState().appendEvent('job-1', {
      event: 'ingestion_progress',
      data: { job_id: 'job-1', message: 'Extracting entities: 3/9 chunks processed' },
      receivedAt: Date.now()
    })
    // The backend has actually already registered this job (the run that
    // started it queued successfully) — reflect that from here on, and rely
    // on the view's activeJobId-change effect to invalidate + refetch
    // `/ingest/jobs` so this becomes visible.
    knownJobIds.add('job-1')

    await waitFor(() => expect(screen.getByText(/3\s*\/\s*9/)).toBeInTheDocument())
    // Regression guard: `interrupted` must not render for a job the server
    // does know about. Wait for the post-activeJobId-change refetch to
    // actually land (not just be dispatched) before asserting the banner's
    // absence, or a still-pending/stale query would let the assertion pass
    // for the wrong reason.
    await waitFor(() => expect(callsMatching('/ingest/jobs')).toBeGreaterThanOrEqual(2))
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(screen.queryByText(/interrupted/i)).not.toBeInTheDocument()
  })

  it('flags a job as interrupted even after it produced events, once the jobs list confirms it is gone', async () => {
    // Regression test: an earlier fix required `jobEvents.length === 0` to
    // flag a job interrupted, reasoning that live SSE evidence proves the
    // server knows about it. But a job that emitted `ingestion_started` and
    // progress *before* the backend restarted still has a non-empty event
    // log forever (nothing ever clears it) — that gate made such a job
    // permanently exempt from ever being flagged interrupted again, no
    // matter how many times the (mocked, permanently job-less) `/ingest/jobs`
    // list confirmed it was gone. `interrupted` must be driven by current
    // list membership alone.
    useIngestRunStore.setState({ activeJobId: 'job-1' })
    const { appendEvent } = useIngestJobsStore.getState()
    appendEvent('job-1', {
      event: 'ingestion_started',
      data: { job_id: 'job-1', collection: 'mydocs' },
      receivedAt: Date.now()
    })
    appendEvent('job-1', {
      event: 'ingestion_progress',
      data: { job_id: 'job-1', message: 'Extracting entities: 3/9 chunks processed' },
      receivedAt: Date.now()
    })
    // `knownJobIds` stays empty (the default) — every `/ingest/jobs` fetch,
    // including the one the activeJobId-change effect triggers, confirms the
    // job is genuinely gone.

    renderIn(<Ingest />)
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /run again|erneut/i })).toBeInTheDocument()
    )
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

  it('does not repeat the effect across an unmount + remount for the same completed job', async () => {
    // `activeJobId` is persisted and the job's event log lives in the
    // module-level job store, so navigating away and back (or a reload)
    // still observes the same terminal frame. A component-local guard (e.g.
    // a `useRef`) resets on remount and would repeat the side effect.
    useIngestRunStore.setState({ activeJobId: 'job-1', collection: 'mydocs' })
    const { appendEvent } = useIngestJobsStore.getState()
    appendEvent('job-1', {
      event: 'ingestion_started',
      data: { job_id: 'job-1', collection: 'mydocs' },
      receivedAt: Date.now()
    })
    appendEvent('job-1', {
      event: 'ingestion_complete',
      data: { job_id: 'job-1', collection: 'mydocs' },
      receivedAt: Date.now()
    })

    const { unmount } = renderIn(<Ingest />)
    await waitFor(() => expect(callsMatching('/collections/select')).toBe(1))
    unmount()

    renderIn(<Ingest />)
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(callsMatching('/collections/select')).toBe(1)
  })
})

describe('Ingest — second run in the same tab', () => {
  it('does not show the previous job as complete while a second run is still uploading', async () => {
    // `activeJobId` still points at the *previous* run's job until
    // `createIngestJob` resolves for the new run (stores/ingestRun.ts). This
    // reproduces that exact window directly via store state, mirroring what
    // `run.start()` produces mid-flight.
    useIngestRunStore.setState({ activeJobId: 'job-1', collection: 'mydocs' })
    // A completed-but-not-dismissed job stays listed by the real backend
    // (dismissal is a separate, explicit action) — so it must not read as
    // interrupted, which would otherwise render a second, unrelated Dismiss
    // control this test isn't exercising.
    knownJobIds.add('job-1')
    const { appendEvent } = useIngestJobsStore.getState()
    appendEvent('job-1', {
      event: 'ingestion_started',
      data: { job_id: 'job-1', collection: 'mydocs' },
      receivedAt: Date.now()
    })
    appendEvent('job-1', {
      event: 'ingestion_complete',
      data: { job_id: 'job-1', collection: 'mydocs' },
      receivedAt: Date.now()
    })

    renderIn(<Ingest />)
    await waitFor(() => expect(callsMatching('/collections/select')).toBe(1))

    useIngestRunStore.setState({
      uploading: true,
      uploadEvents: [
        { event: 'start', data: { collection: 'mydocs', files: ['b.txt'] }, receivedAt: Date.now() },
        { event: 'upload_progress', data: { filename: 'b.txt', bytes_written: 5 }, receivedAt: Date.now() }
      ]
    })

    await waitFor(() => expect(screen.getByText('Uploading')).toBeInTheDocument())
    expect(screen.queryByText('Complete')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /dismiss/i })).not.toBeInTheDocument()
  })
})

describe('Ingest — interrupted run', () => {
  it('re-queues directly (finalize only) instead of re-uploading', async () => {
    useIngestRunStore.setState({ activeJobId: 'ghost', collection: 'mydocs' })
    renderIn(<Ingest />)

    // Let the deployment-defaults seed settle first (it writes `ner`/`hate`
    // on mount), then make an explicit choice — mirrors a user ticking a box
    // before hitting "Run again", and avoids racing the seed effect.
    const nerCheckbox = (await screen.findByLabelText('Extract entities')) as HTMLInputElement
    await waitFor(() => expect(nerCheckbox.checked).toBe(false))
    useIngestRunStore.getState().setNer(true)

    const rerun = await screen.findByRole('button', { name: /run again|erneut/i })
    rerun.click()

    await waitFor(() => expect(callsMatching('/ingest/finalize')).toBe(1))
    expect(callsMatching('/ingest/upload')).toBe(0)
    await waitFor(() => expect(useIngestRunStore.getState().activeJobId).toBe('job-2'))

    const finalizeCall = fetchMock.mock.calls.find((c) => String(c[0]).includes('/ingest/finalize'))!
    const body = JSON.parse((finalizeCall[1] as RequestInit).body as string)
    expect(body).toEqual({ collection: 'mydocs', hybrid: true, ner: true, hate_speech: false })
  })

  it('lets the user dismiss a permanently-interrupted (ghost) job', async () => {
    useIngestRunStore.setState({ activeJobId: 'ghost' })
    renderIn(<Ingest />)

    const dismiss = await screen.findByRole('button', { name: /dismiss/i })
    dismiss.click()

    await waitFor(() => expect(useIngestRunStore.getState().activeJobId).toBeNull())
    expect(screen.queryByText(/interrupted/i)).not.toBeInTheDocument()
  })
})
