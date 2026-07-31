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
/** Job ids the mocked `/ingest/jobs/{id}` endpoint currently reports as
 *  known (resolves 200). Tests populate this to mark a job as *not*
 *  interrupted; the default (empty) is "the server has genuinely
 *  never/no-longer heard of this job" (resolves 404, matching the real
 *  `getIngestJob` contract). */
let knownJobIds: Set<string>
/** Job ids the mocked `/ingest/jobs/{id}` endpoint reports as still
 *  `queued` (waiting on the concurrency semaphore) rather than `running`. */
let queuedJobIds: Set<string>

function jobSnapshot(id: string) {
  return {
    job_id: id,
    collection: 'mydocs',
    status: (queuedJobIds.has(id) ? 'queued' : 'running') as 'queued' | 'running',
    message: null,
    error: null,
    empty: false,
    resolution: null,
    created_at: '',
    started_at: null,
    finished_at: null
  }
}

function notFoundRes() {
  return {
    ok: false,
    status: 404,
    json: async () => ({ detail: 'not found' }),
    text: async () => '{"detail":"not found"}'
  }
}

beforeEach(() => {
  useIngestRunStore.getState().reset()
  useIngestJobsStore.getState().clear()
  knownJobIds = new Set()
  queuedJobIds = new Set()
  fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const u = typeof input === 'string' ? input : input.toString()
    if (u.includes('/collections/select')) return jsonRes({ ok: true, name: 'mydocs' })
    if (u.includes('/collections/list')) return jsonRes([])
    if (u.includes('/config/ingest-defaults')) return jsonRes({ ner: false, hate_speech: false })
    if (u.includes('/ingest/finalize')) return jsonRes({ job_id: 'job-2' })
    if (u.includes('/ingest/jobs/')) {
      const id = u.split('/ingest/jobs/')[1]?.split('?')[0]
      return id && knownJobIds.has(id) ? jsonRes(jobSnapshot(id)) : notFoundRes()
    }
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

  it('renders live progress for the active job and never flashes the interrupted banner', async () => {
    useIngestRunStore.setState({ activeJobId: 'job-1' })
    useIngestJobsStore.getState().appendEvent('job-1', {
      event: 'ingestion_progress',
      data: { job_id: 'job-1', message: 'Extracting entities: 3/9 chunks processed' },
      receivedAt: Date.now()
    })
    // The backend has actually already registered this job — `getIngestJob`
    // will resolve normally once it's asked.
    knownJobIds.add('job-1')

    renderIn(<Ingest />)
    // Assert from the very first render, before `getIngestJob('job-1')` has
    // had any chance to resolve — this is the exact window a prior,
    // list-based approach flashed the banner in (it served stale
    // *previous* data while a background refetch was in flight). The
    // per-job-id query starts with no cached data for a job id it has never
    // seen, so `jobQuery.isError` is `false` immediately, by construction —
    // there is nothing to wait for here.
    expect(screen.queryByText(/interrupted/i)).not.toBeInTheDocument()

    await waitFor(() => expect(screen.getByText(/3\s*\/\s*9/)).toBeInTheDocument())
    // Steady state, after the query has actually settled.
    expect(screen.queryByText(/interrupted/i)).not.toBeInTheDocument()
  })

  it('flags a job as interrupted — with Run-again and Dismiss — once getIngestJob 404s, even if it had produced events', async () => {
    // Folds in a regression an earlier list-based design failed: a job that
    // emitted `ingestion_started`/progress before the backend restarted
    // still has a non-empty event log forever (nothing ever clears it), so
    // "interrupted" must not be gated on whether the job ever had live SSE
    // evidence — only on what the server says about it *now*.
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
    // `knownJobIds` stays empty (the default) — `getIngestJob('job-1')` 404s.

    renderIn(<Ingest />)
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /run again|erneut/i })).toBeInTheDocument()
    )
    expect(screen.getByRole('button', { name: /dismiss/i })).toBeInTheDocument()
  })

  it('offers a re-run when the active job is unknown to the server', async () => {
    useIngestRunStore.setState({ activeJobId: 'ghost' })
    renderIn(<Ingest />)
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /run again|erneut/i })).toBeInTheDocument()
    )
  })

  it('shows a queued notice for a job with no frames yet, instead of rendering nothing', async () => {
    // Regression test: with the default DOCINT_INGEST_CONCURRENCY=1, a
    // second ingest sits in `queued` and emits zero SSE frames until a
    // worker slot frees up — `status.phase` never leaves 'idle' and the
    // whole status block is otherwise gated out, so the run would vanish
    // from view with no card, no spinner, no error.
    useIngestRunStore.setState({ activeJobId: 'job-1' })
    knownJobIds.add('job-1')
    queuedJobIds.add('job-1')
    // No jobEvents at all — this is the exact state a queued job is in.

    renderIn(<Ingest />)
    await waitFor(() =>
      expect(screen.getByText('Waiting for a worker slot — this run will start as soon as the current ingest finishes.')).toBeInTheDocument()
    )
    // Not interrupted — the server knows about the job, it just hasn't started.
    expect(screen.queryByText(/interrupted/i)).not.toBeInTheDocument()
  })

  it('does not read a completed-but-undismissed job as interrupted after a backend restart', async () => {
    // Regression test: `activeJobId` is persisted, so after any backend
    // restart `getIngestJob` 404s regardless of how the job ended. Without
    // consulting `handledJobId` (which already records "this job reached
    // ingestion_complete"), a run that finished successfully would flash the
    // "interrupted" banner and a spurious Run-again button.
    useIngestRunStore.setState({ activeJobId: 'job-1', handledJobId: 'job-1' })
    // `knownJobIds` stays empty — `getIngestJob('job-1')` 404s, as it would
    // after a restart wiped the in-memory job registry.

    renderIn(<Ingest />)
    await waitFor(() => expect(screen.queryByText(/collection/i)).toBeInTheDocument())
    // Give the 404 a chance to resolve and the (absent) banner a chance to render.
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(screen.queryByText(/interrupted/i)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /run again|erneut/i })).not.toBeInTheDocument()
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

  it('re-runs against the job\'s captured collection, not an edited live form field', async () => {
    // Regression test: `run.collection` is the live, user-editable form
    // field — if the user changes it between the interruption and clicking
    // "Run again", the button must still finalize the collection the
    // interrupted job actually targeted (`activeJobCollection`, captured at
    // queue time), not whatever the field currently holds.
    useIngestRunStore.setState({
      activeJobId: 'ghost',
      collection: 'edited-after-interruption',
      activeJobCollection: 'original-mydocs'
    })
    renderIn(<Ingest />)

    const rerun = await screen.findByRole('button', { name: /run again|erneut/i })
    rerun.click()

    await waitFor(() => expect(callsMatching('/ingest/finalize')).toBe(1))
    const finalizeCall = fetchMock.mock.calls.find((c) => String(c[0]).includes('/ingest/finalize'))!
    const body = JSON.parse((finalizeCall[1] as RequestInit).body as string)
    expect(body.collection).toBe('original-mydocs')
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
