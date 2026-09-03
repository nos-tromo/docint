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
/** Server-side timestamps per job id; the default (absent) mirrors a job
 *  that has not started. */
let jobTimes: Record<string, { started_at: string | null; finished_at: string | null }>
/** Job ids the mocked list reports as already finished. */
let finishedJobIds: Set<string>
/** Per-job collection names, for tests that run several collections at once. */
let jobCollections: Record<string, string>

/** Track a job the way `run.start()` does once its finalize resolves. */
function track(jobId: string, collection = 'mydocs') {
  useIngestRunStore.getState().trackJob(jobId, collection)
}

function jobSnapshot(id: string) {
  return {
    job_id: id,
    collection: jobCollections[id] ?? 'mydocs',
    kind: 'ingest' as const,
    status: (queuedJobIds.has(id)
      ? 'queued'
      : finishedJobIds.has(id)
        ? 'completed'
        : 'running') as 'queued' | 'running' | 'completed',
    message: null,
    error: null,
    empty: false,
    resolution: null,
    created_at: '',
    started_at: jobTimes[id]?.started_at ?? null,
    finished_at: jobTimes[id]?.finished_at ?? null
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
  finishedJobIds = new Set()
  jobCollections = {}
  jobTimes = {}
  fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const u = typeof input === 'string' ? input : input.toString()
    if (u.includes('/collections/select')) return jsonRes({ ok: true, name: 'mydocs' })
    if (u.includes('/collections/list')) return jsonRes([])
    if (u.includes('/config/ingest-defaults')) return jsonRes({ ner: false, hate_speech: false })
    if (u.includes('/ingest/finalize')) return jsonRes({ job_id: 'job-2' })
    if (u.includes('/ingest/jobs/')) {
      const id = u.split('/ingest/jobs/')[1]?.split('?')[0]
      if (!id || !knownJobIds.has(id)) return notFoundRes()
      // Dismissal really removes the job, so the list stops carrying it —
      // otherwise a dismissed card would come straight back on the refetch.
      if (init?.method === 'DELETE') {
        knownJobIds.delete(id)
        return jsonRes({ ok: true })
      }
      return jsonRes(jobSnapshot(id))
    }
    if (u.includes('/ingest/jobs')) {
      return jsonRes({ jobs: [...knownJobIds].map((id) => jobSnapshot(id)) })
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

  it('renders the enrichment options as toggle buttons that flip the run store', async () => {
    renderIn(<Ingest />)
    // Wait for the deployment-defaults seed (`ner`/`hate` both false here).
    const ner = await screen.findByRole('button', { name: 'Entities' })
    const hate = screen.getByRole('button', { name: 'Hate speech' })
    await waitFor(() => expect(ner).toHaveAttribute('aria-pressed', 'false'))
    expect(hate).toHaveAttribute('aria-pressed', 'false')

    ner.click()
    await waitFor(() => expect(ner).toHaveAttribute('aria-pressed', 'true'))
    expect(useIngestRunStore.getState().ner).toBe(true)
    // The other option is untouched — each toggle owns exactly one flag.
    expect(hate).toHaveAttribute('aria-pressed', 'false')
    expect(useIngestRunStore.getState().hate).toBe(false)
  })

  it('renders live progress for the active job and never flashes the interrupted banner', async () => {
    track('job-1')
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
    track('job-1')
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
    track('ghost')
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
    track('job-1')
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
    track('job-1')
    useIngestRunStore.getState().markJobHandled('job-1')
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

describe('Ingest — elapsed time across a reload', () => {
  it('shows the server-derived duration when the log has no client start frame', async () => {
    // After a reload only the backend's replayed job frames exist — the
    // synthetic upload `start` frame that anchors the client timer is gone,
    // so the elapsed display must fall back to the snapshot's
    // `started_at`/`finished_at` pair.
    track('job-1')
    knownJobIds.add('job-1')
    jobTimes['job-1'] = {
      started_at: '2026-08-14T10:00:00Z',
      finished_at: '2026-08-14T11:02:05Z'
    }
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
    await waitFor(() => expect(screen.getByText('1:02:05')).toBeInTheDocument())
  })
})

describe('Ingest post-ingest side effects', () => {
  it('selects the collection exactly once, even when the terminal event replays', async () => {
    // Regression test for the reconnect-replay case: the SSE stream resets a
    // job's log on `ingestion_started` and re-delivers its whole collapsed
    // history, so a naive effect keyed only on "last event is
    // ingestion_complete" would re-fire the collection-select side effect on
    // every replay.
    track('job-1')
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
    track('job-1')
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

describe('Ingest — several runs at once', () => {
  it('keeps the earlier job on screen while a second run is uploading', async () => {
    // The whole point of the list: starting a second run must not hide the
    // first. The finished job keeps its own card (and its own Dismiss) while
    // the new upload reports its own progress above.
    track('job-1')
    knownJobIds.add('job-1')
    finishedJobIds.add('job-1')
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
    await waitFor(() => expect(screen.getByText('Complete')).toBeInTheDocument())

    useIngestRunStore.setState({
      uploading: true,
      uploadEvents: [
        { event: 'start', data: { collection: 'other', files: ['b.txt'] }, receivedAt: Date.now() },
        { event: 'upload_progress', data: { filename: 'b.txt', bytes_written: 5 }, receivedAt: Date.now() }
      ]
    })

    await waitFor(() => expect(screen.getByText('Uploading')).toBeInTheDocument())
    expect(screen.getByText('Complete')).toBeInTheDocument()
  })

  it('renders one card per tracked job, newest first', async () => {
    track('job-1', 'first')
    track('job-2', 'second')
    knownJobIds.add('job-1')
    knownJobIds.add('job-2')
    const { appendEvent } = useIngestJobsStore.getState()
    for (const id of ['job-1', 'job-2']) {
      appendEvent(id, {
        event: 'ingestion_started',
        data: { job_id: id, collection: id === 'job-1' ? 'first' : 'second' },
        receivedAt: Date.now()
      })
    }

    renderIn(<Ingest />)
    await waitFor(() => expect(screen.getByText('second')).toBeInTheDocument())
    expect(screen.getByText('first')).toBeInTheDocument()
    const cards = screen.getAllByRole('status')
    // `trackJob` prepends, so the most recently started run leads.
    expect(cards[0]).toHaveTextContent('second')
  })

  it('shows a job the server lists but this browser never queued', async () => {
    // Queued from another tab, or before this page was loaded: only the
    // server list knows about it.
    knownJobIds.add('job-9')
    jobCollections['job-9'] = 'from-another-tab'

    renderIn(<Ingest />)
    await waitFor(() => expect(screen.getByText('from-another-tab')).toBeInTheDocument())
  })

  it('queues one run while another is processing, each on its own card', async () => {
    track('job-1', 'running-one')
    track('job-2', 'waiting-one')
    knownJobIds.add('job-1')
    knownJobIds.add('job-2')
    queuedJobIds.add('job-2')
    jobCollections['job-1'] = 'running-one'
    jobCollections['job-2'] = 'waiting-one'
    useIngestJobsStore.getState().appendEvent('job-1', {
      event: 'ingestion_progress',
      data: { job_id: 'job-1', message: 'Extracting entities: 3/9 chunks processed' },
      receivedAt: Date.now()
    })

    renderIn(<Ingest />)
    await waitFor(() =>
      expect(
        screen.getByText(
          'Waiting for a worker slot — this run will start as soon as the current ingest finishes.'
        )
      ).toBeInTheDocument()
    )
    expect(screen.getByText(/3\s*\/\s*9/)).toBeInTheDocument()
  })

  it('dismisses one finished job without touching the other', async () => {
    track('job-1', 'first')
    track('job-2', 'second')
    knownJobIds.add('job-1')
    knownJobIds.add('job-2')
    jobCollections['job-1'] = 'first'
    jobCollections['job-2'] = 'second'
    const { appendEvent } = useIngestJobsStore.getState()
    appendEvent('job-1', {
      event: 'ingestion_complete',
      data: { job_id: 'job-1', collection: 'first' },
      receivedAt: Date.now()
    })
    appendEvent('job-2', {
      event: 'ingestion_progress',
      data: { job_id: 'job-2', message: 'Extracting entities: 1/9 chunks processed' },
      receivedAt: Date.now()
    })

    renderIn(<Ingest />)
    const dismiss = await screen.findByRole('button', { name: /dismiss/i })
    dismiss.click()

    await waitFor(() =>
      expect(useIngestRunStore.getState().trackedJobs.map((j) => j.job_id)).toEqual(['job-2'])
    )
    expect(screen.getByText('second')).toBeInTheDocument()
  })

  it('clears every finished job at once and leaves the running one', async () => {
    track('job-1', 'first')
    track('job-2', 'second')
    track('job-3', 'third')
    for (const id of ['job-1', 'job-2', 'job-3']) knownJobIds.add(id)
    const { appendEvent } = useIngestJobsStore.getState()
    appendEvent('job-1', {
      event: 'ingestion_complete',
      data: { job_id: 'job-1', collection: 'first' },
      receivedAt: Date.now()
    })
    appendEvent('job-2', {
      event: 'error',
      data: { job_id: 'job-2', code: 'ingestion_failed' },
      receivedAt: Date.now()
    })
    appendEvent('job-3', {
      event: 'ingestion_progress',
      data: { job_id: 'job-3', message: 'Extracting entities: 1/9 chunks processed' },
      receivedAt: Date.now()
    })

    renderIn(<Ingest />)
    const clear = await screen.findByRole('button', { name: /clear finished \(2\)/i })
    clear.click()

    await waitFor(() =>
      expect(useIngestRunStore.getState().trackedJobs.map((j) => j.job_id)).toEqual(['job-3'])
    )
    // The running job is untouched, and its card stays.
    expect(knownJobIds.has('job-3')).toBe(true)
    expect(screen.queryByRole('button', { name: /clear finished/i })).not.toBeInTheDocument()
  })
})

describe('Ingest — interrupted run', () => {
  it('re-queues directly (finalize only) instead of re-uploading', async () => {
    track('ghost')
    useIngestRunStore.getState().setCollection('mydocs')
    renderIn(<Ingest />)

    // Let the deployment-defaults seed settle first (it writes `ner`/`hate`
    // on mount), then make an explicit choice — mirrors a user pressing a
    // toggle before hitting "Run again", and avoids racing the seed effect.
    const nerToggle = await screen.findByRole('button', { name: 'Entities' })
    await waitFor(() => expect(nerToggle).toHaveAttribute('aria-pressed', 'false'))
    useIngestRunStore.getState().setNer(true)

    const rerun = await screen.findByRole('button', { name: /run again|erneut/i })
    rerun.click()

    await waitFor(() => expect(callsMatching('/ingest/finalize')).toBe(1))
    expect(callsMatching('/ingest/upload')).toBe(0)
    await waitFor(() =>
      expect(useIngestRunStore.getState().trackedJobs[0].job_id).toBe('job-2')
    )

    const finalizeCall = fetchMock.mock.calls.find((c) => String(c[0]).includes('/ingest/finalize'))!
    const body = JSON.parse((finalizeCall[1] as RequestInit).body as string)
    expect(body).toEqual({ collection: 'mydocs', ner: true, hate_speech: false })
  })

  it('re-runs against the job\'s captured collection, not an edited live form field', async () => {
    // Regression test: `run.collection` is the live, user-editable form
    // field — if the user changes it between the interruption and clicking
    // "Run again", the button must still finalize the collection the
    // interrupted job actually targeted (`activeJobCollection`, captured at
    // queue time), not whatever the field currently holds.
    track('ghost', 'original-mydocs')
    useIngestRunStore.getState().setCollection('edited-after-interruption')
    renderIn(<Ingest />)

    const rerun = await screen.findByRole('button', { name: /run again|erneut/i })
    rerun.click()

    await waitFor(() => expect(callsMatching('/ingest/finalize')).toBe(1))
    const finalizeCall = fetchMock.mock.calls.find((c) => String(c[0]).includes('/ingest/finalize'))!
    const body = JSON.parse((finalizeCall[1] as RequestInit).body as string)
    expect(body.collection).toBe('original-mydocs')
  })

  it('lets the user dismiss a permanently-interrupted (ghost) job', async () => {
    track('ghost')
    renderIn(<Ingest />)

    const dismiss = await screen.findByRole('button', { name: /dismiss/i })
    dismiss.click()

    await waitFor(() => expect(useIngestRunStore.getState().trackedJobs).toEqual([]))
    expect(screen.queryByText(/interrupted/i)).not.toBeInTheDocument()
  })

  it('clears the queued notice once the job starts running', async () => {
    // The queued snapshot is served from a cached query with a long
    // `staleTime`; nothing invalidates it when the job leaves the queue, so
    // without an active poll the notice outlives the condition it describes.
    track('job-1')
    knownJobIds.add('job-1')
    queuedJobIds.add('job-1')

    renderIn(<Ingest />)
    const notice =
      'Waiting for a worker slot — this run will start as soon as the current ingest finishes.'
    await waitFor(() => expect(screen.getByText(notice)).toBeInTheDocument())

    // A worker slot frees up: the server now reports the job as running.
    queuedJobIds.delete('job-1')

    await waitFor(() => expect(screen.queryByText(notice)).not.toBeInTheDocument(), {
      timeout: 6000
    })
  }, 10000)
})
