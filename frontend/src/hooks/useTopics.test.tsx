import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import {
  useStartTopics,
  useTopicChunks,
  useTopicDocuments,
  useTopicEvidence,
  useTopicsJob,
  useTopicsOverview
} from './useTopics'
import { useDocumentsPages } from './useDocuments'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import { useUiStore } from '@/stores/ui'
import type { IngestEvent } from '@/api/types'
import type { TopicsOverview } from '@/api/topics'

const SCOPE = { collection: 'docs', owner: null }
const RUN = '20260102-030405-a1b2c3'

const OVERVIEW: TopicsOverview = {
  run_id: '20260102-030405-a1b2c3',
  created_at: '2026-01-02T03:04:05+00:00',
  updated_at: '2026-01-02T03:04:05+00:00',
  embed_model: 'embed-small',
  embed_model_changed: false,
  coverage: { total: 120, assigned: 100, unassigned: 20, pending: 0 },
  topics: []
}

let client: QueryClient
let clock = 0

function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

/** Deliver one frame as the job stream does: tagged with its job and stamped on arrival. */
function deliver(jobId: string, event: IngestEvent['event'], data: Record<string, unknown> = {}) {
  clock += 1
  act(() =>
    useIngestJobsStore.getState().appendEvent(jobId, { event, data: { job_id: jobId, ...data }, receivedAt: clock })
  )
}

const FIT_PROGRESS = { message: 'Fitting topics: 40 s', stage: 'fitting', done: 40, total: 0 }

/** Every request the stub saw, with its path split from its query. */
function reads(fetchMock: ReturnType<typeof stubApi>['fetchMock']) {
  return fetchMock.mock.calls.map(([input, init]) => {
    const url = new URL(String(input), 'http://localhost')
    return { path: url.pathname, query: url.searchParams, method: init?.method ?? 'GET', body: init?.body }
  })
}

/** Stub the endpoints the topics hooks and the documents list read. */
function stubApi(options: { topicsEnabled?: boolean; overview?: TopicsOverview | null } = {}) {
  const { topicsEnabled = true, overview = null } = options
  const respond = (body: unknown, status = 200) => ({
    ok: status < 400,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body)
  })
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), 'http://localhost')
    if (url.pathname.endsWith('/config')) return respond({ topics_enabled: topicsEnabled })
    if (url.pathname.endsWith('/collections/documents')) return respond({ items: [], next_cursor: null })
    if (url.pathname.endsWith('/topics')) {
      if (init?.method === 'POST') return respond({ job_id: 'job-9' }, 202)
      return overview ? respond(overview) : respond(null, 204)
    }
    if (url.pathname.endsWith('/evidence')) return respond({ topic: 3, sources: [{ filename: 'a.txt' }] })
    if (url.pathname.endsWith('/documents')) {
      return respond({ documents: [{ file_hash: 'h1', filename: 'a.txt', count: 2 }], more: false })
    }
    if (url.pathname.endsWith('/chunks')) {
      return respond(
        url.searchParams.get('cursor')
          ? { items: [{ filename: 'c.txt' }], next_cursor: null }
          : { items: [{ filename: 'b.txt' }], next_cursor: 'c2' }
      )
    }
    return respond({})
  })
  vi.stubGlobal('fetch', fetchMock)
  return { fetchMock }
}

/** The overview reads the stub saw for the docs collection. */
const overviewReads = (fetchMock: ReturnType<typeof stubApi>['fetchMock']) =>
  reads(fetchMock).filter((r) => r.method === 'GET' && r.path.endsWith('/collections/docs/topics'))

/** The documents-list reads the stub saw. */
const documentReads = (fetchMock: ReturnType<typeof stubApi>['fetchMock']) =>
  reads(fetchMock).filter((r) => r.path.endsWith('/collections/documents'))

beforeEach(() => {
  clock = 0
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  useIngestJobsStore.getState().clear()
  useUiStore.setState({ selectedCollection: 'docs', selectedOwner: null })
})

afterEach(() => vi.unstubAllGlobals())

describe('useTopicsJob', () => {
  it("follows only the selected collection's topics job", () => {
    deliver('job-docs', 'topics_started', { collection: 'docs', action: 'build' })
    deliver('job-docs', 'topics_progress', FIT_PROGRESS)
    deliver('job-other', 'topics_started', { collection: 'other', action: 'align' })
    deliver('job-other', 'topics_progress', {
      message: 'Aligning topics: 3/8 chunks processed',
      stage: 'aligning',
      done: 3,
      total: 8
    })
    // A job of another kind on the same collection is not a topics job.
    deliver('job-ingest', 'ingestion_started', { collection: 'docs' })

    const { result } = renderHook(() => useTopicsJob(), { wrapper })
    expect(result.current).toEqual({
      jobId: 'job-docs',
      action: 'build',
      running: true,
      stage: 'fitting',
      done: 40,
      total: 0,
      warnings: [],
      error: null,
      finished: false,
      live: false
    })

    act(() => useUiStore.getState().setSelectedCollection('other'))
    expect(result.current).toMatchObject({ jobId: 'job-other', action: 'align', stage: 'aligning', done: 3, total: 8 })

    act(() => useUiStore.getState().setSelectedCollection('untouched'))
    expect(result.current).toMatchObject({ jobId: null, action: null, running: false, stage: null, finished: false })
  })

  it('reads fields of a type or value it does not know as absent', () => {
    // Version skew or a damaged frame must not put a token the SPA has no copy for on screen.
    deliver('job-1', 'topics_started', { collection: 'docs', action: 'reindex' })
    deliver('job-1', 'topics_progress', { stage: 7, done: 'many', total: null })
    deliver('job-1', 'warning', { code: 4, message: ['x'] })

    const { result } = renderHook(() => useTopicsJob(), { wrapper })

    expect(result.current).toMatchObject({
      action: null,
      stage: null,
      done: 0,
      total: 0,
      warnings: [{ code: '', message: '' }],
      running: true
    })
  })

  it('follows no job while no collection is selected', () => {
    deliver('job-docs', 'topics_started', { collection: 'docs', action: 'build' })
    useUiStore.setState({ selectedCollection: null })

    const { result } = renderHook(() => useTopicsJob(), { wrapper })

    expect(result.current.jobId).toBeNull()
  })

  it('moves on to a newer job on the same collection as soon as it starts', () => {
    deliver('job-1', 'topics_started', { collection: 'docs', action: 'build' })
    deliver('job-1', 'topics_completed', { collection: 'docs', empty: false })
    const { result } = renderHook(() => useTopicsJob(), { wrapper })
    expect(result.current).toMatchObject({ jobId: 'job-1', finished: true, running: false })

    deliver('job-2', 'topics_started', { collection: 'docs', action: 'discover' })

    expect(result.current).toMatchObject({ jobId: 'job-2', action: 'discover', finished: false, running: true })
  })

  it('takes the newest job by when its started frame arrived, not by the order the store holds them', () => {
    act(() => {
      const store = useIngestJobsStore.getState()
      const data = { collection: 'docs', action: 'build' }
      store.appendEvent('job-late', { event: 'topics_started', data, receivedAt: 500 })
      store.appendEvent('job-early', { event: 'topics_started', data, receivedAt: 100 })
    })

    const { result } = renderHook(() => useTopicsJob(), { wrapper })

    expect(result.current.jobId).toBe('job-late')
  })

  it.each([7, undefined])(
    'prefers the job the stream delivered last when two arrive together (receivedAt %s)',
    (receivedAt) => {
      // A reconnect replays a collection's jobs oldest first, often within one millisecond.
      act(() => {
        const store = useIngestJobsStore.getState()
        const data = { collection: 'docs', action: 'build' }
        store.appendEvent('job-old', { event: 'topics_started', data, receivedAt })
        store.appendEvent('job-new', { event: 'topics_started', data, receivedAt })
      })

      const { result } = renderHook(() => useTopicsJob(), { wrapper })

      expect(result.current.jobId).toBe('job-new')
    }
  )

  it.each(['topics_completed', 'topics_cancelled'] as const)('reports a job finished once it is %s', (terminal) => {
    deliver('job-1', 'topics_started', { collection: 'docs', action: 'build' })
    deliver('job-1', 'topics_progress', FIT_PROGRESS)
    const { result } = renderHook(() => useTopicsJob(), { wrapper })
    expect(result.current).toMatchObject({ running: true, finished: false })

    deliver('job-1', terminal, { collection: 'docs' })

    expect(result.current).toMatchObject({ jobId: 'job-1', running: false, finished: true, error: null })
  })

  it('reports a job watched as it ran as live, and one only replayed as not', () => {
    const at = (jobId: string, event: IngestEvent['event'], receivedAt: number, data: Record<string, unknown> = {}) =>
      act(() =>
        useIngestJobsStore.getState().appendEvent(jobId, { event, data: { job_id: jobId, ...data }, receivedAt })
      )
    at('job-replayed', 'topics_started', 100, { collection: 'docs', action: 'build' })
    at('job-replayed', 'topics_completed', 100)
    const { result } = renderHook(() => useTopicsJob(), { wrapper })
    expect(result.current).toMatchObject({ jobId: 'job-replayed', live: false })

    at('job-watched', 'topics_started', 200, { collection: 'docs', action: 'align' })
    expect(result.current).toMatchObject({ jobId: 'job-watched', live: false })
    at('job-watched', 'topics_completed', 1200)
    expect(result.current).toMatchObject({ jobId: 'job-watched', live: true, finished: true })
  })

  it('reports warnings and the failure code', () => {
    deliver('job-1', 'topics_started', { collection: 'docs', action: 'build' })
    deliver('job-1', 'warning', { code: 'topics_unreachable', message: 'The topics service could not be reached.' })
    deliver('job-1', 'warning', { code: 'none_found', message: 'The fit found no topics.' })
    deliver('job-1', 'error', { code: 'topics_failed', message: 'Topics run failed.', duration_ms: 1200 })

    const { result } = renderHook(() => useTopicsJob(), { wrapper })

    expect(result.current.warnings).toEqual([
      { code: 'topics_unreachable', message: 'The topics service could not be reached.' },
      { code: 'none_found', message: 'The fit found no topics.' }
    ])
    expect(result.current).toMatchObject({ error: 'topics_failed', running: false, finished: false })
  })

  it('names a failure that carries no code topics_failed', () => {
    deliver('job-1', 'topics_started', { collection: 'docs', action: 'build' })
    deliver('job-1', 'error', { message: 'Topics run failed.' })

    const { result } = renderHook(() => useTopicsJob(), { wrapper })

    expect(result.current.error).toBe('topics_failed')
  })

  it('counts a refusal as a finished job with its coded warning, not a failure', () => {
    deliver('job-1', 'topics_started', { collection: 'docs', action: 'discover' })
    deliver('job-1', 'warning', { code: 'no_run', message: 'Build topics first.' })
    deliver('job-1', 'topics_completed', { collection: 'docs', empty: true, duration_ms: 3 })

    const { result } = renderHook(() => useTopicsJob(), { wrapper })

    expect(result.current).toMatchObject({ finished: true, running: false, error: null })
    expect(result.current.warnings).toEqual([{ code: 'no_run', message: 'Build topics first.' }])
  })

  it('keeps what a too-small sample found and needed, only when both counts arrive', () => {
    deliver('job-1', 'topics_started', { collection: 'docs', action: 'build' })
    deliver('job-1', 'warning', { code: 'too_few_chunks', message: 'x', found: 5, needed: 8 })
    deliver('job-1', 'warning', { code: 'too_few_chunks', message: 'x', found: 5, needed: '8' })
    deliver('job-1', 'warning', { code: 'too_few_chunks', message: 'x', needed: 8 })
    deliver('job-1', 'topics_completed', { collection: 'docs', empty: true, duration_ms: 3 })

    const { result } = renderHook(() => useTopicsJob(), { wrapper })

    expect(result.current.warnings).toEqual([
      { code: 'too_few_chunks', message: 'x', found: 5, needed: 8 },
      { code: 'too_few_chunks', message: 'x' },
      { code: 'too_few_chunks', message: 'x' }
    ])
  })

  it('invalidates topics and documents when the job ends', async () => {
    const { fetchMock } = stubApi()
    deliver('job-1', 'topics_started', { collection: 'docs', action: 'build' })
    renderHook(() => ({ job: useTopicsJob(), overview: useTopicsOverview(), documents: useDocumentsPages() }), {
      wrapper
    })
    await waitFor(() => expect(overviewReads(fetchMock)).toHaveLength(1))
    await waitFor(() => expect(documentReads(fetchMock)).toHaveLength(1))

    // Progress alone leaves both lists as they are.
    deliver('job-1', 'topics_progress', FIT_PROGRESS)
    deliver('job-1', 'topics_completed', { collection: 'docs', empty: false, duration_ms: 9 })
    await waitFor(() => expect(overviewReads(fetchMock)).toHaveLength(2))
    await waitFor(() => expect(documentReads(fetchMock)).toHaveLength(2))

    // A reconnect replays the same job, finished again: the lists are already fresh.
    deliver('job-1', 'topics_started', { collection: 'docs', action: 'build' })
    deliver('job-1', 'topics_completed', { collection: 'docs', empty: false, duration_ms: 9 })
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(overviewReads(fetchMock)).toHaveLength(2)
    expect(documentReads(fetchMock)).toHaveLength(2)

    // The next run is a new job and refreshes them again, a failed one included.
    deliver('job-2', 'topics_started', { collection: 'docs', action: 'discover' })
    deliver('job-2', 'error', { code: 'topics_failed' })
    await waitFor(() => expect(overviewReads(fetchMock)).toHaveLength(3))
    await waitFor(() => expect(documentReads(fetchMock)).toHaveLength(3))
  })

  it('invalidates the lists under the scope it follows', async () => {
    const spy = vi.spyOn(client, 'invalidateQueries')
    deliver('job-1', 'topics_started', { collection: 'docs', action: 'build' })
    renderHook(() => useTopicsJob(), { wrapper })
    expect(spy).not.toHaveBeenCalled()

    deliver('job-1', 'topics_cancelled', { collection: 'docs' })

    expect(spy).toHaveBeenCalledTimes(2)
    expect(spy).toHaveBeenCalledWith({ queryKey: ['topics', SCOPE] })
    expect(spy).toHaveBeenCalledWith({ queryKey: ['documents-pages', SCOPE] })
  })

  it('refetches once for a job that ended while no caller was mounted', () => {
    const spy = vi.spyOn(client, 'invalidateQueries')
    deliver('job-1', 'topics_started', { collection: 'docs', action: 'build' })
    deliver('job-1', 'topics_completed', { collection: 'docs', empty: false })

    const { rerender } = renderHook(() => useTopicsJob(), { wrapper })
    expect(spy).toHaveBeenCalledTimes(2)

    rerender()
    expect(spy).toHaveBeenCalledTimes(2)
  })
})

describe('useTopicsOverview', () => {
  it('reads the overview once a collection is selected and topics are switched on', async () => {
    stubApi({ overview: OVERVIEW })

    const { result } = renderHook(() => useTopicsOverview(), { wrapper })

    await waitFor(() => expect(result.current.data).toEqual(OVERVIEW))
  })

  it('resolves null for a collection that has no run', async () => {
    stubApi({ overview: null })

    const { result } = renderHook(() => useTopicsOverview(), { wrapper })

    await waitFor(() => expect(result.current.data).toBeNull())
    expect(result.current.isSuccess).toBe(true)
  })

  it('stays idle while the deployment has topics switched off', async () => {
    const { fetchMock } = stubApi({ topicsEnabled: false })

    renderHook(() => useTopicsOverview(), { wrapper })

    await waitFor(() => expect(reads(fetchMock).some((r) => r.path.endsWith('/config'))).toBe(true))
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(overviewReads(fetchMock)).toHaveLength(0)
  })

  it('stays idle until a collection is selected', async () => {
    const { fetchMock } = stubApi()
    useUiStore.setState({ selectedCollection: null })

    renderHook(() => useTopicsOverview(), { wrapper })

    await waitFor(() => expect(reads(fetchMock).some((r) => r.path.endsWith('/config'))).toBe(true))
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(reads(fetchMock).filter((r) => r.path.endsWith('/topics'))).toHaveLength(0)
  })
})

describe('useStartTopics', () => {
  it('queues the action on the selected collection and leaves progress to the stream', async () => {
    const { fetchMock } = stubApi()
    const { result } = renderHook(() => useStartTopics(), { wrapper })

    let queued: unknown
    await act(async () => {
      queued = await result.current.mutateAsync('discover')
    })

    const [post] = reads(fetchMock).filter((r) => r.method === 'POST')
    expect(post.path).toMatch(/\/collections\/docs\/topics$/)
    expect(JSON.parse(String(post.body))).toEqual({ action: 'discover' })
    expect(queued).toEqual({ job_id: 'job-9', adopted: false })
  })

  it('refuses without a collection and sends nothing', async () => {
    const { fetchMock } = stubApi()
    useUiStore.setState({ selectedCollection: null })
    const { result } = renderHook(() => useStartTopics(), { wrapper })

    await act(async () => {
      await expect(result.current.mutateAsync('build')).rejects.toBeInstanceOf(Error)
    })

    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('topic reads', () => {
  it('read nothing until asked to', async () => {
    const { fetchMock } = stubApi()

    renderHook(() => ({ e: useTopicEvidence(RUN, 3, false), d: useTopicDocuments(RUN, 3, false), c: useTopicChunks(RUN, 3, false) }), {
      wrapper
    })

    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('read nothing while no collection is selected', async () => {
    const { fetchMock } = stubApi()
    useUiStore.setState({ selectedCollection: null })

    renderHook(() => ({ e: useTopicEvidence(RUN, 3, true), d: useTopicDocuments(RUN, 3, true), c: useTopicChunks(RUN, 3, true) }), {
      wrapper
    })

    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('all sit under the topics prefix, so one invalidation refreshes them together', async () => {
    stubApi({ overview: OVERVIEW })
    const { result } = renderHook(
      () => ({
        overview: useTopicsOverview(),
        evidence: useTopicEvidence(RUN, 3, true),
        documents: useTopicDocuments(RUN, 3, true),
        chunks: useTopicChunks(RUN, 3, true)
      }),
      { wrapper }
    )

    await waitFor(() => {
      expect(result.current.overview.isSuccess).toBe(true)
      expect(result.current.evidence.data?.sources).toEqual([{ filename: 'a.txt' }])
      expect(result.current.documents.data?.documents).toEqual([{ file_hash: 'h1', filename: 'a.txt', count: 2 }])
      expect(result.current.chunks.data?.pages[0].items).toEqual([{ filename: 'b.txt' }])
    })

    expect(
      client
        .getQueryCache()
        .findAll({ queryKey: ['topics', SCOPE] })
        .map((query) => query.queryKey)
    ).toEqual(
      expect.arrayContaining([
        ['topics', SCOPE],
        ['topics', SCOPE, RUN, 'evidence', 3],
        ['topics', SCOPE, RUN, 'documents', 3],
        ['topics', SCOPE, RUN, 'chunks', 3]
      ])
    )
  })

  it('keep one run’s reads of a topic apart from another run’s', async () => {
    // Topic ids restart at 0 with every build, so the run is part of what a read is.
    const { fetchMock } = stubApi()
    const { result } = renderHook(
      () => ({ before: useTopicEvidence('20260101-000000-aaaaaa', 3, true), after: useTopicEvidence(RUN, 3, true) }),
      { wrapper }
    )

    await waitFor(() => {
      expect(result.current.before.isSuccess).toBe(true)
      expect(result.current.after.isSuccess).toBe(true)
    })
    expect(
      reads(fetchMock).filter((r) => r.path.endsWith('/topics/3/evidence')).map((r) => r.query.get('run_id'))
    ).toEqual(['20260101-000000-aaaaaa', RUN])
    expect(
      client
        .getQueryCache()
        .findAll({ queryKey: ['topics', SCOPE] })
        .map((query) => query.queryKey)
    ).toEqual(
      expect.arrayContaining([
        ['topics', SCOPE, '20260101-000000-aaaaaa', 'evidence', 3],
        ['topics', SCOPE, RUN, 'evidence', 3]
      ])
    )
  })

  it('walk a topic’s chunks by the cursor each page hands back', async () => {
    const { fetchMock } = stubApi()
    const { result } = renderHook(() => useTopicChunks(RUN, 3, true), { wrapper })
    await waitFor(() => expect(result.current.hasNextPage).toBe(true))

    await act(async () => {
      await result.current.fetchNextPage()
    })

    await waitFor(() =>
      expect(result.current.data?.pages.map((page) => page.items[0].filename)).toEqual(['b.txt', 'c.txt'])
    )
    expect(result.current.hasNextPage).toBe(false)
    const chunkReads = reads(fetchMock).filter((r) => r.path.endsWith('/collections/docs/topics/3/chunks'))
    expect(chunkReads.map((r) => r.query.get('cursor'))).toEqual([null, 'c2'])
  })
})
