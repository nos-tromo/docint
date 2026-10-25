import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { TopicsPanel } from './TopicsPanel'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import { useUiStore } from '@/stores/ui'
import { LanguageContext } from '@/i18n/LanguageContext'
import type { IngestEvent } from '@/api/types'
import type { TopicSummary, TopicsOverview } from '@/api/topics'

const COLLECTION = 'projekt-alpha'
const RUN_ID = '20260102-030405-a1b2c3'

function topic(overrides: Partial<TopicSummary> & Pick<TopicSummary, 'id' | 'label'>): TopicSummary {
  return {
    description: '',
    keywords: [],
    size: 10,
    share: 0.1,
    origin: 'build',
    added_at: '2026-01-02T03:04:05+00:00',
    first_seen: null,
    last_seen: null,
    ...overrides
  }
}

const TOPICS: TopicSummary[] = [
  topic({
    id: 2,
    label: 'Orbits and comets',
    keywords: [
      ['orbit', 0.4],
      ['comet', 0.3],
      ['telescope', 0.2],
      ['planet', 0.1],
      ['star', 0.05],
      ['nebula', 0.01]
    ],
    size: 60,
    share: 0.5,
    first_seen: '2024-03-03T10:00:00+00:00',
    last_seen: '2025-09-09T10:00:00+00:00'
  }),
  topic({
    id: 0,
    label: 'Bread, pastry and ovens',
    keywords: [
      ['dough', 0.5],
      ['oven', 0.2]
    ],
    size: 30,
    share: 0.25
  }),
  topic({ id: 5, label: 'Knots and sails', keywords: [['knot', 0.3]], size: 10, share: 0.0833, origin: 'discover' })
]

const OVERVIEW: TopicsOverview = {
  run_id: RUN_ID,
  created_at: '2026-01-02T03:04:05+00:00',
  updated_at: '2026-01-02T03:04:05+00:00',
  embed_model: 'embed-small',
  embed_model_changed: false,
  coverage: { total: 120, assigned: 100, unassigned: 20, pending: 5 },
  topics: TOPICS
}

const IDLE_COVERAGE: TopicsOverview = { ...OVERVIEW, coverage: { total: 100, assigned: 100, unassigned: 0, pending: 0 } }

let client: QueryClient
let clock = 0

/**
 * Stub the API the panel reads: the config flag, the overview and the queueing route.
 *
 * @param options.overview - The overview served, or null for a collection without a run.
 * @param options.post - Answers the queueing POST; a 202 with `job-new` when omitted.
 * @returns The fetch mock.
 */
function stubApi(
  options: { overview?: TopicsOverview | null; post?: () => Promise<{ status: number; body: unknown }> } = {}
) {
  const { overview = OVERVIEW, post } = options
  const respond = (body: unknown, status = 200) => ({
    ok: status < 400,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body)
  })
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), 'http://localhost')
    if (url.pathname.endsWith('/config')) return respond({ topics_enabled: true })
    if (url.pathname.endsWith('/topics')) {
      if (init?.method === 'POST') {
        const answer = post ? await post() : { status: 202, body: { job_id: 'job-new' } }
        return respond(answer.body, answer.status)
      }
      return overview ? respond(overview) : respond(null, 204)
    }
    if (url.pathname.endsWith('/evidence')) return respond({ topic: 0, sources: [] })
    if (url.pathname.endsWith('/documents')) return respond({ documents: [], more: false })
    if (url.pathname.endsWith('/chunks')) return respond({ items: [], next_cursor: null })
    return respond({})
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

/** The queueing POSTs the stub saw, as `{path, body}`. */
function posts(fetchMock: ReturnType<typeof stubApi>) {
  return fetchMock.mock.calls
    .filter(([, init]) => init?.method === 'POST')
    .map(([input, init]) => ({
      path: new URL(String(input), 'http://localhost').pathname,
      body: JSON.parse(String(init?.body))
    }))
}

/** Deliver one frame as the job stream does: tagged with its job and stamped on arrival. */
function deliver(jobId: string, event: IngestEvent['event'], data: Record<string, unknown> = {}) {
  clock += 1
  act(() =>
    useIngestJobsStore.getState().appendEvent(jobId, { event, data: { job_id: jobId, ...data }, receivedAt: clock })
  )
}

function renderPanel(lang: 'en' | 'de' = 'en', onAsk = vi.fn()) {
  return render(
    <QueryClientProvider client={client}>
      <LanguageContext value={lang}>
        <TopicsPanel onAsk={onAsk} />
      </LanguageContext>
    </QueryClientProvider>
  )
}

const header = {
  rebuild: () => screen.getByRole('button', { name: 'Rebuild topics' }),
  discover: () => screen.getByRole('button', { name: 'Find new topics' }),
  align: () => screen.getByRole('button', { name: 'Align pending chunks' }),
  download: () => screen.getByRole('button', { name: 'Download CSV' })
}

beforeEach(() => {
  clock = 0
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  useIngestJobsStore.getState().clear()
  useUiStore.setState({ selectedCollection: COLLECTION, selectedOwner: null })
  vi.stubGlobal('confirm', vi.fn(() => true))
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('TopicsPanel', () => {
  it('shows the empty state with a build button', async () => {
    const fetchMock = stubApi({ overview: null })
    renderPanel()

    expect(await screen.findByText('No topics for this collection yet.')).toBeInTheDocument()
    // Nothing to rebuild, extend or download before a first run.
    expect(screen.queryByRole('button', { name: 'Rebuild topics' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Download CSV' })).not.toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Build topics' }))

    await waitFor(() =>
      expect(posts(fetchMock)).toEqual([{ path: `/collections/${COLLECTION}/topics`, body: { action: 'build' } }])
    )
    // A first build has nothing to overwrite, so it asks nothing.
    expect(window.confirm).not.toHaveBeenCalled()
  })

  it('asks for a collection before anything else', () => {
    stubApi()
    useUiStore.setState({ selectedCollection: null })
    renderPanel()

    expect(screen.getByText('Select a collection to see its topics.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Build topics' })).not.toBeInTheDocument()
  })

  it('lists topics with coverage and keywords', async () => {
    stubApi()
    renderPanel()

    expect(
      await screen.findByText('3 topics · 100 of 120 chunks assigned · 20 unassigned · 5 pending')
    ).toBeInTheDocument()
    const rows = screen.getAllByTestId('topic-row')
    expect(rows).toHaveLength(3)

    const first = within(rows[0])
    expect(first.getByText('T2')).toHaveClass('font-mono')
    expect(first.getByText('Orbits and comets')).toHaveClass('font-semibold')
    // `share` is a fraction on the wire and a percentage on screen.
    expect(first.getByText('60 chunks · 50.0%')).toBeInTheDocument()
    // The first five keywords, and not the sixth.
    for (const word of ['orbit', 'comet', 'telescope', 'planet', 'star']) {
      expect(first.getByText(word)).toBeInTheDocument()
    }
    expect(first.queryByText('nebula')).not.toBeInTheDocument()
    expect(first.getByText('Mar 3, 2024 – Sep 9, 2025')).toBeInTheDocument()
    expect(first.queryByText('New')).not.toBeInTheDocument()

    // A topic the last find-new run added says so.
    expect(within(rows[2]).getByText('New')).toBeInTheDocument()
    expect(within(rows[2]).getByText('10 chunks · 8.3%')).toBeInTheDocument()
  })

  it('lists them in German with German numbers', async () => {
    stubApi()
    renderPanel('de')

    expect(
      await screen.findByText('3 Themen · 100 von 120 Chunks zugeordnet · 20 nicht zugeordnet · 5 ausstehend')
    ).toBeInTheDocument()
    expect(screen.getByText('60 Chunks · 50,0 %')).toBeInTheDocument()
    expect(screen.getByText('Neu')).toBeInTheDocument()
  })

  it('orders the header actions rebuild, find new, align, download', async () => {
    stubApi()
    renderPanel()
    await screen.findByText(/3 topics/)

    const toolbar = header.rebuild().parentElement as HTMLElement
    expect(within(toolbar).getAllByRole('button').map((b) => b.getAttribute('aria-label'))).toEqual([
      'Rebuild topics',
      'Find new topics',
      'Align pending chunks',
      'Download CSV'
    ])
  })

  it('asks before a rebuild', async () => {
    const confirm = vi.fn(() => false)
    vi.stubGlobal('confirm', confirm)
    const fetchMock = stubApi()
    renderPanel()
    await screen.findByText(/3 topics/)

    await userEvent.click(header.rebuild())

    expect(confirm).toHaveBeenCalledWith('Rebuild the topics? Chat filters on the current topics stop matching.')
    expect(posts(fetchMock)).toEqual([])

    confirm.mockReturnValue(true)
    await userEvent.click(header.rebuild())

    await waitFor(() => expect(posts(fetchMock)).toEqual([expect.objectContaining({ body: { action: 'build' } })]))
  })

  it('disables find-new and align when nothing is unassigned or pending', async () => {
    stubApi({ overview: IDLE_COVERAGE })
    renderPanel()
    await screen.findByText(/3 topics/)

    expect(header.discover()).toBeDisabled()
    expect(header.align()).toBeDisabled()
    expect(header.rebuild()).toBeEnabled()
    expect(header.download()).toBeEnabled()
  })

  it.each([
    ['Find new topics', 'discover'],
    ['Align pending chunks', 'align']
  ])('starts %s without asking', async (name, action) => {
    const fetchMock = stubApi()
    renderPanel()
    await screen.findByText(/3 topics/)

    await userEvent.click(screen.getByRole('button', { name }))

    await waitFor(() => expect(posts(fetchMock)).toEqual([expect.objectContaining({ body: { action } })]))
    expect(window.confirm).not.toHaveBeenCalled()
  })

  it('shows the running stage and localized warnings', async () => {
    stubApi()
    deliver('job-1', 'topics_started', { collection: COLLECTION, action: 'build' })
    deliver('job-1', 'topics_progress', { stage: 'assigning', done: 30, total: 120, message: 'Assigning topics: 30/120' })
    deliver('job-1', 'warning', { code: 'none_found', message: 'Backend prose that is not for display.' })
    deliver('job-1', 'warning', { code: 'never_heard_of_it', message: 'More backend prose.' })
    renderPanel()
    await screen.findByText(/3 topics/)

    expect(screen.getByText('Assigning chunks: 30 of 120')).toBeInTheDocument()
    const bar = screen.getByTestId('topics-progress')
    expect(bar).toHaveAttribute('aria-valuenow', '30')
    expect(bar).toHaveAttribute('aria-valuemax', '120')
    expect(screen.getByText('The fit found no topics; the collection keeps its current topics.')).toBeInTheDocument()
    expect(screen.getByText('The topics run reported a problem.')).toBeInTheDocument()
    expect(screen.queryByText(/Backend prose/)).not.toBeInTheDocument()
    // Every action waits for the run; the running one says so.
    expect(header.rebuild()).toBeDisabled()
    expect(header.rebuild()).toHaveAttribute('aria-busy', 'true')
    expect(header.discover()).toBeDisabled()
    expect(header.align()).toBeDisabled()

    // A stage that cannot count draws no bar.
    deliver('job-1', 'topics_progress', { stage: 'fitting', done: 40, total: 0, message: 'Fitting topics: 40 s' })
    expect(screen.getByText('Fitting topics: 40 s')).toBeInTheDocument()
    expect(screen.queryByTestId('topics-progress')).not.toBeInTheDocument()
  })

  it('says a stage it has no words for is under way, not which', async () => {
    stubApi()
    deliver('job-1', 'topics_started', { collection: COLLECTION, action: 'build' })
    deliver('job-1', 'topics_progress', { stage: 'reticulating', done: 2, total: 9, message: 'Backend prose.' })
    renderPanel()

    expect(await screen.findByText('Working on the topics…')).toBeInTheDocument()
    expect(screen.queryByText(/Backend prose/)).not.toBeInTheDocument()
    expect(screen.queryByText('Starting the topics run…')).not.toBeInTheDocument()
  })

  it('words the stages and warnings in German', async () => {
    stubApi()
    deliver('job-1', 'topics_started', { collection: COLLECTION, action: 'discover' })
    deliver('job-1', 'topics_progress', { stage: 'sampling', done: 3, total: 9, message: 'x' })
    deliver('job-1', 'warning', { code: 'none_new', message: 'x' })
    renderPanel('de')

    expect(await screen.findByText('Stichprobe wird gelesen: 3 von 9 Chunks')).toBeInTheDocument()
    expect(
      screen.getByText('Es wurden keine neuen Themen gefunden; die nicht zugeordneten Chunks bleiben nicht zugeordnet.')
    ).toBeInTheDocument()
  })

  it('reports its own run failing as a danger banner with its reason', async () => {
    stubApi()
    renderPanel()
    await screen.findByText(/3 topics/)
    await userEvent.click(header.rebuild())
    await waitFor(() => expect(header.rebuild()).toHaveAttribute('aria-busy', 'true'))
    deliver('job-new', 'topics_started', { collection: COLLECTION, action: 'build' })
    deliver('job-new', 'topics_progress', { stage: 'fitting', done: 5, total: 0, message: 'x' })
    expect(await screen.findByText('Fitting topics: 5 s')).toBeInTheDocument()

    deliver('job-new', 'warning', { code: 'topics_unreachable', message: 'x' })
    deliver('job-new', 'error', { code: 'topics_failed', message: 'x' })

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('The topics run failed.')
    expect(alert).toHaveTextContent('The topics service could not be reached.')
    expect(screen.queryByText('Fitting topics: 5 s')).not.toBeInTheDocument()
    expect(header.rebuild()).toBeEnabled()
  })

  it('shows how a run it did not start is going, but not how it ended', async () => {
    stubApi()
    deliver('job-1', 'topics_started', { collection: COLLECTION, action: 'build' })
    deliver('job-1', 'topics_progress', { stage: 'fitting', done: 5, total: 0, message: 'x' })
    deliver('job-1', 'warning', { code: 'topics_busy', message: 'x' })
    renderPanel()
    await screen.findByText(/3 topics/)
    expect(screen.getByText('Fitting topics: 5 s')).toBeInTheDocument()
    expect(screen.getByText('The topics service stayed busy; try again later.')).toBeInTheDocument()

    deliver('job-1', 'error', { code: 'topics_failed', message: 'x' })

    await waitFor(() => expect(screen.queryByText('Fitting topics: 5 s')).not.toBeInTheDocument())
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.queryByText('The topics service stayed busy; try again later.')).not.toBeInTheDocument()
    expect(header.rebuild()).toBeEnabled()
  })

  it('shows nothing of a finished run the stream replays frame by frame', async () => {
    // A reconnect replays a finished job's history as a burst, so it passes
    // through "running" on its way to finished.
    stubApi()
    renderPanel()
    await screen.findByText(/3 topics/)

    deliver('job-old', 'topics_started', { collection: COLLECTION, action: 'discover' })
    deliver('job-old', 'warning', { code: 'too_few_unassigned', message: 'x' })
    deliver('job-old', 'topics_completed', { collection: COLLECTION, empty: true })

    await waitFor(() => expect(screen.queryByText('Starting the topics run…')).not.toBeInTheDocument())
    expect(screen.queryByText('Too few unassigned chunks to look for new topics.')).not.toBeInTheDocument()
    expect(header.discover()).toBeEnabled()
  })

  it('shows nothing of a finished job from an earlier run', async () => {
    stubApi()
    deliver('job-old', 'topics_started', { collection: COLLECTION, action: 'discover' })
    deliver('job-old', 'warning', { code: 'too_few_unassigned', message: 'x' })
    deliver('job-old', 'topics_completed', { collection: COLLECTION, empty: true })
    deliver('job-older', 'topics_started', { collection: 'another-collection', action: 'build' })
    deliver('job-older', 'error', { code: 'topics_failed' })
    renderPanel()
    await screen.findByText(/3 topics/)

    expect(screen.queryByText('Too few unassigned chunks to look for new topics.')).not.toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.queryByText('Starting the topics run…')).not.toBeInTheDocument()
    expect(header.rebuild()).toBeEnabled()
    expect(header.discover()).toBeEnabled()
    expect(header.align()).toBeEnabled()
  })

  it("shows the starting stage from the click until the job's first frame", async () => {
    let answer!: () => void
    const answered = new Promise<void>((resolve) => {
      answer = resolve
    })
    stubApi({
      post: async () => {
        await answered
        return { status: 202, body: { job_id: 'job-new' } }
      }
    })
    // An earlier run's refusal must not reappear under the new one.
    deliver('job-old', 'topics_started', { collection: COLLECTION, action: 'align' })
    deliver('job-old', 'warning', { code: 'no_run', message: 'x' })
    deliver('job-old', 'topics_completed', { collection: COLLECTION, empty: true })
    renderPanel()
    await screen.findByText(/3 topics/)

    await userEvent.click(header.align())

    // The POST is still out.
    expect(screen.getByText('Starting the topics run…')).toBeInTheDocument()
    expect(header.align()).toHaveAttribute('aria-busy', 'true')
    expect(header.rebuild()).toBeDisabled()
    expect(header.discover()).toBeDisabled()
    expect(screen.queryByText('Build the topics first.')).not.toBeInTheDocument()

    // Queued, but no frame yet: the job store still names the earlier job.
    await act(async () => {
      answer()
      // The mutation reports its answer on a timer; let it land before reading the screen.
      await new Promise((resolve) => setTimeout(resolve, 20))
    })
    expect(screen.getByText('Starting the topics run…')).toBeInTheDocument()
    expect(header.rebuild()).toBeDisabled()
    expect(screen.queryByText('Build the topics first.')).not.toBeInTheDocument()

    deliver('job-new', 'topics_started', { collection: COLLECTION, action: 'align' })
    expect(screen.getByText('Starting the topics run…')).toBeInTheDocument()
    deliver('job-new', 'topics_progress', { stage: 'aligning', done: 1, total: 5, message: 'x' })
    expect(screen.getByText('Aligning pending chunks: 1 of 5')).toBeInTheDocument()

    deliver('job-new', 'warning', { code: 'none_new', message: 'x' })
    deliver('job-new', 'topics_completed', { collection: COLLECTION, empty: false })

    // Its own outcome stays on screen once it ends.
    expect(
      await screen.findByText('No new topics were found; the unassigned chunks stay unassigned.')
    ).toBeInTheDocument()
    expect(screen.queryByText(/Aligning pending chunks:/)).not.toBeInTheDocument()
    await waitFor(() => expect(header.rebuild()).toBeEnabled())
  })

  it("shows its own run's outcome even when the stream reports it before the queueing answer", async () => {
    let answer!: () => void
    const answered = new Promise<void>((resolve) => {
      answer = resolve
    })
    stubApi({
      overview: null,
      post: async () => {
        await answered
        return { status: 202, body: { job_id: 'job-new' } }
      }
    })
    renderPanel()
    await userEvent.click(await screen.findByRole('button', { name: 'Build topics' }))

    // A refusal can start and end before the 202 is read.
    deliver('job-new', 'topics_started', { collection: COLLECTION, action: 'build' })
    deliver('job-new', 'warning', { code: 'too_few_chunks', message: 'x' })
    deliver('job-new', 'topics_completed', { collection: COLLECTION, empty: true })
    await act(async () => answer())

    expect(
      await screen.findByText('The collection has too few chunks with enough text to build topics.')
    ).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Build topics' })).toBeEnabled()
  })

  it('says a start that could not be queued failed, in catalog words', async () => {
    stubApi({ post: async () => ({ status: 500, body: { detail: 'Backend prose that is not for display.' } }) })
    renderPanel()
    await screen.findByText(/3 topics/)

    await userEvent.click(header.discover())

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('Something went wrong (500). Please try again or contact support.')
    expect(alert).not.toHaveTextContent('Backend prose')
    expect(screen.queryByText('Starting the topics run…')).not.toBeInTheDocument()
    expect(header.discover()).toBeEnabled()
  })

  it('drops a start made on another collection', async () => {
    stubApi({ post: () => new Promise(() => {}) })
    renderPanel()
    await screen.findByText(/3 topics/)
    await userEvent.click(header.align())
    expect(screen.getByText('Starting the topics run…')).toBeInTheDocument()

    act(() => useUiStore.getState().setSelectedCollection('projekt-beta'))

    await waitFor(() => expect(screen.queryByText('Starting the topics run…')).not.toBeInTheDocument())
  })

  it('downloads the topics as CSV', async () => {
    let blob: Blob | null = null
    let filename: string | null = null
    vi.spyOn(URL, 'createObjectURL').mockImplementation((value) => {
      blob = value as Blob
      return 'blob:topics'
    })
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      filename = this.download
    })
    stubApi()
    renderPanel()
    await screen.findByText(/3 topics/)

    await userEvent.click(header.download())

    expect(filename).toBe(`${COLLECTION}-topics.csv`)
    expect(blob).not.toBeNull()
    expect((await blob!.text()).split('\n')).toEqual([
      'id,label,size,share,keywords,first_seen,last_seen,origin',
      '2,Orbits and comets,60,0.5,orbit; comet; telescope; planet; star; nebula,2024-03-03T10:00:00+00:00,2025-09-09T10:00:00+00:00,build',
      '0,"Bread, pastry and ovens",30,0.25,dough; oven,,,build',
      '5,Knots and sails,10,0.0833,knot,,,discover'
    ])
  })

  it('flags a changed embedding model', async () => {
    stubApi({ overview: { ...OVERVIEW, embed_model_changed: true } })
    renderPanel()

    const banner = await screen.findByRole('status')
    expect(banner).toHaveTextContent(
      'This collection is now embedded with a different model than its topics were built with.'
    )
  })

  it('raises no flag while the embedding model is unchanged', async () => {
    stubApi()
    renderPanel()
    await screen.findByText(/3 topics/)

    expect(screen.queryByText(/embedded with a different model/)).not.toBeInTheDocument()
  })

  it('opens one topic at a time and hands Ask in chat the topic and its run', async () => {
    const onAsk = vi.fn()
    stubApi()
    renderPanel('en', onAsk)
    await screen.findByText(/3 topics/)

    const show = screen.getByRole('button', { name: 'Show Orbits and comets' })
    await userEvent.click(show)
    expect(show).toHaveAttribute('aria-expanded', 'true')
    await userEvent.click(screen.getByRole('button', { name: 'Show Knots and sails' }))
    expect(screen.getByRole('button', { name: 'Show Orbits and comets' })).toHaveAttribute('aria-expanded', 'false')
    expect(screen.getByRole('button', { name: 'Hide Knots and sails' })).toHaveAttribute('aria-expanded', 'true')
    await userEvent.click(screen.getByRole('button', { name: 'Hide Knots and sails' }))
    expect(screen.getByRole('button', { name: 'Show Knots and sails' })).toHaveAttribute('aria-expanded', 'false')

    await userEvent.click(within(screen.getAllByTestId('topic-row')[1]).getByRole('button', { name: 'Ask in chat' }))
    expect(onAsk).toHaveBeenCalledWith(TOPICS[1], RUN_ID)
  })
})
