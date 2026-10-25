import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { TopicFocusChip } from './TopicFocusChip'
import type { TopicsOverview } from '@/api/topics'
import type { IngestEvent } from '@/api/types'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import { useSearchUiStore } from '@/stores/searchUi'
import { useTopicFocusStore } from '@/stores/topicFocus'
import { useUiStore } from '@/stores/ui'
import { LanguageContext } from '@/i18n/LanguageContext'

const COLLECTION = 'projekt-alpha'
const RUN = '20260102-030405-a1b2c3'
const EARLIER_RUN = '20251201-101010-0f0f0f'
const REBUILT_RUN = '20260103-090000-d4e5f6'

const OVERVIEW: TopicsOverview = {
  run_id: RUN,
  created_at: '2026-01-02T03:04:05+00:00',
  updated_at: '2026-01-02T03:04:05+00:00',
  embed_model: 'embed-small',
  embed_model_changed: false,
  coverage: { total: 120, assigned: 100, unassigned: 20, pending: 0 },
  topics: [
    {
      id: 3,
      label: 'Harbour cranes',
      description: '',
      keywords: [['crane', 0.4]],
      size: 60,
      share: 0.5,
      origin: 'build',
      added_at: '2026-01-02T03:04:05+00:00',
      first_seen: null,
      last_seen: null
    }
  ]
}

/**
 * Stub the reads the chip makes: the topics flag and the collection's active run.
 *
 * @param overview - Answers the run read; null serves a collection without a run (204).
 * @returns The fetch mock.
 */
function stubApi(overview: () => Promise<TopicsOverview | null> = async () => OVERVIEW) {
  const respond = (body: unknown, status = 200) => ({
    ok: status < 400,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body)
  })
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input), 'http://localhost')
    if (url.pathname.endsWith('/config')) return respond({ topics_enabled: true })
    if (url.pathname.endsWith('/topics')) {
      const body = await overview()
      return body ? respond(body) : respond(null, 204)
    }
    return respond({})
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

function renderChip(lang: 'en' | 'de' = 'en') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <LanguageContext value={lang}>
        <TopicFocusChip />
      </LanguageContext>
    </QueryClientProvider>
  )
}

const chip = () => screen.getByRole('group', { name: 'Topic filter' })

/** Deliver one job frame as the shared job stream does, stamped on arrival. */
function deliver(jobId: string, event: IngestEvent['event'], data: Record<string, unknown>, at: number) {
  act(() =>
    useIngestJobsStore.getState().appendEvent(jobId, { event, data: { job_id: jobId, ...data }, receivedAt: at })
  )
}

beforeEach(() => {
  useUiStore.setState({ selectedCollection: COLLECTION, selectedOwner: null, currentSessionId: null })
  useChatFiltersStore.getState().reset()
  useTopicFocusStore.setState({ focus: null })
  useIngestJobsStore.getState().clear()
  useSearchUiStore.setState({ scopes: {} })
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('TopicFocusChip', () => {
  it('shows the applied, stale and visual states and removes the focus', async () => {
    stubApi()
    useTopicFocusStore.setState({ focus: { collection: COLLECTION, owner: null, runId: RUN, topicId: 3, label: 'Harbour cranes' } })
    renderChip()

    expect(await screen.findByRole('group', { name: 'Topic filter' })).toHaveTextContent('T3 · Harbour cranes')
    expect(chip()).toHaveTextContent('Applied to answers and searches.')

    act(() => useChatFiltersStore.getState().setRetrievalTarget('visual'))
    expect(chip()).toHaveTextContent('Not applied to answers from images.')
    expect(chip()).not.toHaveTextContent('Applied to answers and searches.')

    // An earlier build's topic is applied under no target, and the chip says that first.
    act(() =>
      useTopicFocusStore
        .getState()
        .setFocus({ collection: COLLECTION, owner: null, runId: EARLIER_RUN, topicId: 3, label: 'Harbour cranes' })
    )
    expect(chip()).toHaveTextContent('From an earlier build; not applied.')
    expect(chip()).not.toHaveTextContent('Not applied to answers from images.')
    act(() => useChatFiltersStore.getState().setRetrievalTarget('all'))
    expect(chip()).toHaveTextContent('From an earlier build; not applied.')

    await userEvent.click(screen.getByRole('button', { name: 'Remove topic filter' }))

    expect(useTopicFocusStore.getState().focus).toBeNull()
    expect(screen.queryByRole('group', { name: 'Topic filter' })).not.toBeInTheDocument()
  })

  it('says the topic narrows only searches while chunks are selected', async () => {
    stubApi()
    useUiStore.setState({ currentSessionId: 'sess-1' })
    useSearchUiStore.setState({ scopes: { 'sess-1': { tokens: { p1: 120 }, usableTokens: 0, missing: 0 } } })
    useTopicFocusStore.setState({
      focus: { collection: COLLECTION, owner: null, runId: RUN, topicId: 3, label: 'Harbour cranes' }
    })
    renderChip()

    // A scoped answer comes from the selected chunks alone, whatever its filters say.
    const group = await screen.findByRole('group', { name: 'Topic filter' })
    expect(group).toHaveTextContent('Searches only while chunks are selected.')
    expect(group).not.toHaveAttribute('data-applied')
    // Ahead of the target, which a scope outranks too.
    act(() => useChatFiltersStore.getState().setRetrievalTarget('visual'))
    expect(chip()).toHaveTextContent('Searches only while chunks are selected.')
    // Behind an earlier build, which is applied to nothing.
    act(() =>
      useTopicFocusStore
        .getState()
        .setFocus({ collection: COLLECTION, owner: null, runId: EARLIER_RUN, topicId: 3, label: 'Harbour cranes' })
    )
    expect(chip()).toHaveTextContent('From an earlier build; not applied.')

    // The scope is the open chat's own: another chat's selection says nothing about this one.
    act(() => {
      useTopicFocusStore
        .getState()
        .setFocus({ collection: COLLECTION, owner: null, runId: RUN, topicId: 3, label: 'Harbour cranes' })
      useChatFiltersStore.getState().setRetrievalTarget('all')
      useUiStore.setState({ currentSessionId: 'sess-2' })
    })
    expect(chip()).toHaveTextContent('Applied to answers and searches.')
  })

  it('keeps the focus to the owner it was picked under', async () => {
    const fetchMock = stubApi()
    useTopicFocusStore.setState({
      focus: { collection: COLLECTION, owner: null, runId: RUN, topicId: 3, label: 'Harbour cranes' }
    })
    renderChip()
    expect(await screen.findByRole('group', { name: 'Topic filter' })).toBeInTheDocument()

    // An admin moving to another owner's collection of the same name, whose run reads the same.
    act(() => useUiStore.setState({ selectedCollection: COLLECTION, selectedOwner: 'analyst-b' }))
    const theirs = () =>
      fetchMock.mock.calls.filter(([input]) => {
        const url = new URL(String(input), 'http://localhost')
        return url.pathname.endsWith('/topics') && url.searchParams.get('owner') === 'analyst-b'
      })
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(screen.queryByRole('group', { name: 'Topic filter' })).not.toBeInTheDocument()
    // Not read at all: the focus is no topic of theirs.
    expect(theirs()).toEqual([])

    act(() => useUiStore.setState({ selectedCollection: COLLECTION, selectedOwner: null }))
    expect(await screen.findByRole('group', { name: 'Topic filter' })).toHaveTextContent(
      'Applied to answers and searches.'
    )
  })

  it('notices a rebuild that ends while it is shown', async () => {
    let overview = OVERVIEW
    stubApi(async () => overview)
    useTopicFocusStore.setState({ focus: { collection: COLLECTION, owner: null, runId: RUN, topicId: 3, label: 'Harbour cranes' } })
    renderChip()
    expect(await screen.findByRole('group', { name: 'Topic filter' })).toHaveTextContent(
      'Applied to answers and searches.'
    )

    // Only the job's last frame says the run was replaced; the run read before that is the old one.
    overview = { ...OVERVIEW, run_id: REBUILT_RUN }
    deliver('job-1', 'topics_started', { collection: COLLECTION, action: 'build' }, 0)
    deliver('job-1', 'topics_completed', {}, 5000)

    await waitFor(() => expect(chip()).toHaveTextContent('From an earlier build; not applied.'))
  })

  it('says a topic of a collection that has no run any more is not applied', async () => {
    stubApi(async () => null)
    useTopicFocusStore.setState({ focus: { collection: COLLECTION, owner: null, runId: RUN, topicId: 3, label: 'Harbour cranes' } })
    renderChip()

    expect(await screen.findByRole('group', { name: 'Topic filter' })).toHaveTextContent(
      'From an earlier build; not applied.'
    )
  })

  it('says nothing until the collection’s run is known', async () => {
    let release: (overview: TopicsOverview) => void = () => {}
    const pending = new Promise<TopicsOverview>((resolve) => {
      release = resolve
    })
    const fetchMock = stubApi(() => pending)
    useTopicFocusStore.setState({ focus: { collection: COLLECTION, owner: null, runId: RUN, topicId: 3, label: 'Harbour cranes' } })
    renderChip()

    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => String(input).endsWith('/topics'))).toBe(true))
    // Neither "applied" nor "from an earlier build" can be said of a run not read yet.
    expect(screen.queryByRole('group', { name: 'Topic filter' })).not.toBeInTheDocument()

    release(OVERVIEW)

    expect(await screen.findByRole('group', { name: 'Topic filter' })).toHaveTextContent(
      'Applied to answers and searches.'
    )
  })

  it('keeps a focus picked in another collection out of sight until that collection is back', async () => {
    stubApi()
    useTopicFocusStore.setState({ focus: { collection: 'projekt-beta', owner: null, runId: RUN, topicId: 3, label: 'Harbour cranes' } })
    renderChip()

    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(screen.queryByRole('group', { name: 'Topic filter' })).not.toBeInTheDocument()

    act(() => useUiStore.setState({ selectedCollection: 'projekt-beta' }))

    expect(await screen.findByRole('group', { name: 'Topic filter' })).toHaveTextContent(
      'Applied to answers and searches.'
    )
  })

  it('names a topic without a label by its number alone', async () => {
    stubApi()
    useTopicFocusStore.setState({ focus: { collection: COLLECTION, owner: null, runId: RUN, topicId: 3, label: '' } })
    renderChip()

    const group = await screen.findByRole('group', { name: 'Topic filter' })
    expect(group).toHaveTextContent('T3')
    expect(group).not.toHaveTextContent('·')
  })

  it('words its states in German', async () => {
    stubApi()
    useTopicFocusStore.setState({ focus: { collection: COLLECTION, owner: null, runId: RUN, topicId: 3, label: 'Hafenkräne' } })
    renderChip('de')

    const group = await screen.findByRole('group', { name: 'Themenfilter' })
    expect(group).toHaveTextContent('Auf Antworten und Suchen angewendet.')
    act(() => useChatFiltersStore.getState().setRetrievalTarget('visual'))
    expect(group).toHaveTextContent('Nicht auf Antworten aus Bildern angewendet.')
    expect(screen.getByRole('button', { name: 'Themenfilter entfernen' })).toBeInTheDocument()
  })
})
