import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { TopicsPanel } from './TopicsPanel'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'
import { LanguageContext } from '@/i18n/LanguageContext'
import type { TopicSummary, TopicsOverview, TopicsTuning } from '@/api/topics'

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

const OVERVIEW: TopicsOverview = {
  run_id: RUN_ID,
  created_at: '2026-01-02T03:04:05+00:00',
  updated_at: '2026-01-02T03:04:05+00:00',
  embed_model: 'embed-small',
  embed_model_changed: false,
  leaf_count: 3,
  coverage: { total: 120, assigned: 100, unassigned: 20, pending: 0 },
  topics: [
    topic({ id: 2, members: [2], label: 'Orbits and comets', size: 60, share: 0.5 }),
    topic({ id: 0, members: [0], label: 'Bread and ovens', size: 30, share: 0.25 }),
    topic({ id: 5, members: [5], label: 'Knots and sails', size: 10, share: 0.0833 })
  ]
}

/** Topics 0 and 5 are the closest pair; the three join last. */
const TUNING: TopicsTuning = {
  run_id: RUN_ID,
  min_topic_size: 4,
  leaves: [
    { id: 0, label: 'Bread and ovens', fit_size: 25, size: 30 },
    { id: 2, label: 'Orbits and comets', fit_size: 50, size: 60 },
    { id: 5, label: 'Knots and sails', fit_size: 8, size: 10 }
  ],
  merges: [
    [0, 5, 0.82],
    [0, 2, 0.31]
  ],
  groups: [],
  group_labels: {},
  relax: { threshold: null, histogram: null, applied: 0, stale: true }
}

/** Twenty chunks the rule leaves out: one near 0.20, two near 0.30, four near 0.45, the rest far. */
const HISTOGRAM = Array.from({ length: 100 }, (_, bin) => ({ 5: 13, 20: 1, 30: 2, 45: 4 })[bin] ?? 0)

let client: QueryClient

/**
 * Stub the API: the flag, the overview, the tuning view and the queueing route.
 *
 * @param tuning - The tuning view served.
 * @returns The fetch mock.
 */
function stubApi(tuning: TopicsTuning = TUNING) {
  const respond = (body: unknown, status = 200) => ({
    ok: status < 400,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body)
  })
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), 'http://localhost')
    if (url.pathname.endsWith('/config')) return respond({ topics_enabled: true })
    if (url.pathname.endsWith('/topics/tuning')) return respond(tuning)
    if (url.pathname.endsWith('/topics')) {
      return init?.method === 'POST' ? respond({ job_id: 'job-new' }, 202) : respond(OVERVIEW)
    }
    return respond({})
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

/** The bodies of the queueing POSTs the stub saw. */
function posted(fetchMock: ReturnType<typeof stubApi>) {
  return fetchMock.mock.calls
    .filter(([, init]) => init?.method === 'POST')
    .map(([, init]) => JSON.parse(String(init?.body)))
}

async function openLab() {
  render(
    <QueryClientProvider client={client}>
      <LanguageContext value="en">
        <TopicsPanel onAsk={vi.fn()} />
      </LanguageContext>
    </QueryClientProvider>
  )
  await screen.findByText(/3 topics/)
  await userEvent.click(screen.getByRole('button', { name: 'Show fine-tuning' }))
  await screen.findByText('Fine-tuning')
}

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  useIngestJobsStore.getState().clear()
  useReportStore.setState({ activeReportId: null })
  useUiStore.setState({ selectedCollection: COLLECTION, selectedOwner: null })
  vi.stubGlobal('confirm', vi.fn(() => true))
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('TopicsLab', () => {
  it('takes the place of the topic list and reads the tuning view only once opened', async () => {
    const fetchMock = stubApi()
    await openLab()

    expect(screen.queryAllByTestId('topic-row')).toEqual([])
    expect(screen.getByRole('button', { name: 'Hide fine-tuning' })).toHaveAttribute('aria-expanded', 'true')
    const tuningReads = fetchMock.mock.calls.filter(([input]) => String(input).includes('/topics/tuning'))
    expect(tuningReads).toHaveLength(1)

    await userEvent.click(screen.getByRole('button', { name: 'Hide fine-tuning' }))
    expect(await screen.findAllByTestId('topic-row')).toHaveLength(3)
  })

  it('previews a merge without asking the server, and applies it as a reduce', async () => {
    const fetchMock = stubApi()
    await openLab()
    const reads = fetchMock.mock.calls.length

    fireEvent.change(screen.getByRole('slider', { name: 'Number of topics' }), { target: { value: '2' } })

    expect(screen.getByText('Preview of 2 topics, not applied yet.')).toBeInTheDocument()
    expect(screen.getByText('2 of 3 topics · merging from similarity 0.82')).toBeInTheDocument()
    expect(screen.getByText('from 2 topics')).toBeInTheDocument()
    expect(fetchMock.mock.calls).toHaveLength(reads)

    await userEvent.click(screen.getByRole('button', { name: 'Apply' }))

    await waitFor(() => expect(posted(fetchMock)).toEqual([{ action: 'reduce', run_id: RUN_ID, groups: [[0, 5]] }]))
  })

  it('merges and splits topics by hand, and discards the draft', async () => {
    const fetchMock = stubApi()
    await openLab()

    await userEvent.click(screen.getByRole('button', { name: /T2\s*Orbits and comets/ }))
    await userEvent.click(screen.getByRole('button', { name: /T5\s*Knots and sails/ }))
    await userEvent.click(screen.getByRole('button', { name: 'Merge selected' }))

    expect(screen.getByText('Preview of 2 topics, not applied yet.')).toBeInTheDocument()
    // The larger topic names the group until a label is written for it.
    expect(screen.getByRole('button', { name: /T2\s*Orbits and comets/ })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /T5/ })).not.toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Split' }))
    expect(screen.queryByText(/Preview of/)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled()

    fireEvent.change(screen.getByRole('slider', { name: 'Number of topics' }), { target: { value: '2' } })
    await userEvent.click(screen.getByRole('button', { name: 'Discard' }))
    expect(screen.queryByText(/Preview of/)).not.toBeInTheDocument()
    expect(posted(fetchMock)).toEqual([])
  })

  it('restores the original topics of a reduced run', async () => {
    const fetchMock = stubApi({ ...TUNING, groups: [[0, 5]], group_labels: { '0,5': 'Kitchens and harbours' } })
    await openLab()

    expect(screen.getByRole('button', { name: /T0\s*Kitchens and harbours/ })).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Restore the original topics' }))

    expect(window.confirm).toHaveBeenCalledOnce()
    await waitFor(() => expect(posted(fetchMock)).toEqual([{ action: 'reduce', run_id: RUN_ID, groups: [] }]))
  })

  it('restoring asks first and sends nothing when cancelled', async () => {
    const fetchMock = stubApi({ ...TUNING, groups: [[0, 5]], group_labels: { '0,5': 'Kitchens and harbours' } })
    await openLab()
    vi.mocked(window.confirm).mockReturnValueOnce(false)

    await userEvent.click(screen.getByRole('button', { name: 'Restore the original topics' }))

    expect(window.confirm).toHaveBeenCalledOnce()
    expect(posted(fetchMock)).toEqual([])
    expect(screen.getByRole('button', { name: /T0\s*Kitchens and harbours/ })).toBeInTheDocument()
  })

  it('counts the selection above the list', async () => {
    stubApi()
    await openLab()

    expect(screen.getByText('0 of 3 selected')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Merge selected' })).toBeDisabled()
    await userEvent.click(screen.getByRole('button', { name: /T2\s*Orbits and comets/ }))
    await userEvent.click(screen.getByRole('button', { name: /T5\s*Knots and sails/ }))

    expect(screen.getByText('2 of 3 selected')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Merge selected' })).toBeEnabled()
  })

  it('commits from one full-width bar, Apply last', async () => {
    stubApi()
    await openLab()

    const apply = screen.getByRole('button', { name: 'Apply' })
    const bar = apply.parentElement!
    expect(bar).toHaveClass('flex-wrap', '*:flex-[1_1_8rem]')
    expect(bar).toContainElement(screen.getByRole('button', { name: 'Discard' }))
    expect(bar.lastElementChild).toBe(apply)
  })

  it('restore and undo sit in their section headers', async () => {
    stubApi()
    await openLab()

    const restore = screen.getByRole('button', { name: 'Restore the original topics' })
    expect(restore.closest('section')).toContainElement(
      screen.getByRole('heading', { level: 3, name: 'Merge similar topics' })
    )
    const undo = screen.getByRole('button', { name: 'Undo placement' })
    expect(undo.closest('section')).toContainElement(
      screen.getByRole('heading', { level: 3, name: 'Unassigned chunks' })
    )
    // Icon actions: the label is the accessible name, not visible text.
    expect(restore).toHaveTextContent('')
    expect(undo).toHaveTextContent('')
  })

  it('holds the header actions too while a topics job runs', async () => {
    stubApi({
      ...TUNING,
      groups: [[0, 5]],
      group_labels: { '0,5': 'Kitchens and harbours' },
      relax: { threshold: 0.45, histogram: HISTOGRAM, applied: 4, stale: false }
    })
    await openLab()
    expect(screen.getByRole('button', { name: 'Restore the original topics' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Undo placement' })).toBeEnabled()

    act(() =>
      useIngestJobsStore.getState().appendEvent('job-1', {
        event: 'topics_started',
        data: { job_id: 'job-1', collection: COLLECTION, action: 'reduce' },
        receivedAt: 1
      })
    )

    await waitFor(() => expect(screen.getByRole('button', { name: 'Restore the original topics' })).toBeDisabled())
    expect(screen.getByRole('button', { name: 'Undo placement' })).toBeDisabled()
  })

  it('measures the unassigned chunks before offering a bar', async () => {
    const fetchMock = stubApi()
    await openLab()

    expect(screen.getByText('20 unassigned · 0 placed under a lowered bar')).toBeInTheDocument()
    expect(screen.queryByRole('slider', { name: 'Minimum similarity' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Undo placement' })).toBeDisabled()
    await userEvent.click(screen.getByRole('button', { name: 'Measure the distribution' }))

    await waitFor(() => expect(posted(fetchMock)).toEqual([{ action: 'relax', run_id: RUN_ID, threshold: null }]))
  })

  it('previews exactly what a bar places, and places them', async () => {
    const fetchMock = stubApi({
      ...TUNING,
      relax: { threshold: 0.45, histogram: HISTOGRAM, applied: 4, stale: false }
    })
    await openLab()

    expect(screen.getByText('4 chunks at similarity 0.45 or more')).toBeInTheDocument()
    // The bar applied now places nothing new.
    expect(screen.getByRole('button', { name: 'Place them' })).toBeDisabled()
    fireEvent.change(screen.getByRole('slider', { name: 'Minimum similarity' }), { target: { value: '0.3' } })
    expect(screen.getByText('6 chunks at similarity 0.30 or more')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Place them' }))

    await waitFor(() => expect(posted(fetchMock)).toEqual([{ action: 'relax', run_id: RUN_ID, threshold: 0.3 }]))
  })

  it('undoes the chunks placed under a bar', async () => {
    const fetchMock = stubApi({
      ...TUNING,
      relax: { threshold: 0.45, histogram: HISTOGRAM, applied: 4, stale: false }
    })
    await openLab()

    expect(screen.getByText('20 unassigned · 4 placed under a lowered bar')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Undo placement' }))

    await waitFor(() => expect(posted(fetchMock)).toEqual([{ action: 'relax', run_id: RUN_ID, threshold: null }]))
  })

  it('applies a relax again at its own bar once the pool has moved on', async () => {
    const fetchMock = stubApi({ ...TUNING, relax: { threshold: 0.45, histogram: HISTOGRAM, applied: 4, stale: true } })
    await openLab()

    expect(screen.getByText('The distribution was measured before the latest changes.')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Apply again' }))

    await waitFor(() => expect(posted(fetchMock)).toEqual([{ action: 'relax', run_id: RUN_ID, threshold: 0.45 }]))
  })

  it('recomputes with a chosen minimum topic size after asking', async () => {
    const fetchMock = stubApi()
    await openLab()

    expect(screen.getByText('built with 4')).toBeInTheDocument()
    const size = screen.getByRole('spinbutton', { name: 'Minimum topic size' })
    await userEvent.type(size, '1')
    expect(screen.getByRole('button', { name: 'Recompute' })).toBeDisabled()
    await userEvent.clear(size)
    await userEvent.type(size, '12')
    await userEvent.click(screen.getByRole('button', { name: 'Recompute' }))

    expect(window.confirm).toHaveBeenCalledOnce()
    await waitFor(() => expect(posted(fetchMock)).toEqual([{ action: 'build', min_topic_size: 12 }]))
  })

  it('holds every control while a topics job runs', async () => {
    stubApi()
    await openLab()

    act(() =>
      useIngestJobsStore.getState().appendEvent('job-1', {
        event: 'topics_started',
        data: { job_id: 'job-1', collection: COLLECTION, action: 'reduce' },
        receivedAt: 1
      })
    )

    await waitFor(() => expect(screen.getByRole('slider', { name: 'Number of topics' })).toBeDisabled())
    expect(screen.getByRole('button', { name: 'Measure the distribution' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Recompute' })).toBeDisabled()
  })
})
