import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Inspector } from './Inspector'
import type { TopicsOverview } from '@/api/topics'
import type { DocumentRecord, IngestEvent } from '@/api/types'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import { useUiStore } from '@/stores/ui'

const COLLECTION = 'projekt-alpha'
const RUN = '20260102-030405-a1b2c3'

const OVERVIEW: TopicsOverview = {
  run_id: RUN,
  created_at: '2026-01-02T03:04:05+00:00',
  updated_at: '2026-01-02T03:04:05+00:00',
  embed_model: 'embed-small',
  embed_model_changed: false,
  coverage: { total: 20, assigned: 18, unassigned: 2, pending: 0 },
  topics: [
    {
      id: 2,
      label: 'Orbits and comets',
      description: '',
      keywords: [['orbit', 0.4]],
      size: 12,
      share: 0.6,
      origin: 'build',
      added_at: '2026-01-02T03:04:05+00:00',
      first_seen: null,
      last_seen: null
    }
  ]
}

const DOCS: DocumentRecord[] = [
  {
    filename: 'harbour_log.pdf',
    file_hash: '7'.repeat(40),
    mimetype: 'application/pdf',
    page_count: 2,
    row_count: 0,
    node_count: 20,
    entity_types: [],
    topics: [
      { topic: `${RUN}:2`, count: 12 },
      { topic: `${RUN}:-1`, count: 2 }
    ]
  }
]

/** What the stub serves now; a test changes it to stand for a run the backend replaced. */
let serving: { overview: TopicsOverview | null; docs: DocumentRecord[] } = { overview: null, docs: [] }

/**
 * Stub the reads the Inspector makes.
 *
 * @param options.topicsEnabled - Whether `/config` switches topics on.
 * @param options.overview - The collection's run, or null for none (204).
 * @returns The fetch mock.
 */
function stubApi({ topicsEnabled, overview }: { topicsEnabled: boolean; overview: TopicsOverview | null }) {
  const respond = (body: unknown, status = 200) => ({
    ok: status < 400,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body)
  })
  serving = { overview, docs: DOCS }
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input), 'http://localhost')
    if (url.pathname.endsWith('/config')) return respond(topicsEnabled ? { topics_enabled: true } : {})
    if (url.pathname.endsWith('/topics')) return serving.overview ? respond(serving.overview) : respond(null, 204)
    if (url.pathname.endsWith('/collections/documents')) return respond({ items: serving.docs, next_cursor: null })
    if (url.pathname.endsWith('/extracts')) return respond({ extracts: [] })
    return respond({})
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

function renderInspector() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <Inspector />
    </QueryClientProvider>
  )
}

beforeEach(() => {
  useUiStore.setState({ selectedCollection: COLLECTION, selectedOwner: null, currentSessionId: null })
  useIngestJobsStore.getState().clear()
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('Inspector topics column', () => {
  it("shows each document's topics once the collection has a run", async () => {
    stubApi({ topicsEnabled: true, overview: OVERVIEW })
    renderInspector()

    expect(await screen.findByTestId('topic-badge')).toHaveTextContent('T2 Orbits and comets 12')
    expect(screen.getByText('Topics')).toBeInTheDocument()
  })

  it('leaves the column out while the collection has no run', async () => {
    const fetchMock = stubApi({ topicsEnabled: true, overview: null })
    renderInspector()

    await screen.findByText('harbour_log.pdf')
    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => String(input).endsWith('/topics'))).toBe(true))
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(screen.queryByText('Topics')).not.toBeInTheDocument()
  })

  it('leaves the column out while topics are switched off', async () => {
    const fetchMock = stubApi({ topicsEnabled: false, overview: OVERVIEW })
    renderInspector()

    await screen.findByText('harbour_log.pdf')
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(screen.queryByText('Topics')).not.toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([input]) => String(input).endsWith('/topics'))).toBe(false)
  })

  it('reads the column again when a topics job ends while it is open', async () => {
    stubApi({ topicsEnabled: true, overview: OVERVIEW })
    renderInspector()
    expect(await screen.findByTestId('topic-badge')).toHaveTextContent('T2 Orbits and comets 12')

    // A rebuild replaces the run, and the chunks with it; only the job's last frame says so.
    const rebuilt = '20260103-090000-d4e5f6'
    serving = {
      overview: {
        ...OVERVIEW,
        run_id: rebuilt,
        topics: [{ ...OVERVIEW.topics[0], label: 'Comets and meteors' }]
      },
      docs: [{ ...DOCS[0], topics: [{ topic: `${rebuilt}:2`, count: 9 }] }]
    }
    const frame = (event: IngestEvent['event'], at: number) =>
      act(() =>
        useIngestJobsStore.getState().appendEvent('job-1', {
          event,
          data: { job_id: 'job-1', collection: COLLECTION, action: 'build' },
          receivedAt: at
        })
      )
    frame('topics_started', 0)
    frame('topics_completed', 5000)

    await waitFor(() => expect(screen.getByTestId('topic-badge')).toHaveTextContent('T2 Comets and meteors 9'))
  })
})
