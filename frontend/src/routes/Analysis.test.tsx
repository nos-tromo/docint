import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { Analysis } from './Analysis'
import { useUiStore } from '@/stores/ui'
import { useAnalysisUiStore } from '@/stores/analysisUi'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import { useTopicFocusStore } from '@/stores/topicFocus'
import { LanguageContext } from '@/i18n/LanguageContext'
import type { TopicsOverview } from '@/api/topics'

// Counts every render of a topic row; the real row still renders underneath.
const rowRenders = vi.hoisted(() => ({ count: 0 }))
vi.mock('@/components/analysis/TopicRow', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/components/analysis/TopicRow')>()
  return {
    ...actual,
    TopicRow: (props: import('@/components/analysis/TopicRow').TopicRowProps) => {
      rowRenders.count += 1
      return <actual.TopicRow {...props} />
    }
  }
})

/** The Analysis route, with a stand-in chat route to land on. */
function routes() {
  return (
    <MemoryRouter initialEntries={['/analysis']}>
      <Routes>
        <Route path="/analysis" element={<Analysis />} />
        <Route path="/chat" element={<p>Chat route</p>} />
      </Routes>
    </MemoryRouter>
  )
}

function renderAnalysis() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{routes()}</QueryClientProvider>)
}

function renderAnalysisInLanguage(lang: 'en' | 'de') {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <LanguageContext value={lang}>{routes()}</LanguageContext>
    </QueryClientProvider>
  )
}

const TOPICS_RUN = '20260102-030405-a1b2c3'

const TOPICS_OVERVIEW: TopicsOverview = {
  run_id: TOPICS_RUN,
  created_at: '2026-01-02T03:04:05+00:00',
  updated_at: '2026-01-02T03:04:05+00:00',
  embed_model: 'embed-small',
  embed_model_changed: false,
  coverage: { total: 120, assigned: 100, unassigned: 20, pending: 5 },
  topics: [
    {
      id: 2,
      label: 'Orbits and comets',
      description: '',
      keywords: [['orbit', 0.4]],
      size: 60,
      share: 0.5,
      origin: 'build',
      added_at: '2026-01-02T03:04:05+00:00',
      first_seen: null,
      last_seen: null
    },
    {
      id: 0,
      label: 'Bread, pastry and ovens',
      description: '',
      keywords: [['dough', 0.5]],
      size: 40,
      share: 0.3333,
      origin: 'build',
      added_at: '2026-01-02T03:04:05+00:00',
      first_seen: null,
      last_seen: null
    }
  ]
}

const entities = [
  { text: 'Acme', type: 'ORG', mentions: 9 },
  { text: 'Rivertown', type: 'LOC', mentions: 4 }
]

const graphNodes = [
  { id: 'acme::org', text: 'Acme', type: 'ORG', mentions: 9 },
  { id: 'rivertown::loc', text: 'Rivertown', type: 'LOC', mentions: 4 }
]

const graphEdges = [
  { source: 'acme::org', target: 'rivertown::loc', label: 'located_in', kind: 'relation', weight: 3 }
]

function mockFetch({
  topicsEnabled = false,
  overview = null
}: { topicsEnabled?: boolean; overview?: TopicsOverview | null } = {}) {
  return vi.fn(async (input: RequestInfo | URL) => {
    const path = typeof input === 'string' ? input : input.toString()
    if (path.includes('/topics')) {
      return overview
        ? { ok: true, status: 200, json: async () => overview }
        : { ok: true, status: 204, json: async () => null }
    }
    if (path.includes('/collections/ner/stats')) {
      return { ok: true, status: 200, json: async () => ({ top_entities: entities }) }
    }
    if (path.includes('/collections/ner/graph')) {
      return { ok: true, status: 200, json: async () => ({ nodes: graphNodes, edges: graphEdges }) }
    }
    if (path.includes('/collections/ner/sources')) {
      return { ok: true, status: 200, json: async () => ({ items: [], next_cursor: null }) }
    }
    if (path.includes('/collections/ner/warm')) {
      return { ok: true, status: 200, json: async () => ({ ok: true }) }
    }
    if (path.includes('/collections/hate-speech')) {
      return { ok: true, status: 200, json: async () => ({ items: [], next_cursor: null }) }
    }
    if (path.includes('/config')) {
      return {
        ok: true,
        status: 200,
        // Omitted rather than false when off: a backend predating topics sends no flag.
        json: async () => ({ graph_top_k: 80, graph_max_top_k: 500, ...(topicsEnabled ? { topics_enabled: true } : {}) })
      }
    }
    return { ok: true, status: 200, json: async () => ({}) }
  })
}

beforeEach(() => {
  useUiStore.setState({ selectedCollection: 'alpha', selectedOwner: null, graphTopK: null })
  useAnalysisUiStore.setState({ tab: 'ner', nerView: 'table', entity: null })
  useTopicFocusStore.setState({ focus: null })
  useIngestJobsStore.getState().clear()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('Analysis auto-select guard (table view only)', () => {
  it('auto-selects the top entity in table view', async () => {
    vi.stubGlobal('fetch', mockFetch())
    renderAnalysis()

    // Table view is the default; the "Entity" dropdown should surface the
    // auto-picked top entity once the stats query resolves.
    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Entity' })).toHaveTextContent(/Acme/i)
    )
    expect(screen.getByRole('combobox', { name: 'Entity' })).toHaveTextContent('Acme [ORG]')
  })

  it('does not carry a dimming selection into a freshly-mounted graph view', async () => {
    vi.stubGlobal('fetch', mockFetch())
    const { container } = renderAnalysis()

    // Let the table-view auto-select effect run first (selectedEntityKey
    // becomes non-null there) — this is the scenario the graph-view fix
    // guards against: a pre-existing selection from table view should not
    // wash out the graph on arrival.
    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Entity' })).toHaveTextContent(/Acme/i)
    )
    expect(screen.getByRole('combobox', { name: 'Entity' })).toHaveTextContent('Acme [ORG]')

    await userEvent.click(screen.getByRole('button', { name: 'Graph' }))

    // The ForceGraph canvas should render with no dimmed/pressed node despite
    // Analysis already holding a selectedEntityKey from the table view.
    // (Scoped to `svg g[role="button"]` — the NER view toggle above the
    // canvas is also `aria-pressed`, but is not part of the graph itself.)
    const graphNode = await screen.findByRole('button', { name: /Acme \(ORG\)/ })
    expect(graphNode).toHaveAttribute('aria-pressed', 'false')
    expect(graphNode.getAttribute('opacity')).toBe('1')
    const pressedGraphNodes = container.querySelectorAll('svg g[role="button"][aria-pressed="true"]')
    expect(pressedGraphNodes).toHaveLength(0)
  })

  it('does not auto-select when entities (re)load while the graph view is already active', async () => {
    // Regression for the guard itself: switch to graph view first (so no
    // table-view auto-select has happened yet — selectedEntityKey is still
    // null), then let a collection change reset+reload the entity list while
    // staying in graph view. Without the `nerView === 'table'` guard, the
    // effect would auto-pick the top entity here too, dimming the graph the
    // user is currently looking at.
    vi.stubGlobal('fetch', mockFetch())
    const { container } = renderAnalysis()

    await userEvent.click(screen.getByRole('button', { name: 'Graph' }))
    await screen.findByRole('button', { name: /Acme \(ORG\)/ })

    useUiStore.setState({ selectedCollection: 'beta' })
    await screen.findByRole('button', { name: /Acme \(ORG\)/ })
    expect(screen.queryByLabelText('Entity')).not.toBeInTheDocument()

    // Poll rather than assert once: if the guard were missing, the auto-select
    // effect fires asynchronously once the new collection's stats query
    // resolves, so a single immediate check could pass by sheer timing luck.
    await waitFor(() => {
      const pressedGraphNodes = container.querySelectorAll(
        'svg g[role="button"][aria-pressed="true"]'
      )
      expect(pressedGraphNodes).toHaveLength(0)
    })
  })
})

describe('Analysis entity merge mode is fixed to resolved', () => {
  it('does not render a merge-mode control', async () => {
    vi.stubGlobal('fetch', mockFetch())
    renderAnalysis()

    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Entity' })).toHaveTextContent(/Acme/i)
    )
    expect(screen.queryByRole('group', { name: /merge/i })).not.toBeInTheDocument()
  })

  it('requests the entity graph with entity_merge_mode=resolved', async () => {
    const fetchMock = mockFetch()
    vi.stubGlobal('fetch', fetchMock)
    renderAnalysis()

    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Entity' })).toHaveTextContent(/Acme/i)
    )
    await userEvent.click(screen.getByRole('button', { name: 'Graph' }))
    await screen.findByRole('button', { name: /Acme \(ORG\)/ })

    const graphCall = fetchMock.mock.calls.find(([input]) => {
      const url = typeof input === 'string' ? input : input.toString()
      return url.includes('/collections/ner/graph')
    })
    expect(graphCall).toBeDefined()
    const url = typeof graphCall![0] === 'string' ? graphCall![0] : graphCall![0].toString()
    expect(url).toContain('entity_merge_mode=resolved')
  })
})

describe('Analysis position', () => {
  it('restores the persisted tab on mount', () => {
    vi.stubGlobal('fetch', mockFetch())
    useAnalysisUiStore.setState({ tab: 'hate', nerView: 'table', entity: null })
    renderAnalysis()

    expect(screen.getByRole('button', { name: /hate|hass/i })).toHaveClass('border-foreground')
  })
})

describe('Analysis topics tab', () => {
  it('shows the topics tab only when topics are enabled', async () => {
    const off = mockFetch()
    vi.stubGlobal('fetch', off)
    const { unmount } = renderAnalysis()
    await waitFor(() => expect(off.mock.calls.some(([input]) => String(input).includes('/config'))).toBe(true))
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(screen.getByRole('button', { name: 'Summary' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Topics' })).not.toBeInTheDocument()
    unmount()

    vi.stubGlobal('fetch', mockFetch({ topicsEnabled: true }))
    renderAnalysis()
    const tabs = await screen.findByRole('button', { name: 'Topics' })
    // The fourth tab, after the three that came before it.
    expect(within(tabs.parentElement as HTMLElement).getAllByRole('button').map((b) => b.textContent)).toEqual([
      'Entities',
      'Hate speech',
      'Summary',
      'Topics'
    ])

    await userEvent.click(tabs)

    expect(tabs).toHaveClass('border-foreground')
    expect(tabs).toHaveAttribute('aria-current', 'true')
    expect(screen.getByRole('button', { name: 'Entities' })).not.toHaveAttribute('aria-current')
    expect(useAnalysisUiStore.getState().tab).toBe('topics')
    expect(await screen.findByRole('button', { name: 'Start analysis' })).toBeInTheDocument()
  })

  it('falls back to the entities tab when a remembered topics tab is switched off', async () => {
    useAnalysisUiStore.setState({ tab: 'topics', nerView: 'table', entity: null })
    vi.stubGlobal('fetch', mockFetch())
    renderAnalysis()

    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Entity' })).toHaveTextContent(/Acme/i)
    )
    expect(screen.getByRole('button', { name: 'Entities' })).toHaveClass('border-foreground')
    expect(screen.queryByRole('button', { name: 'Start analysis' })).not.toBeInTheDocument()
    // Shown, not rewritten: the remembered tab returns once the flag is back on.
    expect(useAnalysisUiStore.getState().tab).toBe('topics')
  })

  it('asks about a topic in chat with the collection and run it came from', async () => {
    useAnalysisUiStore.setState({ tab: 'topics', nerView: 'table', entity: null })
    // An admin in another owner's collection: the owner is half of which collection it is.
    useUiStore.setState({ selectedCollection: 'alpha', selectedOwner: 'analyst-b' })
    vi.stubGlobal('fetch', mockFetch({ topicsEnabled: true, overview: TOPICS_OVERVIEW }))
    renderAnalysis()

    const [first] = await screen.findAllByTestId('topic-row')
    await userEvent.click(within(first).getByRole('button', { name: 'Ask in chat' }))

    expect(useTopicFocusStore.getState().focus).toEqual({
      collection: 'alpha',
      owner: 'analyst-b',
      runId: '20260102-030405-a1b2c3',
      topicId: 2,
      label: 'Orbits and comets'
    })
    expect(await screen.findByText('Chat route')).toBeInTheDocument()
  })

  it('keeps the topic rows still when the screen re-renders between progress frames', async () => {
    useAnalysisUiStore.setState({ tab: 'topics', nerView: 'table', entity: null })
    vi.stubGlobal('fetch', mockFetch({ topicsEnabled: true, overview: TOPICS_OVERVIEW }))
    const frame = (event: 'topics_started' | 'topics_progress', data: Record<string, unknown>, at: number) =>
      act(() =>
        useIngestJobsStore
          .getState()
          .appendEvent('job-1', { event, data: { job_id: 'job-1', ...data }, receivedAt: at })
      )
    frame('topics_started', { collection: 'alpha', action: 'align' }, 0)
    frame('topics_progress', { stage: 'aligning', done: 1, total: 100, message: 'x' }, 1000)
    renderAnalysis()
    await screen.findAllByTestId('topic-row')
    // Let the screen's other reads land first: each answer re-renders it.
    await waitFor(() => expect(screen.getByText('Aligning pending chunks: 1 of 100')).toBeInTheDocument())
    await new Promise((resolve) => setTimeout(resolve, 30))
    const before = rowRenders.count

    frame('topics_progress', { stage: 'aligning', done: 2, total: 100, message: 'x' }, 2000)
    // Anything the screen itself reads re-renders it, which hands the panel its props again.
    act(() => useUiStore.setState({ graphTopK: 120 }))
    frame('topics_progress', { stage: 'aligning', done: 3, total: 100, message: 'x' }, 3000)

    expect(screen.getByText('Aligning pending chunks: 3 of 100')).toBeInTheDocument()
    expect(rowRenders.count).toBe(before)
  })
})

describe('Analysis German smoke test', () => {
  it('renders the German heading and tab labels', async () => {
    vi.stubGlobal('fetch', mockFetch({ topicsEnabled: true }))
    renderAnalysisInLanguage('de')

    expect(await screen.findByRole('heading', { name: 'Analyse' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Hatespeech' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Entitäten' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Zusammenfassung' })).toBeInTheDocument()
    expect(await screen.findByRole('button', { name: 'Themen' })).toBeInTheDocument()
  })
})

describe('Analysis entity-key resolution', () => {
  it('canonicalizes a differently-cased key handed over from a chat pill', async () => {
    // A chat entity pill builds its key from the raw chunk surface, while the
    // aggregate is merged orthographically. Left unresolved, the findings
    // query asks for an entity the aggregate does not hold and the dropdown
    // shows an unrelated one.
    useAnalysisUiStore.setState({
      tab: 'ner',
      nerView: 'table',
      entity: { key: 'acme::ORG', collection: 'alpha' }
    })
    vi.stubGlobal('fetch', mockFetch())
    renderAnalysis()

    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Entity' })).toHaveTextContent(/Acme/i)
    )
    await waitFor(() => {
      expect(useAnalysisUiStore.getState().entity).toEqual({
        key: 'Acme::ORG',
        collection: 'alpha'
      })
    })
    expect(screen.getByRole('combobox', { name: 'Entity' })).toHaveTextContent('Acme [ORG]')
  })

  it('leaves a key the aggregate cannot place alone', async () => {
    useAnalysisUiStore.setState({
      tab: 'ner',
      nerView: 'table',
      entity: { key: 'Nobody::PER', collection: 'alpha' }
    })
    vi.stubGlobal('fetch', mockFetch())
    renderAnalysis()

    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Entity' })).toHaveTextContent(/Acme/i)
    )
    expect(useAnalysisUiStore.getState().entity).toEqual({
      key: 'Nobody::PER',
      collection: 'alpha'
    })
  })
})
