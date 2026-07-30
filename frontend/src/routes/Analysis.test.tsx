import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import userEvent from '@testing-library/user-event'
import { Analysis } from './Analysis'
import { useUiStore } from '@/stores/ui'
import { LanguageContext } from '@/i18n/LanguageContext'

function renderAnalysis() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <Analysis />
    </QueryClientProvider>
  )
}

function renderAnalysisInLanguage(lang: 'en' | 'de') {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <LanguageContext.Provider value={lang}>
        <Analysis />
      </LanguageContext.Provider>
    </QueryClientProvider>
  )
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

function mockFetch() {
  return vi.fn(async (input: RequestInfo | URL) => {
    const path = typeof input === 'string' ? input : input.toString()
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
        json: async () => ({ graph_top_k: 80, graph_max_top_k: 500 })
      }
    }
    return { ok: true, status: 200, json: async () => ({}) }
  })
}

beforeEach(() => {
  useUiStore.setState({ selectedCollection: 'alpha', graphTopK: null })
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
    await screen.findByRole('option', { name: /Acme/i })
    expect(screen.getByLabelText('Entity')).toHaveValue('Acme::ORG')
  })

  it('does not carry a dimming selection into a freshly-mounted graph view', async () => {
    vi.stubGlobal('fetch', mockFetch())
    const { container } = renderAnalysis()

    // Let the table-view auto-select effect run first (selectedEntityKey
    // becomes non-null there) — this is the scenario the graph-view fix
    // guards against: a pre-existing selection from table view should not
    // wash out the graph on arrival.
    await screen.findByRole('option', { name: /Acme/i })
    expect(screen.getByLabelText('Entity')).toHaveValue('Acme::ORG')

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

    await screen.findByRole('option', { name: /Acme/i })
    expect(screen.queryByRole('group', { name: /merge/i })).not.toBeInTheDocument()
  })

  it('requests the entity graph with entity_merge_mode=resolved', async () => {
    const fetchMock = mockFetch()
    vi.stubGlobal('fetch', fetchMock)
    renderAnalysis()

    await screen.findByRole('option', { name: /Acme/i })
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

describe('Analysis German smoke test', () => {
  it('renders the German heading and tab labels', async () => {
    vi.stubGlobal('fetch', mockFetch())
    renderAnalysisInLanguage('de')

    expect(await screen.findByRole('heading', { name: 'Analyse' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Hatespeech' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Entitäten' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Zusammenfassung' })).toBeInTheDocument()
  })
})
