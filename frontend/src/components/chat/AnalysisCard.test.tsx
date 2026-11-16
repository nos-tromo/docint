import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AnalysisCard } from './AnalysisCard'
import type { Analysis, AnalysisSummary } from '@/api/types'
import { en } from '@/i18n/en'
import { LanguageContext } from '@/i18n/LanguageContext'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'
import { ANALYSIS, PNG_URI } from '@/test/analysis'

const SUMMARY: AnalysisSummary = {
  id: 'a1',
  index: 1,
  title: 'Posts per author',
  tables: [{ name: 'posts.csv', columns: 2, rows: 7 }],
  charts: [{ name: 'chart.png', bytes: 12 }]
}

interface Props {
  entry?: Analysis | AnalysisSummary
  index?: number
  sessionId?: string | null
  inReport?: boolean
  lang?: 'en' | 'de'
}

function renderCard({ entry = ANALYSIS, index = 1, sessionId = 'session-1', inReport = false, lang = 'en' }: Props = {}) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const ui = (current: Analysis | AnalysisSummary) => (
    <QueryClientProvider client={qc}>
      <LanguageContext value={lang}>
        <AnalysisCard entry={current} index={index} sessionId={sessionId} inReport={inReport} />
      </LanguageContext>
    </QueryClientProvider>
  )
  const view = render(ui(entry))
  return { ...view, update: (next: Analysis | AnalysisSummary) => view.rerender(ui(next)) }
}

/** Answer every request with one JSON body, and keep the URLs asked for. */
function serve(body: unknown, status = 200): string[] {
  const urls: string[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (u: string) => {
      urls.push(String(u))
      return { ok: status < 400, status, json: async () => body, text: async () => JSON.stringify(body) }
    })
  )
  return urls
}

beforeEach(() => {
  localStorage.clear()
  useReportStore.setState({ activeReportId: null })
  useUiStore.setState({ selectedCollection: 'docs', selectedOwner: null, previewModal: null })
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('AnalysisCard', () => {
  it('shows the title, the first five rows and every chart', () => {
    renderCard()

    const card = screen.getByTestId('analysis-card')
    expect(within(card).getByText('Posts per author')).toBeInTheDocument()
    const table = within(card).getByRole('table')
    const body = within(table).getAllByRole('row').slice(1)
    expect(body.map((row) => row.textContent)).toEqual(['alice7', 'bob6', 'carol5', 'dave4', 'erin3'])
    expect(within(table).getAllByRole('columnheader').map((cell) => cell.textContent)).toEqual(['author', 'posts'])
    expect(within(card).getByText('2 more rows')).toBeInTheDocument()
    expect(within(card).getByRole('img', { name: 'chart.png' })).toHaveAttribute('src', PNG_URI)
  })

  it('says nothing about more rows when the preview holds them all', () => {
    const table = { ...ANALYSIS.tables[0], rows: ANALYSIS.tables[0].rows.slice(0, 3), row_count: 3 }
    renderCard({ entry: { ...ANALYSIS, tables: [table] } })

    expect(screen.queryByText(/more rows?$/)).not.toBeInTheDocument()
  })

  it('counts the rows the run had, not only the ones it kept', () => {
    const table = { ...ANALYSIS.tables[0], rows: ANALYSIS.tables[0].rows.slice(0, 6), row_count: 600, truncated: true }
    renderCard({ entry: { ...ANALYSIS, tables: [table] } })

    expect(screen.getByText('595 more rows')).toBeInTheDocument()
  })

  it('names an untitled analysis by its place in the turn', () => {
    renderCard({ entry: { ...ANALYSIS, title: '' }, index: 2 })

    expect(within(screen.getByTestId('analysis-card')).getByText('Analysis 2')).toBeInTheDocument()
  })

  it('keeps the code behind a disclosure', async () => {
    renderCard()

    expect(screen.queryByText(/import pandas/)).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Show code' }))

    expect(screen.getByText(/import pandas/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Hide code' })).toHaveAttribute('aria-expanded', 'true')
    await userEvent.click(screen.getByRole('button', { name: 'Hide code' }))
    expect(screen.queryByText(/import pandas/)).not.toBeInTheDocument()
  })

  it('opens a chart in the preview dialog', async () => {
    renderCard()

    await userEvent.click(screen.getByRole('button', { name: 'chart.png' }))

    expect(useUiStore.getState().previewModal).toEqual({ filename: 'chart.png', data_uri: PNG_URI })
  })

  it('names its provenance', () => {
    renderCard()

    const card = screen.getByTestId('analysis-card')
    expect(card).toHaveTextContent('[1] harbour-report.pdf')
    expect(card).toHaveTextContent('workbench 0.2.0')
    expect(card).toHaveTextContent('job job-1')
  })

  it('names its provenance in German', () => {
    renderCard({ lang: 'de' })

    expect(screen.getByTestId('analysis-card')).toHaveTextContent(/Berechnet aus \[1\] harbour-report\.pdf/)
  })

  it('names the workbench alone when no passage was handed over', () => {
    const provenance = { ...ANALYSIS.provenance, sources: [] }
    renderCard({ entry: { ...ANALYSIS, provenance } })

    const card = screen.getByTestId('analysis-card')
    expect(card).toHaveTextContent('workbench 0.2.0, job job-1')
    expect(card).not.toHaveTextContent('Computed from')
  })

  it('adds itself to the report by id alone', async () => {
    const adds: Array<{ url: string; body: unknown }> = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string, init?: RequestInit) => {
        const url = String(u)
        if (url.endsWith('/reports') && init?.method === 'POST') {
          return { ok: true, status: 200, json: async () => ({ id: 1, title: 'Untitled report', items: [] }) }
        }
        if (init?.method === 'POST') adds.push({ url, body: JSON.parse(String(init.body)) })
        return { ok: true, status: 200, json: async () => ({ id: 9, dedupe_key: 'analysis:a1' }) }
      })
    )
    renderCard()

    await userEvent.click(screen.getByRole('button', { name: /add to report/i }))

    await waitFor(() => expect(adds).toHaveLength(1))
    expect(adds[0].url).toContain('/reports/1/items?collection=docs')
    expect(adds[0].body).toEqual({
      artifact_type: 'analysis',
      dedupe_key: 'analysis:a1',
      snapshot: { analysis_id: 'a1' }
    })
  })

  it('shows an analysis already in the report as added', () => {
    renderCard({ inReport: true })

    expect(screen.getByRole('button', { name: en['report.in_report'] })).toHaveAttribute('aria-pressed', 'true')
  })
})

describe('AnalysisCard from a summary', () => {
  it('fetches a summary from history once, through apiFetch', async () => {
    const urls = serve(ANALYSIS)
    renderCard({ entry: SUMMARY })

    expect(screen.getByText('Posts per author')).toBeInTheDocument()
    expect(screen.getByText('Loading analysis…')).toBeInTheDocument()
    await screen.findByText('alice')

    expect(urls).toHaveLength(1)
    expect(urls[0].endsWith('/sessions/session-1/analyses/a1?collection=docs')).toBe(true)
    expect(screen.getByRole('img', { name: 'chart.png' })).toBeInTheDocument()
  })

  it('does not fetch what the final frame already carried', () => {
    const urls = serve(ANALYSIS)
    renderCard({ entry: ANALYSIS })

    expect(urls).toEqual([])
  })

  it('keeps what the final frame carried when the history describes it by its shape', async () => {
    const urls = serve(ANALYSIS)
    const view = renderCard({ entry: ANALYSIS })

    view.update(SUMMARY)

    expect(await screen.findByText('alice')).toBeInTheDocument()
    expect(screen.queryByText('Loading analysis…')).not.toBeInTheDocument()
    expect(urls).toEqual([])
  })

  it('does not fetch without a session to read it through', () => {
    const urls = serve(ANALYSIS)
    renderCard({ entry: SUMMARY, sessionId: null })

    expect(urls).toEqual([])
    expect(screen.getByText('Posts per author')).toBeInTheDocument()
  })

  it('says so when the analysis cannot be read, and keeps the title', async () => {
    serve({ detail: 'Analysis not found.' }, 404)
    renderCard({ entry: SUMMARY })

    expect(await screen.findByText('This analysis could not be loaded.')).toBeInTheDocument()
    expect(screen.getByText('Posts per author')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /add to report/i })).toBeInTheDocument()
  })

  it('names an untitled summary by its place in the turn', () => {
    serve(ANALYSIS)
    renderCard({ entry: { ...SUMMARY, title: null }, index: 2 })

    expect(screen.getByText('Analysis 2')).toBeInTheDocument()
  })
})

describe('AnalysisCard with untrusted content', () => {
  const HOSTILE = '<img src="x" onerror="alert(1)"><b>bold</b>'

  it('renders every text field as text, never as markup', async () => {
    const entry: Analysis = {
      ...ANALYSIS,
      title: HOSTILE,
      code: HOSTILE,
      tables: [{ name: HOSTILE, columns: [HOSTILE], rows: [[HOSTILE]], row_count: 1, truncated: false }],
      charts: [{ name: HOSTILE, data_uri: PNG_URI, bytes: 12 }],
      provenance: {
        ...ANALYSIS.provenance,
        sources: [{ citation_index: 1, chunk_id: 'c1', file_hash: null, filename: HOSTILE, page: null }],
        workbench: { job_id: HOSTILE, version: HOSTILE, tool: 'run_python' }
      }
    }
    renderCard({ entry })
    await userEvent.click(screen.getByRole('button', { name: 'Show code' }))

    const card = screen.getByTestId('analysis-card')
    expect(card.querySelector('b')).toBeNull()
    // The only <img> is the chart, and its source is the data URI, not a path the text named.
    const images = card.querySelectorAll('img')
    expect(images).toHaveLength(1)
    expect(images[0].getAttribute('src')).toBe(PNG_URI)
    expect(images[0].getAttribute('onerror')).toBeNull()
    expect(within(card).getAllByText(HOSTILE, { exact: false }).length).toBeGreaterThan(0)
  })

  it.each([
    ['an SVG', 'data:image/svg+xml;base64,PHN2Zy8+'],
    ['a JPEG', 'data:image/jpeg;base64,/9j/4AAQ'],
    ['a script URL', 'javascript:alert(1)'],
    ['a remote image', 'https://example.test/chart.png'],
    ['an uppercase scheme', 'DATA:IMAGE/PNG;BASE64,iVBORw0KGgo='],
    ['no URI at all', '']
  ])('leaves out a chart that is %s', (_name, data_uri) => {
    const entry: Analysis = {
      ...ANALYSIS,
      charts: [
        { name: 'bad.png', data_uri, bytes: 12 },
        { name: 'good.png', data_uri: PNG_URI, bytes: 12 }
      ]
    }
    renderCard({ entry })

    const card = screen.getByTestId('analysis-card')
    expect(within(card).getAllByRole('img').map((img) => img.getAttribute('alt'))).toEqual(['good.png'])
    expect(card.innerHTML).not.toContain('javascript:')
  })

  it('draws a chart named in the stored row but a non-string data URI as nothing', () => {
    const entry = {
      ...ANALYSIS,
      charts: [{ name: 'bad.png', data_uri: 7, bytes: 12 }]
    } as unknown as Analysis
    renderCard({ entry })

    expect(screen.queryByRole('img')).not.toBeInTheDocument()
  })
})
