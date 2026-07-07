import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Report } from './Report'
import { useReportStore } from '@/stores/report'

const overview = {
  collection: 'docs',
  captured_at: '2026-07-01T09:00:00Z',
  document_count: 1,
  node_count: 6,
  file_types: [{ label: 'PDF', count: 1 }],
  entity_types: ['ORG'],
  documents: [
    { filename: 'c.pdf', type_label: 'PDF', page_count: 4, row_count: null, node_count: 6, file_hash: '0123456789abcdef' }
  ]
}

const reportDetail = {
  id: 1,
  title: 'Case Alpha',
  collection_name: 'docs',
  show_toc: true,
  show_collection_overview: true,
  session_id: null,
  created_at: null,
  updated_at: null,
  item_count: 2,
  items: [
    {
      id: 10,
      artifact_type: 'entity_finding',
      dedupe_key: 'entity:c1',
      position: 0,
      note: null,
      snapshot: { entity_label: 'Acme [ORG]', chunk_text: 'Acme text', filename: 'a.pdf', page: 1 },
      created_at: null
    },
    {
      id: 11,
      artifact_type: 'hate_speech_finding',
      dedupe_key: 'hate:c2',
      position: 1,
      note: null,
      snapshot: { category: 'slur', confidence: 'high', reason: 'bad', filename: 'b.json' },
      created_at: null
    }
  ],
  collection_overview: overview
}

function mockFetch(detail: Record<string, unknown> = reportDetail) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (u: string) => {
      const url = String(u)
      if (url.includes('/reports/1')) {
        return { ok: true, status: 200, json: async () => detail }
      }
      if (url.endsWith('/reports')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            reports: [{ ...detail, items: undefined }]
          })
        }
      }
      return { ok: true, status: 200, json: async () => ({}) }
    })
  )
}

function renderReport() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <Report />
    </QueryClientProvider>
  )
}

beforeEach(() => {
  localStorage.clear()
  useReportStore.setState({ activeReportId: 1 })
  mockFetch()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('Report view', () => {
  it('shows the active report title, grouped items and export links', async () => {
    renderReport()
    expect(await screen.findByDisplayValue('Case Alpha')).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByText('Acme [ORG]')).toBeInTheDocument()
    })
    expect(screen.getByText(/Entity findings/i)).toBeInTheDocument()
    expect(screen.getByText(/Hate-speech findings/i)).toBeInTheDocument()

    const pdf = screen.getByRole('link', { name: 'PDF' })
    expect(pdf).toHaveAttribute('href', expect.stringContaining('/reports/1/export.pdf'))
    expect(screen.getByRole('link', { name: 'HTML' })).toHaveAttribute('target', '_blank')
  })

  it('prompts to select a report when none is active', async () => {
    useReportStore.setState({ activeReportId: null })
    renderReport()
    expect(await screen.findByText(/select a report/i)).toBeInTheDocument()
  })

  it('reflects show_toc and toggles it via PATCH', async () => {
    renderReport()
    const toggle = await screen.findByRole('checkbox', { name: /contents/i })
    expect(toggle).toBeChecked()
    fireEvent.click(toggle)
    await waitFor(() => {
      const patch = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls.find(
        (c) => (c[1] as RequestInit | undefined)?.method === 'PATCH'
      )
      expect(patch).toBeTruthy()
      expect(JSON.parse(String((patch![1] as RequestInit).body))).toMatchObject({ show_toc: false })
    })
  })
})

describe('Report view — document overview', () => {
  it('shows both the items and the overview preview when a report has both', async () => {
    renderReport()
    await screen.findByText('Acme [ORG]')
    expect(screen.getByText('c.pdf')).toBeInTheDocument()
    expect(screen.getByText('0123456789ab')).toBeInTheDocument()
    expect(screen.queryByText(/this report is empty/i)).not.toBeInTheDocument()
  })

  it('shows items without a preview when the report has no overview snapshot yet', async () => {
    mockFetch({ ...reportDetail, collection_overview: null })
    renderReport()
    await screen.findByText('Acme [ORG]')
    expect(screen.queryByText('c.pdf')).not.toBeInTheDocument()
    expect(screen.queryByText(/this report is empty/i)).not.toBeInTheDocument()
    expect(await screen.findByRole('button', { name: /capture overview/i })).toBeInTheDocument()
  })

  it('shows the overview preview instead of the empty message when there are no items', async () => {
    mockFetch({ ...reportDetail, items: [], item_count: 0 })
    renderReport()
    await screen.findByText('c.pdf')
    expect(screen.queryByText(/this report is empty/i)).not.toBeInTheDocument()
  })

  it('shows the empty message when there are no items and no overview', async () => {
    mockFetch({ ...reportDetail, items: [], item_count: 0, collection_overview: null })
    renderReport()
    expect(await screen.findByText(/this report is empty/i)).toBeInTheDocument()
  })

  it('shows the empty message (not a blank area) when items are empty and the overview is toggled off', async () => {
    mockFetch({ ...reportDetail, items: [], item_count: 0, show_collection_overview: false })
    renderReport()
    expect(await screen.findByText(/this report is empty/i)).toBeInTheDocument()
    expect(screen.queryByText('c.pdf')).not.toBeInTheDocument()
  })

  it('shows the empty message when items are empty and the overview snapshot has no documents', async () => {
    mockFetch({
      ...reportDetail,
      items: [],
      item_count: 0,
      collection_overview: { ...overview, documents: [], document_count: 0 }
    })
    renderReport()
    expect(await screen.findByText(/this report is empty/i)).toBeInTheDocument()
    expect(screen.queryByText('c.pdf')).not.toBeInTheDocument()
  })

  it('reflects show_collection_overview and toggles it via PATCH', async () => {
    renderReport()
    const toggle = await screen.findByRole('checkbox', { name: 'Document overview' })
    expect(toggle).toBeChecked()
    fireEvent.click(toggle)
    await waitFor(() => {
      const patch = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls.find(
        (c) => (c[1] as RequestInit | undefined)?.method === 'PATCH'
      )
      expect(patch).toBeTruthy()
      expect(JSON.parse(String((patch![1] as RequestInit).body))).toMatchObject({ show_collection_overview: false })
    })
  })

  it('shows the captured date and refreshes the overview on click', async () => {
    renderReport()
    const button = await screen.findByRole('button', { name: /refresh overview \(captured 2026-07-01\)/i })
    fireEvent.click(button)
    await waitFor(() => {
      const post = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls.find((c) =>
        String(c[0]).includes('/reports/1/collection-overview/refresh')
      )
      expect(post).toBeTruthy()
    })
  })
})
