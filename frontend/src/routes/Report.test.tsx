import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Report } from './Report'
import { useReportStore } from '@/stores/report'

const reportDetail = {
  id: 1,
  title: 'Case Alpha',
  collection_name: 'docs',
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
  ]
}

function mockFetch() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (u: string) => {
      const url = String(u)
      if (url.includes('/reports/1')) {
        return { ok: true, status: 200, json: async () => reportDetail }
      }
      if (url.endsWith('/reports')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            reports: [{ ...reportDetail, items: undefined }]
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
})
