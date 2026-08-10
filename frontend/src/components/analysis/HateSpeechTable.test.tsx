import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, within, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { HateSpeechTable } from './HateSpeechTable'
import { hateSpeechSnapshot } from '@/lib/reportSnapshots'
import type { HateSpeechRow } from '@/api/types'
import { useUiStore } from '@/stores/ui'

afterEach(() => vi.restoreAllMocks())

// Rows render a TranslateControl (mounted whenever a row has chunk text),
// which calls useTranslate()/useMutation() — it needs a QueryClientProvider
// ancestor even though these tests never trigger a translation.
function renderWithClient(ui: React.ReactNode) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

const rows: HateSpeechRow[] = [
  {
    chunk_id: 'h1',
    filename: 'rant.txt',
    page: 2,
    category: 'harassment',
    confidence: 'high',
    reason: 'Targets a protected group.',
    chunk_text: 'Body of the flagged passage.',
    reference_metadata: {
      author: 'Carol',
      network: 'docs',
      timestamp: '2026-02-14',
      posting_timestamp: '2025-09-17 15:15:30.000000'
    }
  }
]

describe('HateSpeechTable', () => {
  it('renders one row per finding with category, source and chunk text inline', () => {
    renderWithClient(<HateSpeechTable rows={rows} collection="alpha" />)
    const row = screen.getByTestId('hate-speech-row')
    expect(within(row).getByText(/harassment/i)).toBeInTheDocument()
    expect(within(row).getByText('rant.txt')).toBeInTheDocument()
    // Chunk text is shown inline — no expansion required.
    expect(within(row).getByText(/Body of the flagged passage/)).toBeInTheDocument()
  })

  it('shows a clamped reason block and metadata pills; drops confidence and chunk id', () => {
    renderWithClient(<HateSpeechTable rows={rows} collection="alpha" />)
    // Reason is prose, no label.
    expect(screen.getByText('Targets a protected group.')).toBeInTheDocument()
    // Reference metadata renders as pills.
    expect(screen.getByTestId('metadata-pills')).toBeInTheDocument()
    expect(screen.getByText('2026-02-14')).toBeInTheDocument()
    // Fractional seconds are trimmed off a timestamp pill.
    expect(screen.getByText('2025-09-17 15:15:30')).toBeInTheDocument()
    // Confidence and chunk id are display-dropped (still in CSV/report).
    expect(screen.queryByText('high')).not.toBeInTheDocument()
    expect(screen.queryByText('h1')).not.toBeInTheDocument()
  })

  it('shows the empty state when nothing was flagged', () => {
    render(<HateSpeechTable rows={[]} collection="alpha" />)
    expect(screen.getByText(/no flagged content/i)).toBeInTheDocument()
  })

  it('renders a streaming CSV download link to the right collection-scoped endpoint', () => {
    renderWithClient(<HateSpeechTable rows={rows} collection="alpha" />)
    // The control is the download icon; "Export CSV" survives as its name.
    const link = screen.getByRole('link', { name: 'Export CSV' })
    expect(link).toHaveAttribute(
      'href',
      expect.stringContaining('/collections/alpha/export/hate-speech.csv')
    )
    expect(link).toHaveAttribute('download')
  })

  it('reveals a Translate toggle in the actions cell that swaps the chunk text in place', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({ ok: true, translation: 'übersetzt', model: 'm', target_lang: 'de' })
      }))
    )
    renderWithClient(<HateSpeechTable rows={rows} collection="alpha" />)
    const row = screen.getByTestId('hate-speech-row')
    await userEvent.click(within(row).getByRole('button', { name: /^translate$/i }))
    await waitFor(() => expect(within(row).getByText('übersetzt')).toBeInTheDocument())
    expect(within(row).queryByText(/Body of the flagged passage/)).not.toBeInTheDocument()
    expect(within(row).getByRole('button', { name: /show original/i })).toBeInTheDocument()
  })

  it('includes the translation in the hate-speech snapshot', () => {
    const snap = hateSpeechSnapshot(
      { chunk_id: 'c1', chunk_text: 'orig' } as never,
      { text: 'übersetzt', target_lang: 'de', model: 'm' }
    )
    expect(snap.snapshot.translation).toEqual({ text: 'übersetzt', target_lang: 'de', model: 'm' })
  })
})

it('breaks unbreakable metadata values instead of overflowing the column', () => {
  // Regression (former `dl` layout): grid items default to min-content
  // minimums, and overflow-wrap:break-word does NOT reduce min-content — a
  // long unbroken value in a plain `1fr` value track widened the metadata
  // cell across the Text column, misaligning the body scroller with the
  // fixed header. The pills cell instead wraps each value with `break-all`.
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const longValue = 'x'.repeat(160)
  render(
    <QueryClientProvider client={qc}>
      <HateSpeechTable
        rows={[
          {
            chunk_id: 'c1',
            category: 'other',
            reason: 'why',
            confidence: 'high',
            chunk_text: 'text',
            reference_metadata: {
              type: longValue
            }
          } as never
        ]}
        collection="c"
      />
    </QueryClientProvider>
  )
  const pillValue = screen.getByText(longValue)
  expect(pillValue.className).toContain('break-all')
})

function renderRow(row: HateSpeechRow) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  })
  return render(
    <QueryClientProvider client={qc}>
      <HateSpeechTable rows={[row]} collection="test-collection" />
    </QueryClientProvider>
  )
}

describe('HateSpeechTable document preview', () => {
  it('opens the shared preview dialog for the flagged chunk source', async () => {
    useUiStore.setState({ selectedCollection: 'test-collection', previewModal: null })
    renderRow({
      chunk_id: 'c1',
      filename: 'a.pdf',
      file_hash: 'hash',
      chunk_text: 'flagged text',
      category: 'insult'
    })

    await userEvent.click(screen.getByRole('button', { name: /preview/i }))

    expect(useUiStore.getState().previewModal).toEqual({
      collection: 'test-collection',
      file_hash: 'hash',
      filename: 'a.pdf'
    })
  })

  it('omits the preview action for a finding with no stored file', () => {
    useUiStore.setState({ selectedCollection: 'test-collection', previewModal: null })
    renderRow({ chunk_id: 'c1', filename: 'a.pdf', chunk_text: 'flagged text', category: 'insult' })

    expect(screen.queryByRole('button', { name: /preview/i })).not.toBeInTheDocument()
  })
})
