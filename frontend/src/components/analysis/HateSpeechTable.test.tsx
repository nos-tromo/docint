import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, within, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { HateSpeechTable } from './HateSpeechTable'
import { hateSpeechSnapshot } from '@/lib/reportSnapshots'
import type { HateSpeechRow } from '@/api/types'

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
      timestamp: '2026-02-14'
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

  it('flattens reason, confidence and reference metadata into the metadata column', () => {
    renderWithClient(<HateSpeechTable rows={rows} collection="alpha" />)
    const dl = screen.getByText(/^Reason$/).closest('dl') as HTMLElement
    expect(within(dl).getByText(/Targets a protected group/)).toBeInTheDocument()
    expect(within(dl).getByText(/^Confidence$/)).toBeInTheDocument()
    expect(within(dl).getByText('high')).toBeInTheDocument()
    expect(within(dl).getByText(/^Author$/)).toBeInTheDocument()
    expect(within(dl).getByText('Carol')).toBeInTheDocument()
    expect(within(dl).getByText(/^Network$/)).toBeInTheDocument()
    expect(within(dl).getByText('docs')).toBeInTheDocument()
  })

  it('shows the empty state when nothing was flagged', () => {
    render(<HateSpeechTable rows={[]} collection="alpha" />)
    expect(screen.getByText(/no flagged content/i)).toBeInTheDocument()
  })

  it('renders a streaming CSV download link to the right collection-scoped endpoint', () => {
    renderWithClient(<HateSpeechTable rows={rows} collection="alpha" />)
    const link = screen.getByRole('link', { name: 'CSV' })
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
