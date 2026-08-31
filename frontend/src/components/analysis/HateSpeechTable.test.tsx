import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest'
import { render, screen, within, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { HateSpeechTable } from './HateSpeechTable'
import { hateSpeechSnapshot } from '@/lib/reportSnapshots'
import type { HateSpeechRow } from '@/api/types'
import { useUiStore } from '@/stores/ui'
import { useReportStore } from '@/stores/report'
import { useTranslationsStore } from '@/stores/translations'

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

beforeEach(() => {
  localStorage.clear()
  useTranslationsStore.setState({ byText: {} })
  useReportStore.setState({ activeReportId: null })
})

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

  it('Add all sends the stored translation for a translated row and none for the rest', async () => {
    // The batch snapshots rows walked from the server, which were never
    // rendered — the translation has to come from the shared store, keyed by
    // the same trimmed text the row files it under (hence the padded
    // chunk_text here, which would miss under an untrimmed key).
    useUiStore.setState({ selectedCollection: 'alpha' })
    useTranslationsStore.setState({
      byText: { 'Erste markierte Zeile.': { text: 'First flagged line.', target_lang: 'en', model: 'm' } }
    })
    const batchRows: HateSpeechRow[] = [
      {
        chunk_id: 'h20',
        filename: 'a.txt',
        category: 'harassment',
        confidence: 'high',
        chunk_text: '  Erste markierte Zeile.  '
      },
      {
        chunk_id: 'h21',
        filename: 'b.txt',
        category: 'harassment',
        confidence: 'low',
        chunk_text: 'Zweite Zeile ohne Übersetzung.'
      }
    ]
    const captured: Record<string, unknown>[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string, init?: RequestInit) => {
        const url = String(u)
        if (url.includes('/collections/hate-speech')) {
          return { ok: true, status: 200, json: async () => ({ items: batchRows, next_cursor: null }) }
        }
        if (url.endsWith('/reports') && init?.method === 'POST') {
          return { ok: true, status: 200, json: async () => ({ id: 1, title: 'Untitled report', items: [] }) }
        }
        if (url.includes('/items/batch')) {
          captured.push(JSON.parse(String(init?.body)))
          return { ok: true, status: 200, json: async () => ({ added: 2, skipped: 0, item_count: 2 }) }
        }
        return { ok: true, status: 200, json: async () => ({}) }
      })
    )

    renderWithClient(<HateSpeechTable rows={batchRows} collection="alpha" reportDedupeKeys={new Set()} />)
    await userEvent.click(screen.getByRole('button', { name: /add all findings to report/i }))

    await waitFor(() => expect(captured).toHaveLength(1))
    const items = captured[0].items as { snapshot: Record<string, unknown> }[]
    expect(items[0].snapshot.translation).toEqual({
      text: 'First flagged line.',
      target_lang: 'en',
      model: 'm'
    })
    expect(items[1].snapshot).not.toHaveProperty('translation')
  })

  it('Translate all walks the section and fills the store for rows never rendered', async () => {
    // Wired to the section's own page walk, not to the rows on screen, so a
    // flagged chunk below the fold is translated too — which is what lets the
    // subsequent "Add all" carry it into the report.
    const walked: HateSpeechRow[] = [
      { chunk_id: 'h30', filename: 'a.txt', category: 'harassment', confidence: 'high', chunk_text: '  Erste Zeile.  ' },
      { chunk_id: 'h31', filename: 'b.txt', category: 'harassment', confidence: 'low', chunk_text: 'Zweite Zeile.' }
    ]
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string, init?: RequestInit) => {
        const url = String(u)
        if (url.includes('/collections/hate-speech')) {
          return { ok: true, status: 200, json: async () => ({ items: walked, next_cursor: null }) }
        }
        if (url.includes('/translate')) {
          const text = String(JSON.parse(String(init?.body)).text)
          return {
            ok: true,
            status: 200,
            json: async () => ({ ok: true, translation: `en:${text}`, model: 'm', target_lang: 'en' })
          }
        }
        return { ok: true, status: 200, json: async () => ({}) }
      })
    )

    renderWithClient(<HateSpeechTable rows={[walked[0]]} collection="alpha" />)
    await userEvent.click(screen.getByRole('button', { name: /translate all findings/i }))

    await waitFor(() =>
      expect(Object.keys(useTranslationsStore.getState().byText).sort()).toEqual(['Erste Zeile.', 'Zweite Zeile.'])
    )
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

describe('HateSpeechTable — evidence thumbnail', () => {
  it('shows the source image when the flagged chunk came from a picture', () => {
    useUiStore.setState({ selectedCollection: 'alpha', previewModal: null })
    renderWithClient(
      <HateSpeechTable
        rows={[
          {
            chunk_id: 'h9',
            filename: 'poster.png',
            file_hash: 'ph',
            image_id: 'ph',
            category: 'harassment',
            confidence: 'high',
            chunk_text: 'Printed slur.'
          }
        ]}
        collection="alpha"
      />
    )

    expect(screen.getByRole('button', { name: /poster\.png/ })).toBeInTheDocument()
    expect(document.querySelector('[data-testid="hate-speech-row"] img')?.getAttribute('src')).toContain(
      'file_hash=ph'
    )
  })

  it('enlarges the picture when the pixels are clicked', async () => {
    useUiStore.setState({ selectedCollection: 'alpha', previewModal: null })
    renderWithClient(
      <HateSpeechTable
        rows={[
          {
            chunk_id: 'h11',
            filename: 'poster.png',
            file_hash: 'ph',
            image_id: 'ph',
            category: 'harassment',
            confidence: 'high',
            chunk_text: 'Printed slur.'
          }
        ]}
        collection="alpha"
      />
    )

    await userEvent.click(document.querySelector('[data-testid="hate-speech-row"] img')!)

    expect(useUiStore.getState().previewModal).toMatchObject({ file_hash: 'ph', filename: 'poster.png' })
  })

  it('shows nothing for a text row', () => {
    useUiStore.setState({ selectedCollection: 'alpha', previewModal: null })
    renderWithClient(
      <HateSpeechTable
        rows={[
          {
            chunk_id: 'h10',
            filename: 'rant.txt',
            file_hash: 'th',
            category: 'harassment',
            confidence: 'high',
            chunk_text: 'Body.'
          }
        ]}
        collection="alpha"
      />
    )

    expect(document.querySelector('[data-testid="hate-speech-row"] img')).toBeNull()
  })
})
