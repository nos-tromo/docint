import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { EntityFindingsTable } from './EntityFindingsTable'
import { useUiStore } from '@/stores/ui'
import { useReportStore } from '@/stores/report'
import { useTranslationsStore } from '@/stores/translations'
import type { NerEntityRow, NerSourceRow } from '@/api/types'

// Rows render a TranslateControl (mounted whenever a row has chunk text),
// which calls useTranslate()/useMutation() — it needs a QueryClientProvider
// ancestor even though these tests never trigger a translation.
function renderWithClient(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

const selected: NerEntityRow = {
  text: 'Berlin',
  type: 'LOC',
  mentions: 4,
  variants: [{ text: 'Berlin-Mitte' }]
}

const findings: NerSourceRow[] = [
  {
    chunk_id: 'c1',
    filename: 'doc.pdf',
    page: 3,
    chunk_text: 'Berlin is the capital.',
    entities: [{ text: 'Berlin', type: 'LOC' }]
  },
  {
    chunk_id: 'c2',
    filename: 'doc.pdf',
    page: 7,
    chunk_text: 'Alice traveled to Berlin-Mitte.',
    entities: [{ text: 'Berlin-Mitte', type: 'LOC' }]
  }
]

beforeEach(() => {
  localStorage.clear()
  useUiStore.setState({ selectedCollection: 'alpha', currentSessionId: null, previewModal: null })
  useTranslationsStore.setState({ byText: {} })
  useReportStore.setState({ activeReportId: null })
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('EntityFindingsTable', () => {
  it('shows the findings count and a table header for the selected entity', () => {
    renderWithClient(<EntityFindingsTable selected={selected} findings={findings} collection="alpha" />)
    const heading = screen.getByText(/findings for/i).parentElement!
    expect(heading).toHaveTextContent('Berlin')
    expect(heading).toHaveTextContent(/2 chunks/i)
    // Column headers present (table layout, not an accordion).
    expect(screen.getByText('Metadata')).toBeInTheDocument()
    expect(screen.getByText('Source')).toBeInTheDocument()
  })

  it('renders one row per finding with its chunk text inline', () => {
    renderWithClient(<EntityFindingsTable selected={selected} findings={findings} collection="alpha" />)
    expect(screen.getAllByTestId('entity-finding-row')).toHaveLength(2)
    // "Berlin" is highlighted (a <mark>), so assert on the trailing segment.
    expect(screen.getByText(/is the capital/)).toBeInTheDocument()
  })

  it('renders the CSV download link with the selected entity in the query string', () => {
    renderWithClient(<EntityFindingsTable selected={selected} findings={findings} collection="alpha" />)
    // The control is the download icon; "Export CSV" survives as its name.
    const link = screen.getByRole('link', { name: 'Export CSV' })
    const href = link.getAttribute('href') ?? ''
    expect(href).toContain('/collections/alpha/export/ner-sources.csv')
    expect(href).toContain('entity_text=Berlin')
    expect(href).toContain('entity_type=LOC')
    expect(link).toHaveAttribute('download')
  })

  it('Add all sends the stored translation for a translated finding and none for the rest', async () => {
    // The batch snapshots rows walked from the server, never rendered, so the
    // translation comes from the store under the row's own trimmed key (hence
    // the padded chunk_text).
    useTranslationsStore.setState({
      byText: { 'Berlin ist die Hauptstadt.': { text: 'Berlin is the capital.', target_lang: 'en', model: 'm' } }
    })
    const batchRows: NerSourceRow[] = [
      { chunk_id: 'c9', filename: 'doc.pdf', chunk_text: '  Berlin ist die Hauptstadt.  ', entities: [{ text: 'Berlin', type: 'LOC' }] },
      { chunk_id: 'c10', filename: 'doc.pdf', chunk_text: 'Ohne Übersetzung.', entities: [{ text: 'Berlin', type: 'LOC' }] }
    ]
    const captured: Record<string, unknown>[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string, init?: RequestInit) => {
        const url = String(u)
        if (url.includes('/collections/ner/sources')) {
          return { ok: true, status: 200, json: async () => ({ items: batchRows, next_cursor: null }) }
        }
        if (url.endsWith('/reports') && init?.method === 'POST') {
          return { ok: true, status: 200, json: async () => ({ id: 1, title: 'Untitled report', items: [] }) }
        }
        if (url.includes('/items/batch')) {
          captured.push(JSON.parse(String(init?.body)))
          return { ok: true, status: 200, json: async () => ({ added: 2, skipped: 0, updated: 0, item_count: 2 }) }
        }
        return { ok: true, status: 200, json: async () => ({}) }
      })
    )

    renderWithClient(
      <EntityFindingsTable
        selected={selected}
        findings={batchRows}
        collection="alpha"
        reportDedupeKeys={new Set()}
      />
    )
    await userEvent.click(screen.getByRole('button', { name: /add all findings to report/i }))

    await waitFor(() => expect(captured).toHaveLength(1))
    const items = captured[0].items as { snapshot: Record<string, unknown> }[]
    expect(items[0].snapshot.translation).toEqual({
      text: 'Berlin is the capital.',
      target_lang: 'en',
      model: 'm'
    })
    expect(items[1].snapshot).not.toHaveProperty('translation')
  })

  it('Translate all walks the section and fills the store for rows never rendered', async () => {
    // Wired to the section's page walk, not the rows on screen, so a finding
    // below the fold is translated too.
    const walked: NerSourceRow[] = [
      { chunk_id: 'c11', filename: 'doc.pdf', chunk_text: '  Berlin ist die Hauptstadt.  ', entities: [] },
      { chunk_id: 'c12', filename: 'doc.pdf', chunk_text: 'Zweiter Fund.', entities: [] }
    ]
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string, init?: RequestInit) => {
        const url = String(u)
        if (url.includes('/collections/ner/sources')) {
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

    renderWithClient(<EntityFindingsTable selected={selected} findings={[walked[0]]} collection="alpha" />)
    await userEvent.click(screen.getByRole('button', { name: /translate all findings/i }))

    await waitFor(() =>
      expect(Object.keys(useTranslationsStore.getState().byText).sort()).toEqual([
        'Berlin ist die Hauptstadt.',
        'Zweiter Fund.'
      ])
    )
  })

  it('prompts to pick an entity when none is selected', () => {
    render(<EntityFindingsTable selected={null} findings={[]} collection="alpha" />)
    expect(screen.getByText(/pick an entity/i)).toBeInTheDocument()
  })

  it('shows an empty state when the selected entity has no matched chunks', () => {
    // The header (and its "Add all" control) renders even with no findings,
    // so this case needs the query client too.
    renderWithClient(<EntityFindingsTable selected={selected} findings={[]} collection="alpha" />)
    expect(screen.getByText(/no chunks were matched/i)).toBeInTheDocument()
  })
})
