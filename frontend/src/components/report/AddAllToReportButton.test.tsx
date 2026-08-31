import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AddAllToReportButton } from './AddAllToReportButton'
import { reportKey } from '@/hooks/useReports'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'
import type { Report, ReportItemInput } from '@/api/types'

interface Row {
  chunk_id: string
  translation?: { text: string; target_lang: string; model: string }
}

const toItem = (row: Row): ReportItemInput => ({
  artifact_type: 'entity_finding',
  dedupe_key: `entity:${row.chunk_id}`,
  snapshot: {
    chunk_id: row.chunk_id,
    // Conditional exactly as the real builders spread it, so a row with no
    // translation produces a snapshot with no such key.
    ...(row.translation ? { translation: row.translation } : {})
  }
})

/** A report holding one item, whose snapshot the tests vary. */
function reportHolding(snapshot: Record<string, unknown>): Report {
  return {
    id: 1,
    title: 'Case',
    collection_name: 'docs',
    operator: null,
    reference_number: null,
    show_toc: true,
    show_collection_overview: true,
    session_id: null,
    created_at: null,
    updated_at: null,
    item_count: 1,
    collection_overview: null,
    items: [
      {
        id: 5,
        artifact_type: 'entity_finding',
        dedupe_key: 'entity:c1',
        position: 0,
        note: null,
        snapshot,
        created_at: null
      }
    ]
  }
}

interface Captured {
  url: string
  body: Record<string, unknown>
}

/**
 * Stub fetch for the two POSTs this flow makes: report creation and the batch
 * add. Rows are supplied directly rather than over the wire — the page walk
 * itself is `fetchAllPages`' own test.
 */
function stubFetch(
  captured: Captured[],
  batch: { added: number; skipped: number; updated?: number } = { added: 0, skipped: 0 },
  report?: Report
) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (u: string, init?: RequestInit) => {
      const url = String(u)
      if (init?.method === 'POST' && init.body) captured.push({ url, body: JSON.parse(String(init.body)) })
      if (url.endsWith('/reports') && init?.method === 'POST') {
        return { ok: true, status: 200, json: async () => ({ id: 1, title: 'Untitled report', items: [] }) }
      }
      if (url.includes('/items/batch')) {
        const items = captured[captured.length - 1].body.items as unknown[]
        return {
          ok: true,
          status: 200,
          json: async () => ({
            added: batch.added || items.length,
            skipped: batch.skipped,
            updated: batch.updated ?? 0,
            item_count: items.length
          })
        }
      }
      if (report && /\/reports\/\d+$/.test(url.split('?')[0])) {
        return { ok: true, status: 200, json: async () => report }
      }
      return { ok: true, status: 200, json: async () => ({}) }
    })
  )
}

function renderButton(rows: Row[], qc: QueryClient) {
  return render(
    <QueryClientProvider client={qc}>
      <AddAllToReportButton fetchAll={async () => rows} toItem={toItem} hasRows={rows.length > 0} />
    </QueryClientProvider>
  )
}

const client = () => new QueryClient({ defaultOptions: { queries: { retry: false } } })
const clickAddAll = () => userEvent.click(screen.getByRole('button', { name: /add all findings to report/i }))

beforeEach(() => {
  localStorage.clear()
  useReportStore.setState({ activeReportId: null })
  useUiStore.setState({ selectedCollection: 'docs' })
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('AddAllToReportButton', () => {
  it('posts every fetched row as ONE batch request', async () => {
    // The point of the batch route: N findings must not become N round-trips.
    const captured: Captured[] = []
    stubFetch(captured)
    useReportStore.setState({ activeReportId: 1 })

    renderButton([{ chunk_id: 'c1' }, { chunk_id: 'c2' }, { chunk_id: 'c3' }], client())
    await clickAddAll()

    await waitFor(() => expect(captured.some((c) => c.url.includes('/items/batch'))).toBe(true))
    const batches = captured.filter((c) => c.url.includes('/items/batch'))
    expect(batches).toHaveLength(1)
    expect((batches[0].body.items as ReportItemInput[]).map((i) => i.dedupe_key)).toEqual([
      'entity:c1',
      'entity:c2',
      'entity:c3'
    ])
    expect(batches[0].body.collection).toBe('docs')
  })

  it('auto-creates a report when none is active, then adds into it', async () => {
    const captured: Captured[] = []
    stubFetch(captured)

    renderButton([{ chunk_id: 'c1' }], client())
    await clickAddAll()

    await waitFor(() => expect(captured.some((c) => c.url.includes('/items/batch'))).toBe(true))
    expect(captured[0].url).toMatch(/\/reports$/)
    expect(captured[1].url).toContain('/reports/1/items/batch')
    expect(useReportStore.getState().activeReportId).toBe(1)
  })

  it('skips rows already in the report instead of resending them', async () => {
    const captured: Captured[] = []
    const qc = client()
    useReportStore.setState({ activeReportId: 1 })
    const existing: Report = {
      id: 1,
      title: 'Case',
      collection_name: 'docs',
      operator: null,
      reference_number: null,
      show_toc: true,
      show_collection_overview: true,
      session_id: null,
      created_at: null,
      updated_at: null,
      item_count: 1,
      collection_overview: null,
      items: [
        {
          id: 5,
          artifact_type: 'entity_finding',
          dedupe_key: 'entity:c1',
          position: 0,
          note: null,
          snapshot: {},
          created_at: null
        }
      ]
    }
    stubFetch(captured, { added: 1, skipped: 0 }, existing)
    qc.setQueryData<Report>(reportKey(1), existing)

    renderButton([{ chunk_id: 'c1' }, { chunk_id: 'c2' }], qc)
    await clickAddAll()

    await waitFor(() => expect(captured.some((c) => c.url.includes('/items/batch'))).toBe(true))
    const sent = captured.find((c) => c.url.includes('/items/batch'))!.body.items as ReportItemInput[]
    expect(sent.map((i) => i.dedupe_key)).toEqual(['entity:c2'])
  })

  it('resends an item the report holds when it can gain a translation', async () => {
    // The one exception to "duplicates are never sent": a stored snapshot with
    // no translation, and a fresh one that has it. Without this a report
    // collected before its corpus was translated could never become readable.
    const captured: Captured[] = []
    const qc = client()
    useReportStore.setState({ activeReportId: 1 })
    const existing = reportHolding({ chunk_id: 'c1' })
    stubFetch(captured, { added: 0, skipped: 0, updated: 1 }, existing)
    qc.setQueryData<Report>(reportKey(1), existing)
    const translation = { text: 'translated', target_lang: 'en', model: 'm' }

    renderButton([{ chunk_id: 'c1', translation }], qc)
    await clickAddAll()

    await waitFor(() => expect(captured.some((c) => c.url.includes('/items/batch'))).toBe(true))
    const sent = captured.find((c) => c.url.includes('/items/batch'))!.body.items as ReportItemInput[]
    expect(sent.map((i) => i.dedupe_key)).toEqual(['entity:c1'])
    expect(sent[0].snapshot.translation).toEqual(translation)
  })

  it('does not resend an item whose stored snapshot already carries a translation', async () => {
    const captured: Captured[] = []
    const qc = client()
    useReportStore.setState({ activeReportId: 1 })
    const existing = reportHolding({
      chunk_id: 'c1',
      translation: { text: 'stored', target_lang: 'en', model: 'm' }
    })
    stubFetch(captured, { added: 0, skipped: 0 }, existing)
    qc.setQueryData<Report>(reportKey(1), existing)

    renderButton([{ chunk_id: 'c1', translation: { text: 'fresher', target_lang: 'en', model: 'm' } }], qc)
    await clickAddAll()

    await waitFor(() => expect(screen.getByTestId('add-all-message')).toBeInTheDocument())
    expect(captured.some((c) => c.url.includes('/items/batch'))).toBe(false)
  })

  it('states the translations a batch backfilled, not only what it added', async () => {
    const captured: Captured[] = []
    const qc = client()
    useReportStore.setState({ activeReportId: 1 })
    const existing = reportHolding({ chunk_id: 'c1' })
    stubFetch(captured, { added: 1, skipped: 0, updated: 1 }, existing)
    qc.setQueryData<Report>(reportKey(1), existing)
    const translation = { text: 'translated', target_lang: 'en', model: 'm' }

    renderButton([{ chunk_id: 'c1', translation }, { chunk_id: 'c2', translation }], qc)
    await clickAddAll()

    await waitFor(() =>
      expect(screen.getByTestId('add-all-message')).toHaveTextContent(/1 added, 1 translations added/i)
    )
  })

  it('reports the outcome, counting locally skipped rows too', async () => {
    const captured: Captured[] = []
    stubFetch(captured, { added: 1, skipped: 0 })
    useReportStore.setState({ activeReportId: 1 })

    // The same chunk twice in the fetched pages is one item, one skip.
    renderButton([{ chunk_id: 'c1' }, { chunk_id: 'c1' }], client())
    await clickAddAll()

    await waitFor(() => expect(screen.getByTestId('add-all-message')).toHaveTextContent(/1 added, 1 already/i))
  })

  it('asks before adding a large batch and adds nothing when refused', async () => {
    const captured: Captured[] = []
    stubFetch(captured)
    useReportStore.setState({ activeReportId: 1 })
    const confirmSpy = vi.fn(() => false)
    vi.stubGlobal('confirm', confirmSpy)
    const rows = Array.from({ length: 150 }, (_, i) => ({ chunk_id: `c${i}` }))

    renderButton(rows, client())
    await clickAddAll()

    await waitFor(() => expect(confirmSpy).toHaveBeenCalledWith(expect.stringContaining('150')))
    expect(captured.some((c) => c.url.includes('/items/batch'))).toBe(false)
  })

  it('adds the large batch once confirmed', async () => {
    const captured: Captured[] = []
    stubFetch(captured)
    useReportStore.setState({ activeReportId: 1 })
    vi.stubGlobal('confirm', vi.fn(() => true))
    const rows = Array.from({ length: 150 }, (_, i) => ({ chunk_id: `c${i}` }))

    renderButton(rows, client())
    await clickAddAll()

    await waitFor(() => expect(captured.some((c) => c.url.includes('/items/batch'))).toBe(true))
    expect((captured.find((c) => c.url.includes('/items/batch'))!.body.items as unknown[]).length).toBe(150)
  })

  it('refuses a batch above the server cap without posting it', async () => {
    const captured: Captured[] = []
    stubFetch(captured)
    useReportStore.setState({ activeReportId: 1 })
    const confirmSpy = vi.fn(() => true)
    vi.stubGlobal('confirm', confirmSpy)
    const rows = Array.from({ length: 2001 }, (_, i) => ({ chunk_id: `c${i}` }))

    renderButton(rows, client())
    await clickAddAll()

    await waitFor(() => expect(screen.getByTestId('add-all-message')).toHaveTextContent(/too many findings/i))
    expect(captured.some((c) => c.url.includes('/items/batch'))).toBe(false)
    expect(confirmSpy).not.toHaveBeenCalled()
  })

  it('walks one row past the cap so an overflowing section is detectable', async () => {
    // fetchAllPages truncates silently at its maxItems. Asking for cap + 1 is
    // what turns "there are more findings than we can add" into an observable
    // fact instead of a batch that quietly carries an arbitrary sample.
    const captured: Captured[] = []
    stubFetch(captured)
    useReportStore.setState({ activeReportId: 1 })
    const fetchAll = vi.fn(async () => [{ chunk_id: 'c1' }])
    const qc = client()
    render(
      <QueryClientProvider client={qc}>
        <AddAllToReportButton fetchAll={fetchAll} toItem={toItem} hasRows />
      </QueryClientProvider>
    )

    await clickAddAll()

    await waitFor(() => expect(fetchAll).toHaveBeenCalledWith(2001))
  })

  it('refuses against the cap the server advertises, not the shipped default', async () => {
    const captured: Captured[] = []
    stubFetch(captured)
    useReportStore.setState({ activeReportId: 1 })
    const qc = client()
    qc.setQueryData(['app-config'], {
      graph_top_k: 80,
      graph_max_top_k: 500,
      collection_timeout: 120,
      max_upload_bytes: 1024,
      report_batch_max_items: 3,
      language: 'en'
    })

    renderButton([{ chunk_id: 'c1' }, { chunk_id: 'c2' }, { chunk_id: 'c3' }, { chunk_id: 'c4' }], qc)
    await clickAddAll()

    await waitFor(() => expect(screen.getByTestId('add-all-message')).toHaveTextContent(/too many findings/i))
    expect(screen.getByTestId('add-all-message')).toHaveTextContent('3')
    expect(captured.some((c) => c.url.includes('/items/batch'))).toBe(false)
  })

  it('explains a 413 as a size problem instead of offering a retry', async () => {
    // nginx refuses an oversize body before FastAPI sees it. Retrying the same
    // body cannot succeed, so this must not wear the retry affordance.
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string) => {
        const url = String(u)
        if (url.endsWith('/reports')) {
          return { ok: true, status: 200, json: async () => ({ id: 1, title: 'Untitled report', items: [] }) }
        }
        if (url.includes('/items/batch')) {
          return new Response('<html><title>413 Request Entity Too Large</title></html>', { status: 413 })
        }
        return { ok: true, status: 200, json: async () => ({}) }
      })
    )
    useReportStore.setState({ activeReportId: 1 })

    renderButton([{ chunk_id: 'c1' }], client())
    await clickAddAll()

    await waitFor(() => expect(screen.getByTestId('add-all-message')).toHaveTextContent(/larger than the server accepts/i))
    expect(screen.queryByRole('button', { name: /retry/i })).not.toBeInTheDocument()
  })

  it('shows a retry state when the batch fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: false, status: 500, json: async () => ({ detail: 'boom' }) }))
    )
    useReportStore.setState({ activeReportId: 1 })

    renderButton([{ chunk_id: 'c1' }], client())
    await clickAddAll()

    await waitFor(() => expect(screen.getByTestId('add-all-message')).toHaveTextContent(/could not add/i))
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument()
  })

  it('is disabled when the section has no findings', () => {
    renderButton([], client())
    expect(screen.getByRole('button', { name: /add all findings to report/i })).toBeDisabled()
  })
})
