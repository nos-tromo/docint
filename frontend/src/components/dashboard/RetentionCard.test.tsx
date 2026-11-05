import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { RetentionCard } from './RetentionCard'
import type { RetentionItem } from '@/api/types'

function item(
  kind: RetentionItem['kind'],
  name: string,
  collection: string,
  expires_at: string,
  owner: string | null = null
): RetentionItem {
  return { kind, collection, owner, id: `${kind}-${name}`, name, expires_at, warning: true }
}

/** A backend with a retention period of `days` (or off) whose notice lists `items`. */
function retentionFetch(days: number | null, items: RetentionItem[], notice = 30) {
  return vi.fn(async (input: RequestInfo | URL) => {
    const path = typeof input === 'string' ? input : input.toString()
    const body = path.includes('/collections/retention')
      ? { period_days: days, notice_days: notice, items }
      : { language: 'en', document_retention_days: days, retention_notice_days: notice }
    return { ok: true, status: 200, json: async () => body }
  })
}

function renderCard() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <RetentionCard />
    </QueryClientProvider>
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('RetentionCard', () => {
  it("renders the owner's due items grouped by collection", async () => {
    vi.stubGlobal(
      'fetch',
      retentionFetch(365, [
        item('document', 'memo.pdf', 'projekt-alpha', '2026-11-20T06:45:00Z'),
        item('session', 'What did the memo say?', 'laufend', '2026-11-22T08:00:00Z'),
        item('report', 'Weekly', 'projekt-alpha', '2026-11-25T08:00:00Z')
      ])
    )
    renderCard()

    const card = await screen.findByRole('region', { name: /automatic deletion/i })
    expect(card).toHaveTextContent(
      'Documents are deleted 365 days after they were added. Chats, reports and extracts built from them are deleted with the earliest of their sources.'
    )
    const groups = within(card).getAllByRole('group')
    expect(groups.map((g) => within(g).getByRole('heading').textContent)).toEqual(['projekt-alpha', 'laufend'])
    expect(within(groups[0]).getAllByRole('listitem').map((li) => li.textContent)).toEqual([
      expect.stringContaining('memo.pdf'),
      expect.stringContaining('Weekly')
    ])
  })

  it('shows kind, name and date', async () => {
    vi.stubGlobal('fetch', retentionFetch(365, [item('session', 'What did the memo say?', 'laufend', '2026-11-22T08:00:00Z')]))
    renderCard()

    const row = await screen.findByRole('listitem')
    expect(row).toHaveTextContent('Chat')
    expect(row).toHaveTextContent('What did the memo say?')
    expect(row).toHaveTextContent('Deleted on Nov 22, 2026')
  })

  it("names the owner of another user's collection", async () => {
    vi.stubGlobal('fetch', retentionFetch(365, [item('document', 'a.pdf', 'shared-notes', '2026-11-22T08:00:00Z', 'a.beispiel')]))
    renderCard()

    expect(await screen.findByRole('heading', { name: /shared-notes \(owner: a\.beispiel\)/ })).toBeInTheDocument()
  })

  it('renders nothing when retention is off, and asks for nothing', async () => {
    const fetchMock = retentionFetch(null, [])
    vi.stubGlobal('fetch', fetchMock)
    const { container } = renderCard()

    await waitFor(() => expect(fetchMock.mock.calls.some((c) => String(c[0]).includes('/config'))).toBe(true))
    expect(container).toBeEmptyDOMElement()
    expect(fetchMock.mock.calls.some((c) => String(c[0]).includes('/collections/retention'))).toBe(false)
  })

  it('shows the empty state with the notice days', async () => {
    vi.stubGlobal('fetch', retentionFetch(365, [], 14))
    renderCard()

    const card = await screen.findByRole('region', { name: /automatic deletion/i })
    expect(card).toHaveTextContent('Nothing is due for deletion in the next 14 days.')
    expect(within(card).queryByRole('list')).not.toBeInTheDocument()
  })

  it('shows the soonest few and counts the rest', async () => {
    const items = Array.from({ length: 10 }, (_, i) => item('document', `doc-${i}.pdf`, 'alpha', `2026-11-${10 + i}T00:00:00Z`))
    vi.stubGlobal('fetch', retentionFetch(365, items))
    renderCard()

    expect(await screen.findByText('doc-7.pdf')).toBeInTheDocument()
    expect(screen.queryByText('doc-8.pdf')).not.toBeInTheDocument()
    expect(screen.getByText(/2 more/)).toBeInTheDocument()
  })
})
