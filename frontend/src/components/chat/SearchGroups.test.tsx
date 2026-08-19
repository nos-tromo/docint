import { describe, it, expect, vi } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { SearchGroups, type SearchGroupsProps } from './SearchGroups'
import type { AggregateResult, SearchHit } from '@/api/types'

const INDEX_STATUS = {
  indexed: true,
  total: 10,
  with_search_text: 10,
  missing: 0,
  complete: true
}

const HIT1: SearchHit = {
  id: 'p1',
  chunk_id: 'c1',
  filename: 'alpha.pdf',
  page: 3,
  row: null,
  preview: 'Der Parteitag beschloss den Tagesordnungspunkt.',
  entity_types: [],
  est_tokens: 1200
}

const HIT2: SearchHit = {
  id: 'p2',
  chunk_id: 'c2',
  filename: 'beta.pdf',
  page: 7,
  row: null,
  preview: 'Zweiter Abschnitt zum Parteitag.',
  entity_types: [],
  est_tokens: 1200
}

const RESULT: AggregateResult = {
  status: 'ok',
  group_by: 'author',
  total: 7,
  unassigned: 0,
  groups: [
    { value: 'acme_news', count: 5, samples: [HIT1, HIT2] },
    { value: 'beta_daily', count: 2, samples: [] }
  ],
  limit: 100,
  index_status: INDEX_STATUS
}

function renderGroups(overrides: Partial<SearchGroupsProps> = {}) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const onToggle = vi.fn()
  const utils = render(
    <QueryClientProvider client={qc}>
      <SearchGroups result={RESULT} keywords={[]} selectedTokens={{}} onToggle={onToggle} {...overrides} />
    </QueryClientProvider>
  )
  return { ...utils, onToggle }
}

describe('SearchGroups', () => {
  it('renders every group with its value and chunk count', () => {
    renderGroups()

    expect(screen.getByText('acme_news')).toBeInTheDocument()
    expect(screen.getByText('5 chunks')).toBeInTheDocument()
    expect(screen.getByText('beta_daily')).toBeInTheDocument()
    expect(screen.getByText('2 chunks')).toBeInTheDocument()
  })

  it("reveals a group's sample chunks when its header row is clicked", async () => {
    renderGroups()

    expect(screen.queryByText(/alpha\.pdf/)).toBeNull()

    const disclosure = screen.getByRole('button', { name: /show sample chunks/i })
    expect(disclosure).toHaveAttribute('aria-expanded', 'false')

    // Click the value text itself, not just the row's bounding element —
    // this is the usability bug the whole-row toggle fixes.
    await userEvent.click(screen.getByText('acme_news'))

    expect(disclosure).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByText(/alpha\.pdf/)).toBeInTheDocument()
    expect(screen.getByText(/beta\.pdf/)).toBeInTheDocument()
  })

  it('toggles the disclosure with Enter and Space on the header row', async () => {
    renderGroups()

    const disclosure = screen.getByRole('button', { name: /show sample chunks/i })
    disclosure.focus()

    await userEvent.keyboard('{Enter}')
    expect(disclosure).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByText(/alpha\.pdf/)).toBeInTheDocument()

    await userEvent.keyboard(' ')
    expect(disclosure).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByText(/alpha\.pdf/)).toBeNull()
  })

  it('calls onToggle with the sample hit when it is clicked, without re-collapsing the group', async () => {
    const { onToggle } = renderGroups()

    await userEvent.click(screen.getByRole('button', { name: /show sample chunks/i }))
    const disclosure = screen.getByRole('button', { name: /hide sample chunks/i })
    await userEvent.click(await screen.findByRole('button', { name: /alpha\.pdf/i }))

    expect(onToggle).toHaveBeenCalledWith(HIT1)
    // The sample tile is a sibling of the header, not nested inside it, so
    // selecting it must not bubble up into the group's own toggle.
    expect(disclosure).toHaveAttribute('aria-expanded', 'true')
  })

  it('renders a non-interactive header for a group with no samples', () => {
    renderGroups()

    const row = screen.getByText('beta_daily').closest('li')
    expect(row).not.toBeNull()
    expect(within(row as HTMLElement).queryByRole('button')).toBeNull()

    const header = screen.getByText('beta_daily').closest('div')
    expect(header).not.toBeNull()
    expect(header).not.toHaveAttribute('tabindex')
  })

  it('shows the empty state when there are no groups', () => {
    renderGroups({ result: { ...RESULT, groups: [] } })

    expect(screen.getByTestId('search-no-groups')).toHaveTextContent(/no chunk matches/i)
    expect(screen.queryByTestId('search-groups')).toBeNull()
  })
})
