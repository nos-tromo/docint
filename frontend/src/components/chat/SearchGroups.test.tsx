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

  it("reveals a group's sample chunks only once its disclosure is opened", async () => {
    renderGroups()

    expect(screen.queryByText(/alpha\.pdf/)).toBeNull()

    const disclosure = screen.getByRole('button', { name: /show sample chunks/i })
    expect(disclosure).toHaveAttribute('aria-expanded', 'false')

    await userEvent.click(disclosure)

    expect(disclosure).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByText(/alpha\.pdf/)).toBeInTheDocument()
    expect(screen.getByText(/beta\.pdf/)).toBeInTheDocument()
  })

  it('calls onToggle with the sample hit when it is clicked', async () => {
    const { onToggle } = renderGroups()

    await userEvent.click(screen.getByRole('button', { name: /show sample chunks/i }))
    await userEvent.click(await screen.findByRole('button', { name: /alpha\.pdf/i }))

    expect(onToggle).toHaveBeenCalledWith(HIT1)
  })

  it('renders no disclosure control for a group with no samples', () => {
    renderGroups()

    const row = screen.getByText('beta_daily').closest('li')
    expect(row).not.toBeNull()
    expect(within(row as HTMLElement).queryByRole('button')).toBeNull()
  })

  it('shows the empty state when there are no groups', () => {
    renderGroups({ result: { ...RESULT, groups: [] } })

    expect(screen.getByTestId('search-no-groups')).toHaveTextContent(/no chunk matches/i)
    expect(screen.queryByTestId('search-groups')).toBeNull()
  })
})
