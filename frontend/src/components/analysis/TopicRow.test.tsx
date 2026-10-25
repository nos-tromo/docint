import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { TopicRow } from './TopicRow'
import { useUiStore } from '@/stores/ui'
import type { TopicSummary } from '@/api/topics'

const RUN_ID = '20260102-030405-a1b2c3'

const TOPIC: TopicSummary = {
  id: 4,
  label: 'Hives and honey',
  description: 'Keeping bees through the seasons.',
  keywords: [
    ['hive', 0.4],
    ['honey', 0.3]
  ],
  size: 42,
  share: 0.35,
  origin: 'build',
  added_at: '2026-01-02T03:04:05+00:00',
  first_seen: '2024-05-01T08:00:00+00:00',
  last_seen: '2024-05-01T18:00:00+00:00'
}

let client: QueryClient

/** Stub the three topic reads with small invented pages. */
function stubApi() {
  const respond = (body: unknown, status = 200) => ({
    ok: status < 400,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body)
  })
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input), 'http://localhost')
    if (url.pathname.endsWith('/topics/4/evidence')) {
      return respond({
        topic: 4,
        sources: [{ id: 'p1', filename: 'field-notes.txt', page: 2, text: 'The colony swarmed in late spring.' }]
      })
    }
    if (url.pathname.endsWith('/topics/4/documents')) {
      return respond({
        documents: [
          { file_hash: 'h1', filename: 'field-notes.txt', count: 30 },
          { file_hash: 'h2', filename: 'season-log.txt', count: 12 }
        ],
        more: true
      })
    }
    if (url.pathname.endsWith('/topics/4/chunks')) {
      return respond(
        url.searchParams.get('cursor')
          ? { items: [{ id: 'c2', filename: 'season-log.txt', text: 'Frames were checked for brood.' }], next_cursor: null }
          : { items: [{ id: 'c1', filename: 'field-notes.txt', text: 'Smoke calms the workers.' }], next_cursor: 'n1' }
      )
    }
    return respond({})
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

/** The topic reads the stub saw, by path suffix. */
function topicReads(fetchMock: ReturnType<typeof stubApi>) {
  return fetchMock.mock.calls
    .map(([input]) => new URL(String(input), 'http://localhost'))
    .filter((url) => url.pathname.includes('/topics/'))
    .map((url) => `${url.pathname.split('/topics/')[1]}${url.searchParams.get('cursor') ? `?cursor=${url.searchParams.get('cursor')}` : ''}`)
}

function renderRow(props: { expanded: boolean; onToggle?: () => void; onAsk?: () => void }) {
  const { onToggle = vi.fn(), onAsk = vi.fn() } = props
  const ui = (expanded: boolean) => (
    <QueryClientProvider client={client}>
      <ul>
        <TopicRow topic={TOPIC} runId={RUN_ID} expanded={expanded} onToggle={onToggle} onAsk={onAsk} />
      </ul>
    </QueryClientProvider>
  )
  const view = render(ui(props.expanded))
  return { ...view, setExpanded: (expanded: boolean) => view.rerender(ui(expanded)), onToggle, onAsk }
}

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  useUiStore.setState({ selectedCollection: 'projekt-alpha', selectedOwner: null })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('TopicRow', () => {
  it('loads evidence, documents and passages only when expanded', async () => {
    const fetchMock = stubApi()
    const { setExpanded } = renderRow({ expanded: false })

    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(topicReads(fetchMock)).toEqual([])
    expect(screen.queryByText('Keeping bees through the seasons.')).not.toBeInTheDocument()

    setExpanded(true)

    expect(screen.getByText('Keeping bees through the seasons.')).toBeInTheDocument()
    const evidence = screen.getByRole('region', { name: 'Evidence' })
    expect(await within(evidence).findByText('The colony swarmed in late spring.')).toBeInTheDocument()
    expect(within(evidence).getByText('field-notes.txt · page 2')).toBeInTheDocument()

    const documents = screen.getByRole('region', { name: 'Documents' })
    expect(await within(documents).findByText('season-log.txt')).toBeInTheDocument()
    expect(within(documents).getByText('12')).toBeInTheDocument()
    expect(within(documents).getByText('More documents are not listed.')).toBeInTheDocument()

    const passages = screen.getByRole('region', { name: 'All passages' })
    expect(await within(passages).findByText('Smoke calms the workers.')).toBeInTheDocument()
    await userEvent.click(within(passages).getByRole('button', { name: 'Load more' }))
    expect(await within(passages).findByText('Frames were checked for brood.')).toBeInTheDocument()
    await waitFor(() => expect(within(passages).queryByRole('button', { name: 'Load more' })).not.toBeInTheDocument())

    expect(topicReads(fetchMock).sort()).toEqual(['4/chunks', '4/chunks?cursor=n1', '4/documents', '4/evidence'])
  })

  it('names the disclosure after the topic and what the next click does', () => {
    stubApi()
    const { setExpanded, onToggle } = renderRow({ expanded: false })

    const toggle = screen.getByRole('button', { name: 'Show Hives and honey' })
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    const panelId = toggle.getAttribute('aria-controls')
    expect(panelId).toBeTruthy()
    expect(document.getElementById(panelId!)).not.toBeNull()

    setExpanded(true)
    expect(screen.getByRole('button', { name: 'Hide Hives and honey' })).toHaveAttribute('aria-expanded', 'true')

    screen.getByRole('button', { name: 'Hide Hives and honey' }).click()
    expect(onToggle).toHaveBeenCalledTimes(1)
  })

  it('shows a one-day span as a single date', () => {
    stubApi()
    renderRow({ expanded: false })

    expect(screen.getByText('May 1, 2024')).toBeInTheDocument()
  })

  it('calls onAsk with the topic and run', async () => {
    stubApi()
    const { onAsk } = renderRow({ expanded: false })

    await userEvent.click(screen.getByRole('button', { name: 'Ask in chat' }))

    expect(onAsk).toHaveBeenCalledWith(TOPIC, RUN_ID)
  })
})
