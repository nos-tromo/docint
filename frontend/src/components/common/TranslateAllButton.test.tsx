import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { TranslateAllButton } from './TranslateAllButton'
import { useTranslationsStore } from '@/stores/translations'

interface Row {
  chunk_text: string
}

const textOf = (row: Row) => row.chunk_text.trim()

interface Deferred {
  resolve: () => void
}

/**
 * Stub fetch for the calls this flow makes. `/translate` is branched on
 * explicitly: a catch-all `{}` reads as `ok: false`, so the breaker would trip
 * and the tests would pass for the wrong reason.
 */
function stubFetch(opts: {
  calls: string[]
  fail?: (text: string) => boolean
  gate?: Deferred[]
} = { calls: [] }) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (u: string, init?: RequestInit) => {
      const url = String(u)
      if (url.includes('/translate')) {
        const text = String(JSON.parse(String(init?.body)).text)
        opts.calls.push(text)
        if (opts.gate) {
          await new Promise<void>((resolve) => opts.gate?.push({ resolve }))
        }
        if (opts.fail?.(text)) {
          return {
            ok: true,
            status: 200,
            json: async () => ({ ok: false, translation: null, model: 'm', target_lang: 'de', error: 'unavailable' })
          }
        }
        return {
          ok: true,
          status: 200,
          json: async () => ({ ok: true, translation: `de:${text}`, model: 'm', target_lang: 'de' })
        }
      }
      return { ok: true, status: 200, json: async () => ({}) }
    })
  )
}

function renderButton(rows: Row[], qc: QueryClient) {
  return render(
    <QueryClientProvider client={qc}>
      <TranslateAllButton fetchAll={async () => rows} textOf={textOf} hasRows={rows.length > 0} />
    </QueryClientProvider>
  )
}

const client = () => new QueryClient({ defaultOptions: { queries: { retry: false } } })
const clickTranslateAll = () => userEvent.click(screen.getByRole('button', { name: /translate all findings/i }))

beforeEach(() => {
  // Keyed by text app-wide: one test's success would otherwise satisfy the
  // next one's already-translated check.
  useTranslationsStore.setState({ byText: {} })
  vi.stubGlobal('confirm', vi.fn(() => true))
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('TranslateAllButton', () => {
  it('translates every finding and files each one in the shared store', async () => {
    const calls: string[] = []
    stubFetch({ calls })

    renderButton([{ chunk_text: 'eins' }, { chunk_text: 'zwei' }], client())
    await clickTranslateAll()

    await waitFor(() => expect(screen.getByTestId('translate-all-message')).toHaveTextContent('2 translated'))
    expect(calls.sort()).toEqual(['eins', 'zwei'])
    expect(useTranslationsStore.getState().byText['eins'].text).toBe('de:eins')
    expect(useTranslationsStore.getState().byText['zwei'].text).toBe('de:zwei')
  })

  it('posts identical text once, however many rows carry it', async () => {
    const calls: string[] = []
    stubFetch({ calls })

    renderButton([{ chunk_text: 'same' }, { chunk_text: 'same' }, { chunk_text: '  same  ' }], client())
    await clickTranslateAll()

    await waitFor(() => expect(screen.getByTestId('translate-all-message')).toBeInTheDocument())
    expect(calls).toEqual(['same'])
  })

  it('skips text already translated this session and says so', async () => {
    const calls: string[] = []
    stubFetch({ calls })
    useTranslationsStore.setState({ byText: { eins: { text: 'de:eins', target_lang: 'de', model: 'm' } } })

    renderButton([{ chunk_text: 'eins' }, { chunk_text: 'zwei' }], client())
    await clickTranslateAll()

    await waitFor(() => expect(screen.getByTestId('translate-all-message')).toHaveTextContent('1 translated'))
    expect(calls).toEqual(['zwei'])
    expect(screen.getByTestId('translate-all-message')).toHaveTextContent('1 already translated')
  })

  it('reports a fully translated section without calling the endpoint', async () => {
    const calls: string[] = []
    stubFetch({ calls })
    useTranslationsStore.setState({ byText: { eins: { text: 'de:eins', target_lang: 'de', model: 'm' } } })

    renderButton([{ chunk_text: 'eins' }], client())
    await clickTranslateAll()

    await waitFor(() =>
      expect(screen.getByTestId('translate-all-message')).toHaveTextContent(/already translated/i)
    )
    expect(calls).toEqual([])
  })

  it('refuses a section larger than the server cap without translating anything', async () => {
    const calls: string[] = []
    stubFetch({ calls })
    const qc = client()
    qc.setQueryData(['app-config'], { report_batch_max_items: 2, language: 'en' })

    renderButton([{ chunk_text: 'a' }, { chunk_text: 'b' }, { chunk_text: 'c' }], qc)
    await clickTranslateAll()

    await waitFor(() => expect(screen.getByTestId('translate-all-message')).toHaveTextContent(/too many findings/i))
    expect(calls).toEqual([])
  })

  it('stops after three consecutive failures instead of grinding through the section', async () => {
    // The point of the breaker: a dead model must not cost one call per finding.
    const calls: string[] = []
    stubFetch({ calls, fail: () => true })
    const rows = Array.from({ length: 12 }, (_, i) => ({ chunk_text: `t${i}` }))

    renderButton(rows, client())
    await clickTranslateAll()

    await waitFor(() => expect(screen.getByTestId('translate-all-message')).toHaveTextContent(/unavailable/i))
    expect(calls.length).toBeLessThan(rows.length)
    expect(screen.getByRole('button', { name: /retry translating/i })).toBeInTheDocument()
  })

  it('counts a failure that is not an outage and still reports what got through', async () => {
    const calls: string[] = []
    stubFetch({ calls, fail: (text) => text === 'b' })

    renderButton([{ chunk_text: 'a' }, { chunk_text: 'b' }, { chunk_text: 'c' }], client())
    await clickTranslateAll()

    await waitFor(() =>
      expect(screen.getByTestId('translate-all-message')).toHaveTextContent('2 translated, 1 could not be translated')
    )
    expect(useTranslationsStore.getState().byText['b']).toBeUndefined()
  })

  it('shows progress and a stop control while it runs, and stops when asked', async () => {
    const calls: string[] = []
    const gate: Deferred[] = []
    stubFetch({ calls, gate })
    const rows = Array.from({ length: 8 }, (_, i) => ({ chunk_text: `t${i}` }))

    renderButton(rows, client())
    await clickTranslateAll()

    // Concurrency 3: three calls are in flight and the button now stops the run.
    await waitFor(() => expect(calls).toHaveLength(3))
    const stopButton = await screen.findByRole('button', { name: /stop translating/i })
    expect(screen.getByTestId('translate-all-message')).toHaveTextContent('0 of 8 translated')

    await userEvent.click(stopButton)
    gate.forEach((d) => d.resolve())

    await waitFor(() => expect(screen.getByTestId('translate-all-message')).toHaveTextContent(/stopped/i))
    // The three in flight finish and file their answers; nothing further starts.
    expect(calls).toHaveLength(3)
    expect(Object.keys(useTranslationsStore.getState().byText)).toHaveLength(3)
  })

  it('asks before translating a large section and translates nothing when refused', async () => {
    const calls: string[] = []
    stubFetch({ calls })
    vi.stubGlobal('confirm', vi.fn(() => false))
    const rows = Array.from({ length: 101 }, (_, i) => ({ chunk_text: `t${i}` }))

    renderButton(rows, client())
    await clickTranslateAll()

    await waitFor(() => expect(window.confirm).toHaveBeenCalled())
    expect(calls).toEqual([])
  })

  it('is disabled when the section has no rows', () => {
    stubFetch({ calls: [] })
    renderButton([], client())
    expect(screen.getByRole('button', { name: /translate all findings/i })).toBeDisabled()
  })
})
