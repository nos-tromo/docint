import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { useHateSpeechPages, useNerStats } from './useNer'
import { useUiStore } from '@/stores/ui'

afterEach(() => vi.restoreAllMocks())

beforeEach(() => {
  useUiStore.setState({ selectedCollection: null, entityMergeMode: 'resolved' })
})

function mockFetch(body: unknown) {
  const fn = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => body,
    text: async () => JSON.stringify(body)
  })
  vi.stubGlobal('fetch', fn)
  return fn
}

function wrapper({ children }: { children: ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>
}

describe('analysis hooks thread the selected collection into requests', () => {
  it('useNerStats sends the store collection as a query param', async () => {
    const fetchMock = mockFetch({})
    useUiStore.setState({ selectedCollection: 'docs' })

    renderHook(() => useNerStats({ top_k: 10, include_relations: false }), { wrapper })

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([u]) =>
        String(u).includes('/collections/ner/stats')
      )
      expect(call).toBeDefined()
      expect(String(call![0])).toContain('collection=docs')
    })
  })

  it('useHateSpeechPages sends the store collection as a query param', async () => {
    const fetchMock = mockFetch({ items: [], next_cursor: null })
    useUiStore.setState({ selectedCollection: 'docs' })

    renderHook(() => useHateSpeechPages(), { wrapper })

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([u]) =>
        String(u).includes('/collections/hate-speech')
      )
      expect(call).toBeDefined()
      expect(String(call![0])).toContain('collection=docs')
    })
  })

  it('does not fire NER requests until a collection is selected', async () => {
    const fetchMock = mockFetch({})

    renderHook(() => useNerStats({ top_k: 10 }), { wrapper })

    // The hook is gated on `enabled: !!collection`; with none selected it
    // must not hit the network at all.
    await new Promise((r) => setTimeout(r, 20))
    expect(
      fetchMock.mock.calls.some(([u]) => String(u).includes('/collections/ner/stats'))
    ).toBe(false)
  })
})
