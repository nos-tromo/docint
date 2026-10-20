import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { useSessionHistory, useSessions } from './useSessions'
import { useUiStore } from '@/stores/ui'

let client: QueryClient

function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

/**
 * Answer as the server does for two owners who both have a collection `docs`:
 * each owner lists only their own session, and a session's history is found
 * only under the owner it belongs to.
 */
function serveTwoOwners(): string[] {
  const urls: string[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (u: string) => {
      const url = String(u)
      urls.push(url)
      const owner = new URL(url, 'http://localhost').searchParams.get('owner')
      let status = 200
      let body: unknown = { sessions: [{ id: `${owner}-s1` }] }
      if (url.includes('/history')) {
        const found = url.includes(`/sessions/${owner}-s1/history`)
        status = found ? 200 : 404
        body = found ? { messages: [{ role: 'assistant', content: `${owner}'s answer` }] } : { detail: 'Not found.' }
      }
      return { ok: status < 400, status, json: async () => body, text: async () => JSON.stringify(body) }
    })
  )
  return urls
}

beforeEach(() => {
  localStorage.clear()
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  useUiStore.setState({ selectedCollection: 'docs', selectedOwner: 'alice', currentSessionId: null })
})

afterEach(() => {
  vi.unstubAllGlobals()
  client.clear()
  useUiStore.setState({ selectedCollection: null, selectedOwner: null })
})

describe("another owner's collection of the same name", () => {
  it('lists the sessions of the owner whose collection is active', async () => {
    serveTwoOwners()
    const { result } = renderHook(() => useSessions(), { wrapper })
    await waitFor(() => expect(result.current.data?.sessions.map((s) => s.id)).toEqual(['alice-s1']))

    act(() => useUiStore.getState().setSelectedCollection('docs', 'bob'))

    await waitFor(() => expect(result.current.data?.sessions.map((s) => s.id)).toEqual(['bob-s1']))
  })

  it("never serves one owner's cached history in the other's collection", async () => {
    // An admin who switches owners and goes Back to the earlier session.
    const urls = serveTwoOwners()
    const { result } = renderHook(() => useSessionHistory('alice-s1'), { wrapper })
    await waitFor(() => expect(result.current.data?.messages).toHaveLength(1))

    act(() => useUiStore.getState().setSelectedCollection('docs', 'bob'))

    await waitFor(() => expect(result.current.isError).toBe(true))
    expect(result.current.data).toBeUndefined()
    expect(urls.at(-1)).toContain('owner=bob')
  })
})
