import { describe, it, expect, vi, afterEach } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { useTranslatable } from './useTranslatable'

afterEach(() => vi.restoreAllMocks())

function wrapper({ children }: { children: ReactNode }) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  })
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>
}

function mockFetchOk(body: unknown) {
  const fn = vi.fn(async () => ({ ok: true, status: 200, json: async () => body }))
  vi.stubGlobal('fetch', fn)
  return fn
}

describe('useTranslatable', () => {
  it('toggle fetches and shows the translation, reporting the nested payload once', async () => {
    mockFetchOk({ ok: true, translation: 'Hallo Welt', model: 'm', target_lang: 'de' })
    const onTranslated = vi.fn()
    const { result } = renderHook(() => useTranslatable('Hello world', onTranslated), { wrapper })

    await act(async () => {
      await result.current.toggle()
    })

    expect(result.current.shown).toBe(true)
    expect(result.current.translation).toBe('Hallo Welt')
    expect(result.current.failed).toBe(false)
    expect(onTranslated).toHaveBeenCalledTimes(1)
    expect(onTranslated).toHaveBeenCalledWith({ text: 'Hallo Welt', target_lang: 'de', model: 'm' })
  })

  it('a second toggle hides the translation; a third reuses it without re-fetching or re-reporting', async () => {
    const fetchMock = mockFetchOk({ ok: true, translation: 'Hallo Welt', model: 'm', target_lang: 'de' })
    const onTranslated = vi.fn()
    const { result } = renderHook(() => useTranslatable('Hello world', onTranslated), { wrapper })

    await act(async () => {
      await result.current.toggle()
    })
    expect(result.current.shown).toBe(true)

    await act(async () => {
      await result.current.toggle()
    })
    expect(result.current.shown).toBe(false)
    expect(result.current.translation).toBeNull()

    await act(async () => {
      await result.current.toggle()
    })
    expect(result.current.shown).toBe(true)
    expect(result.current.translation).toBe('Hallo Welt')
    // Cached: no second network round-trip, no second report to the caller.
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(onTranslated).toHaveBeenCalledTimes(1)
  })

  it('sets failed and does not call onTranslated when the fetch rejects at the transport layer', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('network')
      })
    )
    const onTranslated = vi.fn()
    const { result } = renderHook(() => useTranslatable('Hello world', onTranslated), { wrapper })

    await act(async () => {
      await result.current.toggle()
    })

    await waitFor(() => expect(result.current.failed).toBe(true))
    expect(result.current.shown).toBe(false)
    expect(result.current.translation).toBeNull()
    expect(onTranslated).not.toHaveBeenCalled()
  })
})
