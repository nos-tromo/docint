import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { useTranslatable } from './useTranslatable'
import { useTranslationsStore, storedTranslation } from '@/stores/translations'

afterEach(() => vi.restoreAllMocks())
beforeEach(() => useTranslationsStore.setState({ byText: {} }))

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

  it('writes a successful translation to the shared store', async () => {
    mockFetchOk({ ok: true, translation: 'Hallo Welt', model: 'm', target_lang: 'de' })
    const { result } = renderHook(() => useTranslatable('Hello world'), { wrapper })

    await act(async () => {
      await result.current.toggle()
    })

    expect(storedTranslation('Hello world')).toEqual({
      text: 'Hallo Welt',
      target_lang: 'de',
      model: 'm'
    })
  })

  it('shows a translation already in the shared store without fetching', async () => {
    const fetchMock = mockFetchOk({ ok: true, translation: 'unused', model: 'm', target_lang: 'de' })
    useTranslationsStore.setState({
      byText: { 'Hello world': { text: 'Hallo Welt', target_lang: 'de', model: 'm' } }
    })
    const { result } = renderHook(() => useTranslatable('Hello world'), { wrapper })

    await act(async () => {
      await result.current.toggle()
    })

    expect(result.current.shown).toBe(true)
    expect(result.current.translation).toBe('Hallo Welt')
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('a remounted row reuses the stored translation instead of re-posting it', async () => {
    // The findings tables are virtualized: scrolling a row out unmounts it and
    // destroys its hook state. Without the shared store the translation is lost
    // and the next toggle pays for it again.
    const fetchMock = mockFetchOk({ ok: true, translation: 'Hallo Welt', model: 'm', target_lang: 'de' })
    const first = renderHook(() => useTranslatable('Hello world'), { wrapper })
    await act(async () => {
      await first.result.current.toggle()
    })
    first.unmount()

    const second = renderHook(() => useTranslatable('Hello world'), { wrapper })
    await act(async () => {
      await second.result.current.toggle()
    })

    expect(second.result.current.translation).toBe('Hallo Welt')
    expect(fetchMock).toHaveBeenCalledTimes(1)
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
