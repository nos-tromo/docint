import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { TranslateControl } from './TranslateControl'

afterEach(() => vi.restoreAllMocks())

function renderControl(onTranslated?: (t: unknown) => void) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  })
  return render(
    <QueryClientProvider client={qc}>
      <div className="group">
        <TranslateControl rawText="Hello world" onTranslated={onTranslated} />
      </div>
    </QueryClientProvider>
  )
}

describe('TranslateControl', () => {
  it('renders the original text', () => {
    renderControl()
    expect(screen.getByText('Hello world')).toBeInTheDocument()
  })

  it('swaps the translation in for the original in place, and reports it up', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({ ok: true, translation: 'Hallo Welt', model: 'm', target_lang: 'de' })
      }))
    )
    const onTranslated = vi.fn()
    renderControl(onTranslated)
    await userEvent.click(screen.getByRole('button', { name: /translate/i }))
    await waitFor(() => expect(screen.getByText('Hallo Welt')).toBeInTheDocument())
    // In-place swap: the original is gone, only the translation is rendered.
    expect(screen.queryByText('Hello world')).not.toBeInTheDocument()
    expect(screen.getByText(/^translation$/i)).toBeInTheDocument()
    expect(onTranslated).toHaveBeenCalledWith({ text: 'Hallo Welt', target_lang: 'de', model: 'm' })
  })

  it('toggles back to the original on a second click ("Show original")', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({ ok: true, translation: 'Hallo Welt', model: 'm', target_lang: 'de' })
      }))
    )
    renderControl()
    await userEvent.click(screen.getByRole('button', { name: /translate/i }))
    await waitFor(() => expect(screen.getByText('Hallo Welt')).toBeInTheDocument())
    await userEvent.click(screen.getByRole('button', { name: /show original/i }))
    expect(screen.getByText('Hello world')).toBeInTheDocument()
    expect(screen.queryByText('Hallo Welt')).not.toBeInTheDocument()
  })

  it('shows a fail-soft message (and keeps the original) when the endpoint reports ok:false', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({ ok: false, translation: null, model: 'm', target_lang: 'de', error: 'unavailable' })
      }))
    )
    const onTranslated = vi.fn()
    renderControl(onTranslated)
    await userEvent.click(screen.getByRole('button', { name: /translate/i }))
    await waitFor(() => expect(screen.getByText(/unavailable/i)).toBeInTheDocument())
    expect(screen.getByText('Hello world')).toBeInTheDocument()
    expect(onTranslated).not.toHaveBeenCalled()
  })

  it('fails soft (and does not report up) when the fetch rejects at the transport layer', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('network')
      })
    )
    const onTranslated = vi.fn()
    renderControl(onTranslated)
    await userEvent.click(screen.getByRole('button', { name: /translate/i }))
    await waitFor(() => expect(screen.getByText(/unavailable/i)).toBeInTheDocument())
    expect(screen.getByText('Hello world')).toBeInTheDocument()
    expect(onTranslated).not.toHaveBeenCalled()
  })
})
