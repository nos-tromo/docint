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
        <TranslateControl text="Hello world" onTranslated={onTranslated} />
      </div>
    </QueryClientProvider>
  )
}

describe('TranslateControl', () => {
  it('translates on click, shows the Translation block, and reports it up', async () => {
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
    expect(screen.getByText(/translation/i)).toBeInTheDocument()
    expect(onTranslated).toHaveBeenCalledWith({ text: 'Hallo Welt', target_lang: 'de', model: 'm' })
  })

  it('shows a fail-soft message when the endpoint reports ok:false', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({ ok: false, translation: null, model: 'm', target_lang: 'de', error: 'unavailable' })
      }))
    )
    renderControl()
    await userEvent.click(screen.getByRole('button', { name: /translate/i }))
    await waitFor(() => expect(screen.getByText(/unavailable/i)).toBeInTheDocument())
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
    expect(onTranslated).not.toHaveBeenCalled()
  })
})
