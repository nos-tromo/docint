import { afterEach, beforeEach, describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { clearAllPurposes, setPurpose } from '@infra/ui'
import { Shell } from './Shell'
import { LanguageContext } from '@/i18n/LanguageContext'
import type { Lang } from '@/i18n'
import type { AppConfig } from '@/api/types'

const CONFIG: AppConfig = {
  graph_top_k: 80,
  graph_max_top_k: 500,
  collection_timeout: 120,
  max_upload_bytes: 1024,
  language: 'en',
  require_purpose: []
}

function renderShell(
  whoami?: { username: string; display_name: string | null },
  config: AppConfig = CONFIG,
  lang: Lang = 'en'
) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  if (whoami) qc.setQueryData(['whoami'], whoami)
  qc.setQueryData(['app-config'], config)
  return render(
    <QueryClientProvider client={qc}>
      <LanguageContext value={lang}>
        <MemoryRouter>
          <Shell>
            <p>main content</p>
          </Shell>
        </MemoryRouter>
      </LanguageContext>
    </QueryClientProvider>
  )
}

beforeEach(() => {
  // The shell's own requests (sessions, the job stream) stay pending: these
  // tests are about the header, and must not reach for a network.
  vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})))
})

afterEach(() => {
  clearAllPurposes()
  sessionStorage.clear()
  vi.unstubAllGlobals()
})

describe('Shell', () => {
  it('renders a single header (AppShell title) and the main slot', () => {
    renderShell()
    expect(screen.getByText('docint')).toBeInTheDocument()
    expect(screen.getByText('main content')).toBeInTheDocument()
  })

  it('prefers display_name over username in the header user slot', () => {
    renderShell({ username: 'alice', display_name: 'Alice Example' })
    expect(screen.getByRole('button', { name: /Alice Example/ })).toBeInTheDocument()
  })

  it('falls back to username when display_name is absent', () => {
    renderShell({ username: 'alice', display_name: null })
    expect(screen.getByRole('button', { name: /alice/ })).toBeInTheDocument()
  })
})

describe('Shell purpose bar', () => {
  it('sits in the header, required when the backend refuses requests without a purpose', () => {
    renderShell(undefined, { ...CONFIG, require_purpose: ['disclose', 'query'] })
    const bar = screen.getByRole('button', { name: 'Purpose required' })
    expect(bar).toHaveAttribute('data-state', 'required')
    expect(screen.getByRole('banner')).toContainElement(bar)
  })

  it('stays optional when the backend requires none', () => {
    renderShell()
    expect(screen.getByRole('button', { name: 'No purpose stated' })).toHaveAttribute('data-state', 'empty')
  })

  it('stays optional against a backend that predates the field', () => {
    const older: AppConfig = { ...CONFIG }
    delete older.require_purpose
    renderShell(undefined, older)
    expect(screen.getByRole('button', { name: 'No purpose stated' })).toBeInTheDocument()
  })

  it("shows the purpose stated for this app, and only this app's", () => {
    setPurpose('chorus', { caseRef: 'case-0002' })
    setPurpose('docint', { caseRef: 'case-0001' })
    renderShell()
    expect(screen.getByRole('button', { name: 'Purpose: case-0001' })).toBeInTheDocument()
  })

  it('speaks German in a German deployment', () => {
    renderShell(undefined, { ...CONFIG, language: 'de', require_purpose: ['query'] }, 'de')
    expect(screen.getByRole('button', { name: 'Zweck erforderlich' })).toBeInTheDocument()
  })
})
