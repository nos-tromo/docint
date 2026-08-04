import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Shell } from './Shell'

function renderShell(whoami?: { username: string; display_name: string | null }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  if (whoami) qc.setQueryData(['whoami'], whoami)
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <Shell>
          <p>main content</p>
        </Shell>
      </MemoryRouter>
    </QueryClientProvider>
  )
}

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
