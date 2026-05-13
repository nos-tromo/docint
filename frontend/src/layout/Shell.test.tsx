import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Shell } from './Shell'

function renderShell() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
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
  it('renders sidebar and main slot', () => {
    renderShell()
    expect(screen.getByText(/docint/i)).toBeInTheDocument()
    expect(screen.getByText('main content')).toBeInTheDocument()
  })
})
