import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { describe, it, expect, vi } from 'vitest'
import { VersionBadge } from './VersionBadge'

vi.mock('../api/config', () => ({ getVersion: vi.fn() }))
import { getVersion } from '../api/config'

function renderWithClient(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

describe('VersionBadge', () => {
  it('renders the fetched version', async () => {
    vi.mocked(getVersion).mockResolvedValue({ version: '1.0.0' })
    renderWithClient(<VersionBadge />)
    expect(await screen.findByText('v1.0.0')).toBeInTheDocument()
  })

  it('renders nothing when there is no version', () => {
    vi.mocked(getVersion).mockResolvedValue({ version: '' })
    const { container } = renderWithClient(<VersionBadge />)
    expect(container.textContent).toBe('')
  })
})
