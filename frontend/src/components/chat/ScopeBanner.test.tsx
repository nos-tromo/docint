import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ScopeBanner } from './ScopeBanner'

describe('ScopeBanner', () => {
  it('renders nothing when the session is unscoped', () => {
    const { container } = render(<ScopeBanner count={0} onClear={() => {}} />)

    expect(container).toBeEmptyDOMElement()
  })

  it('states how many chunks the answers are restricted to', () => {
    render(<ScopeBanner count={14} onClear={() => {}} />)

    expect(screen.getByTestId('scope-banner')).toHaveTextContent('Scoped to 14 chunks')
  })

  it('clears the scope on demand', async () => {
    const onClear = vi.fn()
    render(<ScopeBanner count={3} onClear={onClear} />)

    await userEvent.click(screen.getByRole('button', { name: /clear/i }))

    expect(onClear).toHaveBeenCalledTimes(1)
  })

  it('reports scoped chunks the backend can no longer find', () => {
    // Re-ingestion mints new point ids, so a pinned scope can outlive its
    // chunks. Answering from the remainder silently would narrow the evidence
    // the investigator believes they selected.
    render(<ScopeBanner count={14} missing={2} onClear={() => {}} />)

    expect(screen.getByTestId('scope-missing')).toHaveTextContent(
      '2 of 14 chunks no longer exist'
    )
  })

  it('stays quiet when nothing is missing', () => {
    render(<ScopeBanner count={14} missing={0} onClear={() => {}} />)

    expect(screen.queryByTestId('scope-missing')).toBeNull()
  })
})
