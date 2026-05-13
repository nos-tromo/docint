import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { Shell } from './Shell'

describe('Shell', () => {
  it('renders sidebar and main slot', () => {
    render(
      <MemoryRouter>
        <Shell>
          <p>main content</p>
        </Shell>
      </MemoryRouter>
    )
    expect(screen.getByText(/docint/i)).toBeInTheDocument()
    expect(screen.getByText('main content')).toBeInTheDocument()
  })
})
