import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { GraphDebugPanel } from './GraphDebugPanel'

describe('GraphDebugPanel', () => {
  it('renders nothing without debug data', () => {
    const { container } = render(<GraphDebugPanel data={null} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('reveals the payload from its disclosure, and names the next click', async () => {
    render(<GraphDebugPanel data={{ terms: ['nordwind'] }} />)

    const toggle = screen.getByRole('button', { name: 'Show graph debug' })
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    // The heading stays on the bar as text — only the caret is the control.
    expect(screen.getByText(/graph debug/i)).toBeInTheDocument()

    await userEvent.click(toggle)

    expect(toggle).toHaveAttribute('aria-expanded', 'true')
    // The name swaps with the state, so it always says what the next click does.
    expect(screen.getByRole('button', { name: 'Hide graph debug' })).toBe(toggle)
    const panel = document.getElementById(toggle.getAttribute('aria-controls') ?? '')
    expect(panel).not.toBeNull()
    expect(panel).toHaveTextContent('nordwind')

    await userEvent.click(toggle)
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    expect(document.getElementById(toggle.getAttribute('aria-controls') ?? '')).toBeNull()
  })
})
