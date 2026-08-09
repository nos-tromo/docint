import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { SearchControls } from './SearchControls'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { useSearchUiStore } from '@/stores/searchUi'

beforeEach(() => {
  useChatFiltersStore.getState().reset()
  useSearchUiStore.setState({ drafts: {}, queries: {}, scopes: {}, filtersOpen: false })
})

describe('SearchControls band', () => {
  it('carries both settings on one row', () => {
    render(<SearchControls />)

    expect(screen.getByRole('button', { name: /filters/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /retrieval/i })).toBeInTheDocument()
  })

  it('names the active retrieval mode, since the control carries no label', () => {
    render(<SearchControls />)

    // Default is stateful. An icon-only control whose state is invisible is a
    // control people leave set wrong, so the mode is in the accessible name
    // (and, through it, the tooltip) rather than implied by "pressed".
    const toggle = screen.getByRole('button', { name: /retrieval/i })
    expect(toggle).toHaveAccessibleName(/whole chat \(stateful\)/i)
    expect(toggle).toHaveAttribute('aria-pressed', 'true')
  })

  it('flips the retrieval mode, and says so', async () => {
    render(<SearchControls />)

    await userEvent.click(screen.getByRole('button', { name: /retrieval/i }))

    expect(useChatFiltersStore.getState().retrievalMode).toBe('stateless')
    const toggle = screen.getByRole('button', { name: /retrieval/i })
    expect(toggle).toHaveAccessibleName(/last message only \(stateless\)/i)
    expect(toggle).toHaveAttribute('aria-pressed', 'false')

    await userEvent.click(toggle)
    expect(useChatFiltersStore.getState().retrievalMode).toBe('session')
  })

  it('draws different icons for the two modes', async () => {
    // Two shapes, not one shape pressed and unpressed — the difference has to
    // survive a glance without a hover.
    const { container } = render(<SearchControls />)
    const statefulIcon = container.querySelector('[aria-pressed="true"] svg')?.innerHTML

    await userEvent.click(screen.getByRole('button', { name: /retrieval/i }))
    const statelessIcon = container.querySelector('[aria-pressed="false"] svg')?.innerHTML

    expect(statefulIcon).toBeTruthy()
    expect(statelessIcon).toBeTruthy()
    expect(statelessIcon).not.toBe(statefulIcon)
  })

  it('drops the filter panel downward, so it cannot cover the mode control', async () => {
    // The regression this replaced: anchored `bottom-full` at the foot of the
    // column, the panel opened straight over the retrieval control sitting
    // above it, and the two settings read as one confusing toggle.
    render(<SearchControls />)

    await userEvent.click(screen.getByRole('button', { name: /filters/i }))

    const panel = screen.getByText(/enable metadata filters/i).closest('div.absolute')
    expect(panel).not.toBeNull()
    expect(panel?.className).toContain('top-full')
    expect(panel?.className).not.toContain('bottom-full')
  })
})
