import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ChatControls } from './ChatControls'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { useSearchUiStore } from '@/stores/searchUi'

beforeEach(() => {
  useChatFiltersStore.getState().reset()
  useSearchUiStore.setState({ drafts: {}, queries: {}, scopes: {}, filtersOpen: false })
})

describe('ChatControls', () => {
  it('carries both retrieval settings', () => {
    render(<ChatControls />)

    expect(screen.getByRole('button', { name: /filters/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /retrieval/i })).toBeInTheDocument()
  })

  it('names the active retrieval mode, since the control carries no label', () => {
    render(<ChatControls />)

    // Default is stateful. An icon-only control whose state is invisible is a
    // control people leave set wrong, so the mode is in the accessible name
    // (and, through it, the tooltip) rather than implied by "pressed".
    const toggle = screen.getByRole('button', { name: /retrieval/i })
    expect(toggle).toHaveAccessibleName(/whole chat \(stateful\)/i)
    expect(toggle).toHaveAttribute('aria-pressed', 'true')
  })

  it('flips the retrieval mode, and says so', async () => {
    render(<ChatControls />)

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
    const { container } = render(<ChatControls />)
    const statefulIcon = container.querySelector('[aria-pressed="true"] svg')?.innerHTML

    await userEvent.click(screen.getByRole('button', { name: /retrieval/i }))
    const statelessIcon = container.querySelector('[aria-pressed="false"] svg')?.innerHTML

    expect(statefulIcon).toBeTruthy()
    expect(statelessIcon).toBeTruthy()
    expect(statelessIcon).not.toBe(statefulIcon)
  })

  it('drops the filter panel downward, so it cannot cover the mode control', async () => {
    // The regression this replaced: anchored `bottom-full` at the foot of the
    // search column, the panel opened straight over the retrieval control
    // sitting above it, and the two settings read as one confusing toggle.
    render(<ChatControls />)

    await userEvent.click(screen.getByRole('button', { name: /filters/i }))

    const panel = screen.getByText(/enable metadata filters/i).closest('div.absolute')
    expect(panel).not.toBeNull()
    expect(panel?.className).toContain('top-full')
    expect(panel?.className).not.toContain('bottom-full')
  })

  it('carries the reasoning toggle, off by default and named by its state', () => {
    render(<ChatControls />)

    // Off is the safe default: thinking costs latency and tokens, so the user
    // opts in per chat. The state lives in the accessible name, like the
    // retrieval mode, because the control carries no label.
    const toggle = screen.getByRole('button', { name: /reasoning/i })
    expect(toggle).toHaveAccessibleName(/off/i)
    expect(toggle).toHaveAttribute('aria-pressed', 'false')
  })

  it('flips reasoning on and off, and says so', async () => {
    render(<ChatControls />)

    await userEvent.click(screen.getByRole('button', { name: /reasoning/i }))

    expect(useChatFiltersStore.getState().reasoning).toBe(true)
    const toggle = screen.getByRole('button', { name: /reasoning/i })
    expect(toggle).toHaveAccessibleName(/on/i)
    expect(toggle).toHaveAttribute('aria-pressed', 'true')

    await userEvent.click(toggle)
    expect(useChatFiltersStore.getState().reasoning).toBe(false)
  })

  it('lights the brain up when reasoning is on, rather than only tinting it', async () => {
    // Same rule as the retrieval pair: the two states must survive a glance
    // without a hover, so the drawing changes, not just the background.
    const { container } = render(<ChatControls />)
    const off = screen.getByRole('button', { name: /reasoning/i }).querySelector('svg')?.innerHTML

    await userEvent.click(screen.getByRole('button', { name: /reasoning/i }))
    const on = screen.getByRole('button', { name: /reasoning/i }).querySelector('svg')?.innerHTML

    expect(off).toBeTruthy()
    expect(on).toBeTruthy()
    expect(on).not.toBe(off)
    void container
  })
})
