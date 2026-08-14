import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { FilterBuilder } from './FilterBuilder'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { useSearchUiStore } from '@/stores/searchUi'

beforeEach(() => {
  useChatFiltersStore.getState().reset()
  useSearchUiStore.setState({ drafts: {}, queries: {}, scopes: {}, filtersOpen: false })
})

describe('FilterBuilder disclosure', () => {
  it('starts collapsed as a single summary line', () => {
    render(<FilterBuilder />)

    const toggle = screen.getByRole('button', { name: /filters/i })
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByLabelText(/enable metadata filters/i)).toBeNull()
  })

  it('expands to reveal the controls', async () => {
    render(<FilterBuilder />)

    await userEvent.click(screen.getByRole('button', { name: /filters/i }))

    expect(screen.getByRole('button', { name: /filters/i })).toHaveAttribute(
      'aria-expanded',
      'true'
    )
    expect(screen.getByText(/enable metadata filters/i)).toBeInTheDocument()
  })

  it('badges the active filter count so a hidden filter is never silent', async () => {
    useSearchUiStore.getState().setFiltersOpen(true)
    const { container } = render(<FilterBuilder />)

    await userEvent.click(screen.getByRole('checkbox', { name: /enable metadata filters/i }))
    await userEvent.type(screen.getByPlaceholderText('application/pdf'), 'application/pdf')

    // The trigger is an icon with no label, so the count has to reach a screen
    // reader through its name; the corner badge is the sighted half of that.
    expect(screen.getByRole('button', { name: /1 active filters/i })).toBeInTheDocument()
    expect(container.querySelector('[aria-hidden="true"].absolute')).toHaveTextContent('1')
  })

  it('closes on a pointer press outside it, and on Escape', async () => {
    // An overlay you can only shut by hitting the same small icon again is its
    // own trap — it sits over the transcript the user is trying to read.
    useSearchUiStore.getState().setFiltersOpen(true)
    render(
      <div>
        <FilterBuilder />
        <button type="button">elsewhere</button>
      </div>
    )

    await userEvent.click(screen.getByRole('button', { name: /elsewhere/i }))
    expect(useSearchUiStore.getState().filtersOpen).toBe(false)

    useSearchUiStore.getState().setFiltersOpen(true)
    await userEvent.keyboard('{Escape}')
    expect(useSearchUiStore.getState().filtersOpen).toBe(false)
  })

  it('stays open while the panel itself is used', async () => {
    useSearchUiStore.getState().setFiltersOpen(true)
    useChatFiltersStore.getState().setFilterEnabled(true)
    render(<FilterBuilder />)

    await userEvent.click(screen.getByPlaceholderText('application/pdf'))

    expect(useSearchUiStore.getState().filtersOpen).toBe(true)
  })

  it('gives the hate-speech toggle its own checkbox-beside-label row', async () => {
    // It used to sit in a grid cell sized for a text input, orphaned under a
    // floating caption. An accessible name is the proof it is paired again.
    useSearchUiStore.getState().setFiltersOpen(true)
    useChatFiltersStore.getState().setFilterEnabled(true)
    render(<FilterBuilder />)

    const toggle = screen.getByRole('checkbox', { name: /hate-speech only/i })
    await userEvent.click(toggle)

    expect(useChatFiltersStore.getState().hateSpeechOnly).toBe(true)
  })

  it('edits a custom rule through the shared primitives', async () => {
    useSearchUiStore.getState().setFiltersOpen(true)
    useChatFiltersStore.getState().setFilterEnabled(true)
    render(<FilterBuilder />)

    await userEvent.click(screen.getByRole('button', { name: /^rule$/i }))
    await userEvent.type(screen.getByPlaceholderText('field'), 'mimetype')

    expect(useChatFiltersStore.getState().customRules[0].field).toBe('mimetype')

    await userEvent.click(screen.getByRole('button', { name: /remove rule/i }))
    expect(useChatFiltersStore.getState().customRules).toHaveLength(0)
  })
})
