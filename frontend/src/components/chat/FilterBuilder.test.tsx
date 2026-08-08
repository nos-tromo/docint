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
    render(<FilterBuilder />)

    await userEvent.click(screen.getByRole('checkbox', { name: /enable metadata filters/i }))
    await userEvent.type(screen.getByPlaceholderText('application/pdf'), 'application/pdf')

    expect(screen.getByRole('button', { name: /filters/i })).toHaveTextContent('1')
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

    await userEvent.click(screen.getByRole('button', { name: /\+ rule/i }))
    await userEvent.type(screen.getByPlaceholderText('field'), 'mimetype')

    expect(useChatFiltersStore.getState().customRules[0].field).toBe('mimetype')

    await userEvent.click(screen.getByRole('button', { name: /remove rule/i }))
    expect(useChatFiltersStore.getState().customRules).toHaveLength(0)
  })
})
