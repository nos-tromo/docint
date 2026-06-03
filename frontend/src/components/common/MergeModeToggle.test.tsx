import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MergeModeToggle } from './MergeModeToggle'
import { useUiStore } from '@/stores/ui'

beforeEach(() => {
  useUiStore.setState({ entityMergeMode: 'resolved' })
})

describe('MergeModeToggle', () => {
  it('marks the active mode (resolved by default)', () => {
    render(<MergeModeToggle />)
    expect(screen.getByRole('button', { name: 'Resolved' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'Orthographic' })).toHaveAttribute('aria-pressed', 'false')
  })

  it('switches the store mode on click', async () => {
    render(<MergeModeToggle />)
    await userEvent.click(screen.getByRole('button', { name: 'Orthographic' }))
    expect(useUiStore.getState().entityMergeMode).toBe('orthographic')
  })
})
