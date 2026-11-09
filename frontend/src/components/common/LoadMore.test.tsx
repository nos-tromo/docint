import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { LoadMore } from './LoadMore'

describe('LoadMore', () => {
  it('sits centred under a border at the end of a list', () => {
    render(<LoadMore onClick={vi.fn()} busy={false} />)
    const button = screen.getByRole('button', { name: 'Load more' })
    expect(button.parentElement).toHaveClass('justify-center', 'border-t')
  })

  it('asks for the next page', () => {
    const onClick = vi.fn()
    render(<LoadMore onClick={onClick} busy={false} />)
    fireEvent.click(screen.getByRole('button', { name: 'Load more' }))
    expect(onClick).toHaveBeenCalledOnce()
  })

  it('holds while a page loads', () => {
    render(<LoadMore onClick={vi.fn()} busy />)
    expect(screen.getByRole('button', { name: 'Loading…' })).toBeDisabled()
  })
})
