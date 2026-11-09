import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ClampedText } from './ClampedText'

describe('ClampedText', () => {
  it('draws its show-more toggle in the theme colour, not a fixed blue', () => {
    render(<ClampedText length={300}>{'x'.repeat(300)}</ClampedText>)
    const toggle = screen.getByRole('button', { name: /show more/i })
    expect(toggle).toHaveClass('text-primary')
    expect(toggle).not.toHaveClass('text-blue-400')
  })
})
