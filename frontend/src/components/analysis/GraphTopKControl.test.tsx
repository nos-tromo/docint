import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { GraphTopKControl } from './GraphTopKControl'

describe('GraphTopKControl', () => {
  it('shows the current value', () => {
    render(<GraphTopKControl value={80} max={500} onChange={() => {}} />)
    expect(screen.getByLabelText('Graph node count')).toHaveValue(80)
  })

  it('commits a typed value on blur', async () => {
    const onChange = vi.fn()
    render(<GraphTopKControl value={80} max={500} onChange={onChange} />)
    const input = screen.getByLabelText('Graph node count')
    await userEvent.clear(input)
    await userEvent.type(input, '250')
    await userEvent.tab()
    expect(onChange).toHaveBeenCalledWith(250)
  })

  it('clamps above the max on blur', async () => {
    const onChange = vi.fn()
    render(<GraphTopKControl value={80} max={300} onChange={onChange} />)
    const input = screen.getByLabelText('Graph node count')
    await userEvent.clear(input)
    await userEvent.type(input, '9999')
    await userEvent.tab()
    expect(onChange).toHaveBeenCalledWith(300)
  })

  it('does not call onChange when the value is unchanged', async () => {
    const onChange = vi.fn()
    render(<GraphTopKControl value={80} max={500} onChange={onChange} />)
    const input = screen.getByLabelText('Graph node count')
    await userEvent.clear(input)
    await userEvent.type(input, '80')
    await userEvent.tab()
    expect(onChange).not.toHaveBeenCalled()
  })
})
