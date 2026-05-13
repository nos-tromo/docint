import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { KpiCard } from './KpiCard'

describe('KpiCard', () => {
  it('renders label and value', () => {
    render(<KpiCard label="Documents" value={42} />)
    expect(screen.getByText('Documents')).toBeInTheDocument()
    expect(screen.getByText('42')).toBeInTheDocument()
  })

  it('renders dash for missing value', () => {
    render(<KpiCard label="X" value={null} />)
    expect(screen.getByText('—')).toBeInTheDocument()
  })
})
