import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ValidationBanner } from './ValidationBanner'

describe('ValidationBanner', () => {
  it('renders the validated state when the answer is grounded', () => {
    render(
      <ValidationBanner
        v={{
          validation_checked: true,
          validation_mismatch: false,
          validation_reason: 'Answer matches retrieved sources.'
        }}
      />
    )
    expect(screen.getByText(/response validation passed/i)).toBeInTheDocument()
    expect(screen.getByText(/answer matches/i)).toBeInTheDocument()
  })

  it('renders the mismatch state when the validator flags a problem', () => {
    render(
      <ValidationBanner
        v={{
          validation_checked: true,
          validation_mismatch: true,
          validation_reason: 'Answer not supported by sources.'
        }}
      />
    )
    expect(screen.getByText(/flagged a potential mismatch/i)).toBeInTheDocument()
    expect(screen.getByText(/not supported/i)).toBeInTheDocument()
  })

  it('shows an unavailable banner with the reason when validation could not run', () => {
    render(
      <ValidationBanner
        v={{
          validation_checked: false,
          validation_mismatch: null as unknown as boolean,
          validation_reason: 'Validation model unavailable.'
        }}
      />
    )
    expect(
      screen.getByText(/response validation unavailable/i)
    ).toBeInTheDocument()
    expect(screen.getByText(/model unavailable/i)).toBeInTheDocument()
  })

  it('always renders a skipped/unavailable notice even with no validation signal', () => {
    // Matches the Streamlit `response_validation_summary` behavior — the
    // user expects to see *some* validation status under every response.
    render(<ValidationBanner v={{}} />)
    expect(screen.getByText(/response not validated/i)).toBeInTheDocument()
    expect(
      screen.getByText(/skipped or unavailable/i)
    ).toBeInTheDocument()
  })
})
