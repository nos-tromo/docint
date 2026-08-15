import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ValidationBanner } from './ValidationBanner'
import type { ValidationFields } from '@/api/types'

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

  it('shows an unavailable banner with generic catalog copy, never the raw backend reason', () => {
    // The backend reason for "unavailable" can be a raw caught-exception
    // message (see docint/agents/generation.py's exception branch) — it
    // must never reach the DOM even when the field is populated.
    render(
      <ValidationBanner
        v={{
          validation_checked: false,
          validation_mismatch: null as unknown as boolean,
          validation_reason: 'Validation request failed: connection refused to internal-host:9000'
        }}
      />
    )
    expect(
      screen.getByText(/response validation unavailable/i)
    ).toBeInTheDocument()
    expect(screen.getByText(/skipped or unavailable/i)).toBeInTheDocument()
    expect(screen.queryByText(/connection refused/i)).not.toBeInTheDocument()
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

describe('ValidationBanner markers are drawn, never typed', () => {
  // `⚠` and `ⓘ` both carry emoji presentation on some platforms, so the typed
  // form could arrive full-colour beside monochrome chrome.
  it.each([
    ['mismatch', { validation_checked: true, validation_mismatch: true }],
    ['passed', { validation_checked: true, validation_mismatch: false }],
    ['unavailable', { validation_checked: false }]
  ])('draws the %s tone marker as an icon', (_name, fields) => {
    const { container } = render(<ValidationBanner v={fields as ValidationFields} />)
    expect(container.querySelector('svg')).not.toBeNull()
    expect(container.textContent).not.toMatch(/[⚠✓ⓘ]/)
  })
})
