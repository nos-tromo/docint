import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ValidationBanner } from './ValidationBanner'
import type { AgentTrail, ValidationFields } from '@/api/types'

function trail(overrides: Partial<AgentTrail>): AgentTrail {
  return { steps: 1, actions: [], reason: null, answer: 'split', shortened: 0, evidence_chars: 9, ...overrides }
}

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

  it('names the source-backed part when it validated an assessment', () => {
    render(
      <ValidationBanner
        v={{ validation_checked: true, validation_mismatch: false, validation_reason: null, answer_mode: 'assessment' }}
      />
    )
    expect(screen.getByText(/validation passed for the source-backed part/i)).toBeInTheDocument()
  })

  it('says an unseparated assessment was not validated', () => {
    render(
      <ValidationBanner
        v={{
          validation_checked: false,
          validation_reason: 'x',
          answer_mode: 'assessment',
          agent: trail({ answer: 'unsplit', evidence_chars: 0 })
        }}
      />
    )
    expect(screen.getByText(/no separate source-backed part/i)).toBeInTheDocument()
  })

  it('says a split assessment with no source-backed text was not validated', () => {
    render(
      <ValidationBanner
        v={{
          validation_checked: false,
          answer_mode: 'assessment',
          agent: trail({ answer: 'split', evidence_chars: 0 })
        }}
      />
    )
    expect(screen.getByText(/no separate source-backed part/i)).toBeInTheDocument()
  })

  it('says an empty assessment was not validated', () => {
    render(
      <ValidationBanner
        v={{
          validation_checked: false,
          answer_mode: 'assessment',
          agent: trail({ answer: 'empty', evidence_chars: null })
        }}
      />
    )
    expect(screen.getByText(/no separate source-backed part/i)).toBeInTheDocument()
  })

  it('blames the validator, not the answer, when a split assessment went unchecked', () => {
    // The source-backed part exists; validation was disabled or failed.
    render(
      <ValidationBanner
        v={{
          validation_checked: false,
          answer_mode: 'assessment',
          agent: trail({ answer: 'split', evidence_chars: 9 })
        }}
      />
    )
    expect(screen.getByText(/response validation unavailable/i)).toBeInTheDocument()
    expect(screen.queryByText(/no separate source-backed part/i)).not.toBeInTheDocument()
  })

  it('treats a split assessment whose whole answer is source-backed as a validator outage', () => {
    render(
      <ValidationBanner
        v={{
          validation_checked: false,
          answer_mode: 'assessment',
          agent: trail({ answer: 'split', evidence_chars: null })
        }}
      />
    )
    expect(screen.getByText(/response validation unavailable/i)).toBeInTheDocument()
    expect(screen.queryByText(/no separate source-backed part/i)).not.toBeInTheDocument()
  })

  it('falls back to the generic unavailable title when an assessment carries no trail', () => {
    render(
      <ValidationBanner v={{ validation_checked: false, answer_mode: 'assessment', agent: null }} />
    )
    expect(screen.getByText(/response validation unavailable/i)).toBeInTheDocument()
    expect(screen.queryByText(/no separate source-backed part/i)).not.toBeInTheDocument()
  })

  it('names the source-backed part when the validator flagged an assessment', () => {
    render(
      <ValidationBanner
        v={{ validation_checked: true, validation_mismatch: true, validation_reason: null, answer_mode: 'assessment' }}
      />
    )
    expect(screen.getByText(/validation flagged the source-backed part/i)).toBeInTheDocument()
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
