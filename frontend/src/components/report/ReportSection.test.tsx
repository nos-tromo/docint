import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ReportSection } from './ReportSection'

function panelOf(bar: HTMLElement): HTMLElement {
  const panel = document.getElementById(bar.getAttribute('aria-controls') ?? '')
  expect(panel).not.toBeNull()
  return panel as HTMLElement
}

describe('ReportSection', () => {
  it('folds and unfolds from its bar', async () => {
    render(
      <ReportSection title="Entity findings" count="(2)">
        <p>Nordwind Logistik</p>
      </ReportSection>
    )

    const bar = screen.getByRole('button', { name: /entity findings/i })
    expect(bar).toHaveAttribute('aria-expanded', 'true')

    await userEvent.click(bar)
    expect(bar).toHaveAttribute('aria-expanded', 'false')

    await userEvent.click(bar)
    expect(bar).toHaveAttribute('aria-expanded', 'true')
  })

  it('keeps a folded section peeking rather than empty', async () => {
    render(
      <ReportSection title="Chat answers" count="(3)">
        <p>Which colour is the sky?</p>
      </ReportSection>
    )

    const bar = screen.getByRole('button', { name: /chat answers/i })
    await userEvent.click(bar)

    // A row of shut bars says nothing about which pile is which, so the
    // content stays mounted and is clipped to a couple of lines instead.
    expect(screen.getByText('Which colour is the sky?')).toBeInTheDocument()
    const panel = panelOf(bar)
    expect(panel.style.maxHeight).not.toBe('')
    // The cut runs through a card, so what is below it still holds a note
    // field and three buttons — Tab must not walk into them.
    expect(panel).toHaveAttribute('inert')
    expect(panel).toHaveAttribute('data-state', 'collapsed')
  })

  it("carries the count in the bar's accessible name", () => {
    render(
      <ReportSection title="Document overview" count="4 documents · 312 nodes">
        <p>manifest</p>
      </ReportSection>
    )

    // The overview's bar and the "Document overview" checkbox in the metadata
    // row would otherwise be two controls with one name; the totals are what
    // tell them apart, for a screen reader as much as for a test.
    expect(
      screen.getByRole('button', { name: /document overview 4 documents · 312 nodes/i })
    ).toBeInTheDocument()
  })

  it('honours defaultOpen', () => {
    render(
      <ReportSection title="Document overview" defaultOpen={false}>
        <p>manifest</p>
      </ReportSection>
    )
    expect(screen.getByRole('button', { name: /document overview/i })).toHaveAttribute(
      'aria-expanded',
      'false'
    )
  })

  it('lifts the cap entirely when open', () => {
    render(
      <ReportSection title="Summaries">
        <p>body</p>
      </ReportSection>
    )
    const bar = screen.getByRole('button', { name: /summaries/i })
    const panel = panelOf(bar)
    expect(panel.style.maxHeight).toBe('')
    expect(panel).not.toHaveAttribute('inert')
  })
})
