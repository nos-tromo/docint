import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { MetadataPills } from './MetadataPills'

describe('MetadataPills', () => {
  it('renders nothing for an empty list', () => {
    const { container } = render(<MetadataPills items={[]} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('renders value-only and labeled pills', () => {
    render(
      <MetadataPills
        items={[
          { key: 'network', value: 'Instagram' },
          { key: 'posting_author', label: 'Autor', value: 'beispiel_account' }
        ]}
      />
    )
    expect(screen.getByText('Instagram')).toBeInTheDocument()
    expect(screen.getByText('Autor')).toBeInTheDocument()
    expect(screen.getByText('beispiel_account')).toBeInTheDocument()
  })

  it('renders href pills as external links', () => {
    render(
      <MetadataPills
        items={[{ key: 'posting_url', value: 'Beitrag öffnen', href: 'https://ig.example/p' }]}
      />
    )
    const link = screen.getByRole('link', { name: 'Beitrag öffnen' })
    expect(link).toHaveAttribute('href', 'https://ig.example/p')
    expect(link).toHaveAttribute('target', '_blank')
    expect(link).toHaveAttribute('rel', 'noreferrer')
    // The leaving-arrow is drawn and driven by `href`, not appended to the
    // pill's copy — so it survives a translator editing the label, and it
    // cannot arrive as a full-colour emoji the way `↗` can.
    expect(link.querySelector('svg')).not.toBeNull()
    expect(link.textContent).toBe('Beitrag öffnen')
  })
})
