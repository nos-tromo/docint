import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { DownloadButton, DownloadLink } from './DownloadAction'

describe('DownloadAction', () => {
  it('drops the label from the page but not from the accessible name', () => {
    render(<DownloadButton label="Download MD" onClick={() => {}} />)
    const button = screen.getByRole('button', { name: 'Download MD' })
    // The whole point of the swap: no text competing for width…
    expect(button).toHaveTextContent('')
    // …and nothing lost to a screen reader or a hover.
    expect(button).toHaveAttribute('title', 'Download MD')
  })

  it('runs the click handler that builds the file', async () => {
    const onClick = vi.fn()
    render(<DownloadButton label="Download" onClick={onClick} />)
    await userEvent.click(screen.getByRole('button', { name: 'Download' }))
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('keeps an adornment visible and inside the accessible name', () => {
    render(<DownloadButton label="Export GraphML">GraphML</DownloadButton>)
    const button = screen.getByRole('button', { name: 'Export GraphML' })
    // Side-by-side downloads are told apart by the format, so it stays on
    // screen — and the name contains it, as WCAG "Label in Name" requires.
    expect(button).toHaveTextContent('GraphML')
  })

  it('lets the browser fetch a server-streamed file', () => {
    render(<DownloadLink href="/collections/alpha/export/documents.csv" label="Export CSV" />)
    const link = screen.getByRole('link', { name: 'Export CSV' })
    expect(link).toHaveAttribute('href', '/collections/alpha/export/documents.csv')
    expect(link).toHaveAttribute('download')
  })

  it('lets the caller name the downloaded file', () => {
    render(<DownloadLink href="/sessions/s1/sources.zip" download="sources.zip" label="Download sources" />)
    expect(screen.getByRole('link', { name: 'Download sources' })).toHaveAttribute(
      'download',
      'sources.zip'
    )
  })
})
