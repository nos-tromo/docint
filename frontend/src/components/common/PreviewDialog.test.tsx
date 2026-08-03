import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { PreviewDialog } from './PreviewDialog'
import { useUiStore } from '@/stores/ui'

beforeEach(() => {
  useUiStore.setState({ selectedCollection: 'docs', previewModal: null })
})

function openWith(filename: string) {
  useUiStore.getState().openPreview({ collection: 'docs', file_hash: 'h1', filename })
}

describe('PreviewDialog', () => {
  it('renders nothing until a preview is opened', () => {
    const { container } = render(<PreviewDialog />)
    expect(container).toBeEmptyDOMElement()
  })

  it('shows the document inline for a browser-renderable type', async () => {
    render(<PreviewDialog />)
    openWith('handbook.pdf')

    const dialog = await screen.findByRole('dialog')
    expect(dialog).toHaveAttribute('aria-modal', 'true')
    expect(screen.getByTitle('handbook.pdf')).toHaveAttribute(
      'src',
      expect.stringContaining('file_hash=h1')
    )
  })

  it('offers a new-tab link instead of an iframe for a type the browser downloads', async () => {
    render(<PreviewDialog />)
    openWith('report.docx')

    await screen.findByRole('dialog')
    // An iframe pointed at a docx triggers a download rather than a preview,
    // so the dialog must not mount one.
    expect(screen.queryByTitle('report.docx')).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: /new tab/i })).toHaveAttribute('target', '_blank')
  })

  it('closes on Escape', async () => {
    render(<PreviewDialog />)
    openWith('handbook.pdf')
    await screen.findByRole('dialog')

    await userEvent.keyboard('{Escape}')

    await waitFor(() => {
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    })
    expect(useUiStore.getState().previewModal).toBeNull()
  })

  it('closes on a backdrop click but not on a click inside the panel', async () => {
    render(<PreviewDialog />)
    openWith('handbook.pdf')
    const dialog = await screen.findByRole('dialog')

    await userEvent.click(dialog)
    expect(screen.getByRole('dialog')).toBeInTheDocument()

    await userEvent.click(screen.getByTestId('preview-backdrop'))
    await waitFor(() => {
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    })
  })

  it('returns focus to the control that opened it', async () => {
    render(
      <>
        <button type="button" data-testid="opener" onClick={() => openWith('handbook.pdf')}>
          open
        </button>
        <PreviewDialog />
      </>
    )
    const opener = screen.getByTestId('opener')
    await userEvent.click(opener)
    await screen.findByRole('dialog')
    expect(opener).not.toHaveFocus()

    await userEvent.keyboard('{Escape}')

    await waitFor(() => {
      expect(opener).toHaveFocus()
    })
  })
})

describe('PreviewDialog type-based rendering', () => {
  // The dialog must not delegate these types to iframe navigation: subframe
  // heuristics differ from tabs (Chrome refuses application/json in frames,
  // renders image documents at natural size without shrink-to-fit, and hands
  // some text types to the download manager).
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders an image with its full extent visible, not an iframe', async () => {
    render(<PreviewDialog />)
    openWith('shot.png')

    await screen.findByRole('dialog')
    const img = screen.getByRole('img', { name: 'shot.png' })
    expect(img).toHaveAttribute('src', expect.stringContaining('file_hash=h1'))
    // object-contain is what keeps a large PNG zoomed-out inside the panel.
    expect(img.className).toContain('object-contain')
    expect(screen.queryByTitle('shot.png')).not.toBeInTheDocument()
  })

  it.each(['data.json', 'notes.md', 'table.csv', 'plain.txt'])(
    'fetches %s and renders its text itself',
    async (filename) => {
      const fetchMock = vi.fn().mockResolvedValue(
        new Response('body of the file', { status: 200 })
      )
      vi.stubGlobal('fetch', fetchMock)
      render(<PreviewDialog />)
      openWith(filename)

      await screen.findByRole('dialog')
      expect(await screen.findByText('body of the file')).toBeInTheDocument()
      expect(screen.queryByTitle(filename)).not.toBeInTheDocument()
      expect(String(fetchMock.mock.calls[0][0])).toContain('file_hash=h1')
    }
  )

  it('renders HTML sources as text, never as a live document', async () => {
    // With same-origin framing now allowed on the preview route, framing an
    // ingested HTML file would execute its scripts against the app origin.
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('<script>alert(1)</script>', { status: 200 })
    )
    vi.stubGlobal('fetch', fetchMock)
    render(<PreviewDialog />)
    openWith('page.html')

    await screen.findByRole('dialog')
    expect(await screen.findByText('<script>alert(1)</script>')).toBeInTheDocument()
    expect(screen.queryByTitle('page.html')).not.toBeInTheDocument()
  })

  it('falls back to the new-tab hint when the text fetch fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('nope', { status: 500 })))
    render(<PreviewDialog />)
    openWith('data.json')

    await screen.findByRole('dialog')
    expect(await screen.findByText(/could not be loaded/i)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /new tab/i })).toBeInTheDocument()
  })

  it('still frames PDFs, where the browser viewer works', async () => {
    render(<PreviewDialog />)
    openWith('handbook.pdf')

    await screen.findByRole('dialog')
    expect(screen.getByTitle('handbook.pdf')).toBeInTheDocument()
  })
})
