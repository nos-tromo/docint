import { describe, it, expect, beforeEach } from 'vitest'
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
