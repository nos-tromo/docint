import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { SourcePreviewAction } from './SourcePreviewAction'
import { useUiStore } from '@/stores/ui'

beforeEach(() => {
  useUiStore.setState({ selectedCollection: 'docs', previewModal: null })
})

describe('SourcePreviewAction', () => {
  it('opens the shared preview dialog for the source', async () => {
    render(<SourcePreviewAction fileHash="h1" filename="handbook.pdf" />)

    await userEvent.click(screen.getByRole('button', { name: /preview/i }))

    expect(useUiStore.getState().previewModal).toEqual({
      collection: 'docs',
      file_hash: 'h1',
      filename: 'handbook.pdf'
    })
  })

  it('renders nothing without a stored file to preview', () => {
    const { container } = render(<SourcePreviewAction fileHash={undefined} filename="handbook.pdf" />)
    expect(container).toBeEmptyDOMElement()
  })

  it('renders nothing without an active collection to resolve the file against', () => {
    useUiStore.setState({ selectedCollection: null })
    const { container } = render(<SourcePreviewAction fileHash="h1" filename="handbook.pdf" />)
    expect(container).toBeEmptyDOMElement()
  })

  it('still labels the dialog when the source has no filename', async () => {
    render(<SourcePreviewAction fileHash="h1" filename={undefined} />)

    await userEvent.click(screen.getByRole('button', { name: /preview/i }))

    expect(useUiStore.getState().previewModal?.filename).toBeTruthy()
  })
})
