import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { DocumentTable } from './DocumentTable'
import type { DocumentRecord } from '@/api/types'
import { useUiStore } from '@/stores/ui'
import userEvent from '@testing-library/user-event'

const DOCS: DocumentRecord[] = [
  {
    filename: 'field_photo_014.jpg',
    file_hash: 'abd4fc7803e1e8d1c7e2b92e8d8fb9d64f4b26b3',
    mimetype: 'image/jpeg',
    page_count: 0,
    row_count: 0,
    node_count: 1,
    entity_types: ['loc', 'org', 'person']
  },
  {
    filename: 'network_postings.csv',
    file_hash: '05dccacd5843d3f948e2d3cf94e9471ace41207c',
    mimetype: 'text/csv',
    row_count: 138,
    node_count: 138,
    entity_types: ['date', 'event', 'group', 'loc', 'mail', 'org', 'person', 'phone']
  }
]

describe('DocumentTable', () => {
  it('renders aligned column headers', () => {
    render(<DocumentTable docs={DOCS} collection="mydocs" />)
    for (const header of ['Filename', 'Type', 'Units', 'Nodes', 'Entities', 'Hash']) {
      expect(screen.getByText(header)).toBeInTheDocument()
    }
  })

  it('drives header and body rows from one shared grid template (the alignment fix)', () => {
    render(<DocumentTable docs={DOCS} collection="mydocs" />)
    const rows = screen.getAllByRole('row')
    const headerTemplate = (rows[0] as HTMLElement).style.gridTemplateColumns
    const bodyTemplate = (rows[1] as HTMLElement).style.gridTemplateColumns
    expect(headerTemplate).not.toBe('')
    // Identical templates on header and body are what keep the columns aligned.
    expect(bodyTemplate).toBe(headerTemplate)
  })

  it('humanizes MIME types and formats units, using an em dash for image "pages"', () => {
    render(<DocumentTable docs={DOCS} collection="mydocs" />)
    expect(screen.getByText('JPEG')).toBeInTheDocument()
    expect(screen.getByText('CSV')).toBeInTheDocument()
    expect(screen.getByText('138 rows')).toBeInTheDocument()
    // The image has neither pages nor rows -> em dash, not a misleading 0.
    expect(screen.getByText('—')).toBeInTheDocument()
  })

  it('renders entity types as chips with a +N overflow', () => {
    render(<DocumentTable docs={DOCS} collection="mydocs" />)
    expect(screen.getByText('person')).toBeInTheDocument()
    // 8 entity types -> first 4 shown + "+4".
    expect(screen.getByText('+4')).toBeInTheDocument()
  })

  it('truncates the hash and offers a copy control per row', () => {
    render(<DocumentTable docs={DOCS} collection="mydocs" />)
    expect(screen.getByText('abd4fc78')).toBeInTheDocument()
    expect(screen.getByText('05dccacd')).toBeInTheDocument()
    expect(
      screen.getByRole('button', { name: 'Copy hash for field_photo_014.jpg' })
    ).toBeInTheDocument()
  })

  it('shows the document count and a CSV export link', () => {
    render(<DocumentTable docs={DOCS} collection="mydocs" />)
    expect(screen.getByText(/2 documents/)).toBeInTheDocument()
    const link = screen.getByRole('link', { name: 'Export CSV' })
    expect(link).toHaveAttribute('href', expect.stringContaining('/collections/mydocs/export/documents.csv'))
  })

  it('renders an empty state instead of a bare table when there are no documents', () => {
    render(<DocumentTable docs={[]} collection="mydocs" isFetching={false} />)
    expect(screen.getByText('No documents in this collection yet.')).toBeInTheDocument()
    expect(screen.queryByText('Filename')).not.toBeInTheDocument()
  })
})

describe('DocumentTable document preview', () => {
  it('opens the shared preview dialog for a listed document', async () => {
    useUiStore.setState({ selectedCollection: 'mydocs', previewModal: null })
    render(<DocumentTable docs={[DOCS[1]]} collection="mydocs" />)

    await userEvent.click(screen.getByRole('button', { name: /preview/i }))

    expect(useUiStore.getState().previewModal).toEqual({
      collection: 'mydocs',
      file_hash: DOCS[1].file_hash,
      filename: 'network_postings.csv'
    })
  })

  it('makes each row a hover group so the action can reveal itself', () => {
    // HoverIconAction ships opacity-0 + group-hover:opacity-100, so without a
    // `group` ancestor the control is mounted but permanently invisible to
    // mouse users (jsdom cannot see that, hence the class assertion).
    useUiStore.setState({ selectedCollection: 'mydocs', previewModal: null })
    render(<DocumentTable docs={[DOCS[1]]} collection="mydocs" />)

    const action = screen.getByRole('button', { name: /preview/i })
    expect(action.closest('.group')).not.toBeNull()
  })
})
