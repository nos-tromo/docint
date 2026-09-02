import { describe, it, expect } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { DocumentTable } from './DocumentTable'
import type { DocumentRecord } from '@/api/types'
import { useUiStore } from '@/stores/ui'
import userEvent from '@testing-library/user-event'

// Each row carries the extract action, which reads the active report's
// appendix fields — so the table needs the query client the app provides.
function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

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
    render(<DocumentTable docs={DOCS} collection="mydocs" />, { wrapper })
    for (const header of ['Filename', 'Type', 'Units', 'Nodes', 'Entities', 'Hash']) {
      expect(screen.getByText(header)).toBeInTheDocument()
    }
  })

  it('drives header and body rows from one shared grid template (the alignment fix)', () => {
    render(<DocumentTable docs={DOCS} collection="mydocs" />, { wrapper })
    const rows = screen.getAllByRole('row')
    const headerTemplate = (rows[0] as HTMLElement).style.gridTemplateColumns
    const bodyTemplate = (rows[1] as HTMLElement).style.gridTemplateColumns
    expect(headerTemplate).not.toBe('')
    // Identical templates on header and body are what keep the columns aligned.
    expect(bodyTemplate).toBe(headerTemplate)
  })

  it('humanizes MIME types and formats units, using an em dash for image "pages"', () => {
    render(<DocumentTable docs={DOCS} collection="mydocs" />, { wrapper })
    expect(screen.getByText('JPEG')).toBeInTheDocument()
    expect(screen.getByText('CSV')).toBeInTheDocument()
    expect(screen.getByText('138 rows')).toBeInTheDocument()
    // The image has neither pages nor rows -> em dash, not a misleading 0.
    expect(screen.getByText('—')).toBeInTheDocument()
  })

  it('renders entity types as chips with a +N overflow', () => {
    render(<DocumentTable docs={DOCS} collection="mydocs" />, { wrapper })
    expect(screen.getByText('person')).toBeInTheDocument()
    // 8 entity types -> first 4 shown + "+4".
    expect(screen.getByText('+4')).toBeInTheDocument()
  })

  it('truncates the hash and offers a copy control per row', () => {
    render(<DocumentTable docs={DOCS} collection="mydocs" />, { wrapper })
    expect(screen.getByText('abd4fc78')).toBeInTheDocument()
    expect(screen.getByText('05dccacd')).toBeInTheDocument()
    expect(
      screen.getByRole('button', { name: 'Copy hash for field_photo_014.jpg' })
    ).toBeInTheDocument()
  })

  it('shows the document count and a CSV export link', () => {
    render(<DocumentTable docs={DOCS} collection="mydocs" />, { wrapper })
    expect(screen.getByText(/2 documents/)).toBeInTheDocument()
    const link = screen.getByRole('link', { name: 'Export CSV' })
    expect(link).toHaveAttribute('href', expect.stringContaining('/collections/mydocs/export/documents.csv'))
  })

  it('renders an empty state instead of a bare table when there are no documents', () => {
    render(<DocumentTable docs={[]} collection="mydocs" isFetching={false} />, { wrapper })
    expect(screen.getByText('No documents in this collection yet.')).toBeInTheDocument()
    expect(screen.queryByText('Filename')).not.toBeInTheDocument()
  })
})

describe('DocumentTable document preview', () => {
  it('opens the shared preview dialog for a listed document', async () => {
    useUiStore.setState({ selectedCollection: 'mydocs', previewModal: null })
    render(<DocumentTable docs={[DOCS[1]]} collection="mydocs" />, { wrapper })

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
    render(<DocumentTable docs={[DOCS[1]]} collection="mydocs" />, { wrapper })

    const action = screen.getByRole('button', { name: /preview/i })
    expect(action.closest('.group')).not.toBeNull()
  })
})

describe('DocumentTable sorting', () => {
  // Natural ordering (appendix, 9, 10) differs from lexicographic ordering
  // (10 before 9) and the node counts order differently as text than as
  // numbers, so these rows tell the built-in comparators apart.
  const SORT_DOCS: DocumentRecord[] = [
    {
      filename: 'report_10.pdf',
      file_hash: '1111111111111111111111111111111111111111',
      mimetype: 'application/pdf',
      page_count: 4,
      row_count: 0,
      node_count: 138,
      entity_types: []
    },
    {
      filename: 'report_9.pdf',
      file_hash: '2222222222222222222222222222222222222222',
      mimetype: 'application/pdf',
      page_count: 2,
      row_count: 0,
      node_count: 9,
      entity_types: []
    },
    {
      filename: 'appendix_2.pdf',
      file_hash: '3333333333333333333333333333333333333333',
      mimetype: 'application/pdf',
      page_count: 1,
      row_count: 0,
      node_count: 42,
      entity_types: []
    }
  ]

  /** Text of the nth cell of every body row, in rendered order. */
  function columnOrder(index: number): string[] {
    return screen
      .getAllByRole('row')
      .slice(1)
      .map((row) => within(row).getAllByRole('cell')[index].textContent?.trim() ?? '')
  }

  it('sorts filenames naturally, not lexicographically, and reverses on a second click', async () => {
    render(<DocumentTable docs={SORT_DOCS} collection="mydocs" />, { wrapper })

    await userEvent.click(screen.getByRole('button', { name: 'Filename' }))
    // Lexicographic order would put report_10 before report_9.
    expect(columnOrder(0)).toEqual(['appendix_2.pdf', 'report_9.pdf', 'report_10.pdf'])

    await userEvent.click(screen.getByRole('button', { name: 'Filename' }))
    expect(columnOrder(0)).toEqual(['report_10.pdf', 'report_9.pdf', 'appendix_2.pdf'])
  })

  it('sorts the node count numerically, largest first', async () => {
    render(<DocumentTable docs={SORT_DOCS} collection="mydocs" />, { wrapper })

    // A numeric column sorts descending on the first click; as text that same
    // descending pass would read 9, 42, 138.
    await userEvent.click(screen.getByRole('button', { name: 'Nodes' }))
    expect(columnOrder(3)).toEqual(['138', '42', '9'])

    await userEvent.click(screen.getByRole('button', { name: 'Nodes' }))
    expect(columnOrder(3)).toEqual(['9', '42', '138'])
  })

  it('leaves the columns that opt out of sorting unsortable', () => {
    render(<DocumentTable docs={SORT_DOCS} collection="mydocs" />, { wrapper })
    expect(screen.queryByRole('button', { name: 'Entities' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Hash' })).not.toBeInTheDocument()
  })
})
