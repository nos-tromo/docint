import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { clearAllPurposes, setPurpose } from '@infra/ui'
import { PreviewDialog } from './PreviewDialog'
import { PURPOSE_SCOPE, setOwnerParam } from '@/api/client'
import { useUiStore } from '@/stores/ui'
import { captureDownloads, fileResponse } from '@/test/downloads'

type FetchFn = (input: unknown, init?: { headers?: Record<string, string> }) => Promise<Response>

let downloads: ReturnType<typeof captureDownloads>

/** Serve every request with one file, as the preview route would. */
function serve(body: string, type: string) {
  const fn = vi.fn<FetchFn>(async () => fileResponse(body, { type }))
  vi.stubGlobal('fetch', fn)
  return fn
}

beforeEach(() => {
  useUiStore.setState({ selectedCollection: 'docs', selectedOwner: null, previewModal: null })
  downloads = captureDownloads()
})

afterEach(() => {
  clearAllPurposes()
  sessionStorage.clear()
  setOwnerParam(null)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

function openWith(filename: string) {
  useUiStore.getState().openPreview({ collection: 'docs', file_hash: 'h1', filename })
}

describe('PreviewDialog', () => {
  it('renders nothing until a preview is opened', () => {
    const { container } = render(<PreviewDialog />)
    expect(container).toBeEmptyDOMElement()
  })

  it('fetches the document with the stated purpose and frames it from memory', async () => {
    const fetchMock = serve('%PDF-1.7', 'application/pdf')
    setPurpose(PURPOSE_SCOPE, { caseRef: 'case-0001' })
    render(<PreviewDialog />)
    openWith('handbook.pdf')

    const dialog = await screen.findByRole('dialog')
    expect(dialog).toHaveAttribute('aria-modal', 'true')
    // An <iframe src> could not state the purpose, so the frame shows a Blob.
    await waitFor(() => expect(screen.getByTitle('handbook.pdf')).toHaveAttribute('src', 'blob:docint/1'))
    const [target, init] = fetchMock.mock.calls[0]
    expect(String(target)).toBe('/sources/preview?collection=docs&file_hash=h1')
    expect(init?.headers?.['X-Audit-Purpose']).toBe('case_ref=case-0001')
  })

  it('frames a PDF as a PDF whatever it was served as', async () => {
    // The frame is a same-origin document: only the browser's PDF viewer belongs in it.
    serve('<script>alert(1)</script>', 'text/html')
    render(<PreviewDialog />)
    openWith('handbook.pdf')

    await waitFor(() => expect(screen.getByTitle('handbook.pdf')).toBeInTheDocument())
    expect(downloads.blobs.at(-1)?.type).toBe('application/pdf')
  })

  it("previews in the namespace of the owner an admin is working for", async () => {
    const fetchMock = serve('%PDF-1.7', 'application/pdf')
    setOwnerParam('jane.doe')
    useUiStore.setState({ selectedOwner: 'jane.doe' })
    render(<PreviewDialog />)
    openWith('handbook.pdf')

    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    expect(String(fetchMock.mock.calls[0][0])).toContain('owner=jane.doe')
  })

  it('offers a new tab instead of an iframe for a type the browser downloads, fetching only then', async () => {
    const fetchMock = serve('PK', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')
    render(<PreviewDialog />)
    openWith('report.docx')

    await screen.findByRole('dialog')
    // An iframe pointed at a docx triggers a download rather than a preview,
    // so the dialog must not mount one — and has nothing to fetch for itself.
    expect(screen.queryByTitle('report.docx')).not.toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()

    const newTab = screen.getByRole('button', { name: /new tab/i })
    // An icon action beside the close button, not a blue text link.
    expect(newTab).toHaveTextContent('')
    expect(newTab.parentElement).toContainElement(screen.getByRole('button', { name: /close/i }))
    await userEvent.click(newTab)

    // No tab can show a docx: it is saved under its own name instead.
    await waitFor(() => expect(downloads.saved()).toHaveLength(1))
    expect(downloads.saved()[0].download).toBe('report.docx')
    expect(String(fetchMock.mock.calls[0][0])).toBe('/sources/preview?collection=docs&file_hash=h1')
  })

  it('opens what it already fetched in a new tab, without asking the server again', async () => {
    const fetchMock = serve('%PDF-1.7', 'application/pdf')
    render(<PreviewDialog />)
    openWith('handbook.pdf')
    await waitFor(() => expect(screen.getByTitle('handbook.pdf')).toBeInTheDocument())

    await userEvent.click(screen.getByRole('button', { name: /new tab/i }))

    expect(downloads.opened()).toHaveLength(1)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('closes on Escape', async () => {
    serve('%PDF-1.7', 'application/pdf')
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
    serve('%PDF-1.7', 'application/pdf')
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
    serve('%PDF-1.7', 'application/pdf')
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

  it('renders an image with its full extent visible, not an iframe', async () => {
    serve('png-bytes', 'image/png')
    render(<PreviewDialog />)
    openWith('shot.png')

    await screen.findByRole('dialog')
    const img = await screen.findByRole('img', { name: 'shot.png' })
    expect(img).toHaveAttribute('src', 'blob:docint/1')
    // object-contain is what keeps a large PNG zoomed-out inside the panel.
    expect(img.className).toContain('object-contain')
    expect(screen.queryByTitle('shot.png')).not.toBeInTheDocument()
  })

  it.each(['data.json', 'notes.md', 'table.csv', 'plain.txt'])(
    'fetches %s and renders its text itself',
    async (filename) => {
      const fetchMock = serve('body of the file', 'text/plain')
      render(<PreviewDialog />)
      openWith(filename)

      await screen.findByRole('dialog')
      expect(await screen.findByText('body of the file')).toBeInTheDocument()
      expect(screen.queryByTitle(filename)).not.toBeInTheDocument()
      expect(String(fetchMock.mock.calls[0][0])).toContain('file_hash=h1')
    }
  )

  it('renders HTML sources as text, never as a live document', async () => {
    // A framed HTML file would run its scripts against the app origin.
    serve('<script>alert(1)</script>', 'text/html')
    render(<PreviewDialog />)
    openWith('page.html')

    await screen.findByRole('dialog')
    expect(await screen.findByText('<script>alert(1)</script>')).toBeInTheDocument()
    expect(screen.queryByTitle('page.html')).not.toBeInTheDocument()
  })

  it('falls back to the new-tab hint when the fetch fails', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('nope', { status: 500 })))
    render(<PreviewDialog />)
    openWith('data.json')

    await screen.findByRole('dialog')
    expect(await screen.findByText(/could not be loaded/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /new tab/i })).toBeInTheDocument()
  })

  it('still frames PDFs, where the browser viewer works', async () => {
    serve('%PDF-1.7', 'application/pdf')
    render(<PreviewDialog />)
    openWith('handbook.pdf')

    expect(await screen.findByTitle('handbook.pdf')).toBeInTheDocument()
  })
})

describe('PreviewDialog — frozen evidence', () => {
  const DATA_URI = 'data:image/jpeg;base64,/9j/4AAQSkZJRg=='

  it('enlarges a snapshot thumbnail with no collection in play, fetching nothing', async () => {
    // A report outlives its collection, so its evidence must enlarge from the
    // bytes it carries rather than from the source store.
    const fetchMock = serve('', 'image/jpeg')
    render(<PreviewDialog />)
    useUiStore.getState().openPreview({ filename: 'plakat.png', data_uri: DATA_URI })

    await screen.findByRole('dialog')
    expect(screen.getByRole('img', { name: 'plakat.png' })).toHaveAttribute('src', DATA_URI)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('offers no new tab for frozen evidence', async () => {
    // Browsers block top-level navigation to a data: URL, so the control would
    // promise something it cannot do.
    render(<PreviewDialog />)
    useUiStore.getState().openPreview({ filename: 'plakat.png', data_uri: DATA_URI })

    await screen.findByRole('dialog')
    expect(screen.queryByRole('button', { name: /new tab/i })).not.toBeInTheDocument()
  })

  it('shows frozen evidence whatever the filename suggests', async () => {
    // The frozen bytes are always a JPEG thumbnail; the name they carry is the
    // source document's, which may be a PDF or a video.
    render(<PreviewDialog />)
    useUiStore.getState().openPreview({ filename: 'clip.mp4', data_uri: DATA_URI })

    await screen.findByRole('dialog')
    expect(screen.getByRole('img', { name: 'clip.mp4' })).toHaveAttribute('src', DATA_URI)
  })
})

describe('PreviewDialog — a stored image', () => {
  it('enlarges a keyframe as an image although its name is the clip it was cut from', async () => {
    const fetchMock = serve('jpeg-bytes', 'image/jpeg')
    render(<PreviewDialog />)
    useUiStore.getState().openPreview({ filename: 'talk.mp4', collection: 'alpha', image_id: 'frame' })

    expect(await screen.findByRole('img', { name: 'talk.mp4' })).toHaveAttribute('src', 'blob:docint/1')
    expect(String(fetchMock.mock.calls[0][0])).toBe('/sources/thumbnail?collection=alpha&image_id=frame')
  })

  it('offers the stored image in a new tab, from what it fetched', async () => {
    serve('jpeg-bytes', 'image/jpeg')
    render(<PreviewDialog />)
    useUiStore.getState().openPreview({ filename: 'talk.mp4', collection: 'alpha', image_id: 'frame' })
    await screen.findByRole('img', { name: 'talk.mp4' })

    await userEvent.click(screen.getByRole('button', { name: /new tab/i }))

    expect(downloads.opened()).toHaveLength(1)
  })
})
