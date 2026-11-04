import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { clearAllPurposes, setPurpose } from '@infra/ui'
import { ApiError, PURPOSE_SCOPE, setOwnerParam } from './client'
import { csvExportPath } from './collections'
import { downloadFile, fetchFile, filenameFromDisposition, viewFile } from './files'
import { reportExportPath } from './reports'
import { captureDownloads, fileResponse } from '@/test/downloads'

type FetchFn = (input: unknown, init?: { headers?: Record<string, string> }) => Promise<Response>

function stubFetch(response: () => Response) {
  const fn = vi.fn<FetchFn>(async () => response())
  vi.stubGlobal('fetch', fn)
  return fn
}

afterEach(() => {
  clearAllPurposes()
  sessionStorage.clear()
  setOwnerParam(null)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('filenameFromDisposition', () => {
  it.each([
    [`attachment; filename="docs-documents.csv"; filename*=UTF-8''docs-documents.csv`, 'docs-documents.csv'],
    [`attachment; filename="Bericht __ber Alice.pdf"; filename*=UTF-8''Bericht%20%C3%BCber%20Alice.pdf`, 'Bericht über Alice.pdf'],
    [`inline; filename="case-0001.html"`, 'case-0001.html'],
    ['attachment; filename=plain.zip', 'plain.zip'],
    ['attachment; filename="a \\"quoted\\" name.csv"', 'a "quoted" name.csv'],
    // Broken percent-encoding: the plain form is the one to trust.
    [`attachment; filename*=UTF-8''%E0%A4%A.txt; filename="fallback.txt"`, 'fallback.txt'],
    ['attachment', null],
    [null, null]
  ])('reads %s as %s', (header, name) => {
    expect(filenameFromDisposition(header)).toBe(name)
  })
})

describe('fetchFile', () => {
  it('fetches through the wrapper: the stated purpose, and the owner added once', async () => {
    const fetchMock = stubFetch(() => fileResponse('a,b\n', { type: 'text/csv', filename: 'docs-documents.csv' }))
    setPurpose(PURPOSE_SCOPE, { caseRef: 'case-0001' })
    setOwnerParam('jane.doe')

    const file = await fetchFile(csvExportPath('docs', 'documents'))

    const [target, init] = fetchMock.mock.calls[0]
    expect(String(target)).toBe('/collections/docs/export/documents.csv?owner=jane.doe')
    expect(init?.headers?.['X-Audit-Purpose']).toBe('case_ref=case-0001')
    expect(file.filename).toBe('docs-documents.csv')
    expect(await file.blob.text()).toBe('a,b\n')
  })

  it('throws an ApiError carrying the status of a refusal', async () => {
    stubFetch(() => new Response(JSON.stringify({ detail: 'Forbidden.' }), { status: 403 }))

    const err = await fetchFile('/sessions/s1/sources.zip').catch((e: unknown) => e)

    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).status).toBe(403)
  })
})

describe('downloadFile', () => {
  let downloads: ReturnType<typeof captureDownloads>
  beforeEach(() => {
    downloads = captureDownloads()
  })

  it('saves the file under the name the server gave it', async () => {
    stubFetch(() => fileResponse('PK', { type: 'application/zip', filename: 'Bericht über Alice.zip' }))

    await downloadFile(reportExportPath('docs', 1, 'zip'), 'report-1.zip')

    const [anchor] = downloads.saved()
    expect(anchor.download).toBe('Bericht über Alice.zip')
    expect(anchor.getAttribute('href')).toBe('blob:docint/1')
    expect(await downloads.blobs[0].text()).toBe('PK')
  })

  it('falls back to the given name when the server names none', async () => {
    stubFetch(() => fileResponse('PK', { type: 'application/zip' }))

    await downloadFile('/sessions/s1/sources.zip', 'session-s1-sources.zip')

    expect(downloads.saved()[0].download).toBe('session-s1-sources.zip')
  })

  it('saves nothing when the request is refused', async () => {
    stubFetch(() => new Response('{}', { status: 500 }))

    await expect(downloadFile('/sessions/s1/sources.zip', 'x.zip')).rejects.toBeInstanceOf(ApiError)
    expect(downloads.saved()).toEqual([])
  })
})

describe('viewFile', () => {
  let downloads: ReturnType<typeof captureDownloads>
  beforeEach(() => {
    downloads = captureDownloads()
  })

  it('opens a file a tab can show in a new tab, cut loose from this page', async () => {
    stubFetch(() => fileResponse('<h1>Case</h1>', { type: 'text/html; charset=utf-8', filename: 'case.html' }))
    const tab = { opener: window as unknown }
    downloads.open.mockImplementation(() => tab as Window)

    await viewFile(reportExportPath('docs', 1, 'html'), 'report-1.html')

    expect(downloads.opened()).toEqual(['blob:docint/1'])
    expect(downloads.open.mock.calls[0][1]).toBe('_blank')
    expect(tab.opener).toBeNull()
    expect(downloads.saved()).toEqual([])
  })

  it('saves a type no tab can show, under its own name', async () => {
    stubFetch(() =>
      fileResponse('PK', {
        type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        filename: 'brief.docx'
      })
    )

    await viewFile('/sources/preview?collection=docs&file_hash=h1', 'brief.docx')

    expect(downloads.opened()).toEqual([])
    expect(downloads.saved()[0].download).toBe('brief.docx')
  })

  it('saves the file when the browser blocked the tab', async () => {
    stubFetch(() => fileResponse('%PDF-1.7', { type: 'application/pdf', filename: 'handbook.pdf' }))
    downloads.open.mockImplementation(() => null)

    await viewFile('/sources/preview?collection=docs&file_hash=h1', 'handbook.pdf')

    expect(downloads.saved()[0].download).toBe('handbook.pdf')
  })
})
