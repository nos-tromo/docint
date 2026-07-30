import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { Dropzone } from './Dropzone'

describe('Dropzone folder picker', () => {
  it('exposes a folder input with webkitdirectory and forwards picked files', () => {
    const onFiles = vi.fn()
    render(<Dropzone onFiles={onFiles} />)

    expect(screen.getByRole('button', { name: /choose a folder/i })).toBeInTheDocument()

    const folderInput = Array.from(
      document.querySelectorAll('input[type="file"]')
    ).find((el) => el.hasAttribute('webkitdirectory')) as HTMLInputElement
    expect(folderInput).toBeTruthy()

    const f = new File([new Uint8Array([1])], 'a.jpg', { type: 'image/jpeg' })
    Object.defineProperty(folderInput, 'files', { value: [f] })
    fireEvent.change(folderInput)
    expect(onFiles).toHaveBeenCalledWith([f])
  })
})

type Entry = {
  isFile: boolean
  isDirectory: boolean
  fullPath: string
  file?: (cb: (f: File) => void) => void
  createReader?: () => { readEntries: (cb: (e: Entry[]) => void) => void }
}

function fileEntry(path: string): Entry {
  const f = new File([new Uint8Array([1])], path.split('/').pop() as string)
  return {
    isFile: true,
    isDirectory: false,
    fullPath: path,
    file: (cb) => cb(f)
  }
}

/** Directory whose reader returns `pages` in sequence, then an empty page. */
function dirEntry(path: string, pages: Entry[][]): Entry {
  let i = 0
  return {
    isFile: false,
    isDirectory: true,
    fullPath: path,
    createReader: () => ({
      readEntries: (cb) => cb(i < pages.length ? pages[i++] : [])
    })
  }
}

/** A file whose `webkitRelativePath` is already defined non-configurable, so
 *  the traversal's `Object.defineProperty` re-stamp throws — proves one
 *  un-stampable file doesn't sink the whole batch. */
function unstampableFileEntry(path: string): Entry {
  const f = new File([new Uint8Array([1])], path.split('/').pop() as string)
  Object.defineProperty(f, 'webkitRelativePath', { value: 'preset', configurable: false })
  return {
    isFile: true,
    isDirectory: false,
    fullPath: path,
    file: (cb) => cb(f)
  }
}

function dropWith(entries: Entry[]) {
  return {
    preventDefault: () => {},
    dataTransfer: {
      items: entries.map((e) => ({ kind: 'file', webkitGetAsEntry: () => e })),
      files: []
    }
  }
}

describe('Dropzone folder drop', () => {
  it('queues every file in a dropped tree, across readEntries pages, with relative paths', async () => {
    const onFiles = vi.fn()
    render(<Dropzone onFiles={onFiles} />)
    const zone = screen.getByText(/drag files here/i).closest('div') as HTMLElement

    const tree = dirEntry('/export', [
      [fileEntry('/export/a.pdf'), dirEntry('/export/sub', [[fileEntry('/export/sub/b.pdf')], []])],
      [fileEntry('/export/c.pdf')]
    ])
    fireEvent.drop(zone, dropWith([tree]))

    await waitFor(() => expect(onFiles).toHaveBeenCalled())
    const names = (onFiles.mock.calls[0][0] as File[]).map((f) => f.webkitRelativePath)
    expect(names.sort()).toEqual(['export/a.pdf', 'export/c.pdf', 'export/sub/b.pdf'])
  })

  it('still queues plainly dropped files', async () => {
    const onFiles = vi.fn()
    render(<Dropzone onFiles={onFiles} />)
    const zone = screen.getByText(/drag files here/i).closest('div') as HTMLElement

    fireEvent.drop(zone, dropWith([fileEntry('/x.pdf')]))
    await waitFor(() => expect(onFiles).toHaveBeenCalled())
    expect((onFiles.mock.calls[0][0] as File[]).map((f) => f.name)).toEqual(['x.pdf'])
  })

  it('reports an empty drop instead of failing silently', async () => {
    const onFiles = vi.fn()
    const onEmpty = vi.fn()
    render(<Dropzone onFiles={onFiles} onEmpty={onEmpty} />)
    const zone = screen.getByText(/drag files here/i).closest('div') as HTMLElement

    // A dropped directory that contains nothing at all.
    fireEvent.drop(zone, dropWith([dirEntry('/empty', [[]])]))
    await waitFor(() => expect(onEmpty).toHaveBeenCalledTimes(1))
    expect(onFiles).not.toHaveBeenCalled()
  })

  it('falls back to dataTransfer.files when the entries API is unavailable', async () => {
    const onFiles = vi.fn()
    render(<Dropzone onFiles={onFiles} />)
    const zone = screen.getByText(/drag files here/i).closest('div') as HTMLElement

    const f = new File([new Uint8Array([1])], 'legacy.pdf')
    fireEvent.drop(zone, { preventDefault: () => {}, dataTransfer: { items: [], files: [f] } })
    await waitFor(() => expect(onFiles).toHaveBeenCalledWith([f]))
  })

  it('still queues the whole batch when one file cannot be path-stamped', async () => {
    const onFiles = vi.fn()
    render(<Dropzone onFiles={onFiles} />)
    const zone = screen.getByText(/drag files here/i).closest('div') as HTMLElement

    fireEvent.drop(
      zone,
      dropWith([unstampableFileEntry('/export/locked.pdf'), fileEntry('/export/ok.pdf')])
    )

    await waitFor(() => expect(onFiles).toHaveBeenCalled())
    const names = (onFiles.mock.calls[0][0] as File[]).map((f) => f.name)
    expect(names.sort()).toEqual(['locked.pdf', 'ok.pdf'])
  })
})
