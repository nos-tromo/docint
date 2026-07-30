import { useRef, useState, type DragEvent } from 'react'
import { cn } from '@/lib/cn'
import { useT } from '@/i18n/LanguageContext'

/** Minimal shape of the non-standard entries API we consume. */
type FsEntry = {
  isFile: boolean
  isDirectory: boolean
  fullPath: string
  file?: (onOk: (f: File) => void, onErr?: (e: unknown) => void) => void
  createReader?: () => { readEntries: (onOk: (e: FsEntry[]) => void, onErr?: (e: unknown) => void) => void }
}

/** `entry.file()` yields an empty webkitRelativePath; ingest.ts uploads each
 *  file as `webkitRelativePath || name`, so stamp the tree path or dropped
 *  folders flatten and same-named files across subfolders collide. */
function withRelativePath(file: File, fullPath: string): File {
  Object.defineProperty(file, 'webkitRelativePath', {
    value: fullPath.replace(/^\//, ''),
    configurable: true
  })
  return file
}

/** readEntries returns at most ~100 entries per call; loop until it yields an
 *  empty page or large folders are silently truncated. */
async function readAllEntries(dir: FsEntry): Promise<FsEntry[]> {
  const reader = dir.createReader?.()
  if (!reader) return []
  const all: FsEntry[] = []
  for (;;) {
    const page = await new Promise<FsEntry[]>((resolve) => reader.readEntries(resolve, () => resolve([])))
    if (!page.length) return all
    all.push(...page)
  }
}

async function collectFiles(entry: FsEntry): Promise<File[]> {
  if (entry.isFile) {
    const file = await new Promise<File | null>((resolve) =>
      entry.file ? entry.file(resolve, () => resolve(null)) : resolve(null)
    )
    return file ? [withRelativePath(file, entry.fullPath)] : []
  }
  if (entry.isDirectory) {
    const children = await readAllEntries(entry)
    const nested = await Promise.all(children.map(collectFiles))
    return nested.flat()
  }
  return []
}

export function Dropzone({
  onFiles,
  disabled
}: {
  onFiles: (files: File[]) => void
  disabled?: boolean
}) {
  const t = useT()
  const [hover, setHover] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)

  const handle = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setHover(false)
    if (disabled) return
    // DataTransfer is neutered once this handler returns, so pull every entry
    // out synchronously BEFORE awaiting anything.
    const entries = Array.from(e.dataTransfer.items ?? [])
      .map((item) => (item as unknown as { webkitGetAsEntry?: () => FsEntry | null }).webkitGetAsEntry?.() ?? null)
      .filter((entry): entry is FsEntry => entry !== null)
    if (!entries.length) {
      // No entries API (or no entries): keep the plain-file behavior.
      const list = Array.from(e.dataTransfer.files)
      if (list.length) onFiles(list)
      return
    }
    void Promise.all(entries.map(collectFiles)).then((groups) => {
      const list = groups.flat()
      if (list.length) onFiles(list)
    })
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setHover(true)
      }}
      onDragLeave={() => setHover(false)}
      onDrop={handle}
      onClick={() => inputRef.current?.click()}
      className={cn(
        'rounded-lg border-2 border-dashed p-10 text-center cursor-pointer',
        hover ? 'border-foreground bg-muted' : 'border-border',
        disabled && 'opacity-50 pointer-events-none'
      )}
    >
      <p>{t('upload.drop_hint')}</p>
      <button
        type="button"
        className="mt-3 underline"
        onClick={(e) => {
          e.stopPropagation()
          folderInputRef.current?.click()
        }}
      >
        {t('upload.choose_folder')}
      </button>
      <input
        ref={inputRef}
        type="file"
        multiple
        className="hidden"
        onChange={(e) => {
          const list = Array.from(e.target.files ?? [])
          if (list.length) onFiles(list)
          e.target.value = ''
        }}
      />
      <input
        ref={folderInputRef}
        type="file"
        multiple
        className="hidden"
        {...({ webkitdirectory: '', directory: '' } as Record<string, string>)}
        onChange={(e) => {
          const list = Array.from(e.target.files ?? [])
          if (list.length) onFiles(list)
          e.target.value = ''
        }}
      />
    </div>
  )
}
