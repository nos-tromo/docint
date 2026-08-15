import { useRef, useState, type DragEvent } from 'react'
import { Button } from '@infra/ui'
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
  try {
    Object.defineProperty(file, 'webkitRelativePath', {
      value: fullPath.replace(/^\//, ''),
      configurable: true
    })
  } catch {
    // Some engines may reject redefining a non-configurable property. A file
    // with a bare name is far better than losing the whole drop —
    // ingest.ts already falls back to f.name when webkitRelativePath is unset.
  }
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
  onEmpty,
  disabled
}: {
  onFiles: (files: File[]) => void
  /** Called when a drop yields no usable files (empty folder, or a traversal
   *  the browser refused) — otherwise the drop would fail silently. */
  onEmpty?: () => void
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
    // (and the plain-file fallback list) out synchronously BEFORE awaiting
    // anything.
    const plain = Array.from(e.dataTransfer.files)
    const entries = Array.from(e.dataTransfer.items ?? [])
      .map((item) => (item as unknown as { webkitGetAsEntry?: () => FsEntry | null }).webkitGetAsEntry?.() ?? null)
      .filter((entry): entry is FsEntry => entry !== null)
    if (!entries.length) {
      // No entries API (or no entries): keep the plain-file behavior.
      if (plain.length) onFiles(plain)
      else onEmpty?.()
      return
    }
    void Promise.all(entries.map(collectFiles))
      .then((groups) => {
        const list = groups.flat()
        if (list.length) onFiles(list)
        else onEmpty?.()
      })
      .catch(() => {
        // Traversal failed unexpectedly: fall back to whatever plain files
        // were present rather than silently losing the whole drop.
        if (plain.length) onFiles(plain)
        else onEmpty?.()
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
      {/* A real button, not underlined text: it opens a picker like every
          other control on this screen, and an underline in the middle of a
          drop target reads as a link to somewhere. */}
      <div className="mt-3">
        <Button
          type="button"
          variant="secondary"
          onClick={(e) => {
            e.stopPropagation()
            folderInputRef.current?.click()
          }}
        >
          {t('upload.choose_folder')}
        </Button>
      </div>
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
