import { useEffect, useRef } from 'react'
import { sourcePreviewUrl } from '@/api/ingest'
import { useUiStore } from '@/stores/ui'
import { useT } from '@/i18n/LanguageContext'

// Types the browser renders in an iframe. Anything else (docx, xlsx, zip …)
// would be handed to the download manager instead of shown, so those get a
// new-tab link rather than a frame that appears blank.
const INLINE_EXTENSIONS = new Set([
  'pdf',
  'png',
  'jpg',
  'jpeg',
  'gif',
  'webp',
  'svg',
  'bmp',
  'txt',
  'md',
  'csv',
  'json',
  'html'
])

/**
 * Whether the browser will render this filename inline in an iframe.
 *
 * @param filename - The source document's filename.
 * @returns True when an iframe preview is worth mounting.
 */
export function isInlinePreviewable(filename: string): boolean {
  const ext = filename.split('.').pop()?.toLowerCase() ?? ''
  return INLINE_EXTENSIONS.has(ext)
}

/**
 * The app-wide document preview. Mounted once in the Shell and driven by
 * `useUiStore.previewModal`, so any control can open a preview by calling
 * `openPreview` without owning dialog state of its own.
 *
 * Deliberately not a native `<dialog>`: `showModal()` is not reliably
 * implemented in jsdom, and this needs to stay testable.
 */
export function PreviewDialog() {
  const t = useT()
  const modal = useUiStore((s) => s.previewModal)
  const closePreview = useUiStore((s) => s.closePreview)
  const panelRef = useRef<HTMLDivElement | null>(null)
  const openerRef = useRef<HTMLElement | null>(null)

  useEffect(() => {
    if (!modal) return
    // Remember who opened us so focus can go back there on close — closing a
    // dialog that dumps focus on <body> strands keyboard users at the top of
    // the page.
    openerRef.current = document.activeElement as HTMLElement | null
    panelRef.current?.focus()

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closePreview()
    }
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      openerRef.current?.focus?.()
      openerRef.current = null
    }
  }, [modal, closePreview])

  if (!modal) return null

  const url = sourcePreviewUrl(modal.collection, modal.file_hash)
  const inline = isInlinePreviewable(modal.filename)

  return (
    <div
      data-testid="preview-backdrop"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={closePreview}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={modal.filename}
        tabIndex={-1}
        className="flex h-full max-h-[90vh] w-full max-w-5xl flex-col overflow-hidden rounded-md border border-border bg-background shadow-lg outline-none"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-2">
          <h2 className="truncate text-sm font-medium">{modal.filename}</h2>
          <div className="flex items-center gap-3">
            <a
              href={url}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-blue-400 hover:text-blue-300"
            >
              {t('common.preview_new_tab')}
            </a>
            <button
              type="button"
              onClick={closePreview}
              aria-label={t('common.preview_close')}
              className="rounded px-2 py-0.5 text-lg leading-none text-muted-foreground hover:text-foreground"
            >
              ×
            </button>
          </div>
        </div>
        {inline ? (
          <iframe title={modal.filename} src={url} className="h-full w-full flex-1 bg-muted" />
        ) : (
          <div className="flex flex-1 flex-col items-center justify-center gap-2 p-8 text-sm text-muted-foreground">
            <p>{t('common.preview_not_inline')}</p>
          </div>
        )}
      </div>
    </div>
  )
}
