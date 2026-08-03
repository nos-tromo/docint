import { useEffect, useRef, useState } from 'react'
import { sourcePreviewUrl } from '@/api/ingest'
import { useUiStore } from '@/stores/ui'
import { useT } from '@/i18n/LanguageContext'

// How the dialog renders each type. Frame navigation is only trusted for
// PDF, where the browser viewer genuinely works inside a subframe. Subframe
// heuristics differ from tabs everywhere else: Chrome refuses to render
// application/json in a frame (its JSON viewer is top-level only), renders
// image documents at natural size without the tab's shrink-to-fit, and hands
// some text types to the download manager. Those types are rendered by the
// dialog itself instead.
const FRAME_EXTENSIONS = new Set(['pdf'])
const IMAGE_EXTENSIONS = new Set(['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'bmp'])
// html is deliberately in the *text* bucket: with same-origin framing now
// allowed on the preview route, framing an ingested HTML file would execute
// its scripts against the app origin. Showing the source is safe — and more
// useful for evidence review anyway.
const TEXT_EXTENSIONS = new Set(['txt', 'md', 'csv', 'json', 'log', 'html'])

type PreviewKind = 'frame' | 'image' | 'text' | 'none'

/**
 * The dialog's rendering strategy for a filename, by extension.
 *
 * @param filename - The source document's filename.
 * @returns Which renderer the dialog uses for the file.
 */
export function previewKind(filename: string): PreviewKind {
  const ext = filename.split('.').pop()?.toLowerCase() ?? ''
  if (FRAME_EXTENSIONS.has(ext)) return 'frame'
  if (IMAGE_EXTENSIONS.has(ext)) return 'image'
  if (TEXT_EXTENSIONS.has(ext)) return 'text'
  return 'none'
}

type TextPreviewState = { state: 'loading' | 'error' | 'ready'; text: string }

/**
 * Fetched body for the text rendering path.
 *
 * @param url - The preview URL to fetch, or null when the text path is inactive.
 * @returns Loading / error / content state for the current url.
 */
function useTextPreview(url: string | null): TextPreviewState {
  const [result, setResult] = useState<TextPreviewState>({ state: 'loading', text: '' })

  useEffect(() => {
    if (!url) return
    let cancelled = false
    setResult({ state: 'loading', text: '' })
    fetch(url)
      .then(async (resp) => {
        if (!resp.ok) throw new Error(String(resp.status))
        const text = await resp.text()
        if (!cancelled) setResult({ state: 'ready', text })
      })
      .catch(() => {
        if (!cancelled) setResult({ state: 'error', text: '' })
      })
    return () => {
      cancelled = true
    }
  }, [url])

  return result
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

  const kind = modal ? previewKind(modal.filename) : 'none'
  const url = modal ? sourcePreviewUrl(modal.collection, modal.file_hash) : null
  const textPreview = useTextPreview(kind === 'text' ? url : null)

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

  if (!modal || !url) return null

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
        {kind === 'frame' && (
          <iframe title={modal.filename} src={url} className="h-full w-full flex-1 bg-muted" />
        )}
        {kind === 'image' && (
          <div className="flex flex-1 items-center justify-center overflow-auto bg-muted p-4">
            <img src={url} alt={modal.filename} className="max-h-full max-w-full object-contain" />
          </div>
        )}
        {kind === 'text' && textPreview.state === 'loading' && (
          <div className="flex flex-1 items-center justify-center p-8 text-sm text-muted-foreground">
            {t('common.preview_loading')}
          </div>
        )}
        {kind === 'text' && textPreview.state === 'error' && (
          <div className="flex flex-1 flex-col items-center justify-center gap-2 p-8 text-sm text-muted-foreground">
            <p>{t('common.preview_error')}</p>
          </div>
        )}
        {kind === 'text' && textPreview.state === 'ready' && (
          <pre className="flex-1 overflow-auto whitespace-pre-wrap break-words bg-muted p-4 text-xs">
            {textPreview.text}
          </pre>
        )}
        {kind === 'none' && (
          <div className="flex flex-1 flex-col items-center justify-center gap-2 p-8 text-sm text-muted-foreground">
            <p>{t('common.preview_not_inline')}</p>
          </div>
        )}
      </div>
    </div>
  )
}
