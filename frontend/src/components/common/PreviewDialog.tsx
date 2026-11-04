import { useEffect, useMemo, useRef, useState } from 'react'
import { ExternalLinkIcon, IconButton, XIcon } from '@infra/ui'
import { fetchFile, viewFile, type FetchedFile } from '@/api/files'
import { sourcePreviewPath, sourceThumbnailPath } from '@/api/ingest'
import { useFileAction } from '@/hooks/useFileAction'
import { useObjectUrl } from '@/hooks/useObjectUrl'
import { openBlob } from '@/lib/files'
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
// html is deliberately in the *text* bucket: a framed HTML file would be a
// same-origin document running its scripts against the app. Showing the
// source is safe — and more useful for evidence review anyway.
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

type PreviewFile =
  | { state: 'idle' | 'loading' | 'error' }
  | { state: 'ready'; file: FetchedFile; text: string | null }

/**
 * The file a preview shows, fetched through the purpose wrapper.
 *
 * An `<iframe src>` or `<img src>` cannot state the purpose, so the dialog
 * fetches the file itself and shows it from memory.
 *
 * @param path - The API path to fetch, or null while nothing is to be shown.
 * @param asText - Also read the body as text, for the text renderer.
 * @param owner - The admin owner context the fetch resolves under; a change refetches.
 * @returns Loading / error / the file for the current path.
 */
function usePreviewFile(path: string | null, asText: boolean, owner: string | null): PreviewFile {
  const [result, setResult] = useState<PreviewFile>({ state: 'idle' })

  useEffect(() => {
    if (!path) {
      setResult({ state: 'idle' })
      return
    }
    // Abort rather than just ignore: a superseded preview should stop occupying
    // a connection, not run to completion with its result thrown away.
    const controller = new AbortController()
    setResult({ state: 'loading' })
    fetchFile(path, controller.signal)
      .then(async (file) => {
        const text = asText ? await file.blob.text() : null
        if (!controller.signal.aborted) setResult({ state: 'ready', file, text })
      })
      .catch(() => {
        // An abort means this effect was superseded, not that the preview failed.
        if (!controller.signal.aborted) setResult({ state: 'error' })
      })
    return () => {
      controller.abort()
    }
  }, [path, asText, owner])

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
  const owner = useUiStore((s) => s.selectedOwner)
  const closePreview = useUiStore((s) => s.closePreview)
  const panelRef = useRef<HTMLDivElement | null>(null)
  const openerRef = useRef<HTMLElement | null>(null)
  const newTab = useFileAction()

  // Frozen evidence and a stored image are always a JPEG thumbnail, whatever
  // the source document they were cut from was called — a keyframe's filename
  // ends in .mp4. So the extension decides the kind only for a live source file.
  const frozen = modal?.data_uri
  const kind: PreviewKind = frozen || modal?.image_id ? 'image' : modal ? previewKind(modal.filename) : 'none'
  const path = !modal?.collection
    ? null
    : modal.image_id
      ? sourceThumbnailPath(modal.collection, modal.image_id)
      : sourcePreviewPath(modal.collection, modal.file_hash ?? '')
  // What the dialog shows is fetched as it opens; a type it cannot show is
  // fetched only when it is asked for in a new tab.
  const preview = usePreviewFile(kind === 'none' ? null : path, kind === 'text', owner)
  const ready = preview.state === 'ready' ? preview : null
  // Framed as a PDF whatever it was served as: the frame is a same-origin
  // document, and only the browser's PDF viewer belongs in it.
  const shown = useMemo(
    () =>
      ready && kind === 'frame' ? new Blob([ready.file.blob], { type: 'application/pdf' }) : (ready?.file.blob ?? null),
    [ready, kind]
  )
  const shownUrl = useObjectUrl(kind === 'text' ? null : shown)

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

  if (!modal || (!frozen && !path)) return null

  // The file already fetched opens at once, inside the click's user
  // activation; anything else is fetched first.
  const openInNewTab = () => {
    if (ready) openBlob(ready.file.blob, modal.filename)
    else if (path) void newTab.run(() => viewFile(path, modal.filename))
  }
  const imageSrc = frozen ?? shownUrl

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
            {/* No new tab for frozen evidence: browsers block top-level
                navigation to a data: URL, so the control would promise
                something it cannot do. */}
            {!frozen && (
              <button
                type="button"
                onClick={openInNewTab}
                disabled={newTab.busy}
                aria-busy={newTab.busy || undefined}
                className="inline-flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300 disabled:opacity-60"
              >
                {t('common.preview_new_tab')}
                <ExternalLinkIcon className="h-3.5 w-3.5" />
              </button>
            )}
            {/* The base `IconButton` rather than `RemoveButton`: closing a
                preview takes nothing away, so it must not warn in red. */}
            <IconButton icon={<XIcon />} label={t('common.preview_close')} onClick={closePreview} />
          </div>
        </div>
        {kind !== 'none' && !frozen && preview.state !== 'ready' && preview.state !== 'error' && (
          <div className="flex flex-1 items-center justify-center p-8 text-sm text-muted-foreground">
            {t('common.preview_loading')}
          </div>
        )}
        {kind !== 'none' && !frozen && preview.state === 'error' && (
          <div className="flex flex-1 flex-col items-center justify-center gap-2 p-8 text-sm text-muted-foreground">
            <p>{t('common.preview_error')}</p>
          </div>
        )}
        {kind === 'frame' && shownUrl && (
          <iframe title={modal.filename} src={shownUrl} className="h-full w-full flex-1 bg-muted" />
        )}
        {kind === 'image' && imageSrc && (
          <div className="flex flex-1 items-center justify-center overflow-auto bg-muted p-4">
            <img src={imageSrc} alt={modal.filename} className="max-h-full max-w-full object-contain" />
          </div>
        )}
        {kind === 'text' && ready && (
          <pre className="flex-1 overflow-auto whitespace-pre-wrap break-words bg-muted p-4 text-xs">
            {ready.text}
          </pre>
        )}
        {kind === 'none' && (
          <div className="flex flex-1 flex-col items-center justify-center gap-2 p-8 text-sm text-muted-foreground">
            <p>{t('common.preview_not_inline')}</p>
            {newTab.failed && <p role="alert">{t('common.download_failed')}</p>}
          </div>
        )}
      </div>
    </div>
  )
}
