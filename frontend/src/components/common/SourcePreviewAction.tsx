import { HoverIconAction } from '@infra/ui'
import { useUiStore } from '@/stores/ui'
import { useT } from '@/i18n/LanguageContext'

const PreviewGlyph = () => (
  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
    <path d="M3 7V4a1 1 0 0 1 1-1h3M17 3h3a1 1 0 0 1 1 1v3M21 17v3a1 1 0 0 1-1 1h-3M7 21H4a1 1 0 0 1-1-1v-3" />
    <rect x="8" y="8" width="8" height="8" rx="1" />
  </svg>
)

/**
 * The hover/focus-revealed "preview this document" icon, shared by every view
 * that lists sources — chat citations, entity findings, hate-speech findings,
 * and the Inspector's document table.
 *
 * Renders nothing when the source cannot be previewed: the URL needs both the
 * active collection and the file's stored hash. Mount inside a `.group`
 * container, like the sibling `TranslateToggle`.
 */
export function SourcePreviewAction({
  fileHash,
  filename,
  className
}: {
  fileHash?: string | null
  filename?: string | null
  className?: string
}) {
  const t = useT()
  const collection = useUiStore((s) => s.selectedCollection)
  const openPreview = useUiStore((s) => s.openPreview)
  if (!collection || !fileHash) return null
  return (
    <HoverIconAction
      icon={<PreviewGlyph />}
      label={t('common.preview_open')}
      onClick={() =>
        openPreview({
          collection,
          file_hash: fileHash,
          // The dialog is labelled by this, so an unnamed source still gets a
          // meaningful heading rather than an empty title bar.
          filename: filename || t('common.unknown_source')
        })
      }
      className={className}
    />
  )
}
