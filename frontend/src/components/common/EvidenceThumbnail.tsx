import { useState } from 'react'
import { sourcePreviewUrl } from '@/api/ingest'
import { useUiStore } from '@/stores/ui'
import { useT } from '@/i18n/LanguageContext'

/**
 * The source image of a finding whose chunk came from a picture — a
 * screenshot, a photographed page, a video keyframe. A caption and an OCR line
 * say what the picture shows and says; only the pixels say whether the finding
 * is what it claims, and an investigator reading a row of image findings
 * should not have to open each one to see them.
 *
 * The image is the stored source file (`/sources/preview`), the same bytes the
 * preview dialog shows and the same URL the preview action opens — for an
 * image document the file hash and the image id are one content hash, which is
 * why `imageId` gates the render: it marks the row as visual evidence, while
 * `fileHash` addresses the file. Clicking enlarges it in that dialog.
 *
 * Renders nothing when the row is not an image, when the collection or hash is
 * missing, or when the fetch fails — a broken image beside a finding reads as
 * a broken finding.
 */
export function EvidenceThumbnail({
  imageId,
  fileHash,
  filename,
  className
}: {
  imageId?: string | null
  fileHash?: string | null
  filename?: string | null
  className?: string
}) {
  const t = useT()
  const collection = useUiStore((s) => s.selectedCollection)
  const openPreview = useUiStore((s) => s.openPreview)
  const [failed, setFailed] = useState(false)
  if (!imageId || !fileHash || !collection || failed) return null
  const name = filename || t('common.unknown_source')
  return (
    <button
      type="button"
      onClick={() => openPreview({ collection, file_hash: fileHash, filename: name })}
      // The picture is the control, so it says what it is and what it does:
      // a bare "open preview" would drop the filename a sighted reader gets
      // from the pixels, and the button's own name replaces the image's alt.
      aria-label={`${t('common.preview_open')}: ${name}`}
      className={`block cursor-pointer rounded border border-border overflow-hidden hover:border-foreground/40 focus-visible:ring-1 focus-visible:ring-primary outline-none ${className ?? ''}`}
    >
      <img
        src={sourcePreviewUrl(collection, fileHash)}
        alt=""
        loading="lazy"
        onError={() => setFailed(true)}
        className="h-20 w-auto max-w-full object-contain"
      />
    </button>
  )
}
