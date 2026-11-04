import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchFile } from '@/api/files'
import { sourceThumbnailPath } from '@/api/ingest'
import { previewKind } from '@/components/common/PreviewDialog'
import { useActiveCollection } from '@/hooks/useActiveCollection'
import { useObjectUrl } from '@/hooks/useObjectUrl'
import { useUiStore } from '@/stores/ui'
import { useT } from '@/i18n/LanguageContext'

/**
 * How long a thumbnail nothing shows stays in memory: long enough that a row
 * the list virtualizer unmounted and scrolled back into view does not fetch
 * — and record — the same picture again.
 */
const THUMBNAIL_GC_MS = 5 * 60_000

/**
 * The source image of a finding whose chunk came from a picture — a
 * screenshot, a photographed page, a video keyframe. A caption and an OCR line
 * say what the picture shows and says; only the pixels say whether the finding
 * is what it claims, and an investigator reading a row of image findings
 * should not have to open each one to see them.
 *
 * The pixels are addressed by `imageId` (`/sources/thumbnail`), never by the
 * row's `fileHash`: that names the file the finding came from, which for a
 * keyframe is its clip and for a figure its PDF — nothing an `<img>` can show.
 * They are fetched through the purpose wrapper, since an `<img src>` cannot
 * state the purpose, and shown from memory. Clicking enlarges the picture: a
 * still from its own file, at full size, and a keyframe or figure from the
 * stored image, since its file is not one.
 *
 * Renders nothing when the row is not an image, when no collection is
 * selected, or when the fetch fails — a broken image beside a finding reads as
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
  const scope = useActiveCollection()
  const { collection } = scope
  const openPreview = useUiStore((s) => s.openPreview)
  const [failed, setFailed] = useState(false)
  const thumbnail = useQuery({
    queryKey: ['source-thumbnail', scope, imageId],
    queryFn: async () => (await fetchFile(sourceThumbnailPath(collection!, imageId!))).blob,
    enabled: !!collection && !!imageId,
    staleTime: Infinity,
    gcTime: THUMBNAIL_GC_MS
  })
  const src = useObjectUrl(thumbnail.data)
  if (!imageId || !collection || failed || thumbnail.isError) return null
  const name = filename || t('common.unknown_source')
  return (
    <button
      type="button"
      onClick={() =>
        openPreview(
          fileHash && previewKind(name) === 'image'
            ? { collection, file_hash: fileHash, filename: name }
            : { collection, image_id: imageId, filename: name }
        )
      }
      // The picture is the control, so it says what it is and what it does:
      // a bare "open preview" would drop the filename a sighted reader gets
      // from the pixels, and the button's own name replaces the image's alt.
      aria-label={`${t('common.preview_open')}: ${name}`}
      className={`block cursor-pointer rounded border border-border overflow-hidden hover:border-foreground/40 focus-visible:ring-1 focus-visible:ring-primary outline-none ${className ?? ''}`}
    >
      <img src={src} alt="" onError={() => setFailed(true)} className="h-20 w-auto max-w-full object-contain" />
    </button>
  )
}
