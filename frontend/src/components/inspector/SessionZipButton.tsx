import { useUiStore } from '@/stores/ui'
import { url } from '@/api/client'
import { DownloadLink } from '@/components/common/DownloadAction'
import { useT } from '@/i18n/LanguageContext'

/**
 * Triggers the server-side session-sources ZIP stream.
 *
 * Replaces the original in-browser JSZip loop (which fetched every source
 * file individually and assembled the archive on the main thread); the
 * backend now streams ``application/zip`` directly from the
 * ``qdrant-sources`` volume, so the browser only handles the download.
 */
export function SessionZipButton() {
  const t = useT()
  const sessionId = useUiStore((s) => s.currentSessionId)
  const collection = useUiStore((s) => s.selectedCollection)
  if (!sessionId || !collection) return null

  const href = url(`/sessions/${encodeURIComponent(sessionId)}/sources.zip`)
  return (
    <DownloadLink
      href={href}
      download={`session-${sessionId}-sources.zip`}
      label={t('inspector.download_session_sources')}
    />
  )
}
