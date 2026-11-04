import { useUiStore } from '@/stores/ui'
import { downloadFile } from '@/api/files'
import { sessionSourcesPath } from '@/api/sessions'
import { FileDownloadButton } from '@/components/common/FileDownloadButton'
import { useT } from '@/i18n/LanguageContext'

/**
 * Downloads the server-side session-sources ZIP.
 *
 * Replaces the original in-browser JSZip loop (which fetched every source
 * file individually and assembled the archive on the main thread); the
 * backend streams ``application/zip`` straight from the ``qdrant-sources``
 * volume. It is fetched through the purpose wrapper rather than linked, so it
 * carries the stated purpose and, for an admin, the owner it belongs to.
 */
export function SessionZipButton() {
  const t = useT()
  const sessionId = useUiStore((s) => s.currentSessionId)
  const collection = useUiStore((s) => s.selectedCollection)
  if (!sessionId || !collection) return null

  return (
    <FileDownloadButton
      label={t('inspector.download_session_sources')}
      download={() => downloadFile(sessionSourcesPath(sessionId), `session-${sessionId}-sources.zip`)}
    />
  )
}
