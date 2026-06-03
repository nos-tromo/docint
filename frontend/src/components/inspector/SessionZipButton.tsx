import { useUiStore } from '@/stores/ui'
import { url } from '@/api/client'

/**
 * Triggers the server-side session-sources ZIP stream.
 *
 * Replaces the original in-browser JSZip loop (which fetched every source
 * file individually and assembled the archive on the main thread); the
 * backend now streams ``application/zip`` directly from the
 * ``qdrant-sources`` volume, so the browser only handles the download.
 */
export function SessionZipButton() {
  const sessionId = useUiStore((s) => s.currentSessionId)
  const collection = useUiStore((s) => s.selectedCollection)
  if (!sessionId || !collection) return null

  const href = url(`/sessions/${encodeURIComponent(sessionId)}/sources.zip`)
  return (
    <a
      href={href}
      download={`session-${sessionId}-sources.zip`}
      className="px-3 py-1 rounded-md border border-border text-sm"
    >
      Download session sources (ZIP)
    </a>
  )
}
