import { useState } from 'react'
import JSZip from 'jszip'
import { useUiStore } from '@/stores/ui'
import { useSessionHistory } from '@/hooks/useSessions'
import { sourcePreviewUrl } from '@/api/ingest'

export function SessionZipButton() {
  const sessionId = useUiStore((s) => s.currentSessionId)
  const collection = useUiStore((s) => s.selectedCollection)
  const history = useSessionHistory(sessionId)
  const [busy, setBusy] = useState(false)

  const build = async () => {
    if (!sessionId || !collection || !history.data) return
    setBusy(true)
    try {
      const zip = new JSZip()
      const seen = new Set<string>()
      for (const m of history.data.messages) {
        for (const c of m.citations ?? []) {
          if (seen.has(c.file_hash)) continue
          seen.add(c.file_hash)
          const res = await fetch(sourcePreviewUrl(collection, c.file_hash))
          if (!res.ok) continue
          zip.file(c.filename, await res.blob())
        }
      }
      const blob = await zip.generateAsync({ type: 'blob' })
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = `session-${sessionId}-sources.zip`
      a.click()
      URL.revokeObjectURL(a.href)
    } finally {
      setBusy(false)
    }
  }

  if (!sessionId || !collection) return null
  return (
    <button
      type="button"
      onClick={build}
      disabled={busy}
      className="px-3 py-1 rounded-md border border-border text-sm disabled:opacity-50"
    >
      {busy ? 'Building…' : 'Download session sources (ZIP)'}
    </button>
  )
}
