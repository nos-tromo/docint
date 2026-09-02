import { useState } from 'react'
import { DownloadIcon, HoverIconAction } from '@infra/ui'
import { createExtract, sourceExtractHref } from '@/api/extracts'
import { useAppendixFields } from '@/hooks/useAppendixFields'
import { useUiStore } from '@/stores/ui'
import { useT } from '@/i18n/LanguageContext'

/**
 * The hover-revealed "download this source's extract" action on a document row.
 *
 * Fetched rather than linked, because one answer is not a file: a postings
 * table's hash expands to every post recorded in it, and the server refuses
 * that with 413 rather than rendering for minutes on the request. On a 413 the
 * click queues a targeted job instead, so one control either downloads or
 * starts a build and the user never meets a dead end.
 */
export function SourceExtractAction({
  fileHash,
  filename
}: {
  fileHash?: string | null
  filename?: string | null
}) {
  const t = useT()
  const collection = useUiStore((s) => s.selectedCollection)
  const appendix = useAppendixFields()
  const [busy, setBusy] = useState(false)
  if (!collection || !fileHash) return null

  const onClick = async () => {
    setBusy(true)
    try {
      const response = await fetch(sourceExtractHref(collection, fileHash, 'zip', appendix))
      if (response.status === 413) {
        await createExtract(collection, fileHash, appendix)
        return
      }
      if (!response.ok) return
      const blob = await response.blob()
      const anchor = document.createElement('a')
      anchor.href = URL.createObjectURL(blob)
      anchor.download = `${(filename || fileHash).replace(/\.[^.]+$/, '')}-extract.zip`
      anchor.click()
      URL.revokeObjectURL(anchor.href)
    } finally {
      setBusy(false)
    }
  }

  return (
    <HoverIconAction
      icon={<DownloadIcon className="h-4 w-4" />}
      label={t('extract.download_source')}
      onClick={() => void onClick()}
      disabled={busy}
    />
  )
}
