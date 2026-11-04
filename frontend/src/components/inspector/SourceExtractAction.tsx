import { useState } from 'react'
import { DownloadIcon, HoverIconAction } from '@infra/ui'
import { ApiError, isPurposeRefusal } from '@/api/client'
import { createExtract, sourceExtractPath } from '@/api/extracts'
import { fetchFile } from '@/api/files'
import { saveBlob } from '@/lib/files'
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
 * starts a build and the user never meets a dead end. The fetch goes through
 * the purpose wrapper, so it carries the stated purpose.
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
      const { blob } = await fetchFile(sourceExtractPath(collection, fileHash, 'zip', appendix))
      saveBlob(blob, `${(filename || fileHash).replace(/\.[^.]+$/, '')}-extract.zip`)
    } catch (err) {
      // Too large to render on the request: build it in the background.
      if (err instanceof ApiError && err.status === 413) await createExtract(collection, fileHash, appendix)
      else if (!isPurposeRefusal(err)) console.error('Source extract download failed', err)
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
