import { useState } from 'react'
import type { Source } from '@/api/types'
import { sourcePreviewUrl } from '@/api/ingest'
import { useUiStore } from '@/stores/ui'
import { formatScore, sourceLabel } from '@/lib/sourceLabel'
import { referenceMetadataItems } from '@/lib/referenceMetadata'
import { TranslateControl } from '@/components/common/TranslateControl'

export function Citation({ source }: { source: Source }) {
  const [open, setOpen] = useState(false)
  const collection = useUiStore((s) => s.selectedCollection)
  const refMeta = referenceMetadataItems(source.reference_metadata)
  return (
    <div className="rounded-md border border-border bg-zinc-900 px-3 py-2 text-sm">
      <button
        type="button"
        className="flex items-center justify-between w-full gap-2"
        onClick={() => setOpen((v) => !v)}
      >
        <span className="truncate">{sourceLabel(source)}</span>
        <span className="text-xs text-muted-foreground">{formatScore(source.score)}</span>
      </button>
      {open && (
        <div className="mt-2 space-y-2">
          {refMeta.length > 0 && (
            <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
              {refMeta.map(({ label, value }) => (
                <div key={label} className="contents">
                  <dt className="text-muted-foreground">{label}</dt>
                  <dd className="break-words">{value}</dd>
                </div>
              ))}
            </dl>
          )}
          {source.text && (
            <div className="group relative whitespace-pre-wrap text-xs bg-zinc-950 p-2 rounded">
              <TranslateControl text={source.text} className="absolute right-1 top-1" />
              {source.text}
            </div>
          )}
          {collection && source.file_hash && (
            <a
              href={sourcePreviewUrl(collection, source.file_hash)}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-blue-400 hover:text-blue-300"
            >
              Open original ↗
            </a>
          )}
        </div>
      )}
    </div>
  )
}
