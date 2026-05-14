import { useState } from 'react'
import type { Source } from '@/api/types'
import { sourcePreviewUrl } from '@/api/ingest'
import { useUiStore } from '@/stores/ui'
import { formatScore, sourceLabel } from '@/lib/sourceLabel'

export function Citation({ source }: { source: Source }) {
  const [open, setOpen] = useState(false)
  const collection = useUiStore((s) => s.selectedCollection)
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
          {source.text && (
            <pre className="whitespace-pre-wrap text-xs bg-zinc-950 p-2 rounded">
              {source.text}
            </pre>
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
