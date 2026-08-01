import { useState } from 'react'
import type { Source } from '@/api/types'
import { sourceLabel } from '@/lib/sourceLabel'
import { referenceMetadataItems } from '@/lib/referenceMetadata'
import { TranslateControl } from '@/components/common/TranslateControl'
import { SourcePreviewAction } from '@/components/common/SourcePreviewAction'
import { useT } from '@/i18n/LanguageContext'

export function Citation({ source }: { source: Source }) {
  const t = useT()
  const [open, setOpen] = useState(false)
  const refMeta = referenceMetadataItems(source.reference_metadata, {}, t)
  return (
    <div className="group relative rounded-md border border-border bg-muted px-3 py-2 text-sm">
      <div className="flex items-center gap-2">
        {/* The badge sits outside the disclosure button so the button's
            accessible name stays the source label. */}
        {typeof source.citation_index === 'number' && (
          <span
            title={t('chat.source_number', { n: source.citation_index })}
            className="shrink-0 rounded border border-border bg-background px-1.5 py-0.5 text-xs tabular-nums text-muted-foreground"
          >
            {source.citation_index}
          </span>
        )}
        <button
          type="button"
          className="flex min-w-0 flex-1 items-center justify-between gap-2 text-left"
          onClick={() => setOpen((v) => !v)}
        >
          <span className="truncate">{sourceLabel(source, t)}</span>
        </button>
        {/* The preview belongs in the header row, not as its own full-width
            row inside the expanded panel: it is an action on the source
            rather than part of its evidence, and it no longer leaves the app. */}
        <SourcePreviewAction fileHash={source.file_hash} filename={source.filename} />
      </div>
      {open && (
        <div className="mt-2 space-y-2">
          {refMeta.length > 0 && (
            <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-xs">
              {refMeta.map(({ label, value }) => (
                <div key={label} className="contents">
                  <dt className="text-muted-foreground">{label}</dt>
                  <dd className="break-words">{value}</dd>
                </div>
              ))}
            </dl>
          )}
          {source.text && <TranslateControl rawText={source.text} />}
        </div>
      )}
    </div>
  )
}
