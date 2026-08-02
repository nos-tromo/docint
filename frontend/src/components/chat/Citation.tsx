import { useState } from 'react'
import type { Source } from '@/api/types'
import { sourcePreviewUrl } from '@/api/ingest'
import { useUiStore } from '@/stores/ui'
import { sourceLabel } from '@/lib/sourceLabel'
import { referenceMetadataItems } from '@/lib/referenceMetadata'
import { TranslateControl } from '@/components/common/TranslateControl'
import { useT } from '@/i18n/LanguageContext'

export function Citation({ source }: { source: Source }) {
  const t = useT()
  const [open, setOpen] = useState(false)
  const collection = useUiStore((s) => s.selectedCollection)
  const refMeta = referenceMetadataItems(source.reference_metadata, {}, t)
  return (
    <div className="rounded-md border border-border bg-muted px-3 py-2 text-sm">
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
          {collection && source.file_hash && (
            <a
              href={sourcePreviewUrl(collection, source.file_hash)}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-blue-400 hover:text-blue-300"
            >
              {t('chat.open_original')}
            </a>
          )}
        </div>
      )}
    </div>
  )
}
