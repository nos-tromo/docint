import { useState } from 'react'
import type { Source } from '@/api/types'
import { sourceLabel } from '@/lib/sourceLabel'
import { Citation } from './Citation'
import { useT } from '@/i18n/LanguageContext'

function sourceKey(s: Source, i: number): string {
  return s.id ?? `${s.filename}-${s.page ?? ''}-${s.row ?? ''}-${i}`
}

/**
 * An answer's sources as a compact pill row, mirroring the entity pills.
 *
 * Clicking a pill expands the familiar `Citation` detail card below the row
 * (already open, so preview, reference metadata, and Translate are one click
 * away); clicking the same pill again collapses it, another pill switches.
 */
export function SourcePills({ sources }: { sources: Source[] }) {
  const t = useT()
  const [selected, setSelected] = useState<string | null>(null)
  if (sources.length === 0) return null

  const selectedSource = sources.find((s, i) => sourceKey(s, i) === selected)

  return (
    <div>
      <ul className="flex flex-wrap gap-1" data-testid="source-pills">
        {sources.map((s, i) => {
          const key = sourceKey(s, i)
          return (
            <li key={key}>
              <button
                type="button"
                data-testid="source-pill"
                title={t('chat.source_toggle_details')}
                aria-expanded={key === selected}
                onClick={() => setSelected((cur) => (cur === key ? null : key))}
                className="inline-flex max-w-full items-center gap-1 rounded border border-border bg-muted px-1.5 py-0.5 text-[11px] hover:text-blue-300"
              >
                {typeof s.citation_index === 'number' && (
                  <span className="tabular-nums text-muted-foreground">{s.citation_index}</span>
                )}
                <span className="truncate">{sourceLabel(s, t)}</span>
              </button>
            </li>
          )
        })}
      </ul>
      {selectedSource && (
        <div className="mt-2">
          {/* Remount per selection so the card's open state resets when the
              user switches pills. */}
          <Citation key={selected} source={selectedSource} defaultOpen />
        </div>
      )}
    </div>
  )
}
