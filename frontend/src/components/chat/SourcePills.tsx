import { useState } from 'react'
import type { Source } from '@/api/types'
import { sourceLabel } from '@/lib/sourceLabel'
import { Citation } from './Citation'
import { useT } from '@/i18n/LanguageContext'

function sourceKey(s: Source, i: number): string {
  return s.id ?? `${s.filename}-${s.page ?? ''}-${s.row ?? ''}-${i}`
}

/**
 * An answer's sources as a row of round chips, mirroring the entity chips (Ghostwriter's result chips).
 *
 * Clicking a chip marks it and opens its details (`Citation`) below the row:
 * the passage, reference metadata, Translate and preview, one click away. The
 * marked chip is their heading, so the details repeat neither its number nor
 * its label. Clicking the same chip again closes them, another chip switches.
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
                className="inline-flex max-w-full items-center gap-1 rounded-full border border-border bg-muted px-2.5 py-1 text-xs hover:bg-accent aria-expanded:border-foreground/40 aria-expanded:bg-accent"
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
          {/* Remount per selection so a translation shown for one source does
              not carry over to the next. */}
          <Citation key={selected} source={selectedSource} />
        </div>
      )}
    </div>
  )
}
