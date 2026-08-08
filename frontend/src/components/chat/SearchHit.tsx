import { Link } from 'react-router-dom'
import { Badge } from '@infra/ui'
import type { SearchHit } from '@/api/types'
import type { Strings } from '@/i18n'
import { keywordSegments } from '@/lib/highlight'
import { useT } from '@/i18n/LanguageContext'

/** Filename plus the page/row that locates the chunk inside it. */
function hitLabel(
  hit: SearchHit,
  t: (key: keyof Strings, vars?: Record<string, string | number>) => string
): string {
  const name = hit.filename || ''
  if (hit.page !== null && hit.page !== undefined) {
    return `${name} · ${t('common.loc_page', { page: hit.page })}`
  }
  if (hit.row !== null && hit.row !== undefined) {
    return `${name} · ${t('common.loc_row', { row: hit.row })}`
  }
  return name
}

export interface SearchHitRowProps {
  hit: SearchHit
  /** Searched keywords, for client-side highlighting of the preview. */
  keywords: string[]
  selected: boolean
  onToggle: (hit: SearchHit) => void
}

/**
 * One search result: where it came from, what matched, and whether the chat
 * is scoped to it.
 *
 * The checkbox is the scope control — ticking it pins this chunk as evidence
 * the next answer must come from. The preview highlights matched words using
 * the index's own prefix semantics (see `lib/highlight.ts::keywordSegments`),
 * so what is painted is what actually matched.
 */
export function SearchHitRow({ hit, keywords, selected, onToggle }: SearchHitRowProps) {
  const t = useT()
  const label = hitLabel(hit, t)
  const segments = keywordSegments(hit.preview, keywords)

  return (
    <li className="rounded-md border border-border bg-background p-2 text-xs" data-testid="search-hit">
      <div className="flex items-start gap-2">
        <input
          type="checkbox"
          checked={selected}
          onChange={() => onToggle(hit)}
          aria-label={
            selected
              ? t('search.scope_remove_aria', { label: label || hit.id })
              : t('search.scope_add_aria', { label: label || hit.id })
          }
          className="mt-0.5 shrink-0 accent-[var(--app-accent,currentColor)]"
        />
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline justify-between gap-2">
            <span className="truncate font-medium" title={label}>
              {label || t('common.unknown_source')}
            </span>
            <Link
              to="/inspector"
              className="shrink-0 text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
              title={t('search.open_in_inspector')}
            >
              {t('search.open_in_inspector')}
            </Link>
          </div>
          <p className="mt-1 line-clamp-4 whitespace-pre-wrap break-words text-muted-foreground">
            {segments.map((segment, i) =>
              segment.highlight ? (
                <mark key={i} className="rounded bg-yellow-200/70 px-0.5 text-foreground dark:bg-yellow-500/30">
                  {segment.text}
                </mark>
              ) : (
                <span key={i}>{segment.text}</span>
              )
            )}
          </p>
          {hit.entity_types.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {hit.entity_types.map((type) => (
                <Badge key={type} variant="neutral">
                  {type}
                </Badge>
              ))}
            </div>
          )}
        </div>
      </div>
    </li>
  )
}
