import { useId, useState } from 'react'
import { Link } from 'react-router-dom'
import { Badge, Button, Spinner } from '@infra/ui'
import { ApiError } from '@/api/client'
import { describeError } from '@/api/errorMessage'
import type { SearchHit } from '@/api/types'
import { useChunkText } from '@/hooks/useSearch'
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

/**
 * Text with the searched keywords marked, using the index's own semantics.
 *
 * Shared by the capped preview and the expanded full chunk so both paint
 * exactly the same words — an expansion that highlighted differently would
 * read as the search having matched something else.
 */
function Highlighted({ text, keywords }: { text: string; keywords: string[] }) {
  return (
    <>
      {keywordSegments(text, keywords).map((segment, i) =>
        segment.highlight ? (
          <mark key={i} className="rounded bg-yellow-200/70 px-0.5 text-foreground dark:bg-yellow-500/30">
            {segment.text}
          </mark>
        ) : (
          <span key={i}>{segment.text}</span>
        )
      )}
    </>
  )
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
 *
 * Expanding is a *reading* control and is deliberately independent of the
 * checkbox: opening a chunk to check whether it is worth pinning must never
 * pin it. The expanded text is fetched on demand (`useChunkText`) because the
 * hit only carries a capped preview, and the expansion state is component-local
 * — it is throwaway UI state that should not survive a new search or a reload
 * the way the selection does.
 */
export function SearchHitRow({ hit, keywords, selected, onToggle }: SearchHitRowProps) {
  const t = useT()
  const label = hitLabel(hit, t)
  const [expanded, setExpanded] = useState(false)
  const chunk = useChunkText(hit.id, expanded)
  const bodyId = useId()
  const full = expanded ? (chunk.data?.text ?? null) : null

  /** A missing chunk is its own answer, never a generic request failure. */
  const chunkErrorText = (): string => {
    if (chunk.error instanceof ApiError && chunk.error.status === 404) {
      return t('search.chunk_gone')
    }
    const described = describeError(chunk.error)
    return t(described.key, described.vars)
  }

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
          {/* One body element, so an expansion replaces the capped preview
              rather than repeating its first 600 characters below itself. */}
          <p
            id={bodyId}
            data-testid={full === null ? 'hit-preview' : 'hit-full-text'}
            className={
              full === null
                ? 'mt-1 line-clamp-4 whitespace-pre-wrap break-words text-muted-foreground'
                : 'mt-1 max-h-64 overflow-auto whitespace-pre-wrap break-words text-muted-foreground'
            }
          >
            <Highlighted text={full ?? hit.preview} keywords={keywords} />
          </p>
          {expanded && chunk.isPending && (
            <p className="mt-1 flex items-center gap-2 text-muted-foreground" data-testid="hit-loading">
              <Spinner className="h-3 w-3" label={t('search.chunk_loading')} />
              <span aria-hidden="true">{t('search.chunk_loading')}</span>
            </p>
          )}
          {expanded && chunk.isError && (
            <p className="mt-1 text-red-500" role="alert" data-testid="hit-chunk-error">
              {chunkErrorText()}
            </p>
          )}
          <div className="mt-1.5 flex flex-wrap items-center gap-1">
            {/* Offered only where the preview was actually cut short —
                otherwise expanding costs a round-trip to re-fetch the text
                already on screen. Older responses omit the flag, so an
                undefined value keeps the control rather than hiding it. */}
            {hit.truncated !== false && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                aria-expanded={expanded}
                aria-controls={bodyId}
                onClick={() => setExpanded((open) => !open)}
              >
                {expanded ? t('search.collapse_hit') : t('search.expand_hit')}
              </Button>
            )}
            {/* An image hit's body is a caption and tags rather than document
                prose, so it is worth telling apart at a glance. */}
            {hit.kind === 'image' && <Badge variant="accent">{t('search.kind_image')}</Badge>}
            {hit.entity_types.map((type) => (
              <Badge key={type} variant="neutral">
                {type}
              </Badge>
            ))}
          </div>
        </div>
      </div>
    </li>
  )
}
