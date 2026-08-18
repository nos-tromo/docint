import { useId, useState } from 'react'
import { Badge, Button, ChevronDownIcon, Spinner } from '@infra/ui'
import { ApiError } from '@/api/client'
import { describeError } from '@/api/errorMessage'
import type { SearchHit } from '@/api/types'
import { useChunkText } from '@/hooks/useSearch'
import type { Strings } from '@/i18n'
import { cn } from '@/lib/cn'
import { highlightSegments, keywordSegments } from '@/lib/highlight'
import { CheckCircleIcon, CircleIcon } from '@/components/common/icons'
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
  const segments =
    keywords.length > 1
      ? highlightSegments(text, [keywords.join(' ')])
      : keywordSegments(text, keywords)
  return (
    <>
      {segments.map((segment, i) =>
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
 * The **tile itself** is the scope control — clicking anywhere on it pins this
 * chunk as evidence the next answer must come from. This replaced a checkbox:
 * in a 22rem column a checkbox spends a fixed indent on an affordance the whole
 * card can carry, and the hit-sized target is easier to hit than the box was.
 * The circle/check marker stays because *something* has to say the tile is
 * selectable before it is hovered.
 *
 * Expanding is a *reading* control and is deliberately independent of the
 * selection: opening a chunk to check whether it is worth pinning must never
 * pin it. It is therefore a sibling of the pressable region rather than a
 * button nested inside one. The expanded text is fetched on demand
 * (`useChunkText`) because the hit only carries a capped preview, and the
 * expansion state is component-local — it is throwaway UI state that should
 * not survive a new search or a reload the way the selection does.
 */
export function SearchHitRow({ hit, keywords, selected, onToggle }: SearchHitRowProps) {
  const t = useT()
  const label = hitLabel(hit, t)
  const [expanded, setExpanded] = useState(false)
  const chunk = useChunkText(hit.id, expanded)
  const bodyId = useId()
  const full = expanded ? (chunk.data?.text ?? null) : null
  // Older responses omit the flag, so an undefined value keeps the control
  // rather than hiding it. Where the preview was not cut short, expanding
  // would cost a round-trip to re-fetch the text already on screen.
  const canExpand = hit.truncated !== false

  /** A missing chunk is its own answer, never a generic request failure. */
  const chunkErrorText = (): string => {
    if (chunk.error instanceof ApiError && chunk.error.status === 404) {
      return t('search.chunk_gone')
    }
    const described = describeError(chunk.error)
    return t(described.key, described.vars)
  }

  /**
   * Toggle the selection — unless the click ended a drag over the text.
   *
   * Selecting a snippet to copy it finishes with a click on the tile, and an
   * investigator quoting a passage must not silently re-scope the chat by
   * doing so.
   */
  const press = () => {
    const selection = window.getSelection()
    if (selection && !selection.isCollapsed) return
    onToggle(hit)
  }

  return (
    <li
      className={cn(
        'relative rounded-md border p-2 text-xs transition-colors',
        selected
          ? 'border-primary bg-primary/5'
          : 'border-border bg-background hover:border-muted-foreground/40'
      )}
      data-testid="search-hit"
    >
      <div
        role="button"
        tabIndex={0}
        aria-pressed={selected}
        aria-label={
          selected
            ? t('search.scope_remove_aria', { label: label || hit.id })
            : t('search.scope_add_aria', { label: label || hit.id })
        }
        onClick={press}
        onKeyDown={(e) => {
          // A div carrying role="button" gets neither for free; Space would
          // otherwise scroll the panel instead of selecting.
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            onToggle(hit)
          }
        }}
        className="flex cursor-pointer items-start gap-2 rounded-sm outline-none focus-visible:ring-1 focus-visible:ring-primary"
      >
        <span
          className={cn('mt-0.5 shrink-0', selected ? 'text-primary' : 'text-muted-foreground')}
          aria-hidden="true"
        >
          {selected ? (
            <CheckCircleIcon className="h-3.5 w-3.5" />
          ) : (
            <CircleIcon className="h-3.5 w-3.5" />
          )}
        </span>
        <div className="min-w-0 flex-1">
          <span
            className={cn('block truncate font-medium', canExpand && 'pr-6')}
            title={label}
          >
            {label || t('common.unknown_source')}
          </span>
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
          {(hit.kind === 'image' || hit.entity_types.length > 0) && (
            <div className="mt-1.5 flex flex-wrap items-center gap-1">
              {/* An image hit's body is a caption and tags rather than document
                  prose, so it is worth telling apart at a glance. */}
              {hit.kind === 'image' && <Badge variant="accent">{t('search.kind_image')}</Badge>}
              {hit.entity_types.map((type) => (
                <Badge key={type} variant="neutral">
                  {type}
                </Badge>
              ))}
            </div>
          )}
        </div>
      </div>
      {canExpand && (
        <Button
          type="button"
          variant="ghost"
          size="sm"
          aria-expanded={expanded}
          aria-controls={bodyId}
          aria-label={expanded ? t('search.collapse_hit') : t('search.expand_hit')}
          title={expanded ? t('search.collapse_hit') : t('search.expand_hit')}
          onClick={() => setExpanded((open) => !open)}
          className="absolute right-1 top-1 h-6 w-6 px-0"
        >
          <ChevronDownIcon
            className={cn('h-4 w-4 transition-transform', expanded && 'rotate-180')}
          />
        </Button>
      )}
    </li>
  )
}
