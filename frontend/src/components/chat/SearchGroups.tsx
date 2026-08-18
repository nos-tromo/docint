import { useState } from 'react'
import { Button, ChevronDownIcon } from '@infra/ui'
import { cn } from '@/lib/cn'
import type { AggregateGroup, AggregateResult, SearchHit } from '@/api/types'
import { SearchHitRow } from '@/components/chat/SearchHit'
import { useT } from '@/i18n/LanguageContext'

export interface SearchGroupsProps {
  result: AggregateResult
  keywords: string[]
  selectedTokens: Record<string, number>
  onToggle: (hit: SearchHit) => void
}

/**
 * The grouped view of a search: one row per payload value, its match count,
 * and (on demand) its sample chunks — pinnable exactly like a plain hit.
 * Counts are chunks, and the summary line says so.
 */
export function SearchGroups({ result, keywords, selectedTokens, onToggle }: SearchGroupsProps) {
  const t = useT()
  const [open, setOpen] = useState<Record<string, boolean>>({})
  // Mirrors SearchPanel's hits-mode rule: "no matches" is only true for a
  // genuine `ok` empty result. A `not_indexed`/`partial` zero-group result
  // renders nothing here — the panel's banner already explains why, and
  // stacking this text under it would claim the evidence is not there when
  // it merely is not indexed yet.
  if (result.groups.length === 0 && result.status === 'ok') {
    return (
      <p className="text-xs text-muted-foreground" data-testid="search-no-groups">
        {t('search.no_groups')}
      </p>
    )
  }
  return (
    <ul className="min-h-0 flex-1 space-y-1 overflow-auto pr-1" data-testid="search-groups">
      {result.groups.map((g: AggregateGroup) => {
        const isOpen = !!open[g.value]
        const hasSamples = g.samples.length > 0
        return (
          <li key={g.value} className="rounded-md border border-border">
            <div className="flex items-center gap-2 px-2 py-1">
              <span className="min-w-0 flex-1 truncate text-sm">{g.value}</span>
              <span className="shrink-0 text-xs text-muted-foreground">
                {t('search.group_count', { count: g.count })}
              </span>
              {hasSamples && (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-7 w-7 shrink-0 px-0"
                  aria-expanded={isOpen}
                  aria-label={isOpen ? t('search.group_collapse') : t('search.group_expand')}
                  onClick={() => setOpen((s) => ({ ...s, [g.value]: !isOpen }))}
                >
                  <ChevronDownIcon className={cn('h-4 w-4 transition-transform', isOpen && 'rotate-180')} />
                </Button>
              )}
            </div>
            {isOpen && (
              <ul className="space-y-2 p-2 pt-0">
                {g.samples.map((hit) => (
                  <SearchHitRow
                    key={hit.id}
                    hit={hit}
                    keywords={keywords}
                    selected={hit.id in selectedTokens}
                    onToggle={onToggle}
                  />
                ))}
              </ul>
            )}
          </li>
        )
      })}
    </ul>
  )
}
