import { useState } from 'react'
import { Banner, Button, Card, DownloadLink, Input, SearchButton, SelectMenu, XIcon } from '@infra/ui'
import { cn } from '@/lib/cn'
import { ApiError } from '@/api/client'
import { describeError } from '@/api/errorMessage'
import { searchExportHref } from '@/api/search'
import type { GroupByField, SearchHit } from '@/api/types'
import { GROUP_BY_FIELDS } from '@/api/types'
import { useAggregate, useSearch, useScope } from '@/hooks/useSearch'
import {
  scopeChunkIds,
  scopeEstTokens,
  scopeFor,
  searchKeyFor,
  useSearchUiStore
} from '@/stores/searchUi'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { useUiStore } from '@/stores/ui'
import { SearchGroups } from '@/components/chat/SearchGroups'
import { SearchHitRow } from '@/components/chat/SearchHit'
import { CheckAllIcon } from '@/components/common/icons'
import { useT } from '@/i18n/LanguageContext'

/**
 * Compact a token count for the meter (`12400` -> `12.4k`).
 *
 * @param n - A token count.
 * @returns The compacted label.
 */
export function formatTokens(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(Math.round(n))
}

/**
 * Split a submitted query the way the backend does — whitespace-separated
 * keywords, all of which must match.
 *
 * @param query - The submitted query text.
 * @returns The keywords, for client-side highlighting.
 */
export function queryKeywords(query: string): string[] {
  return query.trim().split(/\s+/).filter(Boolean)
}

export interface SearchPanelProps {
  /** The open session, or null before the backend has minted one. */
  sessionId: string | null
}

/**
 * Full-text search over the active collection, with the hits doubling as the
 * chat's evidence picker.
 *
 * Three response states are kept visually distinct and must stay that way: a
 * collection that was never search-indexed, a backfill that is incomplete
 * (hits *plus* a warning that the list is short), and a genuine zero-match
 * search. Collapsing any of them into "no results" tells an investigator the
 * evidence is not there when it merely is not indexed.
 */
export function SearchPanel({ sessionId }: SearchPanelProps) {
  const t = useT()
  const key = searchKeyFor(sessionId)
  const collection = useUiStore((s) => s.selectedCollection)

  const draft = useSearchUiStore((s) => s.drafts[key] ?? '')
  const query = useSearchUiStore((s) => s.queries[key] ?? '')
  const setDraft = useSearchUiStore((s) => s.setDraft)
  const setQuery = useSearchUiStore((s) => s.setQuery)
  const setScopeTokens = useSearchUiStore((s) => s.setScopeTokens)
  const setScopeMeta = useSearchUiStore((s) => s.setScopeMeta)
  const scope = useSearchUiStore((s) => scopeFor(s, key))
  const mode = useSearchUiStore((s) => s.mode)
  const groupBy = useSearchUiStore((s) => s.groupBy)
  const setMode = useSearchUiStore((s) => s.setMode)
  const setGroupBy = useSearchUiStore((s) => s.setGroupBy)

  // Hits mode keeps the ranked top-k query; Groups mode runs the exhaustive
  // facet query instead. Both read the same submitted `query` and filters, so
  // switching modes never re-asks the question.
  const search = useSearch(mode === 'hits' ? query : '')
  const grouped = useAggregate(query, groupBy, mode === 'groups')
  const filters = useChatFiltersStore().buildPayload()
  const active = mode === 'groups' ? grouped : search
  const { set } = useScope(sessionId)
  const [scopeError, setScopeError] = useState<string | null>(null)

  const keywords = queryKeywords(query)
  const selectedIds = scopeChunkIds(scope)
  const estTokens = scopeEstTokens(scope)
  const hits = search.data?.hits ?? []
  const docCount = new Set(hits.map((h) => h.filename ?? '').filter(Boolean)).size
  // What "everything on screen" means in the active mode: Hits' ranked page,
  // or every sample the exhaustive Groups/Social query already loaded — not
  // gated on any one group's disclosure being open, since the data is there
  // either way. This is what lets mark-all work in Social mode at all.
  const visibleHits: SearchHit[] =
    mode === 'hits' ? hits : (grouped.data?.groups.flatMap((g) => g.samples) ?? [])

  // The scope everything currently loaded would produce: what is already
  // picked plus every hit/sample on screen. Selecting all is additive — it
  // must not silently drop chunks picked from an earlier query or mode.
  const allLoadedTokens = (): Record<string, number> => {
    const next = { ...scope.tokens }
    for (const hit of visibleHits) next[hit.id] = hit.est_tokens
    return next
  }
  // Whether the toggle is currently "on". A live selection with nothing loaded
  // counts as on too: the control's only remaining job there is to clear it.
  const allLoadedSelected =
    visibleHits.length > 0
      ? visibleHits.every((hit) => hit.id in scope.tokens)
      : selectedIds.length > 0
  const projectedTokens = Object.values(allLoadedTokens()).reduce((sum, n) => sum + n, 0)
  const projectedOverBudget = scope.usableTokens > 0 && projectedTokens > scope.usableTokens

  /**
   * Write a selection: optimistic locally, authoritative server-side.
   *
   * Shared by the tile, select-all and clear so all three keep the same
   * rollback — the server refuses an oversize scope with 422, and a local
   * selection left standing after that refusal would claim evidence the next
   * answer will not use.
   */
  const commitScope = async (next: Record<string, number>) => {
    const previous = scope.tokens
    // Optimistic: the meter must move on the click, not a round trip later.
    setScopeTokens(key, next)
    setScopeError(null)
    // No session id yet (the backend mints one on the first turn), so there is
    // nothing to write to. The selection is held locally and flushed by Chat
    // as soon as the session exists.
    if (!sessionId) return
    try {
      const result = await set.mutateAsync(Object.keys(next))
      setScopeMeta(key, { usableTokens: result.usable_tokens, missing: result.missing })
    } catch (err) {
      // The server refused, so the local selection would otherwise lie about
      // what the next answer will use. Roll it back.
      setScopeTokens(key, previous)
      if (err instanceof ApiError && err.status === 422) {
        setScopeError(t('search.budget_exceeded'))
      } else {
        const described = describeError(err)
        setScopeError(t(described.key, described.vars))
      }
    }
  }

  const toggle = (hit: SearchHit) => {
    const next = { ...scope.tokens }
    if (hit.id in next) delete next[hit.id]
    else next[hit.id] = hit.est_tokens
    return commitScope(next)
  }

  const activeError = (): string => {
    if (active.error instanceof ApiError && active.error.status === 422) {
      return t('search.error_query')
    }
    const described = describeError(active.error)
    return t(described.key, described.vars)
  }

  return (
    // A Card gives the column its own muted surface; every control inside is
    // an `@infra/ui` primitive on `bg-background`, so nothing sits invisibly
    // muted-on-muted the way the hand-rolled controls did.
    <Card className="flex h-full min-h-0 flex-col gap-2 p-3" data-testid="search-panel">
      <form
        onSubmit={(e) => {
          e.preventDefault()
          setQuery(key, draft)
        }}
        className="flex items-center gap-2"
      >
        <Input
          value={draft}
          onChange={(e) => setDraft(key, e.target.value)}
          placeholder={t('search.placeholder')}
          aria-label={t('search.title')}
          className="min-w-0 flex-1"
        />
        {/* size="md" matches the Input's h-10: the pair never lined up while
            this was a sm button beside a taller field. */}
        <SearchButton label={t('search.submit')} type="submit" variant="secondary" size="md" />
      </form>

      {/* Hits (ranked top-k) vs Groups (exhaustive, faceted by a payload
          field) — a toggle rather than two panels, so the query and the
          scope stay put across the switch. The group-by picker only makes
          sense once Groups is chosen. */}
      <div className="flex items-center gap-2" data-testid="search-mode-row">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          aria-pressed={mode === 'hits'}
          onClick={() => setMode('hits')}
          className={cn('text-xs', mode === 'hits' && 'font-semibold text-foreground')}
        >
          {t('search.mode_hits')}
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          aria-pressed={mode === 'groups'}
          onClick={() => setMode('groups')}
          className={cn('text-xs', mode === 'groups' && 'font-semibold text-foreground')}
        >
          {t('search.mode_groups')}
        </Button>
        {mode === 'groups' && (
          <SelectMenu
            options={GROUP_BY_FIELDS.map((field) => ({
              value: field,
              label: t(`search.group_by.${field}`)
            }))}
            value={groupBy}
            onChange={(value) => setGroupBy(value as GroupByField)}
            label={t('search.group_by')}
            className="min-w-0"
            triggerClassName="text-xs font-medium"
          />
        )}
      </div>

      {!collection ? (
        <p className="text-xs text-muted-foreground">{t('search.select_collection')}</p>
      ) : (
        <>
          {active.isFetching && (
            <p className="text-xs text-muted-foreground">{t('search.searching')}</p>
          )}
          {active.isError && (
            <p className="text-xs text-red-500" role="alert">
              {activeError()}
            </p>
          )}
          {/* One line for everything the result set is — hits or groups,
              documents, what the selection costs — and the two things you can
              do to all of it at once, in both modes. Deliberately one line:
              the meter used to be a row of its own that appeared on the first
              selection and shoved the whole hit list down, so picking
              evidence moved the thing you were reading. Content changes, row
              count does not — and now so does the row itself: Hits and
              Groups/Social used to keep separate summary rows with different
              controls and different alignment, which is exactly what let
              mark-all, export and the token meter drift out of sync between
              tabs.

              The row follows the *selection* as well as what's loaded: after
              picking chunks and then searching for something with no
              matches, the selection is still live and must stay clearable
              from here.

              The trailing controls are icons because their labels were the
              longest text in a 22rem column while saying the least. Their
              tooltips can afford full sentences, so that is where the promise
              ("the results loaded so far, not every match") and the
              projected cost live — the *danger* case keeps its own visible
              line below. Both controls share `h-7 w-7 shrink-0 px-0` so they
              measure identically in both modes; the summary `<p>`'s `flex-1`
              pins them upper-right, flush with the column above.

              The token meter is its own `shrink-0` flex child now, never text
              inside that `<p>`. The counts prose is unbounded — Groups/Social
              alone can stack group count, doc/unassigned count and the capped
              notice into four `·`-joined segments, easily past the ~45
              characters this column gives it at `text-xs` — so *something*
              has to give under `truncate`. It must be the counts, not the
              meter: an ellipsised group count still reads as "more than
              fits", but an ellipsised token count reads as "the selection
              fits" when it doesn't, or vanishes right when it matters most —
              the moment a selection is live. Keeping the meter a sibling
              rather than trailing content also means it needs no leading
              separator of its own to worry about running into the counts. */}
          {(active.data || selectedIds.length > 0) && (
            <div className="flex items-center gap-1" data-testid="search-summary-row">
              <p
                className="min-w-0 flex-1 truncate text-xs text-muted-foreground"
                data-testid="search-summary"
              >
                {mode === 'hits' && search.data && (
                  <>
                    {search.data.total != null
                      ? t('search.hits', { count: search.data.total })
                      : t('search.results')}
                    {' · '}
                    {search.data.next_cursor
                      ? t('search.docs_more', { count: docCount })
                      : t('search.docs', { count: docCount })}
                  </>
                )}
                {mode === 'groups' && grouped.data && (
                  <>
                    {t('search.groups_summary', {
                      groups: grouped.data.groups.length,
                      total: grouped.data.total
                    })}
                    {grouped.data.unassigned > 0 && (
                      <>
                        {' · '}
                        {t('search.groups_unassigned', { count: grouped.data.unassigned })}
                      </>
                    )}
                    {grouped.data.groups.length >= grouped.data.limit && (
                      <>
                        {' · '}
                        {t('search.groups_capped', { limit: grouped.data.limit })}
                      </>
                    )}
                  </>
                )}
              </p>
              {selectedIds.length > 0 && (
                <span
                  className="shrink-0 whitespace-nowrap text-xs text-muted-foreground"
                  data-testid="token-meter"
                >
                  {scope.usableTokens > 0
                    ? t('search.budget', {
                        used: formatTokens(estTokens),
                        total: formatTokens(scope.usableTokens)
                      })
                    : t('search.budget_selected', { used: formatTokens(estTokens) })}
                </span>
              )}
              {collection &&
                (mode === 'hits'
                  ? hits.length > 0 && query.trim() !== ''
                  : (grouped.data?.groups.length ?? 0) > 0) && (
                  <DownloadLink
                    href={searchExportHref(collection, {
                      question: query,
                      groupBy: mode === 'groups' ? groupBy : undefined,
                      filters,
                      sessionId,
                      markedIds: selectedIds
                    })}
                    label={t('search.export_results')}
                    className="h-7 w-7 shrink-0 px-0"
                  />
                )}
              {/* One control, both directions: pick everything loaded, press
                  again to let it all go. Two buttons sat side by side where
                  only one was ever live, and a selection is a state you flip,
                  not two commands you choose between.

                  It also has to stay reachable when a selection outlives a
                  zero-hit search — there are no hits to select, but the
                  chunks picked under an earlier query are still answering
                  the next question, so clearing them must not vanish with
                  the list they came from. */}
              <Button
                type="button"
                variant="ghost"
                size="sm"
                disabled={visibleHits.length === 0 && selectedIds.length === 0}
                aria-label={
                  allLoadedSelected
                    ? t('search.clear_selection')
                    : t('search.select_all_loaded', { count: visibleHits.length })
                }
                title={
                  allLoadedSelected
                    ? t('search.clear_selection')
                    : `${t('search.select_all_loaded_title', { count: visibleHits.length })} ${t(
                        'search.select_all_cost',
                        { tokens: formatTokens(projectedTokens) }
                      )}`
                }
                aria-pressed={allLoadedSelected}
                onClick={() => void commitScope(allLoadedSelected ? {} : allLoadedTokens())}
                className={cn('h-7 w-7 shrink-0 px-0', allLoadedSelected && 'text-primary')}
              >
                {allLoadedSelected ? (
                  <XIcon className="h-4 w-4" />
                ) : (
                  <CheckAllIcon className="h-4 w-4" />
                )}
              </Button>
            </div>
          )}
          {visibleHits.length > 0 && projectedOverBudget && (
            <p className="text-xs text-red-500" data-testid="select-all-over-budget">
              {t('search.select_all_over_budget', { total: formatTokens(scope.usableTokens) })}
            </p>
          )}
        </>
      )}

      {/* No "applies once the chat has started" notice: picking evidence
          before the first turn behaves exactly as it does after it, so the
          line explained nothing while shoving the whole hit list down the
          moment anything was selected. The selection is carried into the
          session the backend mints on that first turn (Chat.tsx). */}
      {scopeError && (
        <p className="text-xs text-red-500" role="alert">
          {scopeError}
        </p>
      )}

      {/* The three states below are deliberately separate branches. Hits and
          Groups share them — a collection with no search index, or a partial
          backfill, means the same thing for either query. */}
      {active.data?.status === 'not_indexed' && (
        <Banner variant="info" data-testid="search-not-indexed">
          {t('search.not_indexed')}
        </Banner>
      )}
      {active.data?.status === 'partial' && (
        <Banner variant="danger" data-testid="search-partial">
          {t('search.partial_warning', { count: active.data.index_status?.missing ?? 0 })}
        </Banner>
      )}
      {mode === 'hits' && active.data?.status === 'ok' && hits.length === 0 && (
        <p className="text-xs text-muted-foreground" data-testid="search-no-matches">
          {t('search.no_matches')}
        </p>
      )}

      {mode === 'hits' && (
        <ul className="min-h-0 flex-1 space-y-2 overflow-auto pr-1">
          {hits.map((hit) => (
            <SearchHitRow
              key={hit.id}
              hit={hit}
              keywords={keywords}
              selected={hit.id in scope.tokens}
              onToggle={toggle}
            />
          ))}
        </ul>
      )}
      {mode === 'groups' && grouped.data && (
        <SearchGroups
          result={grouped.data}
          keywords={keywords}
          selectedTokens={scope.tokens}
          onToggle={toggle}
        />
      )}
    </Card>
  )
}
