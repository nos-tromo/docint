import { Badge, Button, Input, Select } from '@infra/ui'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { useSearchUiStore } from '@/stores/searchUi'
import { cn } from '@/lib/cn'
import { SlidersIcon } from '@/components/common/icons'
import { useT } from '@/i18n/LanguageContext'

const OPERATORS = ['eq', 'neq', 'contains', 'gte', 'lte', 'in']

/**
 * The metadata filters, as a `Filters (N)` disclosure beside the Chat heading.
 *
 * Filters are set occasionally and read constantly, so they must not hold
 * screen permanently: collapsed they are one compact trigger, and expanded they
 * overlay what is beneath rather than pushing it down. The badge keeps the
 * active count visible while collapsed, because a filter that silently narrows
 * every retrieval is a trap.
 *
 * The panel **drops downward** over the transcript. It used to rise from the
 * foot of the search column, where it covered the retrieval control sitting
 * directly above it — two settings taking turns in one patch of screen, which
 * is what made the pair read as a single confusing toggle.
 *
 * It anchors itself: `relative` here, so the panel keeps its own readable width
 * wherever the trigger is mounted instead of stretching or shrinking to
 * whatever box happens to contain it.
 *
 * Every control is an `@infra/ui` primitive — the hand-rolled `bg-muted`
 * inputs this replaces had no contrast at all against the muted panel.
 */
export function FilterBuilder() {
  const t = useT()
  const s = useChatFiltersStore()
  const open = useSearchUiStore((state) => state.filtersOpen)
  const setOpen = useSearchUiStore((state) => state.setFiltersOpen)
  const activeCount = s.buildPayload().length

  return (
    <div className="relative">
      {open && (
        // Its own width, not the trigger's, and capped against the viewport so
        // a short window cannot cut the panel off at the bottom.
        <div className="absolute left-0 top-full z-20 mt-1 w-[22rem] max-w-[calc(100vw-4rem)] max-h-[min(26rem,55vh)] space-y-3 overflow-auto rounded-md border border-border bg-background p-3 text-sm shadow-lg">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={s.filterEnabled}
              onChange={(e) => s.setFilterEnabled(e.target.checked)}
            />
            <span>{t('chat.enable_filters')}</span>
          </label>

          {s.filterEnabled && (
            <>
              <label className="flex flex-col gap-1">
                <span className="text-xs text-muted-foreground">{t('chat.mime_pattern')}</span>
                <Input
                  value={s.mimePattern}
                  onChange={(e) => s.setMimePattern(e.target.value)}
                  placeholder="application/pdf"
                />
              </label>

              <div className="grid grid-cols-2 gap-2">
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-muted-foreground">{t('chat.date_from')}</span>
                  <Input
                    type="date"
                    value={s.dateFrom}
                    onChange={(e) => s.setDateFrom(e.target.value)}
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-muted-foreground">{t('chat.date_to')}</span>
                  <Input
                    type="date"
                    value={s.dateTo}
                    onChange={(e) => s.setDateTo(e.target.value)}
                  />
                </label>
              </div>

              {/* A boolean toggle is not a text field: it gets its own
                  checkbox-beside-label row rather than a grid cell sized for
                  an input, where it sat orphaned under a floating caption. */}
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={s.hateSpeechOnly}
                  onChange={(e) => s.setHateSpeechOnly(e.target.checked)}
                />
                <span>{t('chat.hate_speech_only')}</span>
              </label>

              <div>
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">{t('chat.custom_rules')}</span>
                  <Button type="button" variant="secondary" size="sm" onClick={() => s.addRule()}>
                    {t('chat.add_rule')}
                  </Button>
                </div>
                <ul className="space-y-2">
                  {s.customRules.map((r) => (
                    <li key={r.id} className="grid grid-cols-[1fr_auto_1fr_auto] items-center gap-2">
                      <Input
                        value={r.field}
                        onChange={(e) => s.updateRule(r.id, { field: e.target.value })}
                        placeholder={t('chat.field_placeholder')}
                      />
                      <Select
                        value={r.operator}
                        onChange={(e) => s.updateRule(r.id, { operator: e.target.value })}
                      >
                        {OPERATORS.map((o) => (
                          <option key={o} value={o}>
                            {o}
                          </option>
                        ))}
                      </Select>
                      <Input
                        value={r.value}
                        onChange={(e) => s.updateRule(r.id, { value: e.target.value })}
                        placeholder={t('chat.value_placeholder')}
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => s.removeRule(r.id)}
                        aria-label={t('chat.remove_rule_aria')}
                      >
                        ×
                      </Button>
                    </li>
                  ))}
                </ul>
              </div>
            </>
          )}
        </div>
      )}

      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className={cn(
          'flex items-center gap-1.5 rounded-md border border-border px-2 py-1 text-xs transition-colors hover:text-foreground',
          open ? 'bg-muted text-foreground' : 'text-muted-foreground'
        )}
      >
        <SlidersIcon className="h-3.5 w-3.5" />
        <span>{t('search.filters')}</span>
        {activeCount > 0 && <Badge variant="accent">{activeCount}</Badge>}
      </button>
    </div>
  )
}
