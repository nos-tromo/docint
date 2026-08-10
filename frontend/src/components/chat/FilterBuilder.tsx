import { useEffect, useRef } from 'react'
import { Badge, Button, Input, RemoveButton, Select } from '@infra/ui'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { useSearchUiStore } from '@/stores/searchUi'
import { cn } from '@/lib/cn'
import { SlidersIcon } from '@/components/common/icons'
import { useT } from '@/i18n/LanguageContext'

const OPERATORS = ['eq', 'neq', 'contains', 'gte', 'lte', 'in']

/**
 * The metadata filters, as an icon disclosure beside the Chat heading.
 *
 * Filters are set occasionally and read constantly, so they must not hold
 * screen permanently: collapsed they are one icon, and expanded they overlay
 * what is beneath rather than pushing it down. The trigger carries no label —
 * the sliders say it — but it does carry a count badge, because a filter that
 * silently narrows every retrieval is a trap.
 *
 * The panel **drops downward** over the transcript, right-aligned under the
 * trigger so its edge lands on the chat column's edge, and 22rem wide — the
 * width of the search panel across from it. It used to rise from the foot of
 * that column, where it covered the retrieval control sitting directly above
 * it: two settings taking turns in one patch of screen, which is what made the
 * pair read as a single confusing toggle.
 *
 * It dismisses like a menu: a pointer anywhere else, or Escape. An overlay that
 * can only be closed by hitting the same small icon again is a trap of its own.
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
  const root = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    // `pointerdown`, not `click`: a drag that starts outside and ends inside
    // should still dismiss, and the panel must be gone before whatever was
    // clicked underneath reacts. Anything inside the root — trigger included —
    // is left to its own handler.
    const onPointerDown = (e: PointerEvent) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [open, setOpen])

  return (
    <div className="relative" ref={root}>
      {open && (
        // Capped against the viewport so a short window cannot cut the panel
        // off at the bottom.
        <div className="absolute right-0 top-full z-20 mt-1 w-[22rem] max-w-[calc(100vw-4rem)] max-h-[min(26rem,55vh)] space-y-3 overflow-auto rounded-md border border-border bg-background p-3 text-sm shadow-lg">
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
                      <RemoveButton
                        label={t('chat.remove_rule_aria')}
                        onClick={() => s.removeRule(r.id)}
                      />
                    </li>
                  ))}
                </ul>
              </div>
            </>
          )}
        </div>
      )}

      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        // The badge below is decorative, so the count has to live in the name:
        // with the label gone there is nothing else left to read out.
        aria-label={
          activeCount > 0 ? t('search.filters_badge_aria', { count: activeCount }) : t('search.filters')
        }
        title={
          activeCount > 0 ? t('search.filters_badge_aria', { count: activeCount }) : t('search.filters')
        }
        className={cn('h-8 w-8 shrink-0 px-0', open && 'bg-muted text-foreground')}
      >
        <SlidersIcon className="h-4 w-4" />
      </Button>
      {/* Pinned to the corner rather than set beside the icon, so an active
          filter never changes the control's footprint — and so the count is
          still on screen when the label is not. */}
      {activeCount > 0 && (
        <Badge
          variant="accent"
          aria-hidden="true"
          className="pointer-events-none absolute -right-1 -top-1 min-w-4 justify-center px-1 py-0 text-[0.625rem] leading-4"
        >
          {activeCount}
        </Badge>
      )}
    </div>
  )
}
