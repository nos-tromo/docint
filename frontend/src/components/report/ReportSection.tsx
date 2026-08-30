import { useEffect, useId, useRef, useState, type ReactNode } from 'react'
import { DisclosureButton } from '@infra/ui'
import { cn } from '@/lib/cn'
import { useT } from '@/i18n/LanguageContext'

/** How much of a folded section stays visible, in px — two or three lines. */
const PEEK_HEIGHT = 64

/**
 * One foldable section of a report, with its contents peeking out while shut.
 *
 * The bar is the whole width of the column and the control sits at the far
 * right, so a section reads as something you can open rather than as a caption
 * that happens to sit above some cards. The control is the shared
 * `DisclosureButton` rather than the whole row, which is how the same gesture
 * is drawn everywhere in the federation — one chevron rotated under
 * `aria-expanded`, never a typed `+`/`−` rendering from whatever font the
 * machine falls back to.
 *
 * Folded does not mean hidden. A report is a pile of evidence someone picked by
 * hand, and a row of closed bars tells them nothing about which pile is which —
 * so the first couple of lines stay on screen behind a fade. The fade resolves
 * to the route background, not to `bg-muted`: the cards are muted, the space
 * they sit in is not.
 *
 * Open state is local. Only this app's route-level layout panels persist theirs
 * (the chat's side panel, the search filters); which section an investigator
 * happened to fold is throwaway view state, and the report store deliberately
 * keeps only the active report id.
 */
export function ReportSection({
  title,
  count,
  defaultOpen = true,
  children
}: {
  /** The bar's heading text. */
  title: string
  /** Optional trailing text — a count, or the overview's totals. */
  count?: string
  /** Whether the section arrives open. */
  defaultOpen?: boolean
  children: ReactNode
}) {
  const t = useT()
  const [open, setOpen] = useState(defaultOpen)
  const [overflows, setOverflows] = useState(false)
  const bodyRef = useRef<HTMLDivElement>(null)
  const panelId = useId()

  // Whether the peek is actually hiding anything. A section shorter than the
  // cap must not wear a fade: a gradient over nothing promises more below and
  // there is none, which is worse than no affordance at all. Measured rather
  // than guessed, because item bodies vary from one line to a paragraph.
  useEffect(() => {
    const node = bodyRef.current
    if (!node || open) return
    const measure = () => setOverflows(node.scrollHeight > PEEK_HEIGHT + 1)
    measure()
    if (typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(measure)
    observer.observe(node)
    return () => observer.disconnect()
  }, [open, children])

  return (
    <div className="space-y-2">
      <div className="flex w-full items-center gap-2 py-1 text-left">
        <span className="text-sm font-medium uppercase tracking-wide text-muted-foreground">
          {title}
        </span>
        {count && <span className="truncate text-xs text-muted-foreground">{count}</span>}
        {/* `ml-auto` puts the control at the far edge of the bar, where the
            caret sat when the row itself was the button. The name carries the
            section title: a report holds several of these, and "Show" alone
            would read identically on every one of them. */}
        <DisclosureButton
          expanded={open}
          controls={panelId}
          label={
            open
              ? t('report.section_hide', { title })
              : t('report.section_show', { title })
          }
          onClick={() => setOpen((v) => !v)}
          className="ml-auto h-6 w-6"
        />
      </div>
      <div className="relative">
        {/* `inert` while folded: the peek cuts a card in half, and the half
            below the fold still holds a note field and three buttons. Left
            reachable, Tab would walk into content the operator cannot see. */}
        <div
          id={panelId}
          ref={bodyRef}
          inert={!open}
          data-state={open ? 'expanded' : 'collapsed'}
          className={cn('space-y-2', !open && 'overflow-hidden')}
          style={open ? undefined : { maxHeight: PEEK_HEIGHT }}
        >
          {children}
        </div>
        {!open && overflows && (
          <div
            aria-hidden
            className="pointer-events-none absolute inset-x-0 bottom-0 h-8 bg-gradient-to-b from-transparent to-background"
          />
        )}
      </div>
    </div>
  )
}
