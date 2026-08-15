import { useEffect, useId, useRef, useState, type ReactNode } from 'react'
import { ChevronDownIcon } from '@infra/ui'
import { cn } from '@/lib/cn'

/** How much of a folded section stays visible, in px — two or three lines. */
const PEEK_HEIGHT = 64

/**
 * One foldable section of a report, with its contents peeking out while shut.
 *
 * The bar is the whole width of the column and the caret sits at the far right,
 * so a section reads as something you can open rather than as a caption that
 * happens to sit above some cards. That is the shape of the collapsible this
 * was modelled on; the marker is a rotated chevron rather than that example's
 * typed `+`/`−`, because a character renders from whatever font the machine
 * falls back to and `aria-expanded` already carries the state for anyone not
 * looking at it.
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
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls={panelId}
        className="flex w-full items-center gap-2 rounded-md py-1 text-left transition-colors hover:bg-muted/60"
      >
        <span className="text-sm font-medium uppercase tracking-wide text-muted-foreground">
          {title}
        </span>
        {count && <span className="truncate text-xs text-muted-foreground">{count}</span>}
        {/* One caret rotated, never a pair. `ml-auto` is what pushes it to the
            far edge, which is what makes the whole row read as the control. */}
        <ChevronDownIcon
          className={cn(
            'ml-auto h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform',
            !open && '-rotate-90'
          )}
        />
      </button>
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
