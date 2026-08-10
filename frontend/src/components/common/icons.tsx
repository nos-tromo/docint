import type { SVGProps } from 'react'

/**
 * The SPA's inline icon set.
 *
 * Icons are drawn, never typed: a text glyph (`×`, `▸`, `✓`) renders from
 * whatever font the browser and OS happen to pick, so the same control looks
 * different on every machine — and in a control that carries no label of its
 * own, that is the whole affordance. These are 24×24 stroked outlines that
 * inherit `currentColor` and are sized by the caller, matching the hand-rolled
 * glyphs already in `TranslateToggle` and `SourcePreviewAction`.
 *
 * Every icon is `aria-hidden`: each is mounted inside a control that carries
 * its own accessible name.
 */
const base = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 2,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  'aria-hidden': true
} as const

export type IconProps = SVGProps<SVGSVGElement>

/** Disclosure chevron. Rotate it with a class rather than swapping the icon. */
export const ChevronDownIcon = ({ className = 'h-4 w-4', ...props }: IconProps) => (
  <svg {...base} className={className} {...props}>
    <path d="m6 9 6 6 6-6" />
  </svg>
)

/** An unselected tile's marker — the hollow half of the selection pair. */
export const CircleIcon = ({ className = 'h-4 w-4', ...props }: IconProps) => (
  <svg {...base} className={className} {...props}>
    <circle cx="12" cy="12" r="9" />
  </svg>
)

/** A selected tile's marker. Same circle, so selection reads as a fill, not a jump. */
export const CheckCircleIcon = ({ className = 'h-4 w-4', ...props }: IconProps) => (
  <svg {...base} className={className} {...props}>
    <circle cx="12" cy="12" r="9" />
    <path d="m8.5 12 2.5 2.5 4.5-5" />
  </svg>
)

/** Select every loaded hit. */
export const CheckAllIcon = ({ className = 'h-4 w-4', ...props }: IconProps) => (
  <svg {...base} className={className} {...props}>
    <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
    <path d="m9 11 3 3L22 4" />
  </svg>
)

/** Clear / dismiss. */
export const XIcon = ({ className = 'h-4 w-4', ...props }: IconProps) => (
  <svg {...base} className={className} {...props}>
    <path d="M18 6 6 18M6 6l12 12" />
  </svg>
)

/** The metadata filters — sliders, not a funnel: these tune a search, not gate it. */
export const SlidersIcon = ({ className = 'h-4 w-4', ...props }: IconProps) => (
  <svg {...base} className={className} {...props}>
    <path d="M4 7h4M12 7h8M4 12h10M18 12h2M4 17h6M14 17h6" />
    <circle cx="10" cy="7" r="2" />
    <circle cx="16" cy="12" r="2" />
    <circle cx="12" cy="17" r="2" />
  </svg>
)

/**
 * Stateful retrieval: the whole conversation is context.
 *
 * Deliberately a *different shape* from `SingleMessageIcon` rather than the
 * same icon pressed and unpressed — the toggle carries no label, so the two
 * states have to be told apart without hovering for the tooltip.
 */
export const ChatContextIcon = ({ className = 'h-4 w-4', ...props }: IconProps) => (
  <svg {...base} className={className} {...props}>
    <path d="M14 9a2 2 0 0 1-2 2H6l-4 4V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2z" />
    <path d="M18 9h2a2 2 0 0 1 2 2v11l-4-4h-6a2 2 0 0 1-2-2v-1" />
  </svg>
)

/** Stateless retrieval: only the message just sent is context. */
export const SingleMessageIcon = ({ className = 'h-4 w-4', ...props }: IconProps) => (
  <svg {...base} className={className} {...props}>
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
  </svg>
)
