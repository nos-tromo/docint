import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ReactNode } from 'react'
import { Button, cn } from '@infra/ui'
import { DownloadIcon } from '@/components/common/icons'

/**
 * The SPA's one download control, in its two DOM shapes.
 *
 * Every export used to spell itself out — "Download", "Download MD",
 * "Export CSV", "Export GraphML", "Download session sources (ZIP)" — five
 * phrasings of one action, each competing for width with the content it sat
 * beside. The verb is the icon now. The full phrase stays as the accessible
 * name, so a screen reader and a hover still get the sentence; only the pixels
 * are gone.
 *
 * What the icon does *not* replace is the format. Where several downloads sit
 * side by side — the entity graph's JSON / GraphML / HTML — each keeps its
 * format beside the icon as `children`: three identical icons in a row is a
 * guessing game, and the noun is the only thing telling them apart. Where a
 * control is the only download in its row, it stands alone.
 *
 * `secondary`, not the `ghost` the chat header's setting toggles use: a
 * download produces a file. It is an action, not a preference, and with the
 * word gone the border is what still says "button".
 */

/**
 * Extra classes layered over the button recipe.
 *
 * @param adorned - Whether a format label or caret sits beside the icon.
 * @param className - Caller overrides, applied last.
 * @returns The merged class string.
 */
function shell(adorned: boolean, className?: string): string {
  return cn('shrink-0 gap-1.5', adorned ? 'px-2.5' : 'w-8 px-0', className)
}

/**
 * The bordered square, hand-written for `<a>`.
 *
 * `@infra/ui` exports the `Button` component but not the `button` cva recipe
 * behind it (typed, not exported at runtime), so a link that has to look like
 * a button carries its own copy of `variant="secondary" size="sm"`. Kept in
 * this file, beside the component that uses the real thing, so the two cannot
 * drift out of sight of each other.
 */
const LINK_SHELL =
  'inline-flex h-8 items-center justify-center rounded-md border border-border bg-muted px-3 text-sm font-medium text-foreground transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary'

interface DownloadButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'aria-label' | 'title'> {
  /** Accessible name — drives both `aria-label` and `title`. Required: the icon carries no text. */
  label: string
  /** Optional adornment beside the icon — a format ("JSON") or a menu caret. */
  children?: ReactNode
}

/**
 * A download the browser cannot do on its own: the file is built in the page
 * and handed to `downloadText`, so the control is a `<button>`.
 *
 * @param props - `label` names the action; `children` adorns the icon.
 * @returns The download button.
 */
export function DownloadButton({ label, children, className, ...props }: DownloadButtonProps) {
  return (
    <Button
      type="button"
      variant="secondary"
      size="sm"
      aria-label={label}
      title={label}
      className={shell(children != null, className)}
      {...props}
    >
      <DownloadIcon className="h-4 w-4" />
      {children}
    </Button>
  )
}

interface DownloadLinkProps extends Omit<AnchorHTMLAttributes<HTMLAnchorElement>, 'aria-label' | 'title'> {
  /** Accessible name — drives both `aria-label` and `title`. */
  label: string
  /** Target of the download; the server streams the file. */
  href: string
  /** Optional adornment beside the icon. */
  children?: ReactNode
}

/**
 * A download the server streams: the href is the file, so the control stays an
 * `<a download>` and the browser handles the transfer.
 *
 * @param props - `label` names the action; `href` is the streaming endpoint.
 * @returns The download link, styled as the button it mirrors.
 */
export function DownloadLink({ label, href, children, className, ...props }: DownloadLinkProps) {
  return (
    <a
      href={href}
      download
      aria-label={label}
      title={label}
      className={cn(LINK_SHELL, shell(children != null, className))}
      {...props}
    >
      <DownloadIcon className="h-4 w-4" />
      {children}
    </a>
  )
}
