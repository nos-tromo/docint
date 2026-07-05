import { useState, type ReactNode } from 'react'
import { cn } from '@/lib/cn'

/** Chars past which the body clamps to 4 lines behind a "Show more" toggle. */
const CLAMP_CHARS = 240

/**
 * Body text clamped to 4 lines with a "Show more/less" toggle when it exceeds
 * `length` characters. `length` is the plain-text length of `children` (passed
 * explicitly because children may be highlighted nodes, not a plain string).
 */
export function ClampedText({
  children,
  length,
  className
}: {
  children: ReactNode
  length: number
  className?: string
}) {
  const [expanded, setExpanded] = useState(false)
  const canClamp = length > CLAMP_CHARS
  return (
    <div className="min-w-0">
      <p
        className={cn(
          'whitespace-pre-wrap leading-6 break-words',
          canClamp && !expanded && 'line-clamp-4',
          className
        )}
      >
        {children}
      </p>
      {canClamp && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-1 text-xs text-blue-400 hover:text-blue-300"
        >
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </div>
  )
}
