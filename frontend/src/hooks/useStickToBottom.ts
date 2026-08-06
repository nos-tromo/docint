import { useCallback, useLayoutEffect, useRef } from 'react'

/** Within this many pixels of the bottom still counts as "at the bottom" —
 * fractional scroll positions and sub-pixel rounding mean a user who never
 * scrolled can sit a pixel or two short of `scrollHeight - clientHeight`. */
const PIN_THRESHOLD_PX = 40

/**
 * Keep a scroll container pinned to its bottom edge while content grows.
 *
 * The container follows new content (e.g. streamed chat tokens) only while
 * the user is at the bottom; scrolling up to read detaches it, and scrolling
 * back down re-attaches it. Attach `ref` and `onScroll` to the scrollable
 * element and pass the value whose changes should trigger the follow-up
 * scroll as `dep`.
 *
 * Args:
 *   dep: Value that changes whenever content is appended (e.g. the turns
 *     array). Each change scrolls to the bottom if the user is pinned.
 *
 * Returns:
 *   `ref` for the scrollable element and its `onScroll` handler.
 */
export function useStickToBottom<T extends HTMLElement>(dep: unknown) {
  const ref = useRef<T | null>(null)
  // Pinned by default so a freshly opened transcript starts at the bottom.
  const pinnedRef = useRef(true)

  const onScroll = useCallback(() => {
    const el = ref.current
    if (!el) return
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight <= PIN_THRESHOLD_PX
  }, [])

  useLayoutEffect(() => {
    const el = ref.current
    if (el && pinnedRef.current) el.scrollTop = el.scrollHeight
  }, [dep])

  return { ref, onScroll }
}
