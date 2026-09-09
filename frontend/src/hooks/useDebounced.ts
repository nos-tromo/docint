import { useEffect, useState } from 'react'

/**
 * Follow a value, but only after it has held still.
 *
 * For values a user types that drive a request: the collection field feeds
 * the staged-batch lookup, and without this every keystroke would ask the
 * server.
 *
 * @param value - The value to follow.
 * @param delayMs - How long it must hold still. Defaults to 400 ms.
 * @returns The value as it stood `delayMs` after the last change.
 */
export function useDebounced<T>(value: T, delayMs = 400): T {
  const [settled, setSettled] = useState(value)

  useEffect(() => {
    const timer = setTimeout(() => setSettled(value), delayMs)
    return () => clearTimeout(timer)
  }, [value, delayMs])

  return settled
}
