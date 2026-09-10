import { useEffect } from 'react'

/**
 * Make the browser ask before leaving while something must not be interrupted.
 *
 * An upload in flight is exactly that. Bytes are staged request by request,
 * so a reload or a closed tab stops the transfer wherever it stood, and the
 * page cannot resume on its own afterwards — a reloaded page cannot re-read
 * the user's files. This is the only point at which the loss can still be
 * prevented rather than reported.
 *
 * The prompt itself is the browser's own, wording included: since Chrome 60
 * and Firefox 44 no page may supply its text. Which is why the screen also
 * carries a warning of its own while `active` holds.
 *
 * @param active - Whether leaving should be confirmed.
 */
export function useUnloadGuard(active: boolean): void {
  useEffect(() => {
    if (!active) return
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault()
      // Firefox needs the assignment; Chrome needs preventDefault. Neither
      // shows the value.
      e.returnValue = ''
    }
    window.addEventListener('beforeunload', onBeforeUnload)
    return () => window.removeEventListener('beforeunload', onBeforeUnload)
  }, [active])
}
