import { useCallback, useState } from 'react'
import { isPurposeRefusal } from '@/api/client'

/**
 * Run a file action — a download, a file opened in a new tab — and report how
 * it went to the control that started it.
 *
 * A file fetched through the purpose wrapper has no browser download bar to
 * fail in, so its control says so itself. A purpose the user declined to state
 * is no failure: they chose not to go on, and the control goes back to rest.
 *
 * @returns `run(action)`, and whether an action is in flight or the last one failed.
 */
export function useFileAction(): {
  run: (action: () => Promise<void>) => Promise<void>
  busy: boolean
  failed: boolean
} {
  const [state, setState] = useState<'idle' | 'busy' | 'failed'>('idle')
  const run = useCallback(async (action: () => Promise<void>) => {
    setState('busy')
    try {
      await action()
      setState('idle')
    } catch (err) {
      if (isPurposeRefusal(err)) {
        setState('idle')
        return
      }
      console.error('File request failed', err)
      setState('failed')
    }
  }, [])
  return { run, busy: state === 'busy', failed: state === 'failed' }
}
