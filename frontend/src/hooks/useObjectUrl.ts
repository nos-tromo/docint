import { useEffect, useState } from 'react'

/**
 * An object URL for a Blob, for an `<img>` or `<iframe>` to show it.
 *
 * Made in an effect and revoked when the Blob changes or the component
 * unmounts, so every caller owns its own URL and none outlives its element.
 *
 * @param blob - The Blob to show, or nothing.
 * @returns The URL, undefined until there is a Blob and the effect has run.
 */
export function useObjectUrl(blob: Blob | null | undefined): string | undefined {
  const [href, setHref] = useState<string>()
  useEffect(() => {
    if (!blob) {
      setHref(undefined)
      return
    }
    const next = URL.createObjectURL(blob)
    setHref(next)
    return () => URL.revokeObjectURL(next)
  }, [blob])
  return href
}
