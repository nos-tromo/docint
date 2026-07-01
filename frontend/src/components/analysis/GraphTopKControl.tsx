import { useEffect, useState } from 'react'

interface Props {
  /** Current effective node count (already clamped by the parent). */
  value: number
  /** Upper bound from the deploy config. */
  max: number
  /** Commit a new, clamped node count. */
  onChange: (n: number) => void
}

/**
 * Compact number field for the Analysis graph view's node count. Fully
 * controlled — `value`/`max` derive from the UI store + `/config` in the
 * parent. Commits a clamped integer to `onChange` only when the value settles
 * (blur or Enter), so the graph refetches once per settled value rather than on
 * every keystroke. A non-numeric or unchanged entry resets the field and emits
 * nothing.
 */
export function GraphTopKControl({ value, max, onChange }: Props) {
  const [draft, setDraft] = useState(String(value))

  // Re-sync the draft when the committed value changes from outside (config
  // load, clamp on a lowered ceiling, collection switch).
  useEffect(() => {
    setDraft(String(value))
  }, [value])

  const commit = (raw: string) => {
    const n = Number.parseInt(raw, 10)
    if (Number.isNaN(n)) {
      setDraft(String(value))
      return
    }
    const clamped = Math.min(Math.max(1, n), max)
    setDraft(String(clamped))
    if (clamped !== value) onChange(clamped)
  }

  return (
    <label className="flex items-center gap-2 text-sm text-muted-foreground">
      <span>Nodes</span>
      <input
        type="number"
        min={1}
        max={max}
        value={draft}
        aria-label="Graph node count"
        onChange={(e) => setDraft(e.target.value)}
        onBlur={(e) => commit(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') commit((e.target as HTMLInputElement).value)
        }}
        className="w-20 rounded-md border border-border bg-transparent px-2 py-1 text-foreground"
      />
    </label>
  )
}
