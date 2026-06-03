import type { EntityMergeMode } from '@/api/types'
import { useUiStore } from '@/stores/ui'
import { cn } from '@/lib/cn'

const MODES: { value: EntityMergeMode; label: string }[] = [
  { value: 'resolved', label: 'Resolved' },
  { value: 'orthographic', label: 'Orthographic' },
  { value: 'exact', label: 'Exact' }
]

/**
 * Segmented control for the entity merge mode used by the NER views.
 *
 * Reads and writes the shared `entityMergeMode` in the UI store so Analysis
 * and Dashboard stay in sync. Defaults to "Resolved" (durable canonical
 * entities); "Orthographic"/"Exact" expose the pre-resolution clustering.
 */
export function MergeModeToggle() {
  const mode = useUiStore((s) => s.entityMergeMode)
  const setMode = useUiStore((s) => s.setEntityMergeMode)

  return (
    <div
      role="group"
      aria-label="Entity merge mode"
      className="inline-flex overflow-hidden rounded-md border border-border text-sm"
    >
      {MODES.map((m) => (
        <button
          key={m.value}
          type="button"
          aria-pressed={mode === m.value}
          onClick={() => setMode(m.value)}
          className={cn(
            'px-3 py-1 transition-colors',
            mode === m.value
              ? 'bg-zinc-800 text-foreground'
              : 'text-muted-foreground hover:text-foreground'
          )}
        >
          {m.label}
        </button>
      ))}
    </div>
  )
}
