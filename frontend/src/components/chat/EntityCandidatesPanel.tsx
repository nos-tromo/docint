import type { ChatFinalEvent } from '@/api/types'

function readField(obj: unknown, ...keys: string[]): string | null {
  if (!obj || typeof obj !== 'object') return null
  const o = obj as Record<string, unknown>
  for (const k of keys) {
    const v = o[k]
    if (typeof v === 'string' && v.length > 0) return v
    if (typeof v === 'number') return String(v)
  }
  return null
}

export function EntityCandidatesPanel({ meta }: { meta: ChatFinalEvent }) {
  const candidates = meta.entity_match_candidates ?? []
  const groups = meta.entity_match_groups ?? []
  if (candidates.length === 0 && groups.length === 0) return null

  return (
    <div className="rounded-md border border-border bg-zinc-900 p-3 space-y-3 text-sm">
      <div className="text-xs uppercase text-muted-foreground">Entity disambiguation</div>

      {candidates.length > 0 && (
        <div>
          <div className="text-xs text-muted-foreground mb-1">Candidates</div>
          <ul className="flex flex-wrap gap-1">
            {candidates.map((c, i) => {
              const label = readField(c, 'text', 'name', 'label') ?? `#${i}`
              const type = readField(c, 'type', 'entity_type')
              return (
                <li
                  key={i}
                  className="rounded-md bg-zinc-800 px-2 py-0.5 text-xs"
                  title={type ?? undefined}
                >
                  {label}
                  {type && <span className="ml-1 text-muted-foreground">{type}</span>}
                </li>
              )
            })}
          </ul>
        </div>
      )}

      {groups.length > 0 && (
        <div>
          <div className="text-xs text-muted-foreground mb-1">Groups</div>
          <ul className="space-y-1">
            {groups.map((g, i) => {
              const label = readField(g, 'label', 'name', 'text') ?? `Group ${i + 1}`
              const members = ((g as Record<string, unknown>).members ?? (g as Record<string, unknown>).candidates ?? []) as unknown[]
              return (
                <li key={i} className="rounded-md bg-zinc-950 px-2 py-1">
                  <div className="font-medium text-xs">{label}</div>
                  {members.length > 0 && (
                    <ul className="mt-1 flex flex-wrap gap-1">
                      {members.map((m, j) => {
                        const ml = readField(m, 'text', 'name') ?? typeof m === 'string' ? String(m) : `#${j}`
                        return (
                          <li
                            key={j}
                            className="rounded bg-zinc-800 px-1.5 py-0.5 text-[11px]"
                          >
                            {ml}
                          </li>
                        )
                      })}
                    </ul>
                  )}
                </li>
              )
            })}
          </ul>
        </div>
      )}
    </div>
  )
}
