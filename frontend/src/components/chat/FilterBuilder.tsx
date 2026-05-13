import { useChatFiltersStore } from '@/stores/chatFilters'

const OPERATORS = ['eq', 'neq', 'contains', 'gte', 'lte', 'in']

export function FilterBuilder() {
  const s = useChatFiltersStore()
  return (
    <div className="rounded-md border border-border bg-zinc-900 p-3 space-y-3 text-sm">
      <label className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={s.filterEnabled}
          onChange={(e) => s.setFilterEnabled(e.target.checked)}
        />
        Enable metadata filters
      </label>

      {s.filterEnabled && (
        <>
          <div className="grid grid-cols-2 gap-2">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">MIME pattern</span>
              <input
                value={s.mimePattern}
                onChange={(e) => s.setMimePattern(e.target.value)}
                placeholder="application/pdf"
                className="bg-zinc-950 border border-border rounded-md px-2 py-1"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">Hate-speech only</span>
              <input
                type="checkbox"
                checked={s.hateSpeechOnly}
                onChange={(e) => s.setHateSpeechOnly(e.target.checked)}
                className="self-start"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">Date from</span>
              <input
                type="date"
                value={s.dateFrom}
                onChange={(e) => s.setDateFrom(e.target.value)}
                className="bg-zinc-950 border border-border rounded-md px-2 py-1"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">Date to</span>
              <input
                type="date"
                value={s.dateTo}
                onChange={(e) => s.setDateTo(e.target.value)}
                className="bg-zinc-950 border border-border rounded-md px-2 py-1"
              />
            </label>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-muted-foreground">Custom rules</span>
              <button
                type="button"
                onClick={() => s.addRule()}
                className="text-xs px-2 py-1 rounded-md bg-zinc-800 hover:bg-zinc-700"
              >
                + Rule
              </button>
            </div>
            <ul className="space-y-2">
              {s.customRules.map((r) => (
                <li key={r.id} className="grid grid-cols-[1fr_auto_1fr_auto] gap-2 items-center">
                  <input
                    value={r.field}
                    onChange={(e) => s.updateRule(r.id, { field: e.target.value })}
                    placeholder="field"
                    className="bg-zinc-950 border border-border rounded-md px-2 py-1"
                  />
                  <select
                    value={r.operator}
                    onChange={(e) => s.updateRule(r.id, { operator: e.target.value })}
                    className="bg-zinc-950 border border-border rounded-md px-2 py-1"
                  >
                    {OPERATORS.map((o) => (
                      <option key={o} value={o}>
                        {o}
                      </option>
                    ))}
                  </select>
                  <input
                    value={r.value}
                    onChange={(e) => s.updateRule(r.id, { value: e.target.value })}
                    placeholder="value"
                    className="bg-zinc-950 border border-border rounded-md px-2 py-1"
                  />
                  <button
                    type="button"
                    onClick={() => s.removeRule(r.id)}
                    className="text-xs text-red-400 hover:text-red-300"
                    aria-label="Remove rule"
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </div>
  )
}
