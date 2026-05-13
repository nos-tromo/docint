import { useState } from 'react'

export function GraphDebugPanel({ data }: { data: unknown }) {
  const [open, setOpen] = useState(false)
  if (!data) return null
  return (
    <div className="rounded-md border border-border bg-zinc-900">
      <button
        type="button"
        className="w-full text-left px-3 py-2 text-xs uppercase text-muted-foreground"
        onClick={() => setOpen((v) => !v)}
      >
        Graph debug {open ? '▾' : '▸'}
      </button>
      {open && (
        <pre className="text-xs p-3 overflow-auto max-h-80 bg-zinc-950">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  )
}
