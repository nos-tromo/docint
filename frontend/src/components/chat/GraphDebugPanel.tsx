import { useState } from 'react'
import { ChevronDownIcon } from '@infra/ui'
import { cn } from '@/lib/cn'
import { useT } from '@/i18n/LanguageContext'

export function GraphDebugPanel({ data }: { data: unknown }) {
  const t = useT()
  const [open, setOpen] = useState(false)
  if (!data) return null
  return (
    <div className="rounded-md border border-border bg-muted">
      <button
        type="button"
        className="flex w-full items-center gap-1 text-left px-3 py-2 text-xs uppercase text-muted-foreground"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        {t('chat.graph_debug')}
        {/* One caret rotated, not two icons: `aria-expanded` carries the state
            for anyone not looking at it, which the bare character never did. */}
        <ChevronDownIcon
          className={cn('h-3.5 w-3.5 transition-transform', !open && '-rotate-90')}
        />
      </button>
      {open && (
        <pre className="text-xs p-3 overflow-auto max-h-80 bg-muted">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  )
}
