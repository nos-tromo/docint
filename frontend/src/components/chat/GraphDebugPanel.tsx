import { useId, useState } from 'react'
import { DisclosureButton } from '@infra/ui'
import { useT } from '@/i18n/LanguageContext'

export function GraphDebugPanel({ data }: { data: unknown }) {
  const t = useT()
  const [open, setOpen] = useState(false)
  const bodyId = useId()
  if (!data) return null
  return (
    <div className="rounded-md border border-border bg-muted">
      {/* The caret is the control, at the row's edge — the shared
          `DisclosureButton`, so the rotation, `aria-expanded` and the
          state-swapped name come from one place rather than from a caret
          hand-rotated here. */}
      <div className="flex w-full items-center gap-1 px-3 py-2 text-xs uppercase text-muted-foreground">
        <span>{t('chat.graph_debug')}</span>
        <DisclosureButton
          expanded={open}
          controls={bodyId}
          label={open ? t('chat.graph_debug_hide') : t('chat.graph_debug_show')}
          onClick={() => setOpen((v) => !v)}
          className="h-6 w-6"
        />
      </div>
      {open && (
        <pre id={bodyId} className="text-xs p-3 overflow-auto max-h-80 bg-muted">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  )
}
