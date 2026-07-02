import { NavLink, useNavigate } from 'react-router-dom'
import { Button } from '@infra/ui'
import { ApiError } from '@/api/client'
import { useCollections, useDeleteCollection, useSelectCollection } from '@/hooks/useCollections'
import { useDeleteSession, useSessions } from '@/hooks/useSessions'
import { useUiStore } from '@/stores/ui'
import { cn } from '@/lib/cn'
import { VersionBadge } from '@/components/VersionBadge'

const NAV = [
  { to: '/', label: 'Dashboard' },
  { to: '/chat', label: 'Chat' },
  { to: '/ingest', label: 'Ingest' },
  { to: '/analysis', label: 'Analysis' },
  { to: '/inspector', label: 'Inspector' },
  { to: '/report', label: 'Report' }
]

function getSessionsStatusMessage(error: unknown) {
  if (error instanceof ApiError && error.status === 401) {
    return 'Session history requires an authenticated user or DOCINT_DEFAULT_IDENTITY.'
  }
  return 'Failed to load chats.'
}

export function Sidebar() {
  const navigate = useNavigate()
  const { data: collections } = useCollections()
  const selectMutation = useSelectCollection()
  const deleteCollectionMutation = useDeleteCollection()
  const { data: sessionsData, isLoading: sessionsLoading, error: sessionsError } = useSessions()
  const deleteSessionMutation = useDeleteSession()
  const selected = useUiStore((s) => s.selectedCollection)
  const setSelected = useUiStore((s) => s.setSelectedCollection)
  const currentSessionId = useUiStore((s) => s.currentSessionId)
  const setCurrentSessionId = useUiStore((s) => s.setCurrentSessionId)
  const sessions = sessionsData?.sessions ?? []

  const onSelectCollection = async (name: string) => {
    if (!name) return
    await selectMutation.mutateAsync(name)
    setSelected(name)
  }

  const onDeleteCollection = (name: string) => {
    if (!confirm(`Delete collection "${name}"? This cannot be undone.`)) return
    deleteCollectionMutation.mutate(name, {
      onSuccess: () => {
        if (selected === name) setSelected(null)
      }
    })
  }

  const onNewChat = () => {
    setCurrentSessionId(null)
    navigate('/chat')
  }

  const onPickSession = (id: string) => {
    setCurrentSessionId(id)
    navigate(`/chat/${id}`)
  }

  const onDeleteSession = (id: string) => {
    if (!confirm('Delete this chat?')) return
    deleteSessionMutation.mutate(id, {
      onSuccess: () => {
        if (currentSessionId === id) setCurrentSessionId(null)
      }
    })
  }

  return (
    <aside className="w-72 border-r border-border p-4 flex flex-col gap-4 bg-zinc-950">
      <h2 className="text-lg font-semibold tracking-tight">Document Intelligence</h2>

      <nav className="flex flex-col gap-1">
        {NAV.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              cn(
                'rounded-md px-3 py-2 text-sm hover:bg-zinc-800',
                isActive && 'bg-primary/15 text-primary'
              )
            }
          >
            {label}
          </NavLink>
        ))}
      </nav>

      <section>
        <label className="text-xs uppercase text-muted-foreground">Collection</label>
        <div
          data-testid={selected ? 'active-collection' : undefined}
          className={cn(
            'mt-1 flex items-center gap-2 rounded-md border px-2.5 py-2 transition-colors',
            selected ? 'border-primary/60 bg-primary/5' : 'border-border'
          )}
        >
          <span
            aria-hidden="true"
            className={cn(
              'h-2 w-2 shrink-0 rounded-full',
              selected
                ? 'bg-primary shadow-[0_0_6px_var(--color-primary)]'
                : 'bg-zinc-600'
            )}
          />
          {selected && (
            <span className="text-[10px] uppercase tracking-wide text-primary shrink-0">
              Active
            </span>
          )}
          <select
            aria-label="Select collection"
            className="min-w-0 flex-1 cursor-pointer bg-zinc-950 text-sm text-foreground outline-hidden"
            value={selected ?? ''}
            onChange={(e) => onSelectCollection(e.target.value)}
          >
            <option value="" disabled>
              {collections?.length ? 'Choose a collection…' : 'No collections yet'}
            </option>
            {collections?.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          {selected && (
            <button
              type="button"
              onClick={() => onDeleteCollection(selected)}
              aria-label={`Delete collection ${selected}`}
              title="Delete this collection"
              className="shrink-0 text-zinc-500 transition-colors hover:text-red-400"
            >
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-3.5 w-3.5"
                aria-hidden="true"
              >
                <path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2m2 0v14a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V6" />
                <path d="M10 11v6M14 11v6" />
              </svg>
            </button>
          )}
        </div>
        {!selected && (
          <p className="mt-1.5 text-xs text-muted-foreground">
            No active collection — pick one to query.
          </p>
        )}
      </section>

      <section className="flex-1 min-h-0 flex flex-col">
        <div className="flex items-center justify-between">
          <label className="text-xs uppercase text-muted-foreground">Sessions</label>
          <Button
            variant="primary"
            size="sm"
            onClick={onNewChat}
          >
            + New
          </Button>
        </div>
        <ul className="mt-2 flex-1 overflow-auto space-y-1">
          {sessionsLoading && (
            <li className="px-2 py-1 text-sm text-muted-foreground">Loading chats...</li>
          )}
          {!sessionsLoading && sessionsError && (
            <li role="alert" className="rounded-md border border-amber-900/60 bg-amber-500/10 px-2 py-2 text-sm text-amber-200">
              {getSessionsStatusMessage(sessionsError)}
            </li>
          )}
          {!sessionsLoading && !sessionsError && sessions.length === 0 && (
            <li className="px-2 py-1 text-sm text-muted-foreground">No chats yet.</li>
          )}
          {!sessionsLoading && !sessionsError && sessions.map((s) => {
            const active = currentSessionId === s.id
            return (
              <li key={s.id} className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => onPickSession(s.id)}
                  className={cn(
                    'flex-1 text-left text-sm px-2 py-1 rounded-md truncate',
                    active ? 'bg-primary/10 text-primary' : 'hover:bg-zinc-900'
                  )}
                  title={s.title ?? s.id}
                >
                  {s.title?.trim() || `Session ${s.id.slice(0, 8)}`}
                </button>
                <button
                  type="button"
                  onClick={() => onDeleteSession(s.id)}
                  className="text-xs text-zinc-500 hover:text-red-400 px-1"
                  aria-label="Delete session"
                >
                  ×
                </button>
              </li>
            )
          })}
        </ul>
      </section>
      <div className="pt-4">
        <VersionBadge />
      </div>
    </aside>
  )
}
