import { NavLink, useNavigate } from 'react-router-dom'
import { useCollections, useDeleteCollection, useSelectCollection } from '@/hooks/useCollections'
import { useDeleteSession, useSessions } from '@/hooks/useSessions'
import { useUiStore } from '@/stores/ui'
import { cn } from '@/lib/cn'

const NAV = [
  { to: '/', label: 'Dashboard' },
  { to: '/chat', label: 'Chat' },
  { to: '/ingest', label: 'Ingest' },
  { to: '/analysis', label: 'Analysis' },
  { to: '/inspector', label: 'Inspector' }
]

export function Sidebar() {
  const navigate = useNavigate()
  const { data: collections } = useCollections()
  const selectMutation = useSelectCollection()
  const deleteCollectionMutation = useDeleteCollection()
  const { data: sessionsData } = useSessions()
  const deleteSessionMutation = useDeleteSession()
  const selected = useUiStore((s) => s.selectedCollection)
  const setSelected = useUiStore((s) => s.setSelectedCollection)
  const currentSessionId = useUiStore((s) => s.currentSessionId)
  const setCurrentSessionId = useUiStore((s) => s.setCurrentSessionId)

  const onSelectCollection = async (name: string) => {
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
      <h2 className="text-lg font-semibold tracking-tight">DocInt</h2>

      <nav className="flex flex-col gap-1">
        {NAV.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              cn(
                'rounded-md px-3 py-2 text-sm hover:bg-zinc-800',
                isActive && 'bg-zinc-800 text-foreground'
              )
            }
          >
            {label}
          </NavLink>
        ))}
      </nav>

      <section>
        <label className="text-xs uppercase text-muted-foreground">Collection</label>
        <select
          className="mt-1 w-full bg-zinc-900 border border-border rounded-md px-2 py-1 text-sm"
          value={selected ?? ''}
          onChange={(e) => onSelectCollection(e.target.value)}
        >
          <option value="" disabled>
            — choose —
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
            className="mt-2 text-xs text-red-400 hover:text-red-300"
          >
            Delete this collection
          </button>
        )}
      </section>

      <section className="flex-1 min-h-0 flex flex-col">
        <div className="flex items-center justify-between">
          <label className="text-xs uppercase text-muted-foreground">Sessions</label>
          <button
            type="button"
            onClick={onNewChat}
            className="text-xs px-2 py-1 rounded-md bg-zinc-800 hover:bg-zinc-700"
          >
            + New
          </button>
        </div>
        <ul className="mt-2 flex-1 overflow-auto space-y-1">
          {sessionsData?.sessions.map((s) => {
            const active = currentSessionId === s.session_id
            return (
              <li key={s.session_id} className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => onPickSession(s.session_id)}
                  className={cn(
                    'flex-1 text-left text-sm px-2 py-1 rounded-md truncate',
                    active ? 'bg-zinc-800' : 'hover:bg-zinc-900'
                  )}
                  title={s.title ?? s.session_id}
                >
                  {s.title?.trim() || `Session ${s.session_id.slice(0, 8)}`}
                </button>
                <button
                  type="button"
                  onClick={() => onDeleteSession(s.session_id)}
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
    </aside>
  )
}
