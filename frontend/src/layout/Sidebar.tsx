import { useEffect } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import { Button } from '@infra/ui'
import { ApiError } from '@/api/client'
import { useCollections, useDeleteCollection, useSelectCollection } from '@/hooks/useCollections'
import { useDeleteSession, useSessions, sessionsKey } from '@/hooks/useSessions'
import { useUiStore } from '@/stores/ui'
import { cn } from '@/lib/cn'
import { VersionBadge } from '@/components/VersionBadge'
import { buildCollectionEntries, entryMatches, type CollectionEntry } from '@/lib/collectionEntries'
import { useT } from '@/i18n/LanguageContext'

const NAV = [
  { to: '/', key: 'nav.dashboard' },
  { to: '/chat', key: 'nav.chat' },
  { to: '/ingest', key: 'nav.ingest' },
  { to: '/analysis', key: 'nav.analysis' },
  { to: '/inspector', key: 'nav.inspector' },
  { to: '/report', key: 'nav.report' }
] as const

function getSessionsStatusMessage(
  error: unknown,
  t: ReturnType<typeof useT>
) {
  if (error instanceof ApiError && error.status === 401) {
    return t('common.sessions_error_auth')
  }
  return t('common.sessions_error_default')
}

export function Sidebar() {
  const t = useT()
  const navigate = useNavigate()
  const location = useLocation()
  const { data: collections } = useCollections()
  const selectMutation = useSelectCollection()
  const deleteCollectionMutation = useDeleteCollection()
  const { data: sessionsData, isLoading: sessionsLoading, error: sessionsError } = useSessions()
  const deleteSessionMutation = useDeleteSession()
  const qc = useQueryClient()
  const selected = useUiStore((s) => s.selectedCollection)
  const selectedOwner = useUiStore((s) => s.selectedOwner)
  const setSelected = useUiStore((s) => s.setSelectedCollection)
  const currentSessionId = useUiStore((s) => s.currentSessionId)
  const setCurrentSessionId = useUiStore((s) => s.setCurrentSessionId)
  const sessions = sessionsData?.sessions ?? []
  const entries = collections ? buildCollectionEntries(collections) : []
  const selectedIndex = entries.findIndex((e) => entryMatches(e, selected, selectedOwner))

  // A persisted collection can point at one this user no longer has access to
  // (deleted, or a foreign one no longer shared, since last visit). Once the
  // listing has loaded, clear a stale selection so the UI returns to the
  // no-collection state instead of firing requests that 404.
  useEffect(() => {
    if (!collections || !selected) return
    const stillExists = buildCollectionEntries(collections).some((e) =>
      entryMatches(e, selected, useUiStore.getState().selectedOwner)
    )
    if (!stillExists) {
      setSelected(null)
      setCurrentSessionId(null)
    }
  }, [collections, selected, setSelected, setCurrentSessionId])

  const onSelectCollection = async (entry: CollectionEntry) => {
    if (entryMatches(entry, selected, selectedOwner)) return
    const prevSelected = selected
    const prevOwner = selectedOwner
    const prevSessionId = currentSessionId
    setSelected(entry.name, entry.owner)
    // A session is pinned to the collection it was created under. Switching
    // collections resets any open chat so the next message can't resume it
    // against the wrong collection (which the backend refuses with a 409).
    setCurrentSessionId(null)
    try {
      await selectMutation.mutateAsync(entry.name)
    } catch {
      // A failing select must not leave a dead selection committed — restore
      // exactly what was active before this attempt (both the (name, owner)
      // pair and the open session). The file has no toast mechanism; the
      // mutation's own error state (selectMutation.error) is the surfaced
      // signal, matching how the other mutations here report failure.
      setSelected(prevSelected, prevOwner)
      setCurrentSessionId(prevSessionId)
      return
    }
    // Stay in whatever section the user is currently viewing — switching the
    // active collection must not yank them to chat. The one exception is a
    // pinned chat session sub-route (`/chat/:sessionId`): that session belongs
    // to the old collection, so drop to a fresh chat (still the chat section)
    // rather than leave a stale transcript/URL behind.
    if (location.pathname.startsWith('/chat/')) {
      navigate('/chat')
    }
  }

  const onDeleteCollection = (name: string, owner: string | null) => {
    const label = owner ? `"${name}" (owner: ${owner})` : `"${name}"`
    if (!confirm(t('common.delete_collection_confirm', { label }))) return
    deleteCollectionMutation.mutate(name, {
      onSuccess: () => {
        if (selected === name) {
          setSelected(null)
          // The backend cascade-deleted this collection's chat sessions; drop
          // any open one and clear the now-stale session list.
          setCurrentSessionId(null)
          qc.invalidateQueries({ queryKey: sessionsKey })
        }
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
    if (!confirm(t('common.delete_session_confirm'))) return
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
        {NAV.map(({ to, key }) => (
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
            {t(key)}
          </NavLink>
        ))}
      </nav>

      <section>
        <label className="text-xs uppercase text-muted-foreground">{t('common.collection')}</label>
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
              {t('common.active')}
            </span>
          )}
          {selected && selectedOwner && (
            <span className="text-[10px] text-muted-foreground shrink-0 truncate">
              {selectedOwner}
            </span>
          )}
          <select
            aria-label={t('common.select_collection_aria')}
            className="min-w-0 flex-1 cursor-pointer bg-zinc-950 text-sm text-foreground outline-hidden"
            value={selectedIndex >= 0 ? String(selectedIndex) : ''}
            onChange={(e) => onSelectCollection(entries[Number(e.target.value)])}
          >
            <option value="" disabled>
              {entries.length ? t('common.choose_collection') : t('common.no_collections')}
            </option>
            {collections?.mine.map((c) => (
              <option key={`own:${c}`} value={String(entries.findIndex((e) => entryMatches(e, c, null)))}>
                {c}
              </option>
            ))}
            {collections?.others.map((g) => (
              <optgroup key={g.owner} label={g.owner}>
                {g.collections.map((c) => (
                  <option
                    key={`${g.owner}:${c}`}
                    value={String(entries.findIndex((e) => entryMatches(e, c, g.owner)))}
                  >
                    {c}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
          {selected && (
            <button
              type="button"
              onClick={() => onDeleteCollection(selected, selectedOwner)}
              aria-label={t('common.delete_collection_aria', { name: selected })}
              title={t('common.delete_collection_title')}
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
            {t('common.no_active_collection')}
          </p>
        )}
      </section>

      <section className="flex-1 min-h-0 flex flex-col">
        <div className="flex items-center justify-between">
          <label className="text-xs uppercase text-muted-foreground">{t('common.sessions')}</label>
          <Button
            variant="primary"
            size="sm"
            onClick={onNewChat}
          >
            {t('common.new_session')}
          </Button>
        </div>
        <ul className="mt-2 flex-1 overflow-auto space-y-1">
          {sessionsLoading && (
            <li className="px-2 py-1 text-sm text-muted-foreground">{t('common.loading_chats')}</li>
          )}
          {!sessionsLoading && sessionsError && (
            <li role="alert" className="rounded-md border border-amber-900/60 bg-amber-500/10 px-2 py-2 text-sm text-amber-200">
              {getSessionsStatusMessage(sessionsError, t)}
            </li>
          )}
          {!sessionsLoading && !sessionsError && sessions.length === 0 && (
            <li className="px-2 py-1 text-sm text-muted-foreground">
              {selected
                ? t('common.no_chats_in_collection')
                : t('common.select_collection_to_see_chats')}
            </li>
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
                  {s.title?.trim() || t('common.session_title_fallback', { id: s.id.slice(0, 8) })}
                </button>
                <button
                  type="button"
                  onClick={() => onDeleteSession(s.id)}
                  className="text-xs text-zinc-500 hover:text-red-400 px-1"
                  aria-label={t('common.delete_session_aria')}
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
