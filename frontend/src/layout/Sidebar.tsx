import { useEffect } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import { HoverIconAction, NewButton, SelectMenu, TrashIcon } from '@infra/ui'
import { ApiError } from '@/api/client'
import { useCollections, useDeleteCollection, useSelectCollection } from '@/hooks/useCollections'
import { useDeleteSession, useSessions, sessionsKey } from '@/hooks/useSessions'
import { useUiStore } from '@/stores/ui'
import { useChatUiStore } from '@/stores/chatUi'
import { useIngestJobsStore, selectHasRunningJob } from '@/stores/ingestJobs'
import { cn } from '@/lib/cn'
import {
  buildCollectionEntries,
  entryKey,
  entryMatches,
  type CollectionEntry
} from '@/lib/collectionEntries'
import { useT } from '@/i18n/LanguageContext'

const NAV = [
  { to: '/', key: 'nav.dashboard' },
  { to: '/ingest', key: 'nav.ingest' },
  { to: '/inspector', key: 'nav.inspector' },
  { to: '/chat', key: 'nav.chat' },
  { to: '/analysis', key: 'nav.analysis' },
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
  const hasRunningJob = useIngestJobsStore(selectHasRunningJob)
  const selected = useUiStore((s) => s.selectedCollection)
  const selectedOwner = useUiStore((s) => s.selectedOwner)
  const setSelected = useUiStore((s) => s.setSelectedCollection)
  const currentSessionId = useUiStore((s) => s.currentSessionId)
  const setCurrentSessionId = useUiStore((s) => s.setCurrentSessionId)
  const sessions = sessionsData?.sessions ?? []
  const entries = collections ? buildCollectionEntries(collections) : []
  const selectedEntry = entries.find((e) => entryMatches(e, selected, selectedOwner)) ?? null

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
    const label = owner ? `"${name}"${t('common.owned_by_suffix', { owner })}` : `"${name}"`
    if (!confirm(t('common.delete_collection_confirm', { label }))) return
    // Snapshot before the mutation fires: `sessions` is only this collection's
    // list because it's the active one (the `selected === name` branch below),
    // and it would already be stale/invalidated by the time onSuccess runs.
    const deletedSessionIds = sessions.map((s) => s.id)
    deleteCollectionMutation.mutate(name, {
      onSuccess: () => {
        if (selected === name) {
          setSelected(null)
          // The backend cascade-deleted this collection's chat sessions; drop
          // any open one, prune their drafts (there is no dedicated endpoint
          // to look up which sessions belonged to a now-deleted collection,
          // so this only covers the case where it was the active one — the
          // one case the client can see), and clear the now-stale session list.
          setCurrentSessionId(null)
          const clearDraft = useChatUiStore.getState().clearDraft
          for (const id of deletedSessionIds) clearDraft(id)
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
        // Prune unconditionally, not just when this was the open session —
        // a half-typed draft for a session the user just deleted (often to
        // remove sensitive content) must not linger in localStorage.
        useChatUiStore.getState().clearDraft(id)
      }
    })
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      {/* The collection leads the sidebar because every section below it —
          ingest, chat, analysis, report — is scoped to whichever one is
          active. Sitting between the nav and the sessions it read as one
          section's setting rather than the context they all share.

          It carries no heading and no box: it is the first thing in the
          panel, it shows a collection name, and it drops a chevron, so a
          "SAMMLUNG" label above it only repeated what the row already said.
          It carries no fill either — the primary tint means "where you are"
          everywhere else in this panel, and a second tinted row sitting
          directly above the active nav item competed with it for that. The
          live dot and the name's weight say it instead. */}
      <section>
        <div
          data-testid={selected ? 'active-collection' : undefined}
          className="group flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors hover:bg-accent"
        >
          <span
            aria-hidden="true"
            className={cn(
              'h-2 w-2 shrink-0 rounded-full',
              selected
                ? 'bg-primary shadow-[0_0_6px_var(--color-primary)]'
                : 'bg-muted-foreground/40'
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
          {/* Keyed by owner-and-name rather than by list position: the index
              was recomputed per option against the same array it indexed, and
              two owners may name a collection the same thing. */}
          <SelectMenu
            label={t('common.select_collection_aria')}
            options={entries.map((entry) => ({
              value: entryKey(entry),
              label: entry.name,
              ...(entry.owner ? { group: entry.owner } : {})
            }))}
            value={selectedEntry ? entryKey(selectedEntry) : null}
            onChange={(key) => {
              const entry = entries.find((e) => entryKey(e) === key)
              if (entry) void onSelectCollection(entry)
            }}
            placeholder={t('common.choose_collection')}
            emptyLabel={t('common.no_collections')}
            className="min-w-0 flex-1"
            triggerClassName="text-sm font-medium"
          />
          {/* Trash, not ×: deleting a collection destroys every ingested
              document in it. The icon is what says how far the action goes.
              Hover-revealed, because destroying a collection is the rarest
              thing done here and an always-lit trash beside the one row that
              leads the panel reads as an invitation. */}
          {selected && (
            <HoverIconAction
              icon={<TrashIcon />}
              label={t('common.delete_collection_aria', { name: selected })}
              onClick={() => onDeleteCollection(selected, selectedOwner)}
              className="-my-1 h-7"
            />
          )}
        </div>
        {!selected && (
          <p className="mt-1.5 px-3 text-xs text-muted-foreground">
            {t('common.no_active_collection')}
          </p>
        )}
        {/* A failed delete is otherwise indistinguishable from the button
            doing nothing (the collection list renders from the ownership DB
            and stays intact) — surface the mutation error in the same opaque
            chip style as the sessions error below. Clears automatically when
            the next delete attempt starts or succeeds. */}
        {deleteCollectionMutation.isError && (
          <p role="alert" className="mt-1.5 rounded-md border border-amber-700 bg-amber-950 px-2 py-2 text-xs text-amber-200">
            {t('common.delete_collection_error', {
              name: `"${String(deleteCollectionMutation.variables ?? '')}"`
            })}
          </p>
        )}
      </section>

      <nav className="flex flex-col gap-1">
        {NAV.map(({ to, key }) => {
          // The open chat lives on the server and its id is persisted; the
          // plain /chat link is the only reason leaving and returning showed
          // an empty transcript. Resolve the target here rather than
          // redirecting inside Chat.tsx, whose param->store sync would null
          // the id first. `end` stays keyed on `to` so active-state matching
          // is unaffected.
          const target = to === '/chat' && currentSessionId ? `/chat/${currentSessionId}` : to
          return (
            <NavLink
              key={to}
              to={target}
              end={to === '/'}
              className={({ isActive }) =>
                cn(
                  'rounded-md px-3 py-2 text-sm hover:bg-accent',
                  isActive && 'bg-primary/15 text-primary'
                )
              }
            >
              {t(key)}
              {to === '/ingest' && hasRunningJob && (
                <span
                  aria-label={t('nav.ingest_running')}
                  className="ml-2 inline-block h-2 w-2 animate-pulse rounded-full bg-primary align-middle"
                />
              )}
            </NavLink>
          )
        })}
      </nav>

      <section className="flex-1 min-h-0 flex flex-col">
        {/* The one heading in the panel, and it earns its keep: the rows below
            are drawn exactly like the nav rows above, so without a label the
            two runs read as one long list of sections. It also gives the
            new-chat action a home directly over the list it adds to, which is
            reachable from every route — unlike the copy in the chat header. */}
        <div className="flex items-center justify-between gap-2">
          <span className="px-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">
            {t('common.sessions')}
          </span>
          <NewButton label={t('common.new_chat')} onClick={onNewChat} className="h-7" />
        </div>
        <ul className="mt-2 flex-1 overflow-auto space-y-1">
          {sessionsLoading && (
            <li className="px-3 py-2 text-sm text-muted-foreground">{t('common.loading_chats')}</li>
          )}
          {/* Opaque self-contained chip (fixed dark bg + light fg), matching
              ValidationBanner's convention — not a translucent tint over the
              theme-reactive sidebar background, which loses contrast in the
              light theme. */}
          {!sessionsLoading && sessionsError && (
            <li role="alert" className="rounded-md border border-amber-700 bg-amber-950 px-2 py-2 text-sm text-amber-200">
              {getSessionsStatusMessage(sessionsError, t)}
            </li>
          )}
          {!sessionsLoading && !sessionsError && sessions.length === 0 && (
            <li className="px-3 py-2 text-sm text-muted-foreground">
              {selected
                ? t('common.no_chats_in_collection')
                : t('common.select_collection_to_see_chats')}
            </li>
          )}
          {!sessionsLoading && !sessionsError && sessions.map((s) => {
            const active = currentSessionId === s.id
            return (
              // The trash is positioned out of the flow rather than sitting
              // beside the button: `opacity-0` still occupies its column, so
              // as a flex sibling it would hold every session row short of
              // the nav rows they are meant to match.
              <li key={s.id} className="group relative flex items-center">
                <button
                  type="button"
                  onClick={() => onPickSession(s.id)}
                  className={cn(
                    'min-w-0 flex-1 truncate rounded-md px-3 py-2 pr-9 text-left text-sm hover:bg-accent',
                    active && 'bg-primary/15 text-primary'
                  )}
                  title={s.title ?? s.id}
                >
                  {s.title?.trim() || t('common.session_title_fallback', { id: s.id.slice(0, 8) })}
                </button>
                <HoverIconAction
                  icon={<TrashIcon />}
                  label={t('common.delete_session_aria')}
                  onClick={() => onDeleteSession(s.id)}
                  className="absolute right-1 h-7"
                />
              </li>
            )
          })}
        </ul>
      </section>
    </div>
  )
}
