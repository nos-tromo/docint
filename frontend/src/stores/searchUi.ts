import { create } from 'zustand'
import { persist } from 'zustand/middleware'

/**
 * Return the per-chat key a search draft, query and scope are stored under.
 *
 * Mirrors `stores/chatUi.ts::draftKey`: an unstarted chat has no session id
 * yet, so its work lives under `'new'` until the backend mints one on the
 * first turn (there is no create-session endpoint) — see `adoptScope`.
 *
 * @param sessionId - The open session id, or null for an unstarted chat.
 * @returns The key under which this chat's search state is stored.
 */
export const searchKeyFor = (sessionId: string | null): string => sessionId ?? 'new'

/**
 * A chat's selected chunks and what the last server round trip said they cost.
 *
 * `tokens` maps chunk id to its `est_tokens`, so the token meter can be summed
 * locally on every selection instead of costing a request per click.
 * `usableTokens` is only known once a scope has been written (the PUT returns
 * it); 0 means "not measured yet".
 */
export interface ScopeState {
  tokens: Record<string, number>
  usableTokens: number
  missing: number
}

const EMPTY_SCOPE: ScopeState = { tokens: {}, usableTokens: 0, missing: 0 }

export interface SearchUiState {
  /** Unsubmitted keyword input, keyed per chat. */
  drafts: Record<string, string>
  /** The keywords actually searched for, keyed per chat. */
  queries: Record<string, string>
  scopes: Record<string, ScopeState>
  /** Whether the `Filters (N)` disclosure at the column's foot is open. */
  filtersOpen: boolean
  setDraft: (key: string, value: string) => void
  setQuery: (key: string, value: string) => void
  setScopeTokens: (key: string, tokens: Record<string, number>) => void
  setScopeMeta: (key: string, meta: { usableTokens: number; missing: number }) => void
  clearScope: (key: string) => void
  adoptScope: (from: string, to: string) => void
  setFiltersOpen: (open: boolean) => void
}

/** Read a chat's scope, falling back to an empty one. */
export const scopeFor = (state: SearchUiState, key: string): ScopeState =>
  state.scopes[key] ?? EMPTY_SCOPE

/** The chunk ids a chat is scoped to, in insertion order. */
export const scopeChunkIds = (scope: ScopeState): string[] => Object.keys(scope.tokens)

/** Summed `est_tokens` of a chat's selection — the token meter's numerator. */
export const scopeEstTokens = (scope: ScopeState): number =>
  Object.values(scope.tokens).reduce((sum, n) => sum + n, 0)

export const useSearchUiStore = create<SearchUiState>()(
  persist(
    (set) => ({
      drafts: {},
      queries: {},
      scopes: {},
      filtersOpen: false,
      setDraft: (key, value) => set((s) => ({ drafts: { ...s.drafts, [key]: value } })),
      setQuery: (key, value) => set((s) => ({ queries: { ...s.queries, [key]: value } })),
      setScopeTokens: (key, tokens) =>
        set((s) => ({
          scopes: { ...s.scopes, [key]: { ...scopeFor(s, key), tokens } }
        })),
      setScopeMeta: (key, meta) =>
        set((s) => ({
          scopes: {
            ...s.scopes,
            [key]: { ...scopeFor(s, key), usableTokens: meta.usableTokens, missing: meta.missing }
          }
        })),
      clearScope: (key) =>
        set((s) => {
          if (!(key in s.scopes)) return s // stable reference -> no needless re-render
          const scopes = { ...s.scopes }
          delete scopes[key]
          return { scopes }
        }),
      // The backend mints the session id on the first turn, so a selection made
      // before the chat started lives under 'new'. Carrying it over (rather
      // than dropping it) is what makes "search, pick evidence, ask" work.
      adoptScope: (from, to) =>
        set((s) => {
          const carried = s.scopes[from]
          if (!carried || from === to) return s
          const scopes = { ...s.scopes, [to]: carried }
          delete scopes[from]
          const queries = { ...s.queries }
          if (from in queries) {
            queries[to] = queries[from]
            delete queries[from]
          }
          return { scopes, queries }
        }),
      setFiltersOpen: (filtersOpen) => set({ filtersOpen })
    }),
    {
      // The selection is the only client-side record of a session's scope —
      // the API has no GET for it — so persisting is what lets a reload still
      // report honestly what the chat is scoped to.
      name: 'docint-search-ui',
      version: 1
    }
  )
)
