import { describe, it, expect, beforeEach } from 'vitest'
import {
  scopeChunkIds,
  scopeEstTokens,
  scopeFor,
  searchKeyFor,
  useSearchUiStore
} from './searchUi'

beforeEach(() => {
  useSearchUiStore.setState({
    drafts: {},
    queries: {},
    scopes: {},
    filtersOpen: false,
    mode: 'hits',
    groupBy: 'author'
  })
})

describe('searchKeyFor', () => {
  it('keys an unstarted chat under new', () => {
    expect(searchKeyFor(null)).toBe('new')
    expect(searchKeyFor('sess-1')).toBe('sess-1')
  })
})

describe('scope selectors', () => {
  it('reads an absent scope as empty rather than undefined', () => {
    const scope = scopeFor(useSearchUiStore.getState(), 'missing')

    expect(scopeChunkIds(scope)).toEqual([])
    expect(scopeEstTokens(scope)).toBe(0)
  })

  it('sums the token meter locally, with no round trip per checkbox', () => {
    useSearchUiStore.getState().setScopeTokens('sess-1', { p1: 1200, p2: 800 })

    const scope = scopeFor(useSearchUiStore.getState(), 'sess-1')
    expect(scopeChunkIds(scope)).toEqual(['p1', 'p2'])
    expect(scopeEstTokens(scope)).toBe(2000)
  })

  it('keeps the measured budget when the selection changes', () => {
    useSearchUiStore.getState().setScopeTokens('sess-1', { p1: 1200 })
    useSearchUiStore.getState().setScopeMeta('sess-1', { usableTokens: 22000, missing: 0 })
    useSearchUiStore.getState().setScopeTokens('sess-1', { p1: 1200, p2: 800 })

    expect(scopeFor(useSearchUiStore.getState(), 'sess-1').usableTokens).toBe(22000)
  })
})

describe('adoptScope', () => {
  it('carries a pre-session selection onto the id the backend minted', () => {
    // There is no create-session endpoint: the id appears on the first turn.
    // Dropping the selection then would delete the evidence the user picked
    // precisely in order to ask about it.
    useSearchUiStore.getState().setScopeTokens('new', { p1: 1200 })
    useSearchUiStore.getState().setQuery('new', 'partei')

    useSearchUiStore.getState().adoptScope('new', 'sess-9')

    const state = useSearchUiStore.getState()
    expect(state.scopes['new']).toBeUndefined()
    expect(scopeChunkIds(scopeFor(state, 'sess-9'))).toEqual(['p1'])
    expect(state.queries['sess-9']).toBe('partei')
  })

  it('is a no-op when there is nothing to carry', () => {
    const before = useSearchUiStore.getState().scopes

    useSearchUiStore.getState().adoptScope('new', 'sess-9')

    expect(useSearchUiStore.getState().scopes).toBe(before)
  })
})

describe('clearScope', () => {
  it('removes the entry entirely', () => {
    useSearchUiStore.getState().setScopeTokens('sess-1', { p1: 1200 })

    useSearchUiStore.getState().clearScope('sess-1')

    expect(useSearchUiStore.getState().scopes['sess-1']).toBeUndefined()
  })

  it('keeps a stable reference when there is nothing to clear', () => {
    const before = useSearchUiStore.getState().scopes

    useSearchUiStore.getState().clearScope('sess-1')

    expect(useSearchUiStore.getState().scopes).toBe(before)
  })
})

describe('searchUi mode state', () => {
  it('defaults to hits mode grouped by author', () => {
    const s = useSearchUiStore.getState()
    expect(s.mode).toBe('hits')
    expect(s.groupBy).toBe('author')
  })

  it('switches mode and group field', () => {
    useSearchUiStore.getState().setMode('groups')
    useSearchUiStore.getState().setGroupBy('network')

    expect(useSearchUiStore.getState().mode).toBe('groups')
    expect(useSearchUiStore.getState().groupBy).toBe('network')
  })
})

describe('searchUi persistence migration', () => {
  it('migrates a version-1 blob by filling the new defaults', () => {
    const migrate = useSearchUiStore.persist.getOptions().migrate!
    const migrated = migrate({ drafts: {}, queries: {}, scopes: {}, filtersOpen: false }, 1) as unknown as {
      mode: string
      groupBy: string
    }

    expect(migrated.mode).toBe('hits')
    expect(migrated.groupBy).toBe('author')
  })

  it('keeps the rest of a version-1 blob intact', () => {
    const migrate = useSearchUiStore.persist.getOptions().migrate!
    const migrated = migrate(
      { drafts: { new: 'partei' }, queries: {}, scopes: {}, filtersOpen: true },
      1
    ) as unknown as { drafts: Record<string, string>; filtersOpen: boolean }

    expect(migrated.drafts).toEqual({ new: 'partei' })
    expect(migrated.filtersOpen).toBe(true)
  })
})
