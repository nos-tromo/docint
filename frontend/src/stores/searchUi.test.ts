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
    field: 'text'
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

describe('searchUi field state', () => {
  it('defaults to searching the text', () => {
    expect(useSearchUiStore.getState().field).toBe('text')
  })

  it('switches the search field', () => {
    useSearchUiStore.getState().setField('author_id')
    expect(useSearchUiStore.getState().field).toBe('author_id')
  })
})

describe('searchUi persistence migration', () => {
  it('drops a v3 field the picker no longer offers', () => {
    // A user whose last search was "Author ID" would otherwise reload into a
    // blank trigger sending a field the API now refuses.
    const migrate = useSearchUiStore.persist.getOptions().migrate!
    const migrated = migrate(
      { drafts: {}, queries: {}, scopes: {}, filtersOpen: false, field: 'author_id' },
      3
    ) as unknown as { field: string }

    expect(migrated.field).toBe('text')
  })

  it('keeps a v3 field the picker still offers', () => {
    const migrate = useSearchUiStore.persist.getOptions().migrate!
    const migrated = migrate(
      { drafts: {}, queries: {}, scopes: {}, filtersOpen: false, field: 'author' },
      3
    ) as unknown as { field: string }

    expect(migrated.field).toBe('author')
  })

  it('migrates a version-1 blob by filling the field default', () => {
    const migrate = useSearchUiStore.persist.getOptions().migrate!
    const migrated = migrate({ drafts: {}, queries: {}, scopes: {}, filtersOpen: false }, 1) as unknown as {
      field: string
    }
    expect(migrated.field).toBe('text')
  })

  it('drops a version-2 blob’s mode and group-by, keeping the rest', () => {
    const migrate = useSearchUiStore.persist.getOptions().migrate!
    const migrated = migrate(
      { drafts: { new: 'partei' }, queries: {}, scopes: {}, filtersOpen: true, mode: 'groups', groupBy: 'network' },
      2
    ) as unknown as Record<string, unknown>
    expect(migrated.field).toBe('text')
    expect(migrated.drafts).toEqual({ new: 'partei' })
    expect(migrated.filtersOpen).toBe(true)
    expect('mode' in migrated).toBe(false)
    expect('groupBy' in migrated).toBe(false)
  })
})
