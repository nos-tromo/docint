import { describe, expect, it } from 'vitest'
import { buildCollectionEntries, entryMatches } from './collectionEntries'
import type { CollectionsView } from '@/api/collections'

const view: CollectionsView = {
  mine: ['own-a', 'own-b'],
  others: [
    { owner: 'jane.doe', collections: ['own-a', 'reports'] },
    { owner: 'john.roe', collections: ['scans'] }
  ],
  isAdmin: true
}

describe('buildCollectionEntries', () => {
  it('flattens own collections first, then per-owner groups', () => {
    expect(buildCollectionEntries(view)).toEqual([
      { owner: null, name: 'own-a' },
      { owner: null, name: 'own-b' },
      { owner: 'jane.doe', name: 'own-a' },
      { owner: 'jane.doe', name: 'reports' },
      { owner: 'john.roe', name: 'scans' }
    ])
  })

  it('same logical name under different owners stays distinct', () => {
    const entries = buildCollectionEntries(view)
    expect(entries.filter((e) => e.name === 'own-a')).toHaveLength(2)
  })
})

describe('entryMatches', () => {
  it('matches on the (name, owner) pair', () => {
    expect(entryMatches({ owner: null, name: 'own-a' }, 'own-a', null)).toBe(true)
    expect(entryMatches({ owner: 'jane.doe', name: 'own-a' }, 'own-a', null)).toBe(false)
  })
})
