import { describe, expect, it } from 'vitest'
import {
  cutSimilarity,
  cutTree,
  draftTopics,
  highestBar,
  mergeGroups,
  normalizeGroups,
  relaxCount,
  sameGroups,
  splitGroup,
  topicCount
} from './topicMerge'
import type { TopicsTuning } from '@/api/topics'

/** Topics 1 and 6 closest, then 4 and 9, then the two pairs. */
const LEAVES = [1, 4, 6, 9]
const MERGES: TopicsTuning['merges'] = [
  [1, 6, 0.95],
  [4, 9, 0.9],
  [1, 4, 0.4]
]

describe('cutTree', () => {
  it('applies the closest merges first, keeping the lower id', () => {
    expect(cutTree(LEAVES, MERGES, 4)).toEqual([])
    expect(cutTree(LEAVES, MERGES, 3)).toEqual([[1, 6]])
    expect(cutTree(LEAVES, MERGES, 2)).toEqual([
      [1, 6],
      [4, 9]
    ])
    expect(cutTree(LEAVES, MERGES, 1)).toEqual([[1, 4, 6, 9]])
    // Asked for more topics than there are, nothing merges.
    expect(cutTree(LEAVES, MERGES, 9)).toEqual([])
  })

  it('says how similar the last merged pair was, and how many topics are left', () => {
    expect(cutSimilarity(4, MERGES, 4)).toBeNull()
    expect(cutSimilarity(4, MERGES, 3)).toBe(0.95)
    expect(cutSimilarity(4, MERGES, 1)).toBe(0.4)
    expect(topicCount(4, cutTree(LEAVES, MERGES, 2))).toBe(2)
  })
})

describe('manual edits', () => {
  it('merges shown topics, a group standing for all its members', () => {
    expect(mergeGroups([], [9, 4])).toEqual([[4, 9]])
    expect(mergeGroups([[1, 6]], [1, 9])).toEqual([[1, 6, 9]])
    expect(mergeGroups([[1, 6], [4, 9]], [4, 1])).toEqual([[1, 4, 6, 9]])
  })

  it('splits a group back into its topics', () => {
    expect(
      splitGroup(
        [
          [1, 6],
          [4, 9]
        ],
        4
      )
    ).toEqual([[1, 6]])
  })

  it('compares groups whatever their order', () => {
    expect(sameGroups([[9, 4], [6, 1]], [[1, 6], [4, 9]])).toBe(true)
    expect(sameGroups([[1, 6]], [])).toBe(false)
    expect(normalizeGroups([[3], [7, 2, 7]])).toEqual([[2, 7]])
  })
})

describe('draftTopics', () => {
  const tuning = {
    leaves: [
      { id: 1, label: 'Cranes', fit_size: 30, size: 40 },
      { id: 4, label: 'Ferries', fit_size: 10, size: 12 },
      { id: 6, label: 'Docks', fit_size: 50, size: 20 },
      { id: 9, label: 'Tides', fit_size: 5, size: 3 }
    ],
    group_labels: { '4,9': 'Ferries and tides' }
  }

  it('shows each group once, sized by its members and named as the backend names it', () => {
    expect(draftTopics(tuning, [[1, 6], [4, 9]])).toEqual([
      // No label written for 1 and 6: the larger fit names the group.
      { id: 1, members: [1, 6], label: 'Docks', size: 60 },
      { id: 4, members: [4, 9], label: 'Ferries and tides', size: 15 }
    ])
    expect(draftTopics(tuning, []).map((topic) => topic.id)).toEqual([1, 6, 4, 9])
  })
})

describe('relax previews', () => {
  const histogram = Array.from({ length: 100 }, (_, bin) => ({ 20: 1, 30: 2, 45: 4 })[bin] ?? 0)

  it('counts the bins a bar admits, as the relax does', () => {
    expect(relaxCount(histogram, 0.3)).toBe(6)
    expect(relaxCount(histogram, 0.31)).toBe(4)
    expect(relaxCount(histogram, 0.01)).toBe(7)
    expect(relaxCount(histogram, 0.46)).toBe(0)
  })

  it('names the highest bar that still places a chunk', () => {
    expect(highestBar(histogram)).toBe(0.45)
    expect(highestBar(Array(100).fill(0))).toBeNull()
  })
})
