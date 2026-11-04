import type { TopicsTuning } from '@/api/topics'

/** Groups of two or more topic ids, each ascending, ordered by their lowest id, as the API stores them. */
export type TopicGroups = number[][]

/** One topic of a draft: a group under its lowest id, or a topic in no group. */
export interface DraftTopic {
  id: number
  members: number[]
  label: string
  size: number
}

/**
 * Put groups in the order the API stores them, dropping any of fewer than two topics.
 *
 * @param groups - Groups of topic ids, in any order.
 * @returns Each group ascending, the groups by their lowest id.
 */
export function normalizeGroups(groups: number[][]): TopicGroups {
  return groups
    .map((group) => [...new Set(group)].sort((a, b) => a - b))
    .filter((group) => group.length > 1)
    .sort((a, b) => a[0] - b[0])
}

/**
 * The groups the merge tree makes of `leafIds` when cut to `n` topics.
 *
 * The tree names each step's two clusters by their lowest ids, so applying its
 * first `leafIds.length - n` steps, each keeping the lower id, is the cut. The
 * backend computes the same tree and never cuts it itself: it stores the groups
 * it is sent.
 *
 * @param leafIds - The run's topic ids.
 * @param merges - The tree, closest step first.
 * @param n - Topics to keep.
 * @returns The groups.
 */
export function cutTree(leafIds: number[], merges: TopicsTuning['merges'], n: number): TopicGroups {
  const clusters = new Map<number, number[]>(leafIds.map((id) => [id, [id]]))
  for (const [a, b] of merges.slice(0, Math.max(0, leafIds.length - n))) {
    const keep = Math.min(a, b)
    const gone = Math.max(a, b)
    clusters.set(keep, [...(clusters.get(keep) ?? []), ...(clusters.get(gone) ?? [])])
    clusters.delete(gone)
  }
  return normalizeGroups([...clusters.values()])
}

/**
 * The similarity at which a cut to `n` topics made its last merge.
 *
 * @param leafCount - The run's topics.
 * @param merges - The tree.
 * @param n - Topics kept.
 * @returns The last merged step's similarity, or null when the cut merges nothing.
 */
export function cutSimilarity(leafCount: number, merges: TopicsTuning['merges'], n: number): number | null {
  const steps = leafCount - n
  return steps > 0 ? (merges[steps - 1]?.[2] ?? null) : null
}

/**
 * How many topics groups leave of `leafCount`.
 *
 * @param leafCount - The run's topics.
 * @param groups - The groups.
 * @returns One per group plus every topic in none.
 */
export function topicCount(leafCount: number, groups: TopicGroups): number {
  return leafCount - groups.reduce((sum, group) => sum + group.length - 1, 0)
}

/**
 * Merge the topics shown under `ids` into one group.
 *
 * @param groups - The draft's groups.
 * @param ids - Shown topic ids: a group's lowest id stands for the whole group.
 * @returns The groups with those topics, and every member of those groups, as one.
 */
export function mergeGroups(groups: TopicGroups, ids: number[]): TopicGroups {
  const chosen = new Set(ids)
  const kept = groups.filter((group) => !chosen.has(group[0]))
  const merged = ids.flatMap((id) => groups.find((group) => group[0] === id) ?? [id])
  return normalizeGroups([...kept, merged])
}

/**
 * Dissolve the group shown under `id` back into its topics.
 *
 * @param groups - The draft's groups.
 * @param id - The group's lowest id.
 * @returns The groups without it.
 */
export function splitGroup(groups: TopicGroups, id: number): TopicGroups {
  return groups.filter((group) => group[0] !== id)
}

/**
 * Whether two sets of groups merge the same topics.
 *
 * @param a - Groups.
 * @param b - Groups.
 * @returns True when they are equal once put in order.
 */
export function sameGroups(a: TopicGroups, b: TopicGroups): boolean {
  return JSON.stringify(normalizeGroups(a)) === JSON.stringify(normalizeGroups(b))
}

/**
 * The topics a draft shows, largest first.
 *
 * A group takes the label written for exactly its members, else its largest
 * topic's by fit size, as the backend names it until a label is written.
 *
 * @param tuning - The run's topics, as the tuning view lists them.
 * @param groups - The draft's groups.
 * @returns One topic per group and per topic in no group.
 */
export function draftTopics(tuning: Pick<TopicsTuning, 'leaves' | 'group_labels'>, groups: TopicGroups): DraftTopic[] {
  const byId = new Map(tuning.leaves.map((leaf) => [leaf.id, leaf]))
  const grouped = new Set(groups.flat())
  const shown: DraftTopic[] = groups.map((members) => {
    const leaves = members.flatMap((id) => byId.get(id) ?? [])
    const largest = [...leaves].sort((a, b) => b.fit_size - a.fit_size || a.id - b.id)[0]
    return {
      id: members[0],
      members,
      label: tuning.group_labels[members.join(',')] ?? largest?.label ?? '',
      size: leaves.reduce((sum, leaf) => sum + leaf.size, 0)
    }
  })
  for (const leaf of tuning.leaves) {
    if (!grouped.has(leaf.id)) shown.push({ id: leaf.id, members: [leaf.id], label: leaf.label, size: leaf.size })
  }
  return shown.sort((a, b) => b.size - a.size || a.id - b.id)
}

/** The bins of a relax histogram, each 0.01 wide; a bar is a bin's lower edge. */
export const RELAX_BINS = 100

/**
 * How many chunks a bar would place: the histogram's bins at or above it, as the relax itself admits.
 *
 * @param histogram - Chunks by best cosine in bins of 0.01.
 * @param threshold - The bar, 0.01 to 0.99.
 * @returns The chunks.
 */
export function relaxCount(histogram: number[], threshold: number): number {
  const from = Math.round(threshold * RELAX_BINS)
  return histogram.slice(from).reduce((sum, n) => sum + n, 0)
}

/**
 * The highest bar that still places a chunk: the lower edge of the histogram's highest filled bin.
 *
 * @param histogram - Chunks by best cosine in bins of 0.01.
 * @returns The bar, or null when no bin from 0.01 up holds a chunk.
 */
export function highestBar(histogram: number[]): number | null {
  for (let bin = Math.min(histogram.length, RELAX_BINS) - 1; bin >= 1; bin -= 1) {
    if (histogram[bin] > 0) return bin / RELAX_BINS
  }
  return null
}
