import { useInfiniteQuery, useQuery } from '@tanstack/react-query'
import {
  getHateSpeechPage,
  getIeStats,
  getNerGraph,
  getNerSourcesPage,
  getNerStats
} from '@/api/collections'
import { useActiveCollection } from '@/hooks/useActiveCollection'
import { ENTITY_MERGE_MODE } from '@/api/types'

export function useNerStats(params: Parameters<typeof getNerStats>[0]) {
  const scope = useActiveCollection()
  const { collection } = scope
  return useQuery({
    queryKey: ['ner-stats', scope, params],
    queryFn: () => getNerStats({ ...params, collection: collection ?? undefined }),
    enabled: !!collection
  })
}

/**
 * Derived entity graph (nodes + edges) for the active collection. Gated on
 * `enabled` so it only fetches while the graph view is mounted — the force
 * layout and Qdrant scroll are wasted work in the table view.
 */
export function useNerGraph(params: { topKNodes: number; enabled?: boolean }) {
  const scope = useActiveCollection()
  const { collection } = scope
  const topKNodes = params.topKNodes
  return useQuery({
    queryKey: ['ner-graph', scope, ENTITY_MERGE_MODE, topKNodes],
    queryFn: () =>
      getNerGraph({
        top_k_nodes: topKNodes,
        entity_merge_mode: ENTITY_MERGE_MODE,
        collection: collection ?? undefined
      }),
    enabled: !!collection && (params.enabled ?? true)
  })
}

/**
 * Paginated source rows for one entity. ``entityKey`` is the
 * ``${text}::${type}`` shorthand emitted by ``Analysis.tsx``'s ``keyOf`` so the
 * hook fires only when a concrete entity is selected — this is the lazy
 * counterpart to the deleted ``useNer`` hook that previously triggered a
 * full-collection scroll on every collection switch.
 */
export function useNerSources(entityKey: string | null) {
  const scope = useActiveCollection()
  const { collection } = scope
  return useInfiniteQuery({
    queryKey: ['ner-sources', scope, entityKey, ENTITY_MERGE_MODE],
    queryFn: ({ pageParam }) =>
      getNerSourcesPage({
        cursor: pageParam as string | null,
        limit: 50,
        entity_key: entityKey ?? undefined,
        entity_merge_mode: ENTITY_MERGE_MODE,
        collection: collection ?? undefined
      }),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor,
    enabled: !!collection && !!entityKey
  })
}

export function useHateSpeechPages() {
  const scope = useActiveCollection()
  const { collection } = scope
  return useInfiniteQuery({
    queryKey: ['hate-speech-pages', scope],
    queryFn: ({ pageParam }) =>
      getHateSpeechPage({
        cursor: pageParam as string | null,
        limit: 50,
        collection: collection ?? undefined
      }),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor,
    enabled: !!collection
  })
}

export function useIeStats() {
  const scope = useActiveCollection()
  const { collection } = scope
  return useQuery({
    queryKey: ['ie-stats', scope],
    queryFn: () => getIeStats(collection!),
    enabled: !!collection
  })
}
