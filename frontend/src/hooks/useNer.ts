import { useInfiniteQuery, useQuery } from '@tanstack/react-query'
import {
  getHateSpeechPage,
  getIeStats,
  getNerGraph,
  getNerSourcesPage,
  getNerStats
} from '@/api/collections'
import { useUiStore } from '@/stores/ui'

export function useNerStats(params: Parameters<typeof getNerStats>[0]) {
  const collection = useUiStore((s) => s.selectedCollection)
  return useQuery({
    queryKey: ['ner-stats', collection, params],
    queryFn: () => getNerStats(params),
    enabled: !!collection
  })
}

/**
 * Derived entity graph (nodes + edges) for the active collection. Gated on
 * `enabled` so it only fetches while the graph view is mounted — the force
 * layout and Qdrant scroll are wasted work in the table view.
 */
export function useNerGraph(params: { topKNodes?: number; enabled?: boolean }) {
  const collection = useUiStore((s) => s.selectedCollection)
  const mergeMode = useUiStore((s) => s.entityMergeMode)
  const topKNodes = params.topKNodes ?? 80
  return useQuery({
    queryKey: ['ner-graph', collection, mergeMode, topKNodes],
    queryFn: () =>
      getNerGraph({ top_k_nodes: topKNodes, entity_merge_mode: mergeMode }),
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
  const collection = useUiStore((s) => s.selectedCollection)
  const mergeMode = useUiStore((s) => s.entityMergeMode)
  return useInfiniteQuery({
    queryKey: ['ner-sources', collection, entityKey, mergeMode],
    queryFn: ({ pageParam }) =>
      getNerSourcesPage({
        cursor: pageParam as string | null,
        limit: 50,
        entity_key: entityKey ?? undefined,
        entity_merge_mode: mergeMode
      }),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor,
    enabled: !!collection && !!entityKey
  })
}

export function useHateSpeechPages() {
  const collection = useUiStore((s) => s.selectedCollection)
  return useInfiniteQuery({
    queryKey: ['hate-speech-pages', collection],
    queryFn: ({ pageParam }) =>
      getHateSpeechPage({ cursor: pageParam as string | null, limit: 50 }),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor,
    enabled: !!collection
  })
}

export function useIeStats() {
  const collection = useUiStore((s) => s.selectedCollection)
  return useQuery({
    queryKey: ['ie-stats', collection],
    queryFn: () => getIeStats(collection!),
    enabled: !!collection
  })
}
