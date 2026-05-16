import { useQuery } from '@tanstack/react-query'
import { getHateSpeech, getIeStats, getNer, getNerStats } from '@/api/collections'
import { useUiStore } from '@/stores/ui'

export function useNerStats(params: Parameters<typeof getNerStats>[0]) {
  const collection = useUiStore((s) => s.selectedCollection)
  return useQuery({
    queryKey: ['ner-stats', collection, params],
    queryFn: () => getNerStats(params),
    enabled: !!collection
  })
}

export function useNer(refresh?: boolean) {
  const collection = useUiStore((s) => s.selectedCollection)
  return useQuery({
    queryKey: ['ner', collection, refresh ?? false],
    queryFn: () => getNer(refresh),
    enabled: !!collection
  })
}

export function useHateSpeech() {
  const collection = useUiStore((s) => s.selectedCollection)
  return useQuery({
    queryKey: ['hate-speech', collection],
    queryFn: getHateSpeech,
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
