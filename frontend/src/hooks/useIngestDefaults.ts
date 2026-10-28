import { useQuery } from '@tanstack/react-query'
import { apiGet } from '@/api/client'

export interface IngestDefaults {
  ner: boolean
  hate_speech: boolean
  summary: boolean
  /** `TOPICS_ON_INGEST`, off whenever topics are; absent from an API older than the option. */
  topics?: boolean
}

/** Deployment-default enrichment toggles for seeding the ingest toggles. */
export function useIngestDefaults() {
  return useQuery({
    queryKey: ['ingest-defaults'],
    queryFn: () => apiGet<IngestDefaults>('/config/ingest-defaults'),
    staleTime: Infinity,
    gcTime: Infinity
  })
}
