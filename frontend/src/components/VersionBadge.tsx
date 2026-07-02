import { useQuery } from '@tanstack/react-query'
import { Badge } from '@infra/ui'
import { getVersion } from '../api/config'

export function VersionBadge() {
  const { data } = useQuery({
    queryKey: ['version'],
    queryFn: getVersion,
    staleTime: Infinity
  })
  if (!data?.version) return null
  return <Badge variant="neutral">v{data.version}</Badge>
}
