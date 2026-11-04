import { useMemo } from 'react'
import { PageHeader } from '@infra/ui'
import { topicValue, type TopicSummary } from '@/api/topics'
import { useDocumentsPages, useDocumentsSummary } from '@/hooks/useDocuments'
import { useTopicsOverview, useTopicsRefreshOnJobEnd } from '@/hooks/useTopics'
import { useUiStore } from '@/stores/ui'
import { DocumentTable } from '@/components/inspector/DocumentTable'
import { DocumentSummary } from '@/components/inspector/DocumentSummary'
import { SessionZipButton } from '@/components/inspector/SessionZipButton'
import { ExtractsPanel } from '@/components/inspector/ExtractsPanel'
import { useT } from '@/i18n/LanguageContext'

export function Inspector() {
  const t = useT()
  const collection = useUiStore((s) => s.selectedCollection)
  const query = useDocumentsPages()
  const { data: summary } = useDocumentsSummary()
  // Read only while topics are switched on; without a run there is no column.
  const { data: overview } = useTopicsOverview()
  // A topics job ending while this is open rewrites the column; nothing else here would refetch it.
  useTopicsRefreshOnJobEnd()
  const docs = useMemo(
    () => (query.data?.pages ?? []).flatMap((p) => p.items),
    [query.data]
  )
  // Keyed by the values a document's topics carry, so another run's never finds a topic here; a
  // merged topic answers for each of its members' values.
  const topics = useMemo(
    () =>
      overview
        ? new Map<string, TopicSummary>(
            overview.topics.flatMap((topic) =>
              (topic.members ?? [topic.id]).map((member) => [topicValue(overview.run_id, member), topic] as const)
            )
          )
        : undefined,
    [overview]
  )
  return (
    <div className="p-8 space-y-6">
      <PageHeader
        title={t('inspector.title')}
        caption={t('inspector.caption')}
        actions={<SessionZipButton />}
      />
      {!collection ? (
        <div className="text-sm text-muted-foreground">{t('inspector.select_collection')}</div>
      ) : query.isLoading ? (
        <div className="text-sm text-muted-foreground">{t('common.loading_ellipsis')}</div>
      ) : (
        <>
          <DocumentSummary summary={summary} />
          <ExtractsPanel />
          <DocumentTable
            docs={docs}
            isFetching={query.isFetching}
            hasNextPage={!!query.hasNextPage}
            onLoadMore={() => query.fetchNextPage()}
            collection={collection}
            topics={topics}
          />
        </>
      )}
    </div>
  )
}
