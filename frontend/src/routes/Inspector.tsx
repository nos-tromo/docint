import { useMemo } from 'react'
import { PageHeader } from '@infra/ui'
import { useDocumentsPages, useDocumentsSummary } from '@/hooks/useDocuments'
import { useUiStore } from '@/stores/ui'
import { DocumentTable } from '@/components/inspector/DocumentTable'
import { DocumentSummary } from '@/components/inspector/DocumentSummary'
import { SessionZipButton } from '@/components/inspector/SessionZipButton'
import { useT } from '@/i18n/LanguageContext'

export function Inspector() {
  const t = useT()
  const collection = useUiStore((s) => s.selectedCollection)
  const query = useDocumentsPages()
  const { data: summary } = useDocumentsSummary()
  const docs = useMemo(
    () => (query.data?.pages ?? []).flatMap((p) => p.items),
    [query.data]
  )
  return (
    <div className="p-8 space-y-6">
      <div className="flex justify-between items-center">
        <PageHeader title={t('inspector.title')} caption={t('inspector.caption')} />
        <SessionZipButton />
      </div>
      {!collection ? (
        <div className="text-sm text-muted-foreground">{t('inspector.select_collection')}</div>
      ) : query.isLoading ? (
        <div className="text-sm text-muted-foreground">{t('common.loading_ellipsis')}</div>
      ) : (
        <>
          <DocumentSummary summary={summary} />
          <DocumentTable
            docs={docs}
            isFetching={query.isFetching}
            hasNextPage={!!query.hasNextPage}
            onLoadMore={() => query.fetchNextPage()}
            collection={collection}
          />
        </>
      )}
    </div>
  )
}
