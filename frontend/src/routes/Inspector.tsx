import { useMemo } from 'react'
import { useDocumentsPages, useDocumentsSummary } from '@/hooks/useDocuments'
import { useUiStore } from '@/stores/ui'
import { DocumentTable } from '@/components/inspector/DocumentTable'
import { DocumentSummary } from '@/components/inspector/DocumentSummary'
import { SessionZipButton } from '@/components/inspector/SessionZipButton'

export function Inspector() {
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
        <h1 className="text-2xl font-semibold">Inspector</h1>
        <SessionZipButton />
      </div>
      {!collection ? (
        <div className="text-sm text-muted-foreground">Select a collection.</div>
      ) : query.isLoading ? (
        <div className="text-sm text-muted-foreground">Loading…</div>
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
