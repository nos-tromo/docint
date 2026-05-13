import { useDocuments } from '@/hooks/useDocuments'
import { DocumentTable } from '@/components/inspector/DocumentTable'

export function Inspector() {
  const { data, isLoading } = useDocuments()
  return (
    <div className="p-8 space-y-4">
      <h1 className="text-2xl font-semibold">Inspector</h1>
      {isLoading ? (
        <div className="text-sm text-muted-foreground">Loading…</div>
      ) : (
        <DocumentTable docs={data?.documents ?? []} />
      )}
    </div>
  )
}
