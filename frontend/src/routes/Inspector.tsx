import { useDocuments } from '@/hooks/useDocuments'
import { DocumentTable } from '@/components/inspector/DocumentTable'
import { SessionZipButton } from '@/components/inspector/SessionZipButton'

export function Inspector() {
  const { data, isLoading } = useDocuments()
  return (
    <div className="p-8 space-y-4">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-semibold">Inspector</h1>
        <SessionZipButton />
      </div>
      {isLoading ? (
        <div className="text-sm text-muted-foreground">Loading…</div>
      ) : (
        <DocumentTable docs={data?.documents ?? []} />
      )}
    </div>
  )
}
