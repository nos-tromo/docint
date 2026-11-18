import type { Source } from '@/api/types'
import { referenceMetadataItems } from '@/lib/referenceMetadata'
import { TranslateControl } from '@/components/common/TranslateControl'
import { SourcePreviewAction } from '@/components/common/SourcePreviewAction'
import { useT } from '@/i18n/LanguageContext'

/**
 * A source's details, under the pill that opened them (`SourcePills`).
 *
 * The pill above already carries the citation number and the label, so the
 * details repeat neither: a quiet block on a left rule with the passage (its
 * Translate control revealed on hover or focus), the reference metadata and
 * the preview action. A source with no passage says so in one muted line
 * rather than opening an empty box.
 */
export function Citation({ source }: { source: Source }) {
  const t = useT()
  const refMeta = referenceMetadataItems(source.reference_metadata, {}, t)
  return (
    <div className="group space-y-2 border-l-2 border-border pl-3 text-xs" data-testid="source-detail">
      {source.text ? (
        <TranslateControl rawText={source.text} className="rounded-none bg-transparent p-0 pr-8" />
      ) : (
        <p className="text-muted-foreground">{t('chat.source_no_excerpt')}</p>
      )}
      {(refMeta.length > 0 || source.file_hash) && (
        <div className="flex items-start gap-3">
          {refMeta.length > 0 && (
            <dl className="grid min-w-0 flex-1 grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1">
              {refMeta.map(({ label, value }) => (
                <div key={label} className="contents">
                  <dt className="text-muted-foreground">{label}</dt>
                  <dd className="break-words">{value}</dd>
                </div>
              ))}
            </dl>
          )}
          {/* An action on the source rather than part of its evidence, at the
              block's right edge; it opens the preview dialog in the app. */}
          <SourcePreviewAction
            fileHash={source.file_hash}
            filename={source.filename}
            className="ml-auto shrink-0"
          />
        </div>
      )}
    </div>
  )
}
