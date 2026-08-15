import { ExternalLinkIcon } from '@infra/ui'
import type { MetadataPillItem } from '@/lib/referenceMetadata'

/**
 * Compact metadata chips for the Analysis tables' metadata cells — same visual
 * language as the entity table's matched-mention pills. Purely presentational;
 * curation happens in referenceMetadataPills().
 */
export function MetadataPills({ items }: { items: MetadataPillItem[] }) {
  if (!items.length) return null
  return (
    <ul className="flex flex-wrap gap-1" data-testid="metadata-pills">
      {items.map((item) => (
        <li
          key={item.key}
          className="inline-flex items-center gap-1 rounded border border-border bg-muted px-1.5 py-0.5 text-[11px]"
        >
          {item.href ? (
            // The leaving-arrow is drawn and driven by `href`, not appended to
            // the pill's copy — a symbol living in a translation string is one
            // a translator can drop, and `↗` renders as an emoji on some
            // platforms.
            <a
              href={item.href}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-blue-400 hover:text-blue-300 break-all"
            >
              {item.value}
              <ExternalLinkIcon className="h-3 w-3 shrink-0" />
            </a>
          ) : (
            <>
              {item.label && <span className="text-muted-foreground">{item.label}</span>}
              <span className="break-all">{item.value}</span>
            </>
          )}
        </li>
      ))}
    </ul>
  )
}
