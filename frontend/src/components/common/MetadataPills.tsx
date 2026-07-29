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
          className="inline-flex items-center gap-1 rounded border border-border bg-zinc-950 px-1.5 py-0.5 text-[11px]"
        >
          {item.href ? (
            <a
              href={item.href}
              target="_blank"
              rel="noreferrer"
              className="text-blue-400 hover:text-blue-300 break-all"
            >
              {item.value}
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
