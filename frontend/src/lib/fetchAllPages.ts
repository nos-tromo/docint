import type { Page } from '@/api/collections'

/** How many items one walk may collect before it stops asking for more. */
export const DEFAULT_MAX_ITEMS = 2000

/**
 * Collect every item of a cursor-paginated endpoint.
 *
 * The Analysis tables render one page at a time behind a "Load more", so a
 * section-wide action ("add all findings to the report", an export) must walk
 * the cursors itself rather than trust what happens to be on screen. Deliberately
 * outside React Query: the walk feeds one action, and caching it as the table's
 * page state would force the rendered list to expand with it.
 *
 * Two stops besides the end of the data: `maxItems`, so an unexpectedly large
 * collection cannot be walked forever, and a cursor that fails to advance,
 * which would otherwise spin the loop. A failing page rejects — a short list
 * returned as if complete is exactly the silent sample this walk exists to
 * avoid.
 *
 * @param fetchPage Fetch one page, given the previous page's cursor (`null` first).
 * @param opts.maxItems Ceiling on collected items (default {@link DEFAULT_MAX_ITEMS}).
 * @returns Every item, in page order.
 */
export async function fetchAllPages<T>(
  fetchPage: (cursor: string | null) => Promise<Page<T>>,
  opts: { maxItems?: number } = {}
): Promise<T[]> {
  const maxItems = opts.maxItems ?? DEFAULT_MAX_ITEMS
  const items: T[] = []
  const seenCursors = new Set<string>()
  let cursor: string | null = null

  while (items.length < maxItems) {
    const page: Page<T> = await fetchPage(cursor)
    items.push(...(page.items ?? []))
    const next = page.next_cursor
    if (!next || seenCursors.has(next)) break
    seenCursors.add(next)
    cursor = next
  }
  return items.slice(0, maxItems)
}
