// Build the React fragments needed to highlight all case-insensitive
// occurrences of `terms` inside `text`. Returns an array suitable for
// rendering directly (or wrapping in <pre>/<span>). Longer terms are
// matched first so "European Union" wins over "Union" when both are in
// the list.
export interface HighlightSegment {
  text: string
  highlight: boolean
}

export function highlightSegments(text: string, terms: string[]): HighlightSegment[] {
  if (!text) return []
  const cleaned = terms
    .map((t) => t.trim())
    .filter((t) => t.length > 0)
    .sort((a, b) => b.length - a.length)
  if (cleaned.length === 0) return [{ text, highlight: false }]

  const seen = new Set<string>()
  const escaped: string[] = []
  for (const term of cleaned) {
    const lowered = term.toLowerCase()
    if (seen.has(lowered)) continue
    seen.add(lowered)
    escaped.push(term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
  }
  const pattern = new RegExp(`(${escaped.join('|')})`, 'gi')

  const out: HighlightSegment[] = []
  let cursor = 0
  for (const match of text.matchAll(pattern)) {
    const start = match.index ?? 0
    if (start > cursor) out.push({ text: text.slice(cursor, start), highlight: false })
    out.push({ text: match[0], highlight: true })
    cursor = start + match[0].length
  }
  if (cursor < text.length) out.push({ text: text.slice(cursor), highlight: false })
  return out
}

/** Word-ish runs, matching how the search index tokenizes: letters, digits
 *  and underscores, with everything else acting as a separator. */
const WORD_RUN = /[\p{L}\p{N}_]+/gu

/**
 * Highlight the words a full-text search matched, replicating the index.
 *
 * Qdrant's `MatchText` over the lowercase `search_text` index matches a word
 * **prefix**, so `Partei` matches `Parteitag` while `tag` matches neither —
 * unlike {@link highlightSegments}, which is a plain substring search. Using
 * that here would paint matches the backend never made and, worse, imply
 * mid-word matching works.
 *
 * The whole matched word is highlighted, not just its matching prefix: the
 * compound is what the hit is, and half-painted words read as a rendering bug.
 *
 * @param text - The preview to mark up.
 * @param keywords - The searched keywords, as typed.
 * @returns Contiguous segments, each flagged as matched or not.
 */
export function keywordSegments(text: string, keywords: string[]): HighlightSegment[] {
  if (!text) return []
  const needles = keywords.map((k) => k.trim().toLowerCase()).filter((k) => k.length > 0)
  if (needles.length === 0) return [{ text, highlight: false }]

  const out: HighlightSegment[] = []
  let cursor = 0
  for (const match of text.matchAll(WORD_RUN)) {
    const start = match.index ?? 0
    const word = match[0]
    if (!needles.some((needle) => word.toLowerCase().startsWith(needle))) continue
    if (start > cursor) out.push({ text: text.slice(cursor, start), highlight: false })
    out.push({ text: word, highlight: true })
    cursor = start + word.length
  }
  if (cursor < text.length) out.push({ text: text.slice(cursor), highlight: false })
  return out
}
