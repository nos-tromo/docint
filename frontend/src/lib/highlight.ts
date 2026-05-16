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
