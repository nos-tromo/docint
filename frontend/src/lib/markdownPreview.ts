// Only a short excerpt is ever shown, and the inline rewrites below are not all linear.
const SOURCE_CHARS = 2000
const FENCE = /^\s*(```|~~~)/
const DELIMITER_CELL = /^\s*:?-+:?\s*$/
const BREAK_MARKS = /^(?:-{3,}|\*{3,}|_{3,})$/

/** A table's delimiter row (`| --- | :--: |`), checked cell by cell so no pattern spans the line. */
function isTableRule(line: string): boolean {
  return line
    .trim()
    .replace(/^\||\|$/g, '')
    .split('|')
    .every((cell) => DELIMITER_CELL.test(cell))
}

/** A thematic break: three or more of one of `-`, `*`, `_`, spaced or not. */
function isThematicBreak(line: string): boolean {
  return BREAK_MARKS.test(line.replace(/\s/g, ''))
}

/** A table row's cells, or `null` for a line that is not one. */
function tableCells(line: string): string[] | null {
  const row = line.trim()
  return row.length > 1 && row.startsWith('|') && row.endsWith('|') ? row.slice(1, -1).split('|') : null
}

/**
 * Markdown flattened to readable plain text, for a short excerpt.
 *
 * The Report tab previews a frozen chat answer or summary as a few lines of
 * text; without this, its Markdown syntax (`###`, `**`, table rules) would be
 * shown verbatim. Line structure is kept, blank lines dropped, and citation
 * markers such as `[3]` survive, since they are not link syntax. Only the
 * first `SOURCE_CHARS` characters are read: a stored answer has no size
 * limit, and the preview never shows more than its opening.
 *
 * @param text Markdown source.
 * @returns The same content as plain text.
 */
export function markdownPreview(text: string): string {
  return text
    .slice(0, SOURCE_CHARS)
    .split('\n')
    .filter((line) => !FENCE.test(line) && !isTableRule(line) && !isThematicBreak(line))
    .map((line) => {
      const cells = tableCells(line)
      const flat = cells ? cells.map((cell) => cell.trim()).join(' · ') : line
      return flat
        .replace(/^\s{0,3}#{1,6}\s+/, '')
        .replace(/^\s*>\s?/, '')
        .replace(/^\s*(?:[-*+]|\d+[.)])\s+/, '')
        .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
        .replace(/(\*\*|__|~~)(.+?)\1/g, '$2')
        .replace(/(^|[^\w*])\*(?!\s)([^*]+?)\*(?!\w)/g, '$1$2')
        .replace(/(^|\W)_(?!\s)([^_]+?)_(?!\w)/g, '$1$2')
        .replace(/`([^`]+)`/g, '$1')
        .trim()
    })
    .filter(Boolean)
    .join('\n')
}
