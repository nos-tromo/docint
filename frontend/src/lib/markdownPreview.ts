const FENCE = /^\s*(```|~~~)/
const TABLE_RULE = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/
const THEMATIC_BREAK = /^\s*([-*_])(\s*\1){2,}\s*$/
const TABLE_ROW = /^\s*\|(.*)\|\s*$/

/**
 * Markdown flattened to readable plain text, for a short excerpt.
 *
 * The Report tab previews a frozen chat answer or summary as a few lines of
 * text; without this, its Markdown syntax (`###`, `**`, table rules) would be
 * shown verbatim. Line structure is kept, blank lines dropped, and citation
 * markers such as `[3]` survive, since they are not link syntax.
 *
 * @param text Markdown source.
 * @returns The same content as plain text.
 */
export function markdownPreview(text: string): string {
  return text
    .split('\n')
    .filter((line) => !FENCE.test(line) && !TABLE_RULE.test(line) && !THEMATIC_BREAK.test(line))
    .map((line) => {
      const row = TABLE_ROW.exec(line)
      const flat = row
        ? row[1]
            .split('|')
            .map((cell) => cell.trim())
            .join(' · ')
        : line
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
