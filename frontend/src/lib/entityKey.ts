/**
 * Entity selection keys.
 *
 * The Analysis screen indexes entities by a `${text}::${type}` shorthand, and
 * the backend resolves that key against its aggregate. The aggregate is merged
 * *orthographically* by default (`Africa`, `africa`, and `AFRICA` collapse into
 * one row), so a key built from a raw chunk surface — as a chat entity pill
 * does — often names the same entity under a different surface. Matching only
 * on the literal string would strand those clicks on an empty or unrelated
 * selection, and it would do so precisely for the entities the merge collapses.
 */

export interface EntityLike {
  text?: string | null
  type?: string | null
}

/**
 * Build the `${text}::${type}` selection key.
 *
 * @param text - Entity surface form.
 * @param type - Entity type label.
 * @returns The selection key.
 */
export function entityKey(text: string | null | undefined, type: string | null | undefined): string {
  return `${text ?? ''}::${type ?? ''}`
}

/**
 * Split a selection key back into its surface and type.
 *
 * @param key - A `${text}::${type}` key.
 * @returns The parts, or null when the key is malformed.
 */
function splitKey(key: string): { text: string; type: string } | null {
  const idx = key.lastIndexOf('::')
  if (idx < 0) return null
  return { text: key.slice(0, idx), type: key.slice(idx + 2) }
}

/**
 * Compact a surface the way the backend's `orthographic` merge mode does:
 * drop every non-alphanumeric character and casefold.
 *
 * @param value - A surface form.
 * @returns The compacted form.
 */
function compact(value: string): string {
  return value.replace(/[^\p{L}\p{N}]/gu, '').toLowerCase()
}

/**
 * Resolve a selection key to the matching key in a loaded entity aggregate.
 *
 * Tries exact, then case-insensitive, then orthographically-compacted
 * matching. The type must match exactly at every rung — two entities sharing a
 * surface under different types are different entities.
 *
 * @param key - The requested selection key, e.g. from a chat entity pill.
 * @param candidates - Entities currently loaded for the active collection.
 * @returns The aggregate's own key for that entity, or null when it holds none.
 */
export function resolveEntityKey(key: string | null | undefined, candidates: EntityLike[]): string | null {
  if (!key) return null
  const wanted = splitKey(key)
  if (!wanted) return null

  const sameType = candidates.filter((c) => (c.type ?? '') === wanted.type)
  if (sameType.length === 0) return null

  const exact = sameType.find((c) => (c.text ?? '') === wanted.text)
  if (exact) return entityKey(exact.text, exact.type)

  const folded = wanted.text.toLowerCase()
  const caseInsensitive = sameType.find((c) => (c.text ?? '').toLowerCase() === folded)
  if (caseInsensitive) return entityKey(caseInsensitive.text, caseInsensitive.type)

  const compacted = compact(wanted.text)
  if (!compacted) return null
  const orthographic = sameType.find((c) => compact(c.text ?? '') === compacted)
  return orthographic ? entityKey(orthographic.text, orthographic.type) : null
}
