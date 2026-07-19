/**
 * Sanitizes a string for use as a filename component.
 *
 * - Lowercases the input
 * - Replaces any character outside [a-z0-9._-] with dashes
 * - Collapses consecutive dashes
 * - Trims leading/trailing dashes
 * - Returns 'docint' if the result is empty
 */
export function sanitizeExportFilename(name: string, fallback = 'docint'): string {
  // Lowercase and replace non-alphanumeric (except . _ -) with dashes
  const sanitized = name
    .toLowerCase()
    .replace(/[^a-z0-9._-]/g, '-')
    // Collapse consecutive dashes
    .replace(/-+/g, '-')
    // Trim leading/trailing dashes
    .replace(/^-+|-+$/g, '')

  // Return fallback if empty
  return sanitized || fallback
}
