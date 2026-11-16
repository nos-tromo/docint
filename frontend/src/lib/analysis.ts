const PNG_DATA_URI = 'data:image/png;base64,'

/**
 * Whether a stored chart's `data_uri` may reach an `<img src>`.
 *
 * A chart is a PNG that workbench's sandbox drew from retrieved passages, so what the server stored
 * is shown only if it is exactly that: a base64 PNG data URI. Anything else, an SVG or a remote or
 * script URL included, is left out.
 *
 * @param value - The stored `data_uri`.
 * @returns Whether it is a base64 PNG data URI.
 */
export function isPngDataUri(value: unknown): value is string {
  return typeof value === 'string' && value.startsWith(PNG_DATA_URI)
}
