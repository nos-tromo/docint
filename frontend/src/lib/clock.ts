/** Parse a clip position into seconds.
 *
 *  A time range over a video is written the way a player shows it, so the
 *  field accepts `90`, `1:30` and `01:02:03` alike. Returns `null` for
 *  anything else, including an empty string — a bound that cannot be read is
 *  not a bound of zero, and sending one would silently narrow the search.
 *
 *  @param input The typed value.
 *  @returns The position in seconds, or `null` when it is not a time.
 */
export function parseClockSeconds(input: string): number | null {
  const trimmed = input.trim()
  if (!trimmed) return null

  const parts = trimmed.split(':')
  if (parts.length > 3) return null
  if (!parts.every((part) => /^\d+(\.\d+)?$/.test(part))) return null

  const numbers = parts.map(Number)
  if (numbers.some((value) => !Number.isFinite(value))) return null
  // Minutes and seconds past 59 are a typo, not a position: `1:75` reads as
  // "one seventy-five" far more often than it means 2:15.
  if (parts.length > 1 && numbers.slice(1).some((value) => value >= 60)) return null

  return numbers.reduce((total, value) => total * 60 + value, 0)
}

/** Format a position in seconds the way the time fields accept it.
 *
 *  @param seconds The position in seconds.
 *  @returns A `m:ss` or `h:mm:ss` string.
 */
export function formatClockSeconds(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds))
  const hours = Math.floor(whole / 3600)
  const minutes = Math.floor((whole % 3600) / 60)
  const secs = whole % 60
  const pad = (value: number) => String(value).padStart(2, '0')
  return hours ? `${hours}:${pad(minutes)}:${pad(secs)}` : `${minutes}:${pad(secs)}`
}
