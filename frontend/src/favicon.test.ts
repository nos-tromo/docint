// Reads the shipped files from disk, so this test needs Node's types.
/// <reference types="node" />
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import indexHtml from '../index.html?raw'

const srcDir = dirname(fileURLToPath(import.meta.url))
const publicDir = join(srcDir, '..', 'public')

/** `hsl(h s% l%)` as the `#rrggbb` a favicon tile is filled with. */
function hslToHex(h: number, s: number, l: number): string {
  const a = (s / 100) * Math.min(l / 100, 1 - l / 100)
  const channel = (n: number) => {
    const k = (n + h / 30) % 12
    const v = l / 100 - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))
    return Math.round(v * 255).toString(16).padStart(2, '0')
  }
  return `#${channel(0)}${channel(8)}${channel(4)}`
}

/**
 * The `?v=` a head link carries: the first 8 hex digits of the file's SHA-256.
 * A browser keeps a favicon by its URL, whatever the cache headers say, so a
 * redrawn icon has to arrive under a new one.
 */
function version(file: string): string {
  return createHash('sha256').update(readFileSync(join(publicDir, file))).digest('hex').slice(0, 8)
}

describe('favicon', () => {
  it('links the SVG icon, its PNG fallback and the touch icon in the page head, versioned by content', () => {
    const head = indexHtml.split('</head>')[0]
    expect(head).toContain(
      `<link rel="icon" href="/favicon-32.png?v=${version('favicon-32.png')}" type="image/png" sizes="32x32" />`,
    )
    expect(head).toContain(`<link rel="icon" href="/favicon.svg?v=${version('favicon.svg')}" type="image/svg+xml" />`)
    expect(head).toContain(`<link rel="apple-touch-icon" href="/apple-touch-icon.png?v=${version('apple-touch-icon.png')}" />`)
  })

  it('ships every file the page head links', () => {
    const hrefs = [...indexHtml.matchAll(/<link rel="(?:icon|apple-touch-icon)" href="\/([^"?]+)/g)].map((m) => m[1])
    expect(hrefs).toHaveLength(3)
    for (const href of hrefs) expect(readFileSync(join(publicDir, href)).byteLength).toBeGreaterThan(0)
  })

  it('fills the tile with the light app accent', () => {
    const svg = readFileSync(join(publicDir, 'favicon.svg'), 'utf8')
    const tile = /<rect width="32" height="32" rx="7" fill="(#[0-9a-f]{6})"/.exec(svg)?.[1]
    // Read from disk: Vitest hands CSS imports over empty, `?raw` included.
    const css = readFileSync(join(srcDir, 'styles', 'globals.css'), 'utf8')
    const [, h, s, l] = /--app-accent: hsl\((\d+) (\d+)% (\d+)%\)/.exec(css) ?? []
    expect(tile).toBe(hslToHex(Number(h), Number(s), Number(l)))
  })
})
