import type { Config } from 'tailwindcss'
import typography from '@tailwindcss/typography'

export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      // Both families intentionally resolve to the same var so all text
      // shares one font. The fallback chain lives in --app-font
      // (src/styles/globals.css), the single place to change the app font.
      fontFamily: {
        sans: ['var(--app-font)'],
        mono: ['var(--app-font)']
      }
    }
  },
  plugins: [typography]
} satisfies Config
