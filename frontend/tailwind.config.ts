import type { Config } from 'tailwindcss'

export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        background: 'rgb(9 9 11)',
        foreground: 'rgb(244 244 245)',
        muted: 'rgb(39 39 42)',
        'muted-foreground': 'rgb(161 161 170)',
        border: 'rgb(39 39 42)',
        accent: 'rgb(82 82 91)',
        primary: 'rgb(244 244 245)',
        'primary-foreground': 'rgb(9 9 11)'
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui'],
        mono: ['ui-monospace', 'SFMono-Regular']
      }
    }
  },
  plugins: []
} satisfies Config
