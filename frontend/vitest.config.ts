import { fileURLToPath } from 'node:url'
import { mergeConfig } from 'vite'
import { defineConfig } from 'vitest/config'
import viteConfig from './vite.config'

const frontendRoot = fileURLToPath(new URL('./', import.meta.url))

export default mergeConfig(
  viteConfig,
  defineConfig({
    root: frontendRoot,
    test: {
      globals: true,
      environment: 'jsdom',
      include: ['src/__tests__/**/*.test.{ts,tsx}'],
      reporters: process.env.CI ? ['default', 'junit'] : 'default',
      outputFile: process.env.CI ? { junit: 'test-results.xml' } : undefined,
      coverage: {
        provider: 'v8',
        reportsDirectory: 'coverage',
        reporter: ['text', 'lcov'],
      },
    },
  })
)
