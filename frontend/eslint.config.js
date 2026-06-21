// eslint.config.js - shared flat ESLint config (ESLint 9) for nos-tromo React/TS frontends.
//
// Canonical source: nos-tromo/.github/configs/frontend-eslint/eslint.config.js
// Vendored verbatim into each frontend at frontend/eslint.config.js; CI fails on
// drift via scripts/validate_eslint_config.py (the same canonical-config + drift-check
// pattern used for make/common.mk). Do not edit the vendored copy - change the
// canonical file and re-vendor.
//
// Consuming repos need these frontend devDeps: eslint, @eslint/js, typescript-eslint,
// globals, eslint-plugin-react, eslint-plugin-react-hooks.
import js from '@eslint/js'
import globals from 'globals'
import reactPlugin from 'eslint-plugin-react'
import reactHooks from 'eslint-plugin-react-hooks'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  { ignores: ['dist/**', 'coverage/**'] },
  {
    files: ['**/*.{ts,tsx}'],
    plugins: { 'react-hooks': reactHooks },
    extends: [
      js.configs.recommended,
      ...tseslint.configs.recommended,
      reactPlugin.configs.flat.recommended,
      reactPlugin.configs.flat['jsx-runtime']
    ],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: { ...globals.browser, ...globals.node },
      parserOptions: { ecmaFeatures: { jsx: true } }
    },
    settings: { react: { version: 'detect' } },
    rules: {
      // TypeScript enforces prop types and the new JSX transform makes
      // `import React` unnecessary, so these eslint-plugin-react rules are noise.
      'react/prop-types': 'off',
      // The two stable react-hooks rules only. eslint-plugin-react-hooks v7's
      // experimental React Compiler rules (set-state-in-effect, incompatible-library,
      // ...) flag legitimate TanStack and reset-on-prop-change patterns, so they are
      // intentionally left off until the codebases are ready for them.
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
      // Allow intentionally-unused args via a leading underscore (e.g. `_init`),
      // a convention already used across the frontends.
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }]
    }
  },
  {
    // Tests use `as any` casts and run under node/vitest globals.
    files: ['**/*.test.{ts,tsx}', 'src/test/**/*.{ts,tsx}'],
    languageOptions: { globals: { ...globals.node } },
    rules: {
      '@typescript-eslint/no-explicit-any': 'off'
    }
  }
)
