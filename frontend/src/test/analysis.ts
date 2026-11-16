import type { Analysis } from '@/api/types'

/** The smallest PNG data URI the card accepts as a chart. */
export const PNG_URI = 'data:image/png;base64,iVBORw0KGgo='

/** A synthetic analysis: seven rows in one table, one chart, one source. */
export const ANALYSIS: Analysis = {
  id: 'a1',
  schema: 1,
  title: 'Posts per author',
  question: 'Who posted most?',
  code: "import pandas as pd\nprint('posts per author')",
  stdout: 'posts per author',
  tables: [
    {
      name: 'posts.csv',
      columns: ['author', 'posts'],
      rows: ['alice', 'bob', 'carol', 'dave', 'erin', 'frank', 'grace'].map((name, i) => [name, String(7 - i)]),
      row_count: 7,
      truncated: false
    }
  ],
  charts: [{ name: 'chart.png', data_uri: PNG_URI, bytes: 12 }],
  provenance: {
    session_id: 'session-1',
    turn_idx: 0,
    collection: 'docs',
    sources: [{ citation_index: 1, chunk_id: 'c1', file_hash: 'h1', filename: 'harbour-report.pdf', page: 3 }],
    inputs: { passages: 1, keyword_checks: 0, topic_listings: 0 },
    workbench: { job_id: 'job-1', version: '0.2.0', tool: 'run_python' },
    created_at: '2026-11-16T10:00:00+00:00'
  }
}
