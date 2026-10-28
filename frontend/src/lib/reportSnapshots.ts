import type { HateSpeechRow, NerSourceRow, ReportItemInput, Source } from '@/api/types'
import type { TopicsOverview } from '@/api/topics'
import type { TranslationPayload } from '@/hooks/useTranslatable'

/**
 * Pure builders that turn a view's already-loaded artifact data into a
 * `ReportItemInput` (artifact type + type-prefixed dedupe key + frozen
 * snapshot). Snapshots carry everything the server renderers need, so adding
 * an item needs no extra round-trip and the report is immune to re-ingestion.
 */

/**
 * The canonical text of a finding row: what a text finding displays, and what
 * its Translate control sends (see `translatableTextOf`).
 */
export function chunkTextOf(row: { chunk_text?: string | null; text?: string | null }): string {
  return (row.chunk_text ?? row.text ?? '').trim()
}

/** An image finding's printed words, description and tags, as the API keeps them apart. */
export interface ImageParts {
  ocr_text?: string | null
  image_description?: string | null
  image_tags?: string[] | null
}

/** True when a finding row or frozen snapshot carries an image's parts apart. */
export function hasImageParts(row: ImageParts): boolean {
  return Boolean(row.ocr_text?.trim() || row.image_description?.trim() || row.image_tags?.some((tag) => tag.trim()))
}

/**
 * The text a finding's Translate control sends. One derivation shared by the
 * translate POST, the store key and the "Add all" lookup, so a translation is
 * always found under the key the row filed it under. An image's description
 * and tags are docint's own words, written in `RESPONSE_LANGUAGE`, so only its
 * printed words are sent — and an image with none has nothing to translate.
 */
export function translatableTextOf(
  row: { chunk_text?: string | null; text?: string | null } & ImageParts
): string {
  return hasImageParts(row) ? (row.ocr_text ?? '').trim() : chunkTextOf(row)
}

/** The image parts a snapshot freezes; absent keys keep a text finding's snapshot byte-identical. */
function imagePartsOf(row: ImageParts): ImageParts {
  return {
    ...(row.ocr_text ? { ocr_text: row.ocr_text } : {}),
    ...(row.image_description ? { image_description: row.image_description } : {}),
    ...(row.image_tags?.length ? { image_tags: row.image_tags } : {})
  }
}

export function chatAnswerSnapshot(params: {
  sessionId: string
  turnIdx: number
  userText: string
  modelResponse: string
  reasoning?: string | null
  sources?: Source[]
}): ReportItemInput {
  return {
    artifact_type: 'chat_answer',
    dedupe_key: `chat:${params.sessionId}:${params.turnIdx}`,
    snapshot: {
      session_id: params.sessionId,
      turn_idx: params.turnIdx,
      user_text: params.userText,
      model_response: params.modelResponse,
      reasoning: params.reasoning ?? null,
      sources: (params.sources ?? []).map((s) => ({
        filename: s.filename,
        page: s.page ?? null,
        row: s.row ?? null,
        score: s.score ?? null,
        text: s.text ?? s.preview_text ?? '',
        reference_metadata: s.reference_metadata ?? null,
        // Image identity rides along so the server can freeze the source's
        // thumbnail into the snapshot at add-time (and file_hash as cheap
        // provenance). Conditional like `translation`: absent keys keep old
        // snapshots byte-identical.
        ...(s.image_id ? { image_id: s.image_id, image_collection: s.image_collection ?? null } : {}),
        ...(s.file_hash ? { file_hash: s.file_hash } : {}),
        // The number the generator cited this snippet by. Exports print it
        // beside the source and under its figure, so a reader can tell which
        // of several images the answer's [2] meant. Absent for sources that
        // never reached the prompt — a positional number invented here would
        // contradict the answer's own markers.
        ...(typeof s.citation_index === 'number' ? { citation_index: s.citation_index } : {})
      }))
    }
  }
}

export function entityFindingSnapshot(
  row: NerSourceRow,
  entityLabel: string,
  translation?: { text: string; target_lang: string; model: string }
): ReportItemInput {
  const chunkId = row.chunk_id ?? ''
  return {
    artifact_type: 'entity_finding',
    dedupe_key: `entity:${chunkId}`,
    snapshot: {
      chunk_id: chunkId,
      entity_label: entityLabel,
      chunk_text: row.chunk_text ?? row.text ?? '',
      filename: row.filename ?? '',
      page: row.page ?? null,
      row: row.row ?? null,
      score: row.score ?? null,
      entities: (row.entities ?? []).map((e) => ({ text: e.text, type: e.type, score: e.score ?? null })),
      reference_metadata: row.reference_metadata ?? null,
      ...(row.image_id ? { image_id: row.image_id } : {}),
      ...imagePartsOf(row),
      ...(translation ? { translation } : {})
    }
  }
}

export function summarySnapshot(params: { collection: string; text: string }): ReportItemInput {
  return {
    artifact_type: 'summary',
    dedupe_key: `summary:${params.collection}`,
    snapshot: { collection: params.collection, text: params.text }
  }
}

/** Keywords a topic carries into a report: the ones its row shows. */
const REPORT_TOPIC_KEYWORDS = 5

/**
 * The active run's topics as one report item: every topic's label, counts,
 * leading keywords and summary, largest first, as the Topics tab lists them.
 * One per report like the summary, so a rebuilt run is added again by
 * removing the old item first.
 *
 * @param params.collection - The logical collection the run belongs to.
 * @param params.overview - The run as `GET /collections/{name}/topics` returned it.
 * @returns The report item.
 */
export function topicsSnapshot(params: { collection: string; overview: TopicsOverview }): ReportItemInput {
  const { collection, overview } = params
  return {
    artifact_type: 'topics',
    dedupe_key: `topics:${collection}`,
    snapshot: {
      collection,
      run_id: overview.run_id,
      run_updated_at: overview.updated_at,
      coverage: { total: overview.coverage.total, assigned: overview.coverage.assigned },
      topics: overview.topics.map((topic) => ({
        id: topic.id,
        label: topic.label,
        keywords: topic.keywords.slice(0, REPORT_TOPIC_KEYWORDS).map(([word]) => word),
        size: topic.size,
        share: topic.share,
        description: topic.description
      }))
    }
  }
}

export function hateSpeechSnapshot(row: HateSpeechRow, translation?: TranslationPayload): ReportItemInput {
  const chunkId = row.chunk_id ?? ''
  return {
    artifact_type: 'hate_speech_finding',
    dedupe_key: `hate:${chunkId}`,
    snapshot: {
      chunk_id: chunkId,
      category: row.category ?? '',
      confidence: row.confidence ?? '',
      reason: row.reason ?? '',
      chunk_text: row.chunk_text ?? row.text ?? '',
      filename: row.filename ?? row.source_ref ?? '',
      page: row.page ?? null,
      row: row.row ?? null,
      reference_metadata: row.reference_metadata ?? null,
      ...(row.image_id ? { image_id: row.image_id } : {}),
      ...(row.basis ? { basis: row.basis } : {}),
      ...imagePartsOf(row),
      ...(translation ? { translation } : {})
    }
  }
}
