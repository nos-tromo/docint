# Migrations

Changes that existing collections do not pick up on their own. Each entry says
what changed, who is affected, and the one action that fixes it.

Most docint changes need no migration at all: ingestion is idempotent by file
hash, so re-running an unchanged batch is cheap and a new payload field is
picked up by re-ingesting the files that need it. The entries below are the
cases where that is *not* enough.

## Sparse and dense embeddings moved to bge-m3

**Affects:** dev collections on a non-vLLM provider created before the change.
Production (vLLM) collections already used bge-m3 for both and are unaffected.

The sparse model changed from BM42
(`Qdrant/all_miniLM_L6_v2_with_attentions`) to `BAAI/bge-m3` for
non-vLLM providers, and dense embeddings on the `embed-only` shape now
come from that same bge-m3 instance (fp32 transformers) instead of
Ollama's quantised GGUF build. Dev collections created before this
change need **bge-m3 vectors, which a plain re-ingest will not give
them** — for two reasons: file-hash dedup skips any source file
already recorded as ingested, and swapping models mid-collection would
leave old and new points computed by different models side by side
with no way to tell them apart. The dense dimension itself doesn't
change — bge-m3 is 1024-wide both as Ollama's GGUF build and as fp32
transformers — so checking the vector width is not a valid way to
confirm this migration is unnecessary; the drift is in the numeric
values (quantised vs. fp32) and, for sparse, the switch away from BM42
entirely.

**Fix:** **delete the collection and ingest it again from scratch**. That
covers dense and sparse in one migration, not two.

See [configuration.md](configuration.md#dense-embedding-client--embedclientconfig)
for `EMBED_API_BASE` / `SPARSE_API_BASE` and the `embed-only` deployment shape.

## Payload fields added after a collection was ingested

These are ordinary re-ingests — no deletion needed — but they *do* need one,
because there is no payload migration:

| Change | Fix |
|---|---|
| Posting reference metadata on linked social artifacts (`posting_network`, `posting_author`, …) | Re-ingest the export. Cached Nextext transcripts make this cheap: only embedding is redone, not transcription. See [ingestion.md](ingestion.md#social-media-exports). |
| `ocr_text` on images (`IMAGE_OCR_ENABLED`) | Re-ingest. Images are cached by hash, so clear the `_images` companion or ingest into a fresh collection to have them read. See [configuration.md](configuration.md#image-ingestion--imageingestionconfig). |
| Report evidence thumbnails | Re-ingest the same files. Video keyframes need a fresh transcript run. See [reports.md](reports.md). |
| Full-text `search_text` index | Run `make search-index COLLECTION=<name>` (or `make search-index-all` once per host). Payload-only — no re-embedding, no inference. See [cli-reference.md](cli-reference.md#search-index--full-text-search-backfill). |
