/**
 * Split a flat list of files into sequentially-uploadable batches whose total
 * size each stays at or under `budgetBytes`.
 *
 * The Ingest view POSTs one multipart request per batch to `/ingest/upload`;
 * nginx caps each *request* body at `client_max_body_size`, so packing files
 * under a per-request byte budget stops a large selection from being rejected
 * with 413 as one oversized body. Because the backend ingestion is idempotent
 * by file hash, uploading batches sequentially to the same collection is safe
 * (files already saved by an earlier batch are skipped, not re-ingested).
 *
 * Packing is greedy and order-preserving: files accumulate into the current
 * batch until the next file would exceed the budget, then a new batch starts.
 * A single file larger than the budget gets its own solo batch (it cannot be
 * split here); if it also exceeds the hard server limit the caller surfaces the
 * resulting 413 for that one file without losing the other batches.
 *
 * @param files - Files to upload, in the order the user added them.
 * @param budgetBytes - Max total bytes per batch (should already fold in any
 *   safety margin under the real server limit). Non-positive budgets degrade to
 *   one file per batch.
 * @returns An array of non-empty file batches; concatenating them reproduces
 *   `files` in order.
 */
export function planUploadBatches(files: File[], budgetBytes: number): File[][] {
  const batches: File[][] = []
  let current: File[] = []
  let currentBytes = 0

  for (const file of files) {
    const size = Number.isFinite(file.size) && file.size > 0 ? file.size : 0
    // Flush the current batch when adding this file would overflow the budget —
    // unless the batch is empty, in which case a lone oversize file must still
    // go somewhere (its own solo batch) rather than producing an empty batch.
    if (current.length > 0 && currentBytes + size > budgetBytes) {
      batches.push(current)
      current = []
      currentBytes = 0
    }
    current.push(file)
    currentBytes += size
  }

  if (current.length > 0) batches.push(current)
  return batches
}
