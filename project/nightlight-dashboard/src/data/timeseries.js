/**
 * Compute summary stats from a recovery series
 * @param {{ day: number, R_buffer: number | null, R_nonBuffer: number | null, isPostDisaster?: boolean }[]} series
 */
export function computeResilienceStats(series) {
  const post = series.filter(d => (
    d.isPostDisaster
    && Number.isFinite(d.R_buffer)
    && Number.isFinite(d.R_nonBuffer)
  ))
  if (!post.length) return null

  const meanBuf    = post.reduce((s, d) => s + /** @type {number} */ (d.R_buffer), 0) / post.length
  const meanNonBuf = post.reduce((s, d) => s + /** @type {number} */ (d.R_nonBuffer), 0) / post.length
  const minR       = Math.min(...post.map(d => /** @type {number} */ (d.R_nonBuffer)))
  const ra         = parseFloat((meanBuf - meanNonBuf).toFixed(3))

  // Recovery day = first day non-buffer R > 0.9
  const recoveryDay = post.find(d => /** @type {number} */ (d.R_nonBuffer) > 0.9)?.day ?? null

  return { meanBuf, meanNonBuf, minR, ra, recoveryDay }
}
