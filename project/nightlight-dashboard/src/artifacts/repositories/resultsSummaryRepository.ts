import { ArtifactError } from '../errors'
import type { ArtifactReadOptions, ArtifactSource } from '../ports'
import {
  parseResultsSummaryArtifact,
  type ResultsSummary,
  type ValidatedArtifact,
} from '../schemas/legacyArtifacts'
import { HttpArtifactSource } from '../sources/httpArtifactSource'

const RESULTS_SUMMARY_PATH = 'data/results_summary.json'

export class ResultsSummaryRepository {
  readonly source: ArtifactSource
  #validatedCache: ValidatedArtifact<ResultsSummary> | null = null

  constructor(source: ArtifactSource) {
    this.source = source
  }

  async get(options: ArtifactReadOptions = {}): Promise<ValidatedArtifact<ResultsSummary>> {
    if (options.signal?.aborted) {
      throw new ArtifactError('aborted', 'Artifact request was aborted before it started', {
        path: RESULTS_SUMMARY_PATH,
      })
    }
    if (this.#validatedCache) return this.#validatedCache

    const rawValue = await this.source.read(RESULTS_SUMMARY_PATH, options)
    const validated = parseResultsSummaryArtifact(rawValue)
    this.#validatedCache = validated
    return validated
  }

  clearCache(): void {
    this.#validatedCache = null
  }
}

export const resultsSummaryRepository = new ResultsSummaryRepository(
  new HttpArtifactSource(import.meta.env.BASE_URL),
)
