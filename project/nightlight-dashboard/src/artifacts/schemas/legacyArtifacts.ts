import { ArtifactSchemaError, UnsupportedArtifactVersionError } from '../errors'

export type WireVersion = 'legacy-v0' | '1'

export interface ValidatedArtifact<T> {
  wireVersion: WireVersion
  data: T
  limitations: readonly string[]
}

export interface ResultsFeatureImportance {
  feature: string
  rf_imp: number
  xgb_imp: number
  avg_imp: number
}

export interface ResultsModelComparison {
  name: string
  mean_auc: number
  std: number
  f1: number
}

export interface ResultsLoeoRow {
  held_out: string
  rf_auc: number
  xgb_auc: number
  logit_auc: number
}

export interface ResultsProbabilityStats {
  n: number
  min: number
  max: number
  mean: number
  p25: number
  p50: number
  p75: number
  above_50: number
  above_80: number
}

export interface ResultsSummary {
  feature_importance: ResultsFeatureImportance[]
  model_comparison: Record<string, ResultsModelComparison>
  loeo_by_model: Record<string, ResultsLoeoRow[]>
  prob_stats: Record<string, ResultsProbabilityStats>
  [key: string]: unknown
}

export interface LegacyFacilityRow {
  name: string
  type: string
  coords: [number, number]
  probability: number
  radiusM: number
}

export type FacilityProbabilityStatus =
  | 'available'
  | 'unavailable'
  | 'not_assessed'
  | 'computation_failed'
  | 'validation_failed'

export interface FacilityProbabilityV1 {
  value: number | null
  status: FacilityProbabilityStatus
  reason: string | null
  provenance: {
    eligiblePixelCount: number | null
    finiteProbabilityCount: number | null
    aggregationMethod: string
    [key: string]: unknown
  }
}

export interface FacilityRecordV1 {
  facilityId: string
  name: string
  type: string
  coordinates: [number, number]
  radiusM: number
  probability: FacilityProbabilityV1
}

export interface FacilityArtifactV1 {
  schemaVersion: '1.0.0'
  artifactType: 'nightlight-facility-probabilities'
  source: Record<string, unknown>
  provenance: Record<string, unknown>
  records: FacilityRecordV1[]
}

export interface LegacyTimeSeriesRow {
  day: number
  R_buffer: number | null
  R_nonBuffer: number | null
  isPostDisaster: boolean
}

const LEGACY_RESULTS_LIMITATIONS = [
  'unversioned-results-summary',
] as const

const LEGACY_FACILITY_LIMITATIONS = [
  'unversioned-facility-rows',
  'probability-0.5-may-be-an-exporter-fallback-without-missingness-provenance',
] as const

const LEGACY_TIME_SERIES_LIMITATIONS = [
  'unversioned-time-series',
  'paired-null-is-missing-observation-not-zero',
] as const

const FACILITY_REASON_STATUS = {
  no_eligible_pixels_in_facility_type_buffer: 'unavailable',
  all_eligible_probabilities_missing: 'unavailable',
  source_probability_pixels_unavailable: 'unavailable',
  facility_outside_assessment_scope: 'not_assessed',
  required_facility_metadata_missing: 'not_assessed',
  pixel_probability_computation_failed: 'computation_failed',
  facility_aggregation_failed: 'computation_failed',
  nonfinite_aggregate: 'computation_failed',
  source_version_unverified: 'validation_failed',
  source_receipt_missing: 'validation_failed',
  probability_out_of_range: 'validation_failed',
  producer_record_invalid: 'validation_failed',
} as const satisfies Record<string, Exclude<FacilityProbabilityStatus, 'available'>>

type FacilityUnavailableReason = keyof typeof FACILITY_REASON_STATUS

const FORBIDDEN_VERSION_VALUES = new Set(['', 'latest', 'unknown', 'unversioned'])

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function schemaError(path: string, expectation: string): never {
  throw new ArtifactSchemaError(`${path} ${expectation}`)
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (!isRecord(value)) schemaError(path, 'must be an object')
  return value
}

function requireArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) schemaError(path, 'must be an array')
  return value
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== 'string' || value.length === 0) schemaError(path, 'must be a non-empty string')
  return value
}

function requireFiniteNumber(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) schemaError(path, 'must be a finite number')
  return value
}

function requireProbability(value: unknown, path: string): number {
  const probability = requireFiniteNumber(value, path)
  if (probability < 0 || probability > 1) schemaError(path, 'must be between 0 and 1')
  return probability
}

function requireNonNegativeInteger(value: unknown, path: string): number {
  const number = requireFiniteNumber(value, path)
  if (!Number.isInteger(number) || number < 0) schemaError(path, 'must be a non-negative integer')
  return number
}

function requireVersion(value: unknown, path: string): string {
  const version = requireString(value, path)
  if (FORBIDDEN_VERSION_VALUES.has(version)) schemaError(path, 'must identify a pinned version')
  return version
}

function parseFeatureImportance(value: unknown, path: string): ResultsFeatureImportance {
  const row = requireRecord(value, path)
  return {
    feature: requireString(row.feature, `${path}.feature`),
    rf_imp: requireFiniteNumber(row.rf_imp, `${path}.rf_imp`),
    xgb_imp: requireFiniteNumber(row.xgb_imp, `${path}.xgb_imp`),
    avg_imp: requireFiniteNumber(row.avg_imp, `${path}.avg_imp`),
  }
}

function parseModelComparison(value: unknown, path: string): ResultsModelComparison {
  const row = requireRecord(value, path)
  return {
    name: requireString(row.name, `${path}.name`),
    mean_auc: requireFiniteNumber(row.mean_auc, `${path}.mean_auc`),
    std: requireFiniteNumber(row.std, `${path}.std`),
    f1: requireFiniteNumber(row.f1, `${path}.f1`),
  }
}

function parseLoeoRow(value: unknown, path: string): ResultsLoeoRow {
  const row = requireRecord(value, path)
  return {
    held_out: requireString(row.held_out, `${path}.held_out`),
    rf_auc: requireFiniteNumber(row.rf_auc, `${path}.rf_auc`),
    xgb_auc: requireFiniteNumber(row.xgb_auc, `${path}.xgb_auc`),
    logit_auc: requireFiniteNumber(row.logit_auc, `${path}.logit_auc`),
  }
}

function parseProbabilityStats(value: unknown, path: string): ResultsProbabilityStats {
  const row = requireRecord(value, path)
  return {
    n: requireFiniteNumber(row.n, `${path}.n`),
    min: requireProbability(row.min, `${path}.min`),
    max: requireProbability(row.max, `${path}.max`),
    mean: requireProbability(row.mean, `${path}.mean`),
    p25: requireProbability(row.p25, `${path}.p25`),
    p50: requireProbability(row.p50, `${path}.p50`),
    p75: requireProbability(row.p75, `${path}.p75`),
    above_50: requireFiniteNumber(row.above_50, `${path}.above_50`),
    above_80: requireFiniteNumber(row.above_80, `${path}.above_80`),
  }
}

function parseRecord<T>(
  value: unknown,
  path: string,
  parseValue: (entry: unknown, entryPath: string) => T,
): Record<string, T> {
  const record = requireRecord(value, path)
  const entries = Object.entries(record)
  if (entries.length === 0) schemaError(path, 'must not be empty')
  return Object.fromEntries(entries.map(([key, entry]) => [key, parseValue(entry, `${path}.${key}`)]))
}

function validateResultsSummary(value: unknown): ResultsSummary {
  const summary = requireRecord(value, 'results summary')
  return {
    ...summary,
    feature_importance: requireArray(summary.feature_importance, 'feature_importance')
      .map((entry, index) => parseFeatureImportance(entry, `feature_importance[${index}]`)),
    model_comparison: parseRecord(summary.model_comparison, 'model_comparison', parseModelComparison),
    loeo_by_model: parseRecord(summary.loeo_by_model, 'loeo_by_model', (entry, path) => (
      requireArray(entry, path).map((row, index) => parseLoeoRow(row, `${path}[${index}]`))
    )),
    prob_stats: parseRecord(summary.prob_stats, 'prob_stats', parseProbabilityStats),
  }
}

export function parseResultsSummaryArtifact(value: unknown): ValidatedArtifact<ResultsSummary> {
  if (isRecord(value) && 'schemaVersion' in value) {
    if (value.schemaVersion !== 1) throw new UnsupportedArtifactVersionError(value.schemaVersion)
    return {
      wireVersion: '1',
      data: validateResultsSummary(value.data),
      limitations: [],
    }
  }

  return {
    wireVersion: 'legacy-v0',
    data: validateResultsSummary(value),
    limitations: LEGACY_RESULTS_LIMITATIONS,
  }
}

export function parseLegacyFacilityArtifact(value: unknown): ValidatedArtifact<LegacyFacilityRow[]> {
  const rows = requireArray(value, 'facility artifact').map((entry, index) => {
    const path = `facility artifact[${index}]`
    const row = requireRecord(entry, path)
    const coords = requireArray(row.coords, `${path}.coords`)
    if (coords.length !== 2) schemaError(`${path}.coords`, 'must contain longitude and latitude')
    return {
      name: requireString(row.name, `${path}.name`),
      type: requireString(row.type, `${path}.type`),
      coords: [
        requireFiniteNumber(coords[0], `${path}.coords[0]`),
        requireFiniteNumber(coords[1], `${path}.coords[1]`),
      ] as [number, number],
      probability: requireProbability(row.probability, `${path}.probability`),
      radiusM: requireFiniteNumber(row.radiusM, `${path}.radiusM`),
    }
  })

  return { wireVersion: 'legacy-v0', data: rows, limitations: LEGACY_FACILITY_LIMITATIONS }
}

function parseFacilityProbabilityV1(value: unknown, path: string): FacilityProbabilityV1 {
  const probability = requireRecord(value, path)
  const status = requireString(probability.status, `${path}.status`) as FacilityProbabilityStatus
  const provenance = requireRecord(probability.provenance, `${path}.provenance`)
  const aggregationMethod = requireString(provenance.aggregationMethod, `${path}.provenance.aggregationMethod`)
  const eligiblePixelCount = provenance.eligiblePixelCount === null
    ? null
    : requireNonNegativeInteger(provenance.eligiblePixelCount, `${path}.provenance.eligiblePixelCount`)
  const finiteProbabilityCount = provenance.finiteProbabilityCount === null
    ? null
    : requireNonNegativeInteger(provenance.finiteProbabilityCount, `${path}.provenance.finiteProbabilityCount`)

  if (status === 'available') {
    const availableValue = requireProbability(probability.value, `${path}.value`)
    if (probability.reason !== null) schemaError(`${path}.reason`, 'must be null when status is available')
    if (eligiblePixelCount === null || finiteProbabilityCount === null
      || finiteProbabilityCount < 1 || eligiblePixelCount < finiteProbabilityCount) {
      schemaError(`${path}.provenance`, 'must satisfy eligiblePixelCount >= finiteProbabilityCount >= 1')
    }
    return {
      value: availableValue,
      status,
      reason: null,
      provenance: { ...provenance, eligiblePixelCount, finiteProbabilityCount, aggregationMethod },
    }
  }

  if (!['unavailable', 'not_assessed', 'computation_failed', 'validation_failed'].includes(status)) {
    schemaError(`${path}.status`, 'is not a supported producer status')
  }
  if (probability.value !== null) schemaError(`${path}.value`, 'must be null when status is not available')
  const reason = requireString(probability.reason, `${path}.reason`)
  if (!(reason in FACILITY_REASON_STATUS)
    || FACILITY_REASON_STATUS[reason as FacilityUnavailableReason] !== status) {
    schemaError(`${path}.reason`, `is not valid for status ${status}`)
  }

  if (reason === 'no_eligible_pixels_in_facility_type_buffer'
    && (eligiblePixelCount !== 0 || finiteProbabilityCount !== 0)) {
    schemaError(`${path}.provenance`, 'must use zero counts when no eligible pixels exist')
  }
  if (reason === 'all_eligible_probabilities_missing'
    && (eligiblePixelCount === null || eligiblePixelCount < 1 || finiteProbabilityCount !== 0)) {
    schemaError(`${path}.provenance`, 'must record eligible pixels and zero finite probabilities')
  }
  if (reason === 'source_probability_pixels_unavailable'
    || reason === 'facility_outside_assessment_scope'
    || reason === 'required_facility_metadata_missing') {
    if (eligiblePixelCount !== null || finiteProbabilityCount !== null) {
      schemaError(`${path}.provenance`, 'must use null counts for an unassessed or unavailable source')
    }
  }
  if (status === 'computation_failed') {
    requireString(provenance.failureStage, `${path}.provenance.failureStage`)
  }
  if (status === 'validation_failed') {
    const validationErrors = requireArray(provenance.validationErrors, `${path}.provenance.validationErrors`)
    if (validationErrors.length === 0 || !validationErrors.every(error => typeof error === 'string')) {
      schemaError(`${path}.provenance.validationErrors`, 'must contain at least one validation error')
    }
  }

  return {
    value: null,
    status,
    reason,
    provenance: { ...provenance, eligiblePixelCount, finiteProbabilityCount, aggregationMethod },
  }
}

function parseFacilityRecordV1(value: unknown, path: string): FacilityRecordV1 {
  const record = requireRecord(value, path)
  const coordinates = requireArray(record.coordinates, `${path}.coordinates`)
  if (coordinates.length !== 2) schemaError(`${path}.coordinates`, 'must contain longitude and latitude')
  return {
    facilityId: requireString(record.facilityId, `${path}.facilityId`),
    name: requireString(record.name, `${path}.name`),
    type: requireString(record.type, `${path}.type`),
    coordinates: [
      requireFiniteNumber(coordinates[0], `${path}.coordinates[0]`),
      requireFiniteNumber(coordinates[1], `${path}.coordinates[1]`),
    ],
    radiusM: requireFiniteNumber(record.radiusM, `${path}.radiusM`),
    probability: parseFacilityProbabilityV1(record.probability, `${path}.probability`),
  }
}

export function parseFacilityArtifact(
  value: unknown,
): ValidatedArtifact<LegacyFacilityRow[] | FacilityArtifactV1> {
  if (Array.isArray(value)) return parseLegacyFacilityArtifact(value)

  const artifact = requireRecord(value, 'facility artifact')
  if (artifact.schemaVersion !== '1.0.0') {
    throw new UnsupportedArtifactVersionError(artifact.schemaVersion)
  }
  if (artifact.artifactType !== 'nightlight-facility-probabilities') {
    schemaError('facility artifact.artifactType', 'must identify nightlight facility probabilities')
  }
  const source = requireRecord(artifact.source, 'facility artifact.source')
  for (const field of [
    'producer',
    'producerVersion',
    'producerReceipt',
    'model',
    'modelVersion',
    'modelReceipt',
    'inputArtifact',
    'inputVersion',
    'inputReceipt',
  ]) {
    if (field.endsWith('Version')) requireVersion(source[field], `facility artifact.source.${field}`)
    else requireString(source[field], `facility artifact.source.${field}`)
  }
  const provenance = requireRecord(artifact.provenance, 'facility artifact.provenance')
  for (const field of [
    'generatedAtUtc',
    'eventId',
    'facilityCatalogVersion',
    'facilityCatalogReceipt',
    'facilityTypeMatchRule',
    'bufferRuleVersion',
    'aggregationMethod',
  ]) {
    if (field.endsWith('Version')) requireVersion(provenance[field], `facility artifact.provenance.${field}`)
    else requireString(provenance[field], `facility artifact.provenance.${field}`)
  }
  const records = requireArray(artifact.records, 'facility artifact.records')
    .map((record, index) => parseFacilityRecordV1(record, `facility artifact.records[${index}]`))

  return {
    wireVersion: '1',
    data: {
      schemaVersion: '1.0.0',
      artifactType: 'nightlight-facility-probabilities',
      source,
      provenance,
      records,
    },
    limitations: [],
  }
}

export function parseLegacyTimeSeriesArtifact(value: unknown): ValidatedArtifact<LegacyTimeSeriesRow[]> {
  const rows = requireArray(value, 'time-series artifact').map((entry, index) => {
    const path = `time-series artifact[${index}]`
    const row = requireRecord(entry, path)
    const pairedNull = row.R_buffer === null && row.R_nonBuffer === null
    const pairedObserved = typeof row.R_buffer === 'number' && Number.isFinite(row.R_buffer)
      && typeof row.R_nonBuffer === 'number' && Number.isFinite(row.R_nonBuffer)
    if (!pairedNull && !pairedObserved) schemaError(path, 'must contain paired finite observations or paired nulls')
    if (typeof row.isPostDisaster !== 'boolean') schemaError(`${path}.isPostDisaster`, 'must be a boolean')
    return {
      day: requireFiniteNumber(row.day, `${path}.day`),
      R_buffer: row.R_buffer as number | null,
      R_nonBuffer: row.R_nonBuffer as number | null,
      isPostDisaster: row.isPostDisaster,
    }
  })

  if (rows.length === 0) schemaError('time-series artifact', 'must not be empty')
  return { wireVersion: 'legacy-v0', data: rows, limitations: LEGACY_TIME_SERIES_LIMITATIONS }
}
