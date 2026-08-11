export const PUBLIC_SOURCE_VALUE_SCHEMA_VERSION = 1
export const PUBLIC_STALE_SNAPSHOT_MAX_AGE_DAYS = 30

export const PUBLIC_SOURCE_VALUE_STATUSES = Object.freeze([
  'available',
  'stale',
  'unavailable',
  'offline',
  'rate_limited',
  'auth_required',
  'source_failure',
  'validation_failure',
])

const STATUSES = new Set(PUBLIC_SOURCE_VALUE_STATUSES)
const VALUE_STATUSES = new Set(['available', 'stale'])
const NULL_STATUSES = new Set(PUBLIC_SOURCE_VALUE_STATUSES.filter((status) => !VALUE_STATUSES.has(status)))
const SOURCE_FIELDS = new Set(['id', 'version', 'effectiveDate', 'retrievedAt', 'validatedAt'])
const AVAILABLE_FIELDS = new Set(['schemaVersion', 'status', 'source', 'value'])
const REASONED_FIELDS = new Set(['schemaVersion', 'status', 'source', 'value', 'reasonCode'])
const STALE_REASON_CODES = new Set(['offline', 'rate-limited', 'auth-required', 'source-failure'])
const REASON_CODES_BY_STATUS = Object.freeze({
  unavailable: new Set(['snapshot-not-published', 'source-not-assessed', 'source-not-applicable']),
  offline: new Set(['network-offline']),
  rate_limited: new Set(['upstream-rate-limited']),
  auth_required: new Set(['upstream-auth-required']),
  source_failure: new Set(['upstream-source-failure']),
  validation_failure: new Set(['snapshot-validation-failure']),
})

function unknownFields(record, allowedFields) {
  return Object.keys(record).filter((field) => !allowedFields.has(field))
}

function isIsoDate(value) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false
  const instant = Date.parse(`${value}T00:00:00.000Z`)
  return Number.isFinite(instant) && new Date(instant).toISOString().slice(0, 10) === value
}

function isIsoInstant(value) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$/.test(value)) return false
  const instant = Date.parse(value)
  if (!Number.isFinite(instant)) return false
  const normalized = value.includes('.') ? value : value.replace('Z', '.000Z')
  return new Date(instant).toISOString() === normalized
}

function hasCompleteSnapshotSource(source) {
  return typeof source?.version === 'string'
    && /^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/.test(source.version)
    && isIsoDate(source.effectiveDate)
    && isIsoInstant(source.retrievedAt)
    && isIsoInstant(source.validatedAt)
}

function snapshotAgeDays(source, asOf) {
  if (!isIsoInstant(asOf) || !isIsoDate(source?.effectiveDate)) return null
  return Math.floor((Date.parse(asOf) - Date.parse(`${source.effectiveDate}T00:00:00.000Z`)) / 86_400_000)
}

export function staleSnapshotDisplay(record, { asOf } = {}) {
  const ageDays = snapshotAgeDays(record?.source, asOf)
  const displayable = record?.status === 'stale'
    && Number.isFinite(record?.value)
    && hasCompleteSnapshotSource(record?.source)
    && STALE_REASON_CODES.has(record?.reasonCode)
    && Number.isInteger(ageDays)
    && ageDays >= 0
    && ageDays <= PUBLIC_STALE_SNAPSHOT_MAX_AGE_DAYS

  return Object.freeze({
    displayable,
    ageDays,
    maximumAgeDays: PUBLIC_STALE_SNAPSHOT_MAX_AGE_DAYS,
    label: displayable ? `Stale validated snapshot · as of ${record.source.effectiveDate}` : null,
  })
}

export function validatePublicSourceValue(record, { asOf } = {}) {
  if (!record || typeof record !== 'object' || Array.isArray(record)) {
    return ['source value must be an object']
  }

  const violations = []
  for (const field of ['schemaVersion', 'status', 'source', 'value']) {
    if (!Object.hasOwn(record, field)) violations.push(`source value must own the ${field} field`)
  }
  if (record.schemaVersion !== PUBLIC_SOURCE_VALUE_SCHEMA_VERSION) {
    violations.push('source value has an unsupported schema version')
  }
  if (!STATUSES.has(record.status)) {
    violations.push('source value has an unsupported status')
    return violations
  }
  if (!record.source || typeof record.source !== 'object' || Array.isArray(record.source)) {
    violations.push('source metadata must be an object')
    return violations
  }
  const unknownSourceFields = unknownFields(record.source, SOURCE_FIELDS)
  if (unknownSourceFields.length) violations.push(`source metadata has unexpected field: ${unknownSourceFields[0]}`)
  if (typeof record.source.id !== 'string' || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(record.source.id)) {
    violations.push('source metadata needs a stable kebab-case id')
  }

  if (VALUE_STATUSES.has(record.status)) {
    if (!Number.isFinite(record.value)) violations.push(`${record.status} source value must contain a finite number`)
    if (!hasCompleteSnapshotSource(record.source)) {
      violations.push(`${record.status} source value needs complete version/effective/retrieved/validated metadata`)
    } else if (Date.parse(record.source.validatedAt) < Date.parse(record.source.retrievedAt)) {
      violations.push('source validation time must not precede retrieval time')
    }
  } else {
    if (record.value !== null) violations.push(`${record.status} source value must contain null, never a numeric fallback`)
    for (const field of ['version', 'effectiveDate', 'retrievedAt', 'validatedAt']) {
      if (record.source[field] !== null) violations.push(`${record.status} source metadata ${field} must be null without a validated snapshot`)
    }
  }

  if (record.status === 'available') {
    const unexpected = unknownFields(record, AVAILABLE_FIELDS)
    if (unexpected.length) violations.push(`available source value has unexpected field: ${unexpected[0]}`)
  } else {
    if (typeof record.reasonCode !== 'string' || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(record.reasonCode)) {
      violations.push(`${record.status} source value needs a stable kebab-case reasonCode`)
    }
    const allowedReasons = record.status === 'stale' ? STALE_REASON_CODES : REASON_CODES_BY_STATUS[record.status]
    if (allowedReasons && !allowedReasons.has(record.reasonCode)) {
      violations.push(`${record.status} source value has an unsupported reasonCode`)
    }
    const unexpected = unknownFields(record, REASONED_FIELDS)
    if (unexpected.length) violations.push(`${record.status} source value has unexpected field: ${unexpected[0]}`)
  }

  if (record.status === 'stale' && !staleSnapshotDisplay(record, { asOf }).displayable) {
    violations.push(`stale snapshot is not displayable within the ${PUBLIC_STALE_SNAPSHOT_MAX_AGE_DAYS}-day maximum age`)
  }

  return violations
}

export function assertPublicSourceValue(record, options) {
  const violations = validatePublicSourceValue(record, options)
  if (violations.length) throw new TypeError(violations.join('; '))
  return Object.freeze({ ...record, source: Object.freeze({ ...record.source }) })
}

export function unavailableSourceValue(sourceId, status = 'unavailable', reasonCode = 'snapshot-not-published') {
  return assertPublicSourceValue({
    schemaVersion: PUBLIC_SOURCE_VALUE_SCHEMA_VERSION,
    status,
    source: {
      id: sourceId,
      version: null,
      effectiveDate: null,
      retrievedAt: null,
      validatedAt: null,
    },
    value: null,
    reasonCode,
  })
}
