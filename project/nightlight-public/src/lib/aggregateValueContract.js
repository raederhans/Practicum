export const PUBLIC_AGGREGATE_VALUE_SCHEMA_VERSION = 1

export const PUBLIC_AGGREGATE_VALUE_STATUSES = Object.freeze([
  'available',
  'unavailable',
  'not_assessed',
  'not_applicable',
  'suppressed',
  'load_failure',
  'validation_failure',
])

const STATUSES = new Set(PUBLIC_AGGREGATE_VALUE_STATUSES)
const NON_AVAILABLE_STATUSES = new Set(PUBLIC_AGGREGATE_VALUE_STATUSES.filter((status) => status !== 'available'))
const AVAILABLE_FIELDS = new Set(['schemaVersion', 'status', 'value'])
const NON_AVAILABLE_FIELDS = new Set(['schemaVersion', 'status', 'value', 'reasonCode'])

function unknownFields(record, allowedFields) {
  return Object.keys(record).filter((field) => !allowedFields.has(field))
}

export function validatePublicAggregateValue(record) {
  if (!record || typeof record !== 'object' || Array.isArray(record)) {
    return ['aggregate value must be an object']
  }

  const violations = []
  for (const field of ['schemaVersion', 'status', 'value']) {
    if (!Object.hasOwn(record, field)) violations.push(`aggregate value must own the ${field} field`)
  }
  if (record.schemaVersion !== PUBLIC_AGGREGATE_VALUE_SCHEMA_VERSION) {
    violations.push('aggregate value has an unsupported schema version')
  }
  if (!STATUSES.has(record.status)) {
    violations.push('aggregate value has an unsupported status')
    return violations
  }

  if (record.status === 'available') {
    if (!Number.isFinite(record.value)) {
      violations.push('available aggregate value must contain a finite number')
    }
    const unexpected = unknownFields(record, AVAILABLE_FIELDS)
    if (unexpected.length) violations.push(`available aggregate value has unexpected field: ${unexpected[0]}`)
    return violations
  }

  if (NON_AVAILABLE_STATUSES.has(record.status)) {
    if (record.value !== null) {
      violations.push(`${record.status} aggregate value must contain null, never a numeric fallback`)
    }
    if (
      !Object.hasOwn(record, 'reasonCode')
      || typeof record.reasonCode !== 'string'
      || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(record.reasonCode)
    ) {
      violations.push(`${record.status} aggregate value needs a stable kebab-case reasonCode`)
    }
    const unexpected = unknownFields(record, NON_AVAILABLE_FIELDS)
    if (unexpected.length) violations.push(`${record.status} aggregate value has unexpected field: ${unexpected[0]}`)
  }

  return violations
}

export function assertPublicAggregateValue(record) {
  const violations = validatePublicAggregateValue(record)
  if (violations.length) throw new TypeError(violations.join('; '))
  return Object.freeze({ ...record })
}
