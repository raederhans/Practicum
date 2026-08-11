import {
  assertPublicSourceValue,
  unavailableSourceValue,
  validatePublicSourceValue,
} from './sourceFreshnessContract.js'

export function createBundledSourceAdapter(records) {
  if (!Array.isArray(records)) throw new TypeError('Bundled source adapter requires an array of reviewed records')
  const bySourceId = new Map()
  for (const record of records) {
    if (typeof record?.source?.id !== 'string' || bySourceId.has(record.source.id)) {
      throw new TypeError('Bundled source records require unique source ids')
    }
    bySourceId.set(record.source.id, record)
  }

  return Object.freeze({
    kind: 'bundled-static-snapshot',
    read(sourceId, { asOf } = {}) {
      const record = bySourceId.get(sourceId)
      if (!record) return unavailableSourceValue(sourceId)
      const violations = validatePublicSourceValue(record, { asOf })
      if (violations.length) {
        return unavailableSourceValue(sourceId, 'validation_failure', 'snapshot-validation-failure')
      }
      return assertPublicSourceValue(record, { asOf })
    },
  })
}
