import { describe, expect, it } from 'vitest'

import { createBundledSourceAdapter } from '../src/lib/bundledSourceAdapter.js'
import {
  PUBLIC_SOURCE_VALUE_SCHEMA_VERSION,
  PUBLIC_SOURCE_VALUE_STATUSES,
  PUBLIC_STALE_SNAPSHOT_MAX_AGE_DAYS,
  assertPublicSourceValue,
  staleSnapshotDisplay,
  unavailableSourceValue,
  validatePublicSourceValue,
} from '../src/lib/sourceFreshnessContract.js'

const SOURCE = Object.freeze({
  id: 'reviewed-example',
  version: '2026-08.v1',
  effectiveDate: '2026-08-01',
  retrievedAt: '2026-08-02T03:00:00Z',
  validatedAt: '2026-08-02T04:00:00Z',
})

function available(value = 0) {
  return {
    schemaVersion: PUBLIC_SOURCE_VALUE_SCHEMA_VERSION,
    status: 'available',
    source: { ...SOURCE },
    value,
  }
}

describe('versioned source freshness and error contract', () => {
  it('accepts numeric zero only when a complete validated snapshot is available', () => {
    expect(validatePublicSourceValue(available(0), { asOf: '2026-08-11T00:00:00Z' })).toEqual([])
    expect(assertPublicSourceValue(available(0), { asOf: '2026-08-11T00:00:00Z' })).toEqual(available(0))
  })

  it.each([
    ['unavailable', 'snapshot-not-published'],
    ['offline', 'network-offline'],
    ['rate_limited', 'upstream-rate-limited'],
    ['auth_required', 'upstream-auth-required'],
    ['source_failure', 'upstream-source-failure'],
    ['validation_failure', 'snapshot-validation-failure'],
  ])('requires %s to carry null source metadata, null value, and its admitted reason', (status, reasonCode) => {
    const record = unavailableSourceValue('reviewed-example', status, reasonCode)
    expect(validatePublicSourceValue(record)).toEqual([])
    expect(record.value).toBeNull()
    expect(Object.values(record.source).slice(1)).toEqual([null, null, null, null])
  })

  it.each(PUBLIC_SOURCE_VALUE_STATUSES.filter((status) => !['available', 'stale'].includes(status)))('rejects zero fallback for %s', (status) => {
    const reasonCode = {
      unavailable: 'snapshot-not-published',
      offline: 'network-offline',
      rate_limited: 'upstream-rate-limited',
      auth_required: 'upstream-auth-required',
      source_failure: 'upstream-source-failure',
      validation_failure: 'snapshot-validation-failure',
    }[status]
    const record = { ...unavailableSourceValue('reviewed-example', status, reasonCode), value: 0 }
    expect(validatePublicSourceValue(record).join('\n')).toMatch(/null, never a numeric fallback/i)
  })

  it('displays a stale validated snapshot only with an admitted cause and within the fixed maximum age', () => {
    const record = {
      ...available(12.5),
      status: 'stale',
      reasonCode: 'offline',
    }

    expect(staleSnapshotDisplay(record, { asOf: '2026-08-31T23:59:59Z' })).toEqual({
      displayable: true,
      ageDays: 30,
      maximumAgeDays: PUBLIC_STALE_SNAPSHOT_MAX_AGE_DAYS,
      label: 'Stale validated snapshot · as of 2026-08-01',
    })
    expect(validatePublicSourceValue(record, { asOf: '2026-08-31T23:59:59Z' })).toEqual([])

    const expired = staleSnapshotDisplay(record, { asOf: '2026-09-01T00:00:00Z' })
    expect(expired).toMatchObject({ displayable: false, ageDays: 31 })
    expect(validatePublicSourceValue(record, { asOf: '2026-09-01T00:00:00Z' }).join('\n')).toMatch(/30-day maximum age/i)
  })

  it.each([
    [{ ...available(1), source: { ...SOURCE, version: null } }, /complete version/i],
    [{ ...available(1), source: { ...SOURCE, version: 'free form version' } }, /complete version/i],
    [{ ...available(1), source: { ...SOURCE, effectiveDate: '2026-02-31' } }, /complete version/i],
    [{ ...available(1), source: { ...SOURCE, validatedAt: '2026-08-02T02:00:00Z' } }, /must not precede retrieval/i],
    [{ ...available(1), status: 'stale', reasonCode: 'validation-failure' }, /unsupported reasonCode|not displayable/i],
    [{ ...available(1), source: { ...SOURCE, extra: 'free-form' } }, /unexpected field/i],
    [{ ...available(1), fallback: 0 }, /unexpected field/i],
    [{ ...available(1), schemaVersion: 2 }, /unsupported schema version/i],
  ])('fails closed on invalid fixture %#', (record, expected) => {
    expect(validatePublicSourceValue(record, { asOf: '2026-08-11T00:00:00Z' }).join('\n')).toMatch(expected)
    expect(() => assertPublicSourceValue(record, { asOf: '2026-08-11T00:00:00Z' })).toThrow(TypeError)
  })
})

describe('bundled static source adapter seam', () => {
  it('returns validated bundled records and never performs runtime acquisition', () => {
    const adapter = createBundledSourceAdapter([available(4.2)])

    expect(adapter.kind).toBe('bundled-static-snapshot')
    expect(adapter.read('reviewed-example', { asOf: '2026-08-11T00:00:00Z' })).toEqual(available(4.2))
    expect(adapter.read('not-published')).toMatchObject({
      status: 'unavailable',
      value: null,
      reasonCode: 'snapshot-not-published',
    })
  })

  it('converts an invalid bundled record into validation_failure without exposing its value', () => {
    const adapter = createBundledSourceAdapter([{ ...available(9), source: { ...SOURCE, version: null } }])

    expect(adapter.read('reviewed-example', { asOf: '2026-08-11T00:00:00Z' })).toMatchObject({
      status: 'validation_failure',
      value: null,
      reasonCode: 'snapshot-validation-failure',
    })
  })

  it('rejects duplicate source ids instead of silently overriding a snapshot', () => {
    expect(() => createBundledSourceAdapter([available(1), available(2)])).toThrow(/unique source ids/i)
  })
})
