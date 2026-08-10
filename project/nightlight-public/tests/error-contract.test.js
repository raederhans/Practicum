import { describe, expect, it } from 'vitest'

import {
  PUBLIC_AGGREGATE_VALUE_SCHEMA_VERSION,
  PUBLIC_AGGREGATE_VALUE_STATUSES,
  assertPublicAggregateValue,
  validatePublicAggregateValue,
} from '../src/lib/aggregateValueContract.js'

const NON_AVAILABLE_STATUSES = PUBLIC_AGGREGATE_VALUE_STATUSES.filter((status) => status !== 'available')

describe('public aggregate value and error contract', () => {
  it('accepts numeric zero only as an explicitly available value', () => {
    const record = {
      schemaVersion: PUBLIC_AGGREGATE_VALUE_SCHEMA_VERSION,
      status: 'available',
      value: 0,
    }

    expect(validatePublicAggregateValue(record)).toEqual([])
    expect(assertPublicAggregateValue(record)).toEqual(record)
  })

  it.each(NON_AVAILABLE_STATUSES)('requires %s to carry null and a stable reason', (status) => {
    const record = {
      schemaVersion: PUBLIC_AGGREGATE_VALUE_SCHEMA_VERSION,
      status,
      value: null,
      reasonCode: `fixture-${status.replaceAll('_', '-')}`,
    }

    expect(validatePublicAggregateValue(record)).toEqual([])
  })

  it.each(NON_AVAILABLE_STATUSES)('rejects numeric fallback zero for %s', (status) => {
    const violations = validatePublicAggregateValue({
      schemaVersion: PUBLIC_AGGREGATE_VALUE_SCHEMA_VERSION,
      status,
      value: 0,
      reasonCode: `fixture-${status.replaceAll('_', '-')}`,
    })

    expect(violations.join('\n')).toMatch(/null, never a numeric fallback/i)
  })

  it.each([
    [{ schemaVersion: 1, status: 'available', value: null }, /finite number/i],
    [{ schemaVersion: 1, status: 'available', value: Number.NaN }, /finite number/i],
    [{ schemaVersion: 1, status: 'available', value: Number.POSITIVE_INFINITY }, /finite number/i],
    [{ schemaVersion: 1, status: 'not_assessed', value: null }, /reasonCode/i],
    [{ schemaVersion: 1, status: 'missing', value: null, reasonCode: 'missing' }, /unsupported status/i],
    [{ schemaVersion: 2, status: 'available', value: 1 }, /schema version/i],
    [{ schemaVersion: 1, status: 'available', value: 1, fallback: 0 }, /unexpected field/i],
  ])('fails closed on invalid fixture %#', (record, expected) => {
    expect(validatePublicAggregateValue(record).join('\n')).toMatch(expected)
    expect(() => assertPublicAggregateValue(record)).toThrow(TypeError)
  })

  it('rejects prototype-inherited status and value fields', () => {
    const record = Object.create({
      schemaVersion: PUBLIC_AGGREGATE_VALUE_SCHEMA_VERSION,
      status: 'available',
      value: 0,
    })

    expect(validatePublicAggregateValue(record).join('\n')).toMatch(/must own.*(?:schemaVersion|status|value)/i)
    expect(() => assertPublicAggregateValue(record)).toThrow(TypeError)
  })
})
