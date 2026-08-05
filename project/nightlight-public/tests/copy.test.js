import { describe, expect, it } from 'vitest'

import { DATA_BOUNDARY, FINDINGS_COPY } from '../src/content/copy.js'

describe('public interpretation language', () => {
  it('labels 0.551 as descriptive sensitivity only', () => {
    expect(FINDINGS_COPY.sensitivity.value).toBe(0.551)
    expect(FINDINGS_COPY.sensitivity.label).toMatch(/descriptive sensitivity/i)
    expect(FINDINGS_COPY.sensitivity.caution).toMatch(/not causal/i)
    expect(FINDINGS_COPY.sensitivity.caution).toMatch(/not.*fairness/i)
  })

  it('states the non-public data boundary without implying missing-data fallbacks', () => {
    expect(DATA_BOUNDARY.status).toMatch(/not published/i)
    expect(DATA_BOUNDARY.excluded).toEqual([
      'raw outage records',
      'time-series extracts',
      'facility locations',
      'pixel-level surfaces',
      'reversible fine-grained tables',
    ])
    expect(DATA_BOUNDARY.status).not.toMatch(/demo|mock|fallback/i)
  })
})
