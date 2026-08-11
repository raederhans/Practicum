import { readFile, readdir } from 'node:fs/promises'
import { resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

import { ArtifactSchemaError, UnsupportedArtifactVersionError } from '../errors'
import {
  parseFacilityArtifact,
  parseLegacyFacilityArtifact,
  parseLegacyTimeSeriesArtifact,
  parseResultsSummaryArtifact,
} from './legacyArtifacts'

const FACILITY_V1_FIXTURE = {
  schemaVersion: '1.0.0',
  artifactType: 'nightlight-facility-probabilities',
  source: {
    producer: 'facility-probability-exporter',
    producerVersion: '1.0.0',
    producerReceipt: 'producer-receipt-001',
    model: 'nightlight-model-d',
    modelVersion: 'model-d-2026-08-11',
    modelReceipt: 'model-receipt-001',
    inputArtifact: 'pixel-probability-map',
    inputVersion: 'pixel-map-2026-08-11',
    inputReceipt: 'input-receipt-001',
  },
  provenance: {
    generatedAtUtc: '2026-08-11T12:00:00Z',
    eventId: 'uri-houston',
    facilityCatalogVersion: 'facility-catalog-2026-08-11',
    facilityCatalogReceipt: 'catalog-receipt-001',
    facilityTypeMatchRule: 'exact-normalized-v1',
    bufferRuleVersion: 'buffer-rule-v1',
    aggregationMethod: 'finite-pixel-mean-v1',
  },
  records: [{
    facilityId: 'hospital-001',
    name: 'Hospital',
    type: 'hospital',
    coordinates: [-66.1, 18.4],
    radiusM: 1000,
    probability: {
      value: 0.5,
      status: 'available',
      reason: null,
      provenance: {
        eligiblePixelCount: 8,
        finiteProbabilityCount: 6,
        aggregationMethod: 'finite-pixel-mean-v1',
      },
    },
  }],
}

const RESULTS_FIXTURE = {
  feature_importance: [{ feature: 'log_dist', rf_imp: 0.54, xgb_imp: 0.24, avg_imp: 0.39 }],
  model_comparison: {
    model_a: { name: 'Model A', mean_auc: 0.967, std: 0.023, f1: 0.83 },
  },
  loeo_by_model: {
    A: [{ held_out: 'Maria_SanJuan', rf_auc: 0.945, xgb_auc: 0.949, logit_auc: 0.914 }],
  },
  prob_stats: {
    maria: { n: 1260, min: 0.05, max: 0.957, mean: 0.456, p25: 0.178, p50: 0.414, p75: 0.659, above_50: 517, above_80: 260 },
  },
}

describe('legacy artifact conformance', () => {
  it('admits the unversioned results shape only as legacy-v0', () => {
    const result = parseResultsSummaryArtifact(RESULTS_FIXTURE)

    expect(result.wireVersion).toBe('legacy-v0')
    expect(result.limitations).toContain('unversioned-results-summary')
  })

  it('admits an explicit version 1 envelope', () => {
    const result = parseResultsSummaryArtifact({ schemaVersion: 1, data: RESULTS_FIXTURE })

    expect(result.wireVersion).toBe('1')
    expect(result.limitations).toEqual([])
  })

  it('fails closed for unsupported versioned results', () => {
    expect(() => parseResultsSummaryArtifact({ schemaVersion: 2, data: RESULTS_FIXTURE }))
      .toThrow(UnsupportedArtifactVersionError)
  })

  it('rejects results with an invalid nested schema', () => {
    expect(() => parseResultsSummaryArtifact({
      ...RESULTS_FIXTURE,
      prob_stats: { maria: { ...RESULTS_FIXTURE.prob_stats.maria, mean: null } },
    })).toThrow(ArtifactSchemaError)
  })

  it('does not promote a legacy facility 0.5 fallback to scientific availability', () => {
    const result = parseLegacyFacilityArtifact([{
      name: 'Hospital',
      type: 'hospital',
      coords: [-66.1, 18.4],
      probability: 0.5,
      radiusM: 1000,
    }])

    expect(result.data[0].probability).toBe(0.5)
    expect(result.limitations).toContain(
      'probability-0.5-may-be-an-exporter-fallback-without-missingness-provenance',
    )
  })

  it('routes an explicit producer v1 artifact without wrapping legacy rows', () => {
    const result = parseFacilityArtifact(FACILITY_V1_FIXTURE)

    expect(result.wireVersion).toBe('1')
    expect(result.limitations).toEqual([])
    expect(result.data).toMatchObject({ schemaVersion: '1.0.0' })
  })

  it('admits a genuine available producer value of 0.5 with finite lineage counts', () => {
    const result = parseFacilityArtifact(FACILITY_V1_FIXTURE)

    expect('records' in result.data && result.data.records[0].probability).toMatchObject({
      value: 0.5,
      status: 'available',
      reason: null,
    })
  })

  it('admits controlled null unavailability without inventing a numeric fallback', () => {
    const unavailable = {
      ...FACILITY_V1_FIXTURE,
      records: [{
        ...FACILITY_V1_FIXTURE.records[0],
        probability: {
          value: null,
          status: 'unavailable',
          reason: 'no_eligible_pixels_in_facility_type_buffer',
          provenance: {
            eligiblePixelCount: 0,
            finiteProbabilityCount: 0,
            aggregationMethod: 'finite-pixel-mean-v1',
          },
        },
      }],
    }

    const result = parseFacilityArtifact(unavailable)
    expect('records' in result.data && result.data.records[0].probability.value).toBeNull()
  })

  it('rejects a numeric fallback when producer status is unavailable', () => {
    const invalid = structuredClone(FACILITY_V1_FIXTURE)
    Object.assign(invalid.records[0].probability, {
      value: 0.5,
      status: 'unavailable',
      reason: 'no_eligible_pixels_in_facility_type_buffer',
      provenance: {
        eligiblePixelCount: 0,
        finiteProbabilityCount: 0,
        aggregationMethod: 'finite-pixel-mean-v1',
      },
    })

    expect(() => parseFacilityArtifact(invalid)).toThrow(ArtifactSchemaError)
  })

  it('rejects available producer values with inconsistent pixel counts', () => {
    const invalid = structuredClone(FACILITY_V1_FIXTURE)
    invalid.records[0].probability.provenance = {
      eligiblePixelCount: 2,
      finiteProbabilityCount: 3,
      aggregationMethod: 'finite-pixel-mean-v1',
    }

    expect(() => parseFacilityArtifact(invalid)).toThrow(ArtifactSchemaError)
  })

  it('rejects unsupported producer versions instead of guessing a migration', () => {
    expect(() => parseFacilityArtifact({ ...FACILITY_V1_FIXTURE, schemaVersion: '2.0.0' }))
      .toThrow(UnsupportedArtifactVersionError)
  })

  it('keeps paired null observations distinct from numeric zero', () => {
    const result = parseLegacyTimeSeriesArtifact([
      { day: 0, R_buffer: null, R_nonBuffer: null, isPostDisaster: true },
      { day: 1, R_buffer: 0, R_nonBuffer: 0, isPostDisaster: true },
    ])

    expect(result.data[0]).toMatchObject({ R_buffer: null, R_nonBuffer: null })
    expect(result.data[1]).toMatchObject({ R_buffer: 0, R_nonBuffer: 0 })
  })

  it('rejects unpaired time-series missingness', () => {
    expect(() => parseLegacyTimeSeriesArtifact([
      { day: 0, R_buffer: null, R_nonBuffer: 0.5, isPostDisaster: true },
    ])).toThrow(ArtifactSchemaError)
  })

  it('conforms the tracked results, facility, and time-series artifacts', async () => {
    const dataDirectory = resolve('public/data')
    const filenames = await readdir(dataDirectory)
    const readJson = async (filename: string): Promise<unknown> => (
      JSON.parse(await readFile(resolve(dataDirectory, filename), 'utf8'))
    )

    expect(parseResultsSummaryArtifact(await readJson('results_summary.json')).wireVersion)
      .toBe('legacy-v0')

    const facilityFiles = filenames.filter(filename => filename.startsWith('facilities_'))
    const timeSeriesFiles = filenames.filter(filename => filename.startsWith('ts_'))
    expect(facilityFiles).toHaveLength(25)
    expect(timeSeriesFiles).toHaveLength(25)

    for (const filename of facilityFiles) {
      expect(parseLegacyFacilityArtifact(await readJson(filename)).wireVersion).toBe('legacy-v0')
    }
    for (const filename of timeSeriesFiles) {
      expect(parseLegacyTimeSeriesArtifact(await readJson(filename)).wireVersion).toBe('legacy-v0')
    }
  })
})
