import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  DataLoadError,
  loadProbabilityGeoJSON,
  loadTimeSeries,
  validateTimeSeries,
} from './loader.js'


const EVENT = { id: 'missing-event', name: 'Missing Event' }


afterEach(() => {
  vi.unstubAllGlobals()
})


describe('dashboard data loader', () => {
  it('rejects a missing export instead of generating replacement data', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 404 }))

    await expect(loadProbabilityGeoJSON(EVENT)).rejects.toMatchObject({
      name: 'DataLoadError',
      status: 404,
      eventId: EVENT.id,
    })
  })

  it('rejects malformed probability GeoJSON', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ type: 'FeatureCollection', features: 'not-an-array' }),
    }))

    await expect(loadProbabilityGeoJSON(EVENT)).rejects.toBeInstanceOf(DataLoadError)
  })

  it.each([
    { type: 'FeatureCollection', features: [] },
    {
      type: 'FeatureCollection',
      features: [{
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [0, 0] },
        properties: { probability: Number.NaN },
      }],
    },
    {
      type: 'FeatureCollection',
      features: [{
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [0, 0] },
        properties: { probability: Number.POSITIVE_INFINITY },
      }],
    },
    {
      type: 'FeatureCollection',
      features: [{
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [0, 0] },
        properties: { probability: -0.01 },
      }],
    },
    {
      type: 'FeatureCollection',
      features: [{
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [0, 0] },
        properties: { probability: 1.01 },
      }],
    },
  ])('rejects unusable probability GeoJSON', async geojson => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => geojson,
    }))

    await expect(loadProbabilityGeoJSON(EVENT)).rejects.toThrow('probability export')
  })

  it('rejects malformed recovery series', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ dates: [], buffer: [], non_buffer: [] }),
    }))

    await expect(loadTimeSeries(EVENT)).rejects.toThrow('recovery time series')
  })

  it('accepts a paired null as a real missing observation', () => {
    const rows = [
      { day: -1, R_buffer: 1.02, R_nonBuffer: 0.98, isPostDisaster: false },
      { day: 0, R_buffer: null, R_nonBuffer: null, isPostDisaster: true },
      { day: 1, R_buffer: 0.72, R_nonBuffer: 0.65, isPostDisaster: true },
    ]

    expect(validateTimeSeries(rows)).toBe(rows)
  })

  it.each([
    [{ day: 0, R_buffer: null, R_nonBuffer: 0.5 }],
    [{ day: 0, R_buffer: 0.5, R_nonBuffer: null }],
    [{ day: 0, R_buffer: 'missing', R_nonBuffer: 'missing' }],
  ])('rejects unpaired nulls and nonnumeric observations', rows => {
    expect(() => validateTimeSeries(rows)).toThrow('recovery time series')
  })
})
