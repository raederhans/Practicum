const BASE = import.meta.env.BASE_URL

/**
 * @typedef {{ id: string }} DashboardEvent
 * @typedef {{ type: 'Feature', geometry: object, properties?: Record<string, unknown> }} GeoJsonFeature
 * @typedef {{ type: 'FeatureCollection', features: GeoJsonFeature[] }} GeoJsonFeatureCollection
 * @typedef {{ name: string, type: string, coords: [number, number], probability: number, radiusM?: number }} FacilityRow
 * @typedef {{ day: number, R_buffer: number | null, R_nonBuffer: number | null, isPostDisaster?: boolean }} TimeSeriesRow
 * @typedef {{ held_out: string, rf_auc: number, xgb_auc: number, logit_auc: number }} LoeoRow
 */

export class DataLoadError extends Error {
  /**
   * @param {string} message
   * @param {{ path: string, eventId?: string | null, status?: number | null, cause?: unknown }} options
   */
  constructor(message, { path, eventId = null, status = null, cause = null }) {
    super(message, cause ? { cause } : undefined)
    this.name = 'DataLoadError'
    this.path = path
    this.eventId = eventId
    this.status = status
  }
}


/** @param {unknown} value @returns {value is number} */
function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value)
}


/**
 * @param {unknown} value
 * @param {string} [label]
 * @returns {GeoJsonFeatureCollection}
 */
export function validateFeatureCollection(value, label = 'GeoJSON') {
  if (!value || typeof value !== 'object' || !('type' in value) || !('features' in value)
    || value.type !== 'FeatureCollection' || !Array.isArray(value.features)) {
    throw new Error(`${label} must be a GeoJSON FeatureCollection`)
  }
  if (!value.features.every(feature => feature?.type === 'Feature' && feature.geometry)) {
    throw new Error(`${label} contains an invalid feature`)
  }
  return /** @type {GeoJsonFeatureCollection} */ (value)
}


/** @param {unknown} value @returns {GeoJsonFeatureCollection} */
export function validateProbabilityFeatureCollection(value) {
  const collection = validateFeatureCollection(value, 'probability export')
  if (collection.features.length === 0) {
    throw new Error('probability export must contain at least one feature')
  }
  if (!collection.features.every(feature => (
    isFiniteNumber(feature.properties?.probability)
    && feature.properties.probability >= 0
    && feature.properties.probability <= 1
  ))) {
    throw new Error('probability export contains an invalid probability')
  }
  return collection
}


/** @param {unknown} value @returns {FacilityRow[]} */
function validateFacilityRows(value) {
  if (!Array.isArray(value)) throw new Error('facility export must be an array')
  if (!value.every(row => (
    typeof row?.name === 'string'
    && typeof row?.type === 'string'
    && Array.isArray(row.coords)
    && row.coords.length === 2
    && row.coords.every(isFiniteNumber)
    && isFiniteNumber(row.probability)
  ))) {
    throw new Error('facility export contains an invalid row')
  }
  return value
}


/** @param {unknown} value @returns {TimeSeriesRow[]} */
export function validateTimeSeries(value) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error('recovery time series must be a non-empty array')
  }
  if (!value.every(row => {
    const pairedMissing = row?.R_buffer === null && row?.R_nonBuffer === null
    const pairedObserved = isFiniteNumber(row?.R_buffer) && isFiniteNumber(row?.R_nonBuffer)
    return isFiniteNumber(row?.day) && (pairedMissing || pairedObserved)
  })) {
    throw new Error('recovery time series contains an invalid row')
  }
  return value
}


/** @param {unknown} value @returns {LoeoRow[]} */
function validateLoeoRows(value) {
  if (!Array.isArray(value) || !value.every(row => (
    typeof row?.held_out === 'string'
    && isFiniteNumber(row?.rf_auc)
    && isFiniteNumber(row?.xgb_auc)
    && isFiniteNumber(row?.logit_auc)
  ))) {
    throw new Error('LOEO results contain an invalid row')
  }
  return value
}


/**
 * @template T
 * @param {string} path
 * @param {{ eventId?: string | null, validate: (value: unknown) => T }} options
 * @returns {Promise<T>}
 */
async function loadJson(path, { eventId = null, validate }) {
  let response
  try {
    response = await fetch(`${BASE}${path}`)
  } catch (cause) {
    throw new DataLoadError(`Data unavailable: could not fetch ${path}`, { path, eventId, cause })
  }

  if (!response.ok) {
    throw new DataLoadError(`Data unavailable: ${path} returned HTTP ${response.status}`, {
      path,
      eventId,
      status: response.status,
    })
  }

  try {
    return validate(await response.json())
  } catch (cause) {
    if (cause instanceof DataLoadError) throw cause
    const reason = cause instanceof Error ? cause.message : String(cause)
    throw new DataLoadError(`Data unavailable: ${path} has an invalid schema (${reason})`, {
      path,
      eventId,
      status: response.status,
      cause,
    })
  }
}


/** @param {DashboardEvent} event */
export function loadProbabilityGeoJSON(event) {
  return loadJson(`data/prob_${event.id}.geojson`, {
    eventId: event.id,
    validate: validateProbabilityFeatureCollection,
  })
}


/** @param {DashboardEvent} event */
export async function loadFacilityGeoJSON(event) {
  const facilities = await loadJson(`data/facilities_${event.id}.json`, {
    eventId: event.id,
    validate: validateFacilityRows,
  })
  return {
    type: 'FeatureCollection',
    features: facilities.map(facility => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: facility.coords },
      properties: {
        name: facility.name,
        type: facility.type,
        probability: facility.probability,
      },
    })),
  }
}


/** @param {DashboardEvent} event */
export function loadTimeSeries(event) {
  return loadJson(`data/ts_${event.id}.json`, {
    eventId: event.id,
    validate: validateTimeSeries,
  })
}


export function loadLoeoResults() {
  return loadJson('data/loeo_results.json', { validate: validateLoeoRows })
}
