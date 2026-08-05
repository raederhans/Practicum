const BASE = import.meta.env.BASE_URL


export class DataLoadError extends Error {
  constructor(message, { path, eventId = null, status = null, cause = null } = {}) {
    super(message, cause ? { cause } : undefined)
    this.name = 'DataLoadError'
    this.path = path
    this.eventId = eventId
    this.status = status
  }
}


function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value)
}


export function validateFeatureCollection(value, label = 'GeoJSON') {
  if (value?.type !== 'FeatureCollection' || !Array.isArray(value.features)) {
    throw new Error(`${label} must be a GeoJSON FeatureCollection`)
  }
  if (!value.features.every(feature => feature?.type === 'Feature' && feature.geometry)) {
    throw new Error(`${label} contains an invalid feature`)
  }
  return value
}


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


async function loadJson(path, { eventId = null, validate = value => value } = {}) {
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
    throw new DataLoadError(`Data unavailable: ${path} has an invalid schema (${cause.message})`, {
      path,
      eventId,
      status: response.status,
      cause,
    })
  }
}


export function loadProbabilityGeoJSON(event) {
  return loadJson(`data/prob_${event.id}.geojson`, {
    eventId: event.id,
    validate: validateProbabilityFeatureCollection,
  })
}


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


export function loadTimeSeries(event) {
  return loadJson(`data/ts_${event.id}.json`, {
    eventId: event.id,
    validate: validateTimeSeries,
  })
}


export function loadLoeoResults() {
  return loadJson('data/loeo_results.json', { validate: validateLoeoRows })
}
