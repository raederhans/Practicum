import { resolveComparisonPeerId } from './compareEvents.js'
import { filterEvents } from './filterEvents.js'
import { resolveSelectedId } from './resolveSelectedId.js'

export const ATLAS_VIEW_MODES = Object.freeze(['explore', 'compare'])
export const ATLAS_MOBILE_VIEWS = Object.freeze(['list', 'map'])
export const ATLAS_SEARCH_MAX_LENGTH = 80

function scalarQueryValue(value) {
  return typeof value === 'string' ? value : ''
}

function validOption(value, options, fallback) {
  return options.includes(value) ? value : fallback
}

function normalizedSearch(value) {
  return scalarQueryValue(value).slice(0, ATLAS_SEARCH_MAX_LENGTH).trim()
}

export function createAtlasRouteStateCodec({ events, hazardFamilies, presets }) {
  if (!Array.isArray(events) || events.length < 2) throw new TypeError('Atlas route state requires at least two events')
  if (!Array.isArray(hazardFamilies) || !hazardFamilies.includes('All')) throw new TypeError('Atlas route state requires the All hazard option')
  if (!Array.isArray(presets)) throw new TypeError('Atlas route state requires a preset array')

  const eventIds = new Set(events.map(({ id }) => id))
  const presetById = new Map(presets.map((preset) => [preset.id, preset]))
  const defaultLeftId = events[0].id
  const defaultRightId = resolveComparisonPeerId(events, defaultLeftId, defaultLeftId)

  function validEventId(value) {
    return eventIds.has(value) ? value : null
  }

  function validHazardFamily(value) {
    return validOption(value, hazardFamilies, 'All')
  }

  function hydrate(routeQuery = {}) {
    const viewMode = validOption(scalarQueryValue(routeQuery.mode), ATLAS_VIEW_MODES, 'explore')
    const query = normalizedSearch(routeQuery.q)
    const selectedHazardFamily = validHazardFamily(scalarQueryValue(routeQuery.hazard))
    const atlasMobileView = validOption(scalarQueryValue(routeQuery.view), ATLAS_MOBILE_VIEWS, 'list')
    const visibleEvents = filterEvents(events, { hazardFamily: selectedHazardFamily, query })
    const selectedId = resolveSelectedId(visibleEvents, validEventId(scalarQueryValue(routeQuery.event)))

    const requestedPreset = presetById.get(scalarQueryValue(routeQuery.preset)) ?? null
    let comparisonLeftId = validEventId(scalarQueryValue(routeQuery.a))
    let comparisonRightId = validEventId(scalarQueryValue(routeQuery.b))
    if (requestedPreset && (!comparisonLeftId || !comparisonRightId)) {
      [comparisonLeftId, comparisonRightId] = requestedPreset.eventIds
    }
    comparisonLeftId ??= defaultLeftId
    comparisonRightId = resolveComparisonPeerId(events, comparisonLeftId, comparisonRightId)
    const selectedPresetId = requestedPreset
      && requestedPreset.eventIds[0] === comparisonLeftId
      && requestedPreset.eventIds[1] === comparisonRightId
      ? requestedPreset.id
      : null

    return Object.freeze({
      viewMode,
      query,
      selectedHazardFamily,
      selectedId,
      atlasMobileView,
      comparisonLeftId,
      comparisonRightId,
      selectedPresetId,
    })
  }

  function serialize(state) {
    const viewMode = validOption(state?.viewMode, ATLAS_VIEW_MODES, 'explore')
    const query = { mode: viewMode }
    if (viewMode === 'explore') {
      const search = normalizedSearch(state?.query)
      if (search) query.q = search
      const hazardFamily = validHazardFamily(state?.selectedHazardFamily)
      if (hazardFamily !== 'All') query.hazard = hazardFamily
      const selectedId = validEventId(state?.selectedId)
      if (selectedId) query.event = selectedId
      const mobileView = validOption(state?.atlasMobileView, ATLAS_MOBILE_VIEWS, 'list')
      if (mobileView !== 'list') query.view = mobileView
      return query
    }

    const leftId = validEventId(state?.comparisonLeftId) ?? defaultLeftId
    const rightId = resolveComparisonPeerId(events, leftId, validEventId(state?.comparisonRightId))
    query.a = leftId
    query.b = rightId
    const preset = presetById.get(state?.selectedPresetId)
    if (preset && preset.eventIds[0] === leftId && preset.eventIds[1] === rightId) query.preset = preset.id
    return query
  }

  function matches(expectedQuery, routeQuery = {}) {
    const routeKeys = Object.keys(routeQuery).sort()
    const expectedKeys = Object.keys(expectedQuery).sort()
    if (routeKeys.length !== expectedKeys.length) return false
    return expectedKeys.every((key, index) => (
      key === routeKeys[index] && scalarQueryValue(routeQuery[key]) === expectedQuery[key]
    ))
  }

  return Object.freeze({
    defaults: Object.freeze({ leftId: defaultLeftId, rightId: defaultRightId }),
    hydrate,
    serialize,
    matches,
    normalizeSearch: normalizedSearch,
    validEventId,
    validHazardFamily,
  })
}
