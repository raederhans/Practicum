import {
  LOCAL_ANALYTICS_CONSENT_VERSION,
  LOCAL_ANALYTICS_MAX_EVENTS,
  LOCAL_ANALYTICS_RESEARCH_QUESTIONS,
  LOCAL_ANALYTICS_SCHEMA_VERSION,
  createLocalAnalyticsEvent,
  isValidLocalAnalyticsEvent,
} from '../domain/localAnalyticsContract.js'

const STORAGE_KEY = 'nightlight.local-research-log.v1'

function browserSessionStorage() {
  try {
    return typeof window === 'undefined' ? null : window.sessionStorage
  } catch {
    return null
  }
}

function emptyState() {
  return {
    schemaVersion: LOCAL_ANALYTICS_SCHEMA_VERSION,
    consent: { state: 'not_granted', version: LOCAL_ANALYTICS_CONSENT_VERSION },
    events: [],
  }
}

function grantedState(events = []) {
  return {
    schemaVersion: LOCAL_ANALYTICS_SCHEMA_VERSION,
    consent: { state: 'granted', version: LOCAL_ANALYTICS_CONSENT_VERSION },
    events,
  }
}

function isValidStoredState(value) {
  if (
    value?.schemaVersion !== LOCAL_ANALYTICS_SCHEMA_VERSION
    || value?.consent?.state !== 'granted'
    || value?.consent?.version !== LOCAL_ANALYTICS_CONSENT_VERSION
    || !Array.isArray(value?.events)
    || value.events.length > LOCAL_ANALYTICS_MAX_EVENTS
  ) return false

  return value.events.every((event, index) => (
    event.ordinal === index + 1 && isValidLocalAnalyticsEvent(event)
  ))
}

export function createLocalResearchLog({ storage = browserSessionStorage() } = {}) {
  let memoryState = emptyState()
  const listeners = new Set()

  function removeStoredState() {
    try {
      storage?.removeItem(STORAGE_KEY)
    } catch {
      // Storage denial keeps the research log in memory for this page only.
    }
  }

  function readState() {
    let parsed = null
    try {
      const stored = storage?.getItem(STORAGE_KEY)
      parsed = stored ? JSON.parse(stored) : null
    } catch {
      parsed = null
    }

    if (isValidStoredState(parsed)) {
      memoryState = grantedState(parsed.events.map((event) => ({ ...event, properties: { ...event.properties } })))
      return memoryState
    }
    if (parsed !== null) removeStoredState()
    return memoryState
  }

  function writeState(nextState) {
    memoryState = nextState
    try {
      storage?.setItem(STORAGE_KEY, JSON.stringify(nextState))
    } catch {
      // The in-memory state remains usable and still leaves the browser process only on export.
    }
    for (const listener of listeners) listener(snapshot())
  }

  function snapshot() {
    const state = readState()
    return Object.freeze({
      consent: state.consent.state,
      consentVersion: LOCAL_ANALYTICS_CONSENT_VERSION,
      count: state.events.length,
      events: Object.freeze(state.events.map((event) => Object.freeze({
        ...event,
        properties: Object.freeze({ ...event.properties }),
      }))),
    })
  }

  function grantConsent() {
    const current = readState()
    if (current.consent.state === 'granted') return snapshot()
    writeState(grantedState())
    return snapshot()
  }

  function withdrawConsent() {
    memoryState = emptyState()
    removeStoredState()
    for (const listener of listeners) listener(snapshot())
    return snapshot()
  }

  function clearEvents() {
    const current = readState()
    if (current.consent.state !== 'granted') return snapshot()
    writeState(grantedState())
    return snapshot()
  }

  function record(name, properties) {
    const current = readState()
    if (current.consent.state !== 'granted' || current.events.length >= LOCAL_ANALYTICS_MAX_EVENTS) return false
    const event = createLocalAnalyticsEvent(name, properties, current.events.length + 1)
    writeState(grantedState([...current.events, event]))
    return true
  }

  function recordSurfaceViewed(surface) {
    return record('surface_viewed', { surface })
  }

  function recordAtlasModeSelected(mode) {
    return record('atlas_mode_selected', { mode })
  }

  function exportSnapshot() {
    const current = readState()
    if (current.consent.state !== 'granted') return null
    return {
      schemaVersion: LOCAL_ANALYTICS_SCHEMA_VERSION,
      consentVersion: LOCAL_ANALYTICS_CONSENT_VERSION,
      localOnly: true,
      retention: 'current-tab-until-clear-or-close',
      researchQuestions: LOCAL_ANALYTICS_RESEARCH_QUESTIONS.map(({ id, question }) => ({ id, question })),
      events: current.events.map((event) => ({ ...event, properties: { ...event.properties } })),
    }
  }

  function subscribe(listener) {
    listeners.add(listener)
    return () => listeners.delete(listener)
  }

  return Object.freeze({
    snapshot,
    grantConsent,
    withdrawConsent,
    clearEvents,
    recordSurfaceViewed,
    recordAtlasModeSelected,
    exportSnapshot,
    subscribe,
  })
}

export const localResearchLog = createLocalResearchLog()
