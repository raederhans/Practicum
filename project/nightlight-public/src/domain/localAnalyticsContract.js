export const LOCAL_ANALYTICS_SCHEMA_VERSION = 1
export const LOCAL_ANALYTICS_CONSENT_VERSION = '2026-08-11.v1'
export const LOCAL_ANALYTICS_MAX_EVENTS = 100

export const LOCAL_ANALYTICS_RESEARCH_QUESTIONS = Object.freeze([
  Object.freeze({
    id: 'rq-surface-navigation',
    question: 'Which fixed public evidence surfaces are opened during an explicitly opted-in local research session?',
  }),
  Object.freeze({
    id: 'rq-atlas-mode',
    question: 'Does an explicitly opted-in Atlas research session use Explore or Compare?',
  }),
])

const SURFACES = Object.freeze(['overview', 'atlas', 'findings', 'methods', 'credits'])
const ATLAS_MODES = Object.freeze(['explore', 'compare'])

export const LOCAL_ANALYTICS_EVENT_SCHEMAS = Object.freeze({
  surface_viewed: Object.freeze({
    researchQuestionId: 'rq-surface-navigation',
    properties: Object.freeze({ surface: SURFACES }),
  }),
  atlas_mode_selected: Object.freeze({
    researchQuestionId: 'rq-atlas-mode',
    properties: Object.freeze({ mode: ATLAS_MODES }),
  }),
})

function hasExactKeys(value, expectedKeys) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const actual = Object.keys(value).sort()
  const expected = [...expectedKeys].sort()
  return JSON.stringify(actual) === JSON.stringify(expected)
}

export function createLocalAnalyticsEvent(name, properties, ordinal) {
  const schema = LOCAL_ANALYTICS_EVENT_SCHEMAS[name]
  if (!schema) throw new TypeError(`Local analytics event is not allowlisted: ${String(name)}`)
  if (!Number.isInteger(ordinal) || ordinal < 1 || ordinal > LOCAL_ANALYTICS_MAX_EVENTS) {
    throw new TypeError('Local analytics ordinal is outside the bounded session range')
  }
  if (!hasExactKeys(properties, Object.keys(schema.properties))) {
    throw new TypeError(`Local analytics properties do not match the ${name} schema`)
  }

  for (const [property, allowedValues] of Object.entries(schema.properties)) {
    if (!allowedValues.includes(properties[property])) {
      throw new TypeError(`Local analytics property ${property} is not allowlisted for ${name}`)
    }
  }

  return Object.freeze({
    ordinal,
    name,
    researchQuestionId: schema.researchQuestionId,
    properties: Object.freeze({ ...properties }),
  })
}

export function isValidLocalAnalyticsEvent(event) {
  try {
    createLocalAnalyticsEvent(event?.name, event?.properties, event?.ordinal)
    return event.researchQuestionId === LOCAL_ANALYTICS_EVENT_SCHEMAS[event.name].researchQuestionId
      && hasExactKeys(event, ['ordinal', 'name', 'researchQuestionId', 'properties'])
  } catch {
    return false
  }
}
