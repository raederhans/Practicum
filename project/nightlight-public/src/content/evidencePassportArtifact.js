import reviewedManifest from './evidencePassportManifest.json' with { type: 'json' }

const PUBLIC_MANIFEST_SHA256 = '14328f59563dbb268ee66e09b964696cdc0c7630bb32f289a749ec8fd2da1570'

const COMPONENT_DEFINITIONS = Object.freeze(
  reviewedManifest.componentDefinitions.map((definition) => Object.freeze({ ...definition })),
)

const BAND_DEFINITIONS = Object.freeze(
  reviewedManifest.bandDefinitions.map((definition) => Object.freeze({ ...definition })),
)

const REVIEWED_SOURCE = Object.freeze({
  id: reviewedManifest.source.id,
  version: reviewedManifest.source.version,
  sha256: PUBLIC_MANIFEST_SHA256,
  canonicalization: reviewedManifest.source.canonicalization,
  rights: reviewedManifest.source.rights,
  attribution: reviewedManifest.source.attribution,
  publicationStatus: reviewedManifest.source.publicationStatus,
})

const COMPARISON_BOUNDARY = Object.freeze({
  status: 'comparability-not-established',
  schemaScope: 'shared-v1-rule-schema-only',
  knownProductContext: 'The reviewed mixed-source attribution includes NASA Black Marble VNP46A2 Collection 2.',
  unknownConditions: Object.freeze([
    'per-event sensor and platform equivalence',
    'acquisition and pre/post window equivalence',
    'spatial-unit and population-denominator equivalence',
    'context and covariate source equivalence',
    'missingness mechanism and sampling-shift equivalence',
  ]),
  privateSourceVerification: 'restricted-environment-verified',
  statement: 'The public artifact supports descriptive v1 rule-bin pairing only; it does not establish cross-event measurement equivalence.',
})

const REVIEWED_PASSPORTS = new Map(
  reviewedManifest.passports.map((passport) => [passport.eventId, passport]),
)

const BAND_CLAIMS = Object.freeze({
  mainline_ready: 'The reviewed components meet the project’s current mainline analysis-admission threshold.',
  sensitivity_only: 'The reviewed components support sensitivity-only use under the current analysis-admission rule.',
  repair_first: 'The reviewed components indicate that repair is required before mainline analytical use.',
})

const UNSUPPORTED_CLAIM = 'This does not measure community recovery, disaster outcome, resilience, fairness, causality, or policy performance, and it is not an event ranking.'

const ARTIFACT_FIELDS = new Set(['version', 'generatedDate', 'title', 'source', 'comparisonBoundary', 'componentDefinitions', 'bandDefinitions', 'passports'])
const SOURCE_FIELDS = new Set(['id', 'version', 'sha256', 'canonicalization', 'rights', 'attribution', 'publicationStatus'])
const COMPARISON_BOUNDARY_FIELDS = new Set(['status', 'schemaScope', 'knownProductContext', 'unknownConditions', 'privateSourceVerification', 'statement'])
const COMPONENT_DEFINITION_FIELDS = new Set(['id', 'label', 'maxPoints', 'meaning'])
const BAND_DEFINITION_FIELDS = new Set(['id', 'label', 'meaning'])
const PASSPORT_FIELDS = new Set(['eventId', 'schemaVersion', 'readinessBand', 'readinessLabel', 'components', 'supportedClaim', 'unsupportedClaim', 'publicationStatus', 'sourceArtifactId'])
const COMPONENT_FIELDS = new Set(['id', 'points', 'maxPoints', 'status'])
const PROHIBITED_FIELD_NAMES = new Set([
  'eventcount', 'observedrate', 'highcensoringshare', 'poicount', 'totalscore',
  'incrementimpactlabel', 'recommendedrole', 'facility', 'facilities', 'coordinates',
  'probability', 'probabilities', 'raster', 'grid', 'grid_id', 'zip_code', 'zip_event',
  'time_series', 'timeseries', 'outage_duration', 'recovery_time', 'local_path',
])

function readinessLabel(band) {
  return BAND_DEFINITIONS.find(({ id }) => id === band)?.label ?? ''
}

function componentStatus(points, maxPoints) {
  if (points === maxPoints) return 'available'
  if (points === 0) return 'unavailable'
  return 'limited'
}

function makePassport(eventId) {
  const reviewedPassport = REVIEWED_PASSPORTS.get(eventId)
  const readinessBand = reviewedPassport.readinessBand
  const components = COMPONENT_DEFINITIONS.map((definition) => Object.freeze({
    id: definition.id,
    points: reviewedPassport.componentPoints[definition.id],
    maxPoints: definition.maxPoints,
    status: componentStatus(reviewedPassport.componentPoints[definition.id], definition.maxPoints),
  }))
  return Object.freeze({
    eventId,
    schemaVersion: reviewedManifest.artifactVersion,
    readinessBand,
    readinessLabel: readinessLabel(readinessBand),
    components: Object.freeze(components),
    supportedClaim: BAND_CLAIMS[readinessBand],
    unsupportedClaim: UNSUPPORTED_CLAIM,
    publicationStatus: 'reviewed-derived-aggregate',
    sourceArtifactId: REVIEWED_SOURCE.id,
  })
}

const PASSPORT_EVENT_IDS = Object.freeze(reviewedManifest.passports.map(({ eventId }) => eventId))

export const PUBLIC_EVIDENCE_PASSPORT_ARTIFACT = Object.freeze({
  version: reviewedManifest.artifactVersion,
  generatedDate: reviewedManifest.generatedDate,
  title: 'Public Evidence Passport Artifact v1',
  source: REVIEWED_SOURCE,
  comparisonBoundary: COMPARISON_BOUNDARY,
  componentDefinitions: COMPONENT_DEFINITIONS,
  bandDefinitions: BAND_DEFINITIONS,
  passports: Object.freeze(PASSPORT_EVENT_IDS.map(makePassport)),
})

const PASSPORTS_BY_EVENT_ID = new Map(
  PUBLIC_EVIDENCE_PASSPORT_ARTIFACT.passports.map((passport) => [passport.eventId, passport]),
)

export function evidencePassportByEventId(eventId) {
  return PASSPORTS_BY_EVENT_ID.get(eventId) ?? null
}

function findProhibitedField(value, path = '') {
  if (!value || typeof value !== 'object') return null
  for (const [key, nested] of Object.entries(value)) {
    const nextPath = path ? `${path}.${key}` : key
    if (PROHIBITED_FIELD_NAMES.has(key.toLowerCase())) return nextPath
    const prohibitedPath = findProhibitedField(nested, nextPath)
    if (prohibitedPath) return prohibitedPath
  }
  return null
}

function unknownField(value, fields, path) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const key = Object.keys(value).find((candidate) => !fields.has(candidate))
  return key ? `${path}.${key}` : null
}

function sameFields(actual, expected, fields) {
  return [...fields].every((field) => actual?.[field] === expected?.[field])
}

export function validatePublicEvidencePassportArtifact(artifact) {
  if (!artifact || typeof artifact !== 'object' || Array.isArray(artifact)) return ['artifact must be an object']

  const violations = []
  const prohibitedPath = findProhibitedField(artifact)
  if (prohibitedPath) violations.push(`overall or restricted input field is not public: ${prohibitedPath}`)

  const unknownArtifactField = unknownField(artifact, ARTIFACT_FIELDS, 'artifact')
  if (unknownArtifactField) violations.push(`unknown artifact field is not public: ${unknownArtifactField}`)
  if (artifact.version !== '1.0.0') violations.push('artifact version is not reviewed')
  if (!/^\d{4}-\d{2}-\d{2}$/.test(artifact.generatedDate || '')) violations.push('generated date must be ISO-8601')

  const unknownSourceField = unknownField(artifact.source, SOURCE_FIELDS, 'source')
  if (unknownSourceField) violations.push(`unknown source field is not public: ${unknownSourceField}`)
  if (!sameFields(artifact.source, REVIEWED_SOURCE, SOURCE_FIELDS)) violations.push('source does not match the reviewed source definition')
  if (!/^[a-f0-9]{64}$/.test(artifact.source?.sha256 || '')) violations.push('source needs a SHA-256 hash')

  const unknownBoundaryField = unknownField(artifact.comparisonBoundary, COMPARISON_BOUNDARY_FIELDS, 'comparisonBoundary')
  if (unknownBoundaryField) violations.push(`unknown comparison-boundary field is not public: ${unknownBoundaryField}`)
  if (
    !sameFields(artifact.comparisonBoundary, COMPARISON_BOUNDARY, new Set([...COMPARISON_BOUNDARY_FIELDS].filter((field) => field !== 'unknownConditions')))
    || JSON.stringify(artifact.comparisonBoundary?.unknownConditions) !== JSON.stringify(COMPARISON_BOUNDARY.unknownConditions)
  ) {
    violations.push('comparison boundary does not match the reviewed public limitation')
  }

  if (!Array.isArray(artifact.componentDefinitions) || artifact.componentDefinitions.length !== COMPONENT_DEFINITIONS.length) {
    violations.push('component definitions are incomplete')
  }
  for (const [index, definition] of (artifact.componentDefinitions || []).entries()) {
    const unknownDefinitionField = unknownField(definition, COMPONENT_DEFINITION_FIELDS, `componentDefinition.${index}`)
    if (unknownDefinitionField) violations.push(`unknown component-definition field is not public: ${unknownDefinitionField}`)
    if (!sameFields(definition, COMPONENT_DEFINITIONS[index], COMPONENT_DEFINITION_FIELDS)) violations.push(`component definition ${definition?.id || index} is not reviewed`)
  }

  if (!Array.isArray(artifact.bandDefinitions) || artifact.bandDefinitions.length !== BAND_DEFINITIONS.length) {
    violations.push('band definitions are incomplete')
  }
  for (const [index, definition] of (artifact.bandDefinitions || []).entries()) {
    const unknownBandField = unknownField(definition, BAND_DEFINITION_FIELDS, `bandDefinition.${index}`)
    if (unknownBandField) violations.push(`unknown band-definition field is not public: ${unknownBandField}`)
    if (!sameFields(definition, BAND_DEFINITIONS[index], BAND_DEFINITION_FIELDS)) violations.push(`band definition ${definition?.id || index} is not reviewed`)
  }

  if (!Array.isArray(artifact.passports) || artifact.passports.length !== PASSPORT_EVENT_IDS.length) {
    violations.push('passport cohort does not match the reviewed nine-event cohort')
  }
  const seenEventIds = new Set()
  for (const passport of artifact.passports || []) {
    const eventId = passport?.eventId || '(unknown)'
    const unknownPassportField = unknownField(passport, PASSPORT_FIELDS, `passport.${eventId}`)
    if (unknownPassportField) violations.push(`unknown passport field is not public: ${unknownPassportField}`)
    if (seenEventIds.has(eventId)) violations.push(`duplicate event passport: ${eventId}`)
    seenEventIds.add(eventId)

    const reviewedPassport = REVIEWED_PASSPORTS.get(eventId)
    const reviewedBand = reviewedPassport?.readinessBand
    if (!reviewedPassport || !reviewedBand) violations.push(`event passport ${eventId} is not reviewed`)
    if (passport?.schemaVersion !== artifact.version) violations.push(`event passport ${eventId} has an unreviewed schema version`)
    if (passport?.readinessBand !== reviewedBand) violations.push(`event passport ${eventId} has an unreviewed band`)
    if (passport?.readinessLabel !== readinessLabel(reviewedBand)) violations.push(`event passport ${eventId} has an unreviewed label`)
    if (passport?.supportedClaim !== BAND_CLAIMS[reviewedBand] || passport?.unsupportedClaim !== UNSUPPORTED_CLAIM) {
      violations.push(`event passport ${eventId} has unreviewed claim language`)
    }
    if (passport?.publicationStatus !== 'reviewed-derived-aggregate' || passport?.sourceArtifactId !== REVIEWED_SOURCE.id) {
      violations.push(`event passport ${eventId} has missing reviewed source lineage`)
    }
    if (!Array.isArray(passport?.components) || passport.components.length !== COMPONENT_DEFINITIONS.length) {
      violations.push(`event passport ${eventId} has incomplete components`)
      continue
    }
    for (const [index, component] of passport.components.entries()) {
      const definition = COMPONENT_DEFINITIONS[index]
      const unknownComponentField = unknownField(component, COMPONENT_FIELDS, `passport.${eventId}.component.${index}`)
      if (unknownComponentField) violations.push(`unknown component field is not public: ${unknownComponentField}`)
      const expectedPoints = reviewedPassport?.componentPoints?.[definition.id]
      if (
        component.id !== definition.id
        || component.maxPoints !== definition.maxPoints
        || component.points !== expectedPoints
        || component.status !== componentStatus(expectedPoints, definition.maxPoints)
      ) {
        violations.push(`event passport ${eventId} component ${component.id || index} does not match its reviewed component value`)
      }
    }
  }

  for (const eventId of PASSPORT_EVENT_IDS) {
    if (!seenEventIds.has(eventId)) violations.push(`reviewed event passport is missing: ${eventId}`)
  }

  return [...new Set(violations)]
}
