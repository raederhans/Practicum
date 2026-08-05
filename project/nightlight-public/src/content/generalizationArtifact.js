const MODEL_ROLES = new Set([
  'explanatory',
  'damage-ranking',
  'recovery-transport',
  'secondary-interpretation',
])
const METRIC_TYPES = new Set(['description', 'ranking', 'calibration'])
const QUALITY_STATUSES = new Set(['reviewed-aggregate', 'withheld'])
const PUBLICATION_STATUSES = new Set(['admitted', 'withheld'])
const UNIT_BY_METRIC = Object.freeze({
  'R-squared': 'coefficient-of-determination [0–1]',
  'Adjusted R-squared': 'coefficient-of-determination [0–1]',
  'Logit AUC': 'area-under-curve [0–1]',
  'Descriptive sensitivity ratio': 'ratio [unitless]',
})
const REVIEWED_SOURCE_HASHES = Object.freeze({
  'public-study-summary-v1': '8d00fde7164d05c3d2912f0d7714a6dc0b5240194ff0ea2eae527403a06565ee',
  'cross-event-stop-decision-v3x': '306c6ce4736bbd73df526fa0176e78d296b7eb6fe0c8ee515880525fd0c3cb1e',
})
const REVIEWED_PUBLIC_METRICS = Object.freeze({
  'in-sample-r-squared': Object.freeze({
    metricName: 'R-squared',
    modelRole: 'explanatory',
    metricType: 'description',
    value: 0.7603,
    unit: 'coefficient-of-determination [0–1]',
    sourceArtifactId: 'public-study-summary-v1',
  }),
  'cross-event-logit-auc': Object.freeze({
    metricName: 'Logit AUC',
    modelRole: 'damage-ranking',
    metricType: 'ranking',
    value: 0.4814,
    unit: 'area-under-curve [0–1]',
    sourceArtifactId: 'cross-event-stop-decision-v3x',
  }),
  'descriptive-sensitivity-ratio': Object.freeze({
    metricName: 'Descriptive sensitivity ratio',
    modelRole: 'secondary-interpretation',
    metricType: 'description',
    value: 0.551,
    unit: 'ratio [unitless]',
    sourceArtifactId: 'public-study-summary-v1',
  }),
})
const PROHIBITED_FIELD_NAMES = new Set([
  'facility', 'facilities', 'coordinates', 'coordinate', 'probability', 'probabilities',
  'raster', 'grid', 'grid_id', 'zip_code', 'zip_event', 'time_series', 'timeseries',
  'outage_duration', 'recovery_time', 'model_binary', 'credentials', 'local_path',
])
const REQUIRED_METRIC_FIELDS = [
  'id', 'cohort', 'sampleLock', 'validationDesign', 'modelFamily', 'modelRole',
  'metricName', 'metricType', 'unit', 'qualityStatus', 'publicationStatus',
  'supportedClaim', 'unsupportedClaim', 'sourceArtifactId',
]
const ARTIFACT_FIELDS = new Set(['version', 'generatedDate', 'title', 'sources', 'metrics'])
const SOURCE_FIELDS = new Set(['id', 'version', 'sha256', 'source', 'license', 'attribution'])
const METRIC_FIELDS = new Set([...REQUIRED_METRIC_FIELDS, 'value', 'withheldReason'])

export const PUBLIC_GENERALIZATION_ARTIFACT = Object.freeze({
  version: '1.0.0',
  generatedDate: '2026-08-05',
  title: 'Public Generalization Artifact v1',
  sources: Object.freeze([
    Object.freeze({
      id: 'public-study-summary-v1',
      version: 'study.js@1',
      sha256: '8d00fde7164d05c3d2912f0d7714a6dc0b5240194ff0ea2eae527403a06565ee',
      source: 'Reviewed aggregate study summary',
      license: 'CC BY 4.0 aggregate derivation with attribution',
      attribution: 'ORNL EAGLE-I aggregate derivation; no endorsement implied',
    }),
    Object.freeze({
      id: 'cross-event-stop-decision-v3x',
      version: 'v3x-r1',
      sha256: '306c6ce4736bbd73df526fa0176e78d296b7eb6fe0c8ee515880525fd0c3cb1e',
      source: 'Reviewed aggregate cross-event stop decision',
      license: 'CC BY 4.0 aggregate derivation with attribution',
      attribution: 'ORNL EAGLE-I aggregate derivation; no endorsement implied',
    }),
  ]),
  metrics: Object.freeze([
    Object.freeze({
      id: 'in-sample-r-squared',
      cohort: 'Stage 3 complete cases: 22 events, 15 U.S. states',
      sampleLock: 'M1+ complete-case analysis, n=977',
      validationDesign: 'in-sample, fixed-control descriptive fit',
      modelFamily: 'M1+ linear descriptive model',
      modelRole: 'explanatory',
      metricName: 'R-squared',
      metricType: 'description',
      value: 0.7603,
      unit: 'coefficient-of-determination [0–1]',
      qualityStatus: 'reviewed-aggregate',
      publicationStatus: 'admitted',
      supportedClaim: 'The specified model describes variation within the analyzed, fixed-control sample.',
      unsupportedClaim: 'Future-event accuracy, causation, or a ranking of community recovery.',
      sourceArtifactId: 'public-study-summary-v1',
    }),
    Object.freeze({
      id: 'cross-event-logit-auc',
      cohort: 'Cross-event stabilization cohort; held-out event folds',
      sampleLock: 'V3x stabilization r1 aggregate stop decision',
      validationDesign: 'leave-one-event-out cross-event damage-ranking',
      modelFamily: 'Cross-event logit damage-ranking model',
      modelRole: 'damage-ranking',
      metricName: 'Logit AUC',
      metricType: 'ranking',
      value: 0.4814,
      unit: 'area-under-curve [0–1]',
      qualityStatus: 'reviewed-aggregate',
      publicationStatus: 'admitted',
      supportedClaim: 'In this held-out-event damage-ranking design, the admitted ranking result is below a 0.50 reference.',
      unsupportedClaim: 'A recovery-transport result, calibrated probability, future-event readiness, or a community recovery ranking.',
      sourceArtifactId: 'cross-event-stop-decision-v3x',
    }),
    Object.freeze({
      id: 'descriptive-sensitivity-ratio',
      cohort: 'Stage 3 descriptive sensitivity aggregate',
      sampleLock: 'Reviewed descriptive sensitivity summary',
      validationDesign: 'descriptive sensitivity; not a cross-event validation',
      modelFamily: 'Descriptive sensitivity diagnostic',
      modelRole: 'secondary-interpretation',
      metricName: 'Descriptive sensitivity ratio',
      metricType: 'description',
      value: 0.551,
      unit: 'ratio [unitless]',
      qualityStatus: 'reviewed-aggregate',
      publicationStatus: 'admitted',
      supportedClaim: 'A descriptive sensitivity value under its stated analysis conditions.',
      unsupportedClaim: 'A fairness conclusion, causal mechanism, or transport improvement.',
      sourceArtifactId: 'public-study-summary-v1',
    }),
  ]),
})

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

function findUnknownField(value, allowedFields, path) {
  if (!value || typeof value !== 'object') return null
  const unknownKey = Object.keys(value).find((key) => !allowedFields.has(key))
  return unknownKey ? `${path}.${unknownKey}` : null
}

export function validatePublicGeneralizationArtifact(artifact) {
  const violations = []
  const prohibitedPath = findProhibitedField(artifact)
  if (prohibitedPath) violations.push(`restricted field is not public: ${prohibitedPath}`)

  if (!artifact || typeof artifact !== 'object') return ['artifact must be an object']
  const unknownArtifactField = findUnknownField(artifact, ARTIFACT_FIELDS, 'artifact')
  if (unknownArtifactField) violations.push(`unknown artifact field is not public: ${unknownArtifactField}`)
  if (!/^\d+\.\d+\.\d+$/.test(artifact.version || '')) violations.push('artifact version must be semver')
  if (!/^\d{4}-\d{2}-\d{2}$/.test(artifact.generatedDate || '')) violations.push('generated date must be ISO-8601')
  if (!Array.isArray(artifact.sources) || artifact.sources.length === 0) violations.push('source lineage is required')
  if (!Array.isArray(artifact.metrics) || artifact.metrics.length === 0) violations.push('at least one public metric is required')

  const sourceIds = new Set()
  for (const source of artifact.sources || []) {
    const unknownSourceField = findUnknownField(source, SOURCE_FIELDS, `source.${source.id || '(unknown)'}`)
    if (unknownSourceField) violations.push(`unknown source field is not public: ${unknownSourceField}`)
    if (!source.id || !source.version || !source.source || !source.license || !source.attribution) {
      violations.push('source lineage is incomplete')
    }
    if (!/^[a-f0-9]{64}$/.test(source.sha256 || '')) violations.push(`source ${source.id || '(unknown)'} needs a SHA-256 hash`)
    if (!REVIEWED_SOURCE_HASHES[source.id]) {
      violations.push(`source ${source.id || '(unknown)'} is not a reviewed source artifact`)
    } else if (source.sha256 !== REVIEWED_SOURCE_HASHES[source.id]) {
      violations.push(`source ${source.id} does not match its reviewed source hash`)
    }
    sourceIds.add(source.id)
  }

  for (const metric of artifact.metrics || []) {
    const unknownMetricField = findUnknownField(metric, METRIC_FIELDS, `metric.${metric.id || '(unknown)'}`)
    if (unknownMetricField) violations.push(`unknown metric field is not public: ${unknownMetricField}`)
    for (const field of REQUIRED_METRIC_FIELDS) {
      if (metric[field] === undefined || metric[field] === null || metric[field] === '') {
        violations.push(`metric ${metric.id || '(unknown)'} is missing ${field}`)
      }
    }
    if (!MODEL_ROLES.has(metric.modelRole)) violations.push(`metric ${metric.id || '(unknown)'} has an unsupported model role`)
    if (!METRIC_TYPES.has(metric.metricType)) violations.push(`metric ${metric.id || '(unknown)'} has an unsupported metric type`)
    if (!QUALITY_STATUSES.has(metric.qualityStatus)) violations.push(`metric ${metric.id || '(unknown)'} has an unsupported quality status`)
    if (!PUBLICATION_STATUSES.has(metric.publicationStatus)) violations.push(`metric ${metric.id || '(unknown)'} has an unsupported publication status`)
    if (metric.publicationStatus === 'admitted') {
      if (metric.qualityStatus !== 'reviewed-aggregate') violations.push(`metric ${metric.id || '(unknown)'} has an inconsistent admitted quality status`)
      if (!Number.isFinite(metric.value)) violations.push(`metric ${metric.id || '(unknown)'} needs a finite public value`)
    }
    if (metric.publicationStatus === 'withheld') {
      if (metric.qualityStatus !== 'withheld') violations.push(`metric ${metric.id || '(unknown)'} has an inconsistent withheld quality status`)
      if (metric.value !== null) violations.push(`metric ${metric.id || '(unknown)'} must not carry a public value while withheld`)
      if (!metric.withheldReason) violations.push(`metric ${metric.id || '(unknown)'} needs a withheld reason`)
    }
    if (Number.isFinite(metric.value) && typeof metric.unit === 'string' && metric.unit.includes('[0–1]') && (metric.value < 0 || metric.value > 1)) {
      violations.push(`metric ${metric.id || '(unknown)'} is outside its declared [0–1] range`)
    }
    if (UNIT_BY_METRIC[metric.metricName] !== metric.unit) violations.push(`metric ${metric.id || '(unknown)'} has an undeclared unit`)
    if (!sourceIds.has(metric.sourceArtifactId)) violations.push(`metric ${metric.id || '(unknown)'} has missing source lineage`)
    const reviewedDefinition = REVIEWED_PUBLIC_METRICS[metric.id]
    if (metric.publicationStatus === 'admitted' && (!reviewedDefinition || Object.entries(reviewedDefinition).some(([field, value]) => metric[field] !== value))) {
      violations.push(`metric ${metric.id || '(unknown)'} does not match its reviewed metric definition`)
    }
  }

  return violations
}
