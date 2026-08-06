export const PRESET_DISCLAIMER = 'These are editorial examples selected to illustrate boundary conditions. They are not a representative sample, benchmark, recommendation, or preferred pair.'

const REVIEWED_COMPONENT_SCHEMA_V1 = Object.freeze([
  Object.freeze({ id: 'observation-quality', maxPoints: 30 }),
  Object.freeze({ id: 'post-event-coverage', maxPoints: 20 }),
  Object.freeze({ id: 'context-coverage', maxPoints: 20 }),
  Object.freeze({ id: 'covariate-completeness', maxPoints: 20 }),
  Object.freeze({ id: 'data-integrity', maxPoints: 10 }),
])

export const PRESET_COMPARISONS = Object.freeze([
  Object.freeze({
    id: 'same-storm-two-places',
    label: 'Same storm, two places',
    context: 'Hurricane Ian · Florida · 2022',
    eventIds: Object.freeze(['ian-charlotte', 'ian-fortmyers']),
    note: 'Use this pair to inspect two public references from the same storm and year. Shared context does not make their local outcomes interchangeable.',
  }),
  Object.freeze({
    id: 'same-hazard-evidence-shift',
    label: 'Same hazard, evidence shift',
    context: 'Tropical cyclone · Southeast',
    eventIds: Object.freeze(['irma', 'michael']),
    note: 'The documented hazard family and broad region align, while the reviewed rule-bin states differ. Read each row without treating either event as better.',
  }),
  Object.freeze({
    id: 'same-place-different-hazards',
    label: 'Same place, different hazards',
    context: 'San Juan · cross-hazard caution',
    eventIds: Object.freeze(['maria', 'eq-pr']),
    note: 'The broad public center is the same, but the documented hazard families and years differ. This is descriptive evidence inspection, not an outcome comparison.',
  }),
  Object.freeze({
    id: 'earthquake-context-boundary',
    label: 'Earthquake context boundary',
    context: 'Caribbean ↔ International',
    eventIds: Object.freeze(['eq-pr', 'eq-hatay']),
    note: 'The documented hazard family aligns while geographic and source context changes. The public artifact does not establish measurement equivalence.',
  }),
])

function hazardFamily(event) {
  return event?.hazardFamily ?? 'Unclassified'
}

export function resolveComparisonPeerId(events, primaryId, peerId) {
  const primary = events.find(({ id }) => id === primaryId)
  const peer = events.find(({ id }) => id === peerId)
  if (primary && peer && primary.id !== peer.id) return peer.id
  if (!primary) return events.find(({ id }) => id !== peerId)?.id ?? null

  const candidates = events
    .map((event, index) => ({ event, index }))
    .filter(({ event }) => event.id !== primary.id)
    .sort((left, right) => {
      const leftFamilyMatch = Number(hazardFamily(left.event) === hazardFamily(primary))
      const rightFamilyMatch = Number(hazardFamily(right.event) === hazardFamily(primary))
      if (leftFamilyMatch !== rightFamilyMatch) return rightFamilyMatch - leftFamilyMatch

      const leftRegionMatch = Number(left.event.region === primary.region)
      const rightRegionMatch = Number(right.event.region === primary.region)
      if (leftRegionMatch !== rightRegionMatch) return rightRegionMatch - leftRegionMatch

      return left.index - right.index
    })

  return candidates[0]?.event.id ?? null
}

function compatibilityFor(leftEvent, rightEvent) {
  if (hazardFamily(leftEvent) !== hazardFamily(rightEvent)) {
    return Object.freeze({
      id: 'cross-hazard',
      label: 'Cross-hazard context',
      tone: 'warning',
      summary: 'The documented hazard families differ. Treat this as descriptive evidence inspection only.',
    })
  }

  if (leftEvent.region === rightEvent.region) {
    return Object.freeze({
      id: 'category-region-aligned',
      label: 'Hazard family + broad region aligned',
      tone: 'aligned',
      summary: 'The documented public categories align; cross-event measurement comparability is still not established.',
    })
  }

  return Object.freeze({
    id: 'category-aligned',
    label: 'Hazard family aligned',
    tone: 'caution',
    summary: 'The documented hazard family matches while broad regional context differs; measurement comparability is not established.',
  })
}

function expectedStatus(points, maximum) {
  if (points === maximum) return 'available'
  if (points === 0) return 'unavailable'
  return 'limited'
}

function reviewedSchema(artifact) {
  if (artifact?.version !== '1.0.0' || !Array.isArray(artifact.componentDefinitions)) return null
  const definitions = artifact.componentDefinitions
  if (definitions.length !== REVIEWED_COMPONENT_SCHEMA_V1.length) return null
  if (!definitions.every((definition, index) => (
    definition?.id === REVIEWED_COMPONENT_SCHEMA_V1[index].id
    && definition?.maxPoints === REVIEWED_COMPONENT_SCHEMA_V1[index].maxPoints
  ))) return null
  return definitions
}

function passportMatchesSchema(event, passport, artifact, definitions) {
  if (!event || !passport || passport.eventId !== event.id || passport.schemaVersion !== artifact.version) return false
  if (!Array.isArray(passport.components) || passport.components.length !== definitions.length) return false

  return definitions.every((definition, index) => {
    const component = passport.components[index]
    if (!component || typeof component !== 'object') return false
    if (component.id !== definition.id || component.maxPoints !== definition.maxPoints) return false
    if (!Number.isFinite(component.points) || component.points < 0 || component.points > component.maxPoints) return false
    return component.status === expectedStatus(component.points, component.maxPoints)
  })
}

function componentPairsFor(leftPassport, rightPassport, definitions) {
  return definitions.map((definition, index) => {
    const left = leftPassport.components[index]
    const right = rightPassport.components[index]
    return Object.freeze({
      id: definition.id,
      left,
      right,
      sameState: left.status === right.status,
      samePublishedValue: left.points === right.points && left.maxPoints === right.maxPoints,
    })
  })
}

function warningsFor(leftEvent, rightEvent, passportCoverage, schemaStatus, measurementBoundary) {
  const warnings = [
    'This compares public context and evidence state—not recovery, severity, resilience, causality, policy performance, or event quality.',
    'No similarity score is computed. Same v1 rule-bin values do not make events or measurement frames equivalent.',
  ]

  if (hazardFamily(leftEvent) !== hazardFamily(rightEvent)) {
    warnings.push(`Documented hazard families differ: ${hazardFamily(leftEvent)} and ${hazardFamily(rightEvent)}.`)
  }
  if (leftEvent.region !== rightEvent.region) {
    warnings.push(`Broad regions differ: ${leftEvent.region} and ${rightEvent.region}. Geographic context is not controlled by this view.`)
  }
  const crossesInternationalBoundary = (leftEvent.region === 'International') !== (rightEvent.region === 'International')
  if (crossesInternationalBoundary) {
    warnings.push('One event crosses the international context boundary, where source coverage and covariate context may differ materially.')
  }
  if (leftEvent.name === rightEvent.name && leftEvent.year === rightEvent.year) {
    warnings.push('These are two indexed locations for the same named event and year, not two independent disasters.')
  }
  if (schemaStatus === 'not-comparable') {
    warnings.push('The Passport schema is invalid, unsupported, or inconsistent. Component pairing is withheld rather than treating missing or malformed rows as differences.')
  } else if (passportCoverage < 2) {
    const verb = passportCoverage === 1 ? 'has' : 'have'
    warnings.push(`${passportCoverage} of 2 events ${verb} a reviewed Evidence Passport. Missing assessment is not zero and is not imputed.`)
  } else {
    warnings.push(measurementBoundary?.statement ?? 'The public artifact supports descriptive rule-bin pairing only; cross-event measurement comparability is not established.')
  }

  return Object.freeze(warnings)
}

export function buildEventComparison(leftEvent, rightEvent, leftPassport, rightPassport, artifact) {
  if (!leftEvent || !rightEvent) throw new TypeError('two public events are required')
  if (leftEvent.id === rightEvent.id) throw new TypeError('comparison requires two distinct events')

  const definitions = reviewedSchema(artifact)
  const suppliedPassportCoverage = Number(Boolean(leftPassport)) + Number(Boolean(rightPassport))
  const leftPassportValid = Boolean(definitions && leftPassport && passportMatchesSchema(leftEvent, leftPassport, artifact, definitions))
  const rightPassportValid = Boolean(definitions && rightPassport && passportMatchesSchema(rightEvent, rightPassport, artifact, definitions))
  const passportCoverage = Number(leftPassportValid) + Number(rightPassportValid)
  const schemaStatus = suppliedPassportCoverage < 2
    ? 'unavailable'
    : leftPassportValid && rightPassportValid
      ? 'paired-v1'
      : 'not-comparable'
  const componentPairs = schemaStatus === 'paired-v1'
    ? componentPairsFor(leftPassport, rightPassport, definitions)
    : []
  const componentMaximum = definitions?.length ?? null
  const measurementBoundary = artifact?.comparisonBoundary ?? Object.freeze({
    status: 'comparability-not-established',
    privateSourceVerification: 'restricted-environment-required',
    statement: 'Cross-event measurement comparability is not established.',
  })

  return Object.freeze({
    compatibility: compatibilityFor(leftEvent, rightEvent),
    passportCoverage,
    schemaStatus,
    measurementBoundary,
    summaries: Object.freeze([
      Object.freeze({
        id: 'reviewed-passports',
        label: 'Reviewed Passports',
        value: passportCoverage,
        maximum: 2,
        prefix: '',
        suffix: 'events',
        note: 'Assessment coverage only—not an event-quality or similarity measure.',
      }),
      Object.freeze({
        id: 'paired-component-rows',
        label: 'Paired v1 component rows',
        value: componentPairs.length ? componentPairs.length : null,
        maximum: componentMaximum,
        prefix: '',
        suffix: 'rows',
        note: 'Side-by-side rule-bin rows only; measurement equivalence is not established.',
      }),
    ]),
    componentPairs: Object.freeze(componentPairs),
    warnings: warningsFor(leftEvent, rightEvent, passportCoverage, schemaStatus, measurementBoundary),
  })
}
