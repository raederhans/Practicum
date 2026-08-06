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
    context: 'Hurricane · Southeast',
    eventIds: Object.freeze(['irma', 'michael']),
    note: 'The hazard family and broad region align, while the reviewed evidence states differ. Read the component rows rather than treating either event as better.',
  }),
  Object.freeze({
    id: 'same-place-different-hazards',
    label: 'Same place, different hazards',
    context: 'San Juan · cross-hazard caution',
    eventIds: Object.freeze(['maria', 'eq-pr']),
    note: 'The broad public center is the same, but the hazard families and years differ. This is an evidence-readiness contrast, not an outcome comparison.',
  }),
  Object.freeze({
    id: 'earthquake-context-boundary',
    label: 'Earthquake context boundary',
    context: 'Caribbean ↔ International',
    eventIds: Object.freeze(['eq-pr', 'eq-hatay']),
    note: 'The hazard family aligns while the broad geographic and source context changes. Use the warning as a limit on interpretation, not a correction factor.',
  }),
])

export function resolveComparisonPeerId(events, primaryId, peerId) {
  const primary = events.find(({ id }) => id === primaryId)
  const peer = events.find(({ id }) => id === peerId)
  if (primary && peer && primary.id !== peer.id) return peer.id
  if (!primary) return events.find(({ id }) => id !== peerId)?.id ?? null

  const candidates = events
    .map((event, index) => ({ event, index }))
    .filter(({ event }) => event.id !== primary.id)
    .sort((left, right) => {
      const leftTypeMatch = Number(left.event.type === primary.type)
      const rightTypeMatch = Number(right.event.type === primary.type)
      if (leftTypeMatch !== rightTypeMatch) return rightTypeMatch - leftTypeMatch

      const leftRegionMatch = Number(left.event.region === primary.region)
      const rightRegionMatch = Number(right.event.region === primary.region)
      if (leftRegionMatch !== rightRegionMatch) return rightRegionMatch - leftRegionMatch

      return left.index - right.index
    })

  return candidates[0]?.event.id ?? null
}

function compatibilityFor(leftEvent, rightEvent) {
  if (leftEvent.type !== rightEvent.type) {
    return Object.freeze({
      id: 'cross-hazard',
      label: 'Cross-hazard context',
      tone: 'warning',
      summary: 'The hazard families differ, so category context must lead the interpretation.',
    })
  }

  if (leftEvent.region === rightEvent.region) {
    return Object.freeze({
      id: 'category-region-aligned',
      label: 'Hazard family + broad region aligned',
      tone: 'aligned',
      summary: 'The public categories align; measurement-frame equivalence is still unverified.',
    })
  }

  return Object.freeze({
    id: 'category-aligned',
    label: 'Hazard family aligned',
    tone: 'caution',
    summary: 'The hazard family matches, while the broad regional context differs.',
  })
}

function componentPairsFor(leftPassport, rightPassport) {
  if (!leftPassport || !rightPassport) return []
  const rightComponents = new Map(rightPassport.components.map((component) => [component.id, component]))

  return leftPassport.components.map((left) => {
    const right = rightComponents.get(left.id)
    return Object.freeze({
      id: left.id,
      left,
      right,
      sameState: Boolean(right && left.status === right.status),
      samePublishedValue: Boolean(right && left.points === right.points && left.maxPoints === right.maxPoints),
    })
  })
}

function warningsFor(leftEvent, rightEvent, passportCoverage) {
  const warnings = [
    'This compares public context and evidence state—not recovery, severity, resilience, causality, or policy performance.',
  ]

  if (leftEvent.type !== rightEvent.type) {
    warnings.push(`Hazard families differ: ${leftEvent.type} and ${rightEvent.type}. Treat component differences as context-sensitive evidence contrasts.`)
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
  if (passportCoverage < 2) {
    const verb = passportCoverage === 1 ? 'has' : 'have'
    warnings.push(`${passportCoverage} of 2 events ${verb} a reviewed Evidence Passport. Missing assessment is not zero and is not imputed.`)
  }

  return Object.freeze(warnings)
}

export function buildEventComparison(leftEvent, rightEvent, leftPassport, rightPassport) {
  if (!leftEvent || !rightEvent) throw new TypeError('two public events are required')
  if (leftEvent.id === rightEvent.id) throw new TypeError('comparison requires two distinct events')

  const passportCoverage = Number(Boolean(leftPassport)) + Number(Boolean(rightPassport))
  const componentPairs = componentPairsFor(leftPassport, rightPassport)
  const matchingComponentStates = componentPairs.length
    ? componentPairs.filter(({ sameState }) => sameState).length
    : null
  const exactPublishedValues = componentPairs.length
    ? componentPairs.filter(({ samePublishedValue }) => samePublishedValue).length
    : null

  return Object.freeze({
    compatibility: compatibilityFor(leftEvent, rightEvent),
    passportCoverage,
    summaries: Object.freeze([
      Object.freeze({
        id: 'reviewed-passports',
        label: 'Reviewed Passports',
        value: passportCoverage,
        maximum: 2,
        prefix: '',
        suffix: 'events',
        note: 'Assessment coverage, not an event-quality measure.',
      }),
      Object.freeze({
        id: 'comparable-components',
        label: 'Comparable components',
        value: componentPairs.length ? componentPairs.length : null,
        maximum: 5,
        prefix: '',
        suffix: 'components',
        note: 'Available only when both reviewed Passports use the same five-component schema.',
      }),
      Object.freeze({
        id: 'exact-published-values',
        label: 'Exact published values',
        value: exactPublishedValues,
        maximum: 5,
        prefix: '',
        suffix: 'components',
        note: 'Matching values do not make the events equivalent.',
      }),
      Object.freeze({
        id: 'different-published-values',
        label: 'Different published values',
        value: exactPublishedValues === null ? null : componentPairs.length - exactPublishedValues,
        maximum: 5,
        prefix: '',
        suffix: 'components',
        note: matchingComponentStates === null
          ? 'No paired component evidence is available.'
          : 'Inspect each component separately; no event-level conclusion is assigned.',
      }),
    ]),
    componentPairs: Object.freeze(componentPairs),
    warnings: warningsFor(leftEvent, rightEvent, passportCoverage),
  })
}
