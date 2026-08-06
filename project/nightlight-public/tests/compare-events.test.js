import { describe, expect, it } from 'vitest'

import {
  PUBLIC_EVIDENCE_PASSPORT_ARTIFACT,
  evidencePassportByEventId,
} from '../src/content/evidencePassportArtifact.js'
import { EVENTS } from '../src/content/study.js'

const compareModuleUrl = new URL('../src/domain/compareEvents.js', import.meta.url)

async function loadCompareModule() {
  try {
    return await import(compareModuleUrl.href)
  } catch (loadError) {
    return { loadError }
  }
}

function eventById(eventId) {
  return EVENTS.find(({ id }) => id === eventId)
}

async function compare(
  leftId,
  rightId,
  leftPassport = evidencePassportByEventId(leftId),
  rightPassport = evidencePassportByEventId(rightId),
  artifact = PUBLIC_EVIDENCE_PASSPORT_ARTIFACT,
) {
  const { buildEventComparison } = await import(compareModuleUrl.href)
  return buildEventComparison(
    eventById(leftId),
    eventById(rightId),
    leftPassport,
    rightPassport,
    artifact,
  )
}

describe('Atlas event comparison rules', () => {
  it('repairs a duplicate peer with a hazard-family-first alternative and preserves a valid peer', async () => {
    const module = await loadCompareModule()
    expect(module.loadError).toBeUndefined()

    expect(module.resolveComparisonPeerId(EVENTS, 'maria', 'maria')).toBe('irma')
    expect(module.resolveComparisonPeerId(EVENTS, 'maria', 'eq-pr')).toBe('eq-pr')
    for (const event of EVENTS) {
      const firstRepair = module.resolveComparisonPeerId(EVENTS, event.id, event.id)
      const secondRepair = module.resolveComparisonPeerId(EVENTS, event.id, event.id)
      expect(firstRepair).not.toBe(event.id)
      expect(secondRepair).toBe(firstRepair)
      if (EVENTS.some((candidate) => candidate.id !== event.id && candidate.hazardFamily === event.hazardFamily)) {
        expect(eventById(firstRepair).hazardFamily).toBe(event.hazardFamily)
      }
    }
  })

  it('uses documented hazard family before broad region in compatibility language', async () => {
    const sameContext = await compare('irma', 'michael')
    expect(sameContext.compatibility.id).toBe('category-region-aligned')
    expect(sameContext.compatibility.label).toMatch(/hazard.*region/i)

    const crossHazard = await compare('maria', 'eq-pr')
    expect(crossHazard.compatibility.id).toBe('cross-hazard')
    expect(crossHazard.warnings.join(' ')).toMatch(/hazard famil/i)

    const internationalBoundary = await compare('eq-pr', 'eq-hatay')
    expect(internationalBoundary.compatibility.id).toBe('category-aligned')
    expect(internationalBoundary.warnings.join(' ')).toMatch(/international context/i)
  })

  it('uses only coverage summaries and keeps value relationships in the component ledger', async () => {
    const comparison = await compare('maria', 'eq-pr')
    const summaries = Object.fromEntries(comparison.summaries.map((summary) => [summary.id, summary]))

    expect(summaries['reviewed-passports'].value).toBe(2)
    expect(summaries['paired-component-rows'].value).toBe(5)
    expect(summaries['exact-published-values']).toBeUndefined()
    expect(summaries['different-published-values']).toBeUndefined()
    expect(comparison.componentPairs).toHaveLength(5)
    expect(comparison.measurementBoundary.status).toBe('comparability-not-established')
    expect(comparison.warnings.join(' ')).toMatch(/measurement.*not established|descriptive.*only/i)
    expect(JSON.stringify(comparison)).not.toMatch(/leaderboard|recovery.?score|rank(?:ing)?/i)
  })

  it('keeps component pairing unavailable when either event lacks a reviewed Passport', async () => {
    const comparison = await compare('maria', 'matthew-jax')
    const evidenceOnlySummaries = comparison.summaries.filter(({ id }) => id !== 'reviewed-passports')

    expect(comparison.passportCoverage).toBe(1)
    expect(evidenceOnlySummaries.every(({ value }) => value === null)).toBe(true)
    expect(comparison.componentPairs).toEqual([])
    expect(comparison.warnings.join(' ')).toMatch(/1 of 2.*reviewed Evidence Passport/i)
  })

  it('supports all 600 directed pairs and rejects self-comparison', async () => {
    const module = await loadCompareModule()
    expect(module.loadError).toBeUndefined()

    let pairCount = 0
    for (const left of EVENTS) {
      expect(() => module.buildEventComparison(
        left,
        left,
        evidencePassportByEventId(left.id),
        evidencePassportByEventId(left.id),
        PUBLIC_EVIDENCE_PASSPORT_ARTIFACT,
      )).toThrow(/distinct events/i)

      for (const right of EVENTS) {
        if (left.id === right.id) continue
        const comparison = module.buildEventComparison(
          left,
          right,
          evidencePassportByEventId(left.id),
          evidencePassportByEventId(right.id),
          PUBLIC_EVIDENCE_PASSPORT_ARTIFACT,
        )
        pairCount += 1
        expect(comparison.summaries).toHaveLength(2)
        expect(comparison.summaries.every(({ value }) => value === null || Number.isFinite(value))).toBe(true)
        expect(comparison.warnings.length).toBeGreaterThan(0)

        const reverse = module.buildEventComparison(
          right,
          left,
          evidencePassportByEventId(right.id),
          evidencePassportByEventId(left.id),
          PUBLIC_EVIDENCE_PASSPORT_ARTIFACT,
        )
        expect(reverse.summaries.map(({ id, value, maximum }) => ({ id, value, maximum }))).toEqual(
          comparison.summaries.map(({ id, value, maximum }) => ({ id, value, maximum })),
        )
        expect(reverse.schemaStatus).toBe(comparison.schemaStatus)
      }
    }
    expect(pairCount).toBe(600)
  })

  it('produces no schema error for any directed pair among the nine controlled reviewed Passports', async () => {
    const reviewedEventIds = PUBLIC_EVIDENCE_PASSPORT_ARTIFACT.passports.map(({ eventId }) => eventId)
    let reviewedPairCount = 0

    for (const leftId of reviewedEventIds) {
      for (const rightId of reviewedEventIds) {
        if (leftId === rightId) continue
        const comparison = await compare(leftId, rightId)
        reviewedPairCount += 1
        expect(comparison.passportCoverage).toBe(2)
        expect(comparison.schemaStatus).toBe('paired-v1')
        expect(comparison.componentPairs).toHaveLength(5)
      }
    }

    expect(reviewedPairCount).toBe(72)
  })

  it.each([
    ['a missing right component', (left, right) => { right.components.pop() }],
    ['an undefined right component', (left, right) => { right.components[2] = undefined }],
    ['a duplicate right component ID', (left, right) => { right.components[4] = { ...right.components[3] } }],
    ['a duplicate left component ID', (left) => { left.components[4] = { ...left.components[3] } }],
    ['a changed maximum', (left, right) => { right.components[0].maxPoints += 1 }],
    ['an invalid status for the same points', (left, right) => { right.components[0].status = 'limited' }],
    ['an empty component array', (left) => { left.components = [] }],
    ['a sixth component', (left) => { left.components.push({ ...left.components[0], id: 'future-sixth' }) }],
    ['a future Passport schema', (left, right) => { left.schemaVersion = '2.0.0'; right.schemaVersion = '2.0.0' }],
  ])('fails closed for %s without throwing or producing asymmetric counts', async (_, mutate) => {
    const left = structuredClone(evidencePassportByEventId('maria'))
    const right = structuredClone(evidencePassportByEventId('eq-pr'))
    mutate(left, right)

    const forward = await compare('maria', 'eq-pr', left, right)
    const reverse = await compare('eq-pr', 'maria', right, left)

    expect(forward.schemaStatus).toBe('not-comparable')
    expect(forward.componentPairs).toEqual([])
    expect(reverse.componentPairs).toEqual([])
    expect(forward.summaries.find(({ id }) => id === 'paired-component-rows')?.value).toBeNull()
    expect(reverse.summaries.find(({ id }) => id === 'paired-component-rows')?.value).toBeNull()
    expect(forward.warnings.join(' ')).toMatch(/schema.*withheld|not comparable/i)
  })

  it('reports an invalid supplied Passport before a missing peer assessment', async () => {
    const malformedLeft = structuredClone(evidencePassportByEventId('maria'))
    malformedLeft.components.pop()

    const comparison = await compare('maria', 'matthew-jax', malformedLeft, undefined)

    expect(comparison.passportCoverage).toBe(0)
    expect(comparison.schemaStatus).toBe('not-comparable')
    expect(comparison.componentPairs).toEqual([])
    expect(comparison.warnings.join(' ')).toMatch(/schema.*withheld/i)
    expect(comparison.warnings.join(' ')).not.toMatch(/0 of 2 events.*reviewed Evidence Passport/i)
  })

  it('rejects an unknown or future artifact schema instead of using a hard-coded five', async () => {
    const futureArtifact = structuredClone(PUBLIC_EVIDENCE_PASSPORT_ARTIFACT)
    futureArtifact.version = '2.0.0'
    futureArtifact.componentDefinitions.push({ id: 'future-sixth', label: 'Future', maxPoints: 10, meaning: 'Unreviewed.' })

    const comparison = await compare('maria', 'eq-pr', undefined, undefined, futureArtifact)

    expect(comparison.schemaStatus).toBe('not-comparable')
    expect(comparison.componentPairs).toEqual([])
    expect(comparison.summaries.find(({ id }) => id === 'paired-component-rows')?.maximum).toBeNull()
  })

  it.each([
    ['reordered v1 definitions', (artifact) => { artifact.componentDefinitions.reverse() }],
    ['a duplicate v1 definition ID', (artifact) => { artifact.componentDefinitions[4].id = artifact.componentDefinitions[3].id }],
    ['a changed v1 definition maximum', (artifact) => { artifact.componentDefinitions[0].maxPoints += 1 }],
    ['a missing v1 definition', (artifact) => { artifact.componentDefinitions.pop() }],
    ['an added v1 definition', (artifact) => { artifact.componentDefinitions.push({ id: 'future-sixth', maxPoints: 20 }) }],
  ])('fails closed for %s even when the artifact still claims version 1.0.0', async (_, mutate) => {
    const artifact = structuredClone(PUBLIC_EVIDENCE_PASSPORT_ARTIFACT)
    mutate(artifact)

    const comparison = await compare(
      'maria',
      'eq-pr',
      evidencePassportByEventId('maria'),
      evidencePassportByEventId('eq-pr'),
      artifact,
    )

    expect(comparison.passportCoverage).toBe(0)
    expect(comparison.schemaStatus).toBe('not-comparable')
    expect(comparison.componentPairs).toEqual([])
    expect(comparison.summaries.find(({ id }) => id === 'paired-component-rows')?.maximum).toBeNull()
    expect(comparison.warnings.join(' ')).toMatch(/schema.*withheld/i)
  })

  it('ships four valid curated presets with an explicit non-representative disclaimer', async () => {
    const module = await loadCompareModule()
    expect(module.loadError).toBeUndefined()

    expect(module.PRESET_COMPARISONS).toEqual([
      expect.objectContaining({ id: 'same-storm-two-places', eventIds: ['ian-charlotte', 'ian-fortmyers'] }),
      expect.objectContaining({ id: 'same-hazard-evidence-shift', eventIds: ['irma', 'michael'] }),
      expect.objectContaining({ id: 'same-place-different-hazards', eventIds: ['maria', 'eq-pr'] }),
      expect.objectContaining({ id: 'earthquake-context-boundary', eventIds: ['eq-pr', 'eq-hatay'] }),
    ])
    for (const preset of module.PRESET_COMPARISONS) {
      expect(preset.eventIds).toHaveLength(2)
      expect(new Set(preset.eventIds).size).toBe(2)
      expect(preset.eventIds.every((eventId) => eventById(eventId))).toBe(true)
      expect(preset.note.length).toBeGreaterThan(40)
    }
    expect(module.PRESET_DISCLAIMER).toMatch(/editorial.*not.*representative/i)
  })
})
