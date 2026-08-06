import { describe, expect, it } from 'vitest'

import { evidencePassportByEventId } from '../src/content/evidencePassportArtifact.js'
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

describe('Atlas event comparison rules', () => {
  it('repairs a duplicate peer with a category-first alternative and preserves a valid peer', async () => {
    const module = await loadCompareModule()
    expect(module.loadError).toBeUndefined()

    expect(module.resolveComparisonPeerId(EVENTS, 'maria', 'maria')).toBe('irma')
    expect(module.resolveComparisonPeerId(EVENTS, 'maria', 'eq-pr')).toBe('eq-pr')
    for (const event of EVENTS) {
      const firstRepair = module.resolveComparisonPeerId(EVENTS, event.id, event.id)
      const secondRepair = module.resolveComparisonPeerId(EVENTS, event.id, event.id)
      expect(firstRepair).not.toBe(event.id)
      expect(secondRepair).toBe(firstRepair)
      if (EVENTS.some((candidate) => candidate.id !== event.id && candidate.type === event.type)) {
        expect(eventById(firstRepair).type).toBe(event.type)
      }
    }
  })

  it('puts hazard family before broad region and evidence availability in its compatibility language', async () => {
    const module = await loadCompareModule()
    expect(module.loadError).toBeUndefined()

    const sameContext = module.buildEventComparison(
      eventById('irma'),
      eventById('michael'),
      evidencePassportByEventId('irma'),
      evidencePassportByEventId('michael'),
    )
    expect(sameContext.compatibility.id).toBe('category-region-aligned')
    expect(sameContext.compatibility.label).toMatch(/hazard.*region/i)

    const crossHazard = module.buildEventComparison(
      eventById('maria'),
      eventById('eq-pr'),
      evidencePassportByEventId('maria'),
      evidencePassportByEventId('eq-pr'),
    )
    expect(crossHazard.compatibility.id).toBe('cross-hazard')
    expect(crossHazard.warnings.join(' ')).toMatch(/hazard famil/i)

    const internationalBoundary = module.buildEventComparison(
      eventById('eq-pr'),
      eventById('eq-hatay'),
      evidencePassportByEventId('eq-pr'),
      evidencePassportByEventId('eq-hatay'),
    )
    expect(internationalBoundary.compatibility.id).toBe('category-aligned')
    expect(internationalBoundary.warnings.join(' ')).toMatch(/international context/i)
  })

  it('returns independent dynamic summaries without inventing a total or ranking', async () => {
    const module = await loadCompareModule()
    expect(module.loadError).toBeUndefined()

    const comparison = module.buildEventComparison(
      eventById('maria'),
      eventById('eq-pr'),
      evidencePassportByEventId('maria'),
      evidencePassportByEventId('eq-pr'),
    )
    const summaries = Object.fromEntries(comparison.summaries.map((summary) => [summary.id, summary]))

    expect(summaries['reviewed-passports'].value).toBe(2)
    expect(summaries['comparable-components'].value).toBe(5)
    expect(summaries['exact-published-values'].value).toBe(4)
    expect(summaries['different-published-values'].value).toBe(1)
    expect(comparison.componentPairs).toHaveLength(5)
    expect(JSON.stringify(comparison)).not.toMatch(/total.?score|leaderboard|recovery.?score|rank(?:ing)?/i)
  })

  it('keeps component comparison unavailable when either event lacks a reviewed Passport', async () => {
    const module = await loadCompareModule()
    expect(module.loadError).toBeUndefined()

    const comparison = module.buildEventComparison(
      eventById('maria'),
      eventById('matthew-jax'),
      evidencePassportByEventId('maria'),
      evidencePassportByEventId('matthew-jax'),
    )
    const evidenceOnlySummaries = comparison.summaries.filter(({ id }) => id !== 'reviewed-passports')

    expect(comparison.passportCoverage).toBe(1)
    expect(evidenceOnlySummaries.every(({ value }) => value === null)).toBe(true)
    expect(comparison.componentPairs).toEqual([])
    expect(comparison.warnings.join(' ')).toMatch(/1 of 2.*reviewed Evidence Passport/i)
  })

  it('supports all 600 directed pairs in the public 25-event index and rejects self-comparison', async () => {
    const module = await loadCompareModule()
    expect(module.loadError).toBeUndefined()

    let pairCount = 0
    for (const left of EVENTS) {
      expect(() => module.buildEventComparison(
        left,
        left,
        evidencePassportByEventId(left.id),
        evidencePassportByEventId(left.id),
      )).toThrow(/distinct events/i)

      for (const right of EVENTS) {
        if (left.id === right.id) continue
        const comparison = module.buildEventComparison(
          left,
          right,
          evidencePassportByEventId(left.id),
          evidencePassportByEventId(right.id),
        )
        pairCount += 1
        expect(comparison.summaries).toHaveLength(4)
        expect(comparison.summaries.every(({ value }) => value === null || Number.isFinite(value))).toBe(true)
        expect(comparison.warnings.length).toBeGreaterThan(0)

        const reverse = module.buildEventComparison(
          right,
          left,
          evidencePassportByEventId(right.id),
          evidencePassportByEventId(left.id),
        )
        expect(reverse.summaries.map(({ id, value, maximum }) => ({ id, value, maximum }))).toEqual(
          comparison.summaries.map(({ id, value, maximum }) => ({ id, value, maximum })),
        )
      }
    }
    expect(pairCount).toBe(600)
  })

  it('ships four valid curated presets with authored notes', async () => {
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
  })
})
