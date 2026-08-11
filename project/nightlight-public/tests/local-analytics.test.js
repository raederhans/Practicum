import { describe, expect, it } from 'vitest'

import {
  LOCAL_ANALYTICS_CONSENT_VERSION,
  LOCAL_ANALYTICS_MAX_EVENTS,
  createLocalAnalyticsEvent,
} from '../src/domain/localAnalyticsContract.js'
import { createLocalResearchLog } from '../src/lib/localResearchAnalytics.js'

function memoryStorage(initial = {}) {
  const values = new Map(Object.entries(initial))
  const writes = []
  return {
    writes,
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => {
      writes.push({ key, value })
      values.set(key, value)
    },
    removeItem: (key) => values.delete(key),
  }
}

describe('local opt-in analytics contract', () => {
  it('records nothing and writes nothing before explicit opt-in', () => {
    const storage = memoryStorage()
    const log = createLocalResearchLog({ storage })

    expect(log.recordSurfaceViewed('overview')).toBe(false)
    expect(log.recordAtlasModeSelected('explore')).toBe(false)
    expect(log.snapshot()).toMatchObject({ consent: 'not_granted', count: 0, events: [] })
    expect(log.exportSnapshot()).toBeNull()
    expect(storage.writes).toEqual([])
  })

  it('keeps an allowlisted, identifier-free, timestamp-free session and supports export and deletion', () => {
    const storage = memoryStorage()
    const log = createLocalResearchLog({ storage })

    log.grantConsent()
    expect(log.recordSurfaceViewed('atlas')).toBe(true)
    expect(log.recordAtlasModeSelected('compare')).toBe(true)

    expect(log.snapshot()).toMatchObject({ consent: 'granted', count: 2 })
    expect(log.exportSnapshot()).toEqual({
      schemaVersion: 1,
      consentVersion: LOCAL_ANALYTICS_CONSENT_VERSION,
      localOnly: true,
      retention: 'current-tab-until-clear-or-close',
      researchQuestions: expect.any(Array),
      events: [
        {
          ordinal: 1,
          name: 'surface_viewed',
          researchQuestionId: 'rq-surface-navigation',
          properties: { surface: 'atlas' },
        },
        {
          ordinal: 2,
          name: 'atlas_mode_selected',
          researchQuestionId: 'rq-atlas-mode',
          properties: { mode: 'compare' },
        },
      ],
    })
    expect(JSON.stringify(log.exportSnapshot())).not.toMatch(/timestamp|user.?agent|ip|identifier|location|query|model.?input/i)

    log.clearEvents()
    expect(log.snapshot()).toMatchObject({ consent: 'granted', count: 0 })
    log.withdrawConsent()
    expect(log.snapshot()).toMatchObject({ consent: 'not_granted', count: 0 })
  })

  it('rejects custom events, free-form properties, extra properties, and out-of-range values', () => {
    expect(() => createLocalAnalyticsEvent('custom_event', {}, 1)).toThrow(/not allowlisted/i)
    expect(() => createLocalAnalyticsEvent('surface_viewed', { surface: 'atlas', note: 'free text' }, 1)).toThrow(/properties/i)
    expect(() => createLocalAnalyticsEvent('surface_viewed', { surface: 'precise-place' }, 1)).toThrow(/not allowlisted/i)
    expect(() => createLocalAnalyticsEvent('atlas_mode_selected', { mode: 'rank' }, 1)).toThrow(/not allowlisted/i)
    expect(() => createLocalAnalyticsEvent('surface_viewed', { surface: 'atlas' }, LOCAL_ANALYTICS_MAX_EVENTS + 1)).toThrow(/bounded session range/i)
  })

  it('fails closed and removes a stale consent/schema envelope', () => {
    const storage = memoryStorage({
      'nightlight.local-research-log.v1': JSON.stringify({
        schemaVersion: 1,
        consent: { state: 'granted', version: 'obsolete' },
        events: [],
      }),
    })
    const log = createLocalResearchLog({ storage })

    expect(log.snapshot()).toMatchObject({ consent: 'not_granted', count: 0 })
    expect(log.exportSnapshot()).toBeNull()
  })
})
