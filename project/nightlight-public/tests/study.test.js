import { describe, expect, it } from 'vitest'

import { EVENTS, STUDY_SUMMARY } from '../src/content/study.js'
import { filterEvents } from '../src/domain/filterEvents.js'
import { projectPoint } from '../src/domain/projectPoint.js'
import { resolveSelectedId } from '../src/domain/resolveSelectedId.js'

const allowedEventKeys = ['id', 'name', 'year', 'location', 'region', 'type', 'center']

describe('public study facts', () => {
  it('publishes only the approved aggregate facts', () => {
    expect(STUDY_SUMMARY).toEqual({
      stage2: { events: 25, jurisdictions: 17 },
      stage3: { observations: 1002, events: 22, states: 15 },
      descriptiveModel: { rSquared: 0.7603, adjustedRSquared: 0.7543, n: 977 },
      descriptiveSensitivity: 0.551,
    })
  })

  it('keeps atlas events at event-level resolution', () => {
    expect(EVENTS.map((event) => event.id)).toEqual([
      'maria',
      'irma',
      'ida',
      'laura',
      'michael',
      'eq-pr',
      'ian-charlotte',
      'ian-fortmyers',
      'eq-hatay',
      'matthew-jax',
      'florence-wilm',
      'zeta-atlanta',
      'zeta-birmingham',
      'isaias-nj',
      'irma-savannah',
      'matthew-nc',
      'florence-sc',
      'isaias-ny',
      'uri-houston',
      'derecho-chicago',
      'severe-detroit',
      'noreaster-boston',
      'icestorm-okc',
      'severe-nashville',
      'atmos-seattle',
    ])
    for (const event of EVENTS) {
      expect(Object.keys(event).sort()).toEqual([...allowedEventKeys].sort())
      expect(event.center).toHaveLength(2)
      for (const coordinate of event.center) {
        expect(Number.isFinite(coordinate)).toBe(true)
        expect(Number(coordinate.toFixed(1))).toBe(coordinate)
      }
      expect(JSON.stringify(event)).not.toMatch(/facility|probab|time.?series|grid|zip/i)
    }
  })
})

describe('atlas filtering', () => {
  it('filters by type and case-insensitive query without mutating the source', () => {
    const source = EVENTS.slice()
    const filtered = filterEvents(EVENTS, { type: 'Hurricane', query: 'maria' })

    expect(filtered).toHaveLength(1)
    expect(filtered[0].name).toContain('Maria')
    expect(EVENTS).toEqual(source)
  })

  it('returns all events for empty filters', () => {
    expect(filterEvents(EVENTS, { type: 'All', query: '' })).toEqual(EVENTS)
  })
})

describe('atlas projection', () => {
  it('projects the public one-decimal center into the SVG field', () => {
    expect(projectPoint([-170, 72])).toEqual([48, 48])
    expect(projectPoint([45, 16])).toEqual([912, 492])
    expect(projectPoint([-95.4, 29.8])).toEqual([347.8, 382.6])
    expect(projectPoint([36.2, 36.2])).toEqual([876.6, 331.8])
  })
})

describe('atlas selection state', () => {
  const visible = [{ id: 'maria' }, { id: 'irma' }]

  it('keeps a selected event that remains visible', () => {
    expect(resolveSelectedId(visible, 'irma')).toBe('irma')
  })

  it('moves selection to the first visible event when a filter hides the old one', () => {
    expect(resolveSelectedId(visible, 'ida')).toBe('maria')
  })

  it('clears selection when a filter has no results', () => {
    expect(resolveSelectedId([], 'maria')).toBeNull()
  })
})
