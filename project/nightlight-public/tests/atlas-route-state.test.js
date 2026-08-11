import { describe, expect, it } from 'vitest'

import { PRESET_COMPARISONS } from '../src/domain/compareEvents.js'
import { createAtlasRouteStateCodec } from '../src/domain/atlasRouteState.js'
import { EVENTS, HAZARD_FAMILIES } from '../src/content/study.js'

const codec = createAtlasRouteStateCodec({
  events: EVENTS,
  hazardFamilies: HAZARD_FAMILIES,
  presets: PRESET_COMPARISONS,
})

describe('Atlas route-state codec', () => {
  it('hydrates and canonicalizes invalid or repeated query values without preserving unknown fields', () => {
    const state = codec.hydrate({
      mode: ['compare', 'explore'],
      q: `  ${'x'.repeat(90)}  `,
      hazard: 'not-a-hazard',
      event: 'not-an-event',
      view: 'table',
      extra: 'drop-me',
    })

    expect(state).toMatchObject({
      viewMode: 'explore',
      query: 'x'.repeat(78),
      selectedHazardFamily: 'All',
      selectedId: null,
      atlasMobileView: 'list',
    })
    expect(codec.serialize(state)).toEqual({ mode: 'explore', q: 'x'.repeat(78) })
  })

  it('hydrates a preset only when the pair is incomplete and clears a mismatched preset', () => {
    const preset = PRESET_COMPARISONS[0]
    const fromPreset = codec.hydrate({ mode: 'compare', preset: preset.id })
    expect(fromPreset).toMatchObject({
      viewMode: 'compare',
      comparisonLeftId: preset.eventIds[0],
      comparisonRightId: preset.eventIds[1],
      selectedPresetId: preset.id,
    })
    expect(codec.serialize(fromPreset)).toEqual({
      mode: 'compare',
      a: preset.eventIds[0],
      b: preset.eventIds[1],
      preset: preset.id,
    })

    const explicitPair = codec.hydrate({
      mode: 'compare',
      preset: preset.id,
      a: EVENTS[2].id,
      b: EVENTS[3].id,
    })
    expect(explicitPair.selectedPresetId).toBeNull()
    expect(codec.serialize(explicitPair)).toEqual({
      mode: 'compare',
      a: EVENTS[2].id,
      b: EVENTS[3].id,
    })
  })

  it('repairs same-event and unknown peers while keeping an exact canonical compare pair', () => {
    const state = codec.hydrate({ mode: 'compare', a: EVENTS[4].id, b: EVENTS[4].id })
    expect(state.comparisonLeftId).toBe(EVENTS[4].id)
    expect(state.comparisonRightId).not.toBe(EVENTS[4].id)
    expect(EVENTS.some(({ id }) => id === state.comparisonRightId)).toBe(true)
    expect(codec.serialize(state)).toEqual({
      mode: 'compare',
      a: state.comparisonLeftId,
      b: state.comparisonRightId,
    })
  })

  it('distinguishes exact query equality from extra, missing, array, and changed values', () => {
    const expected = { mode: 'explore', event: EVENTS[0].id }
    expect(codec.matches(expected, { event: EVENTS[0].id, mode: 'explore' })).toBe(true)
    expect(codec.matches(expected, { mode: 'explore', event: EVENTS[0].id, extra: '1' })).toBe(false)
    expect(codec.matches(expected, { mode: 'explore' })).toBe(false)
    expect(codec.matches(expected, { mode: ['explore'], event: EVENTS[0].id })).toBe(false)
    expect(codec.matches(expected, { mode: 'compare', event: EVENTS[0].id })).toBe(false)
  })
})
