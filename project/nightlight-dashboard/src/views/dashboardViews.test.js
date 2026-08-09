import { readFile } from 'node:fs/promises'

import { describe, expect, it } from 'vitest'


const readView = name => readFile(new URL(`./${name}`, import.meta.url), 'utf8')


describe('dashboard view regressions', () => {
  it('separates the public EAGLE-I release from unproven repository derivative lineage', async () => {
    const [overview, detail] = await Promise.all([
      readView('DocsView.vue'),
      readView('DocsDetailView.vue'),
    ])
    const documentation = `${overview}\n${detail}`
    const overviewText = overview.replace(/\s+/g, ' ')
    const detailText = detail.replace(/\s+/g, ' ')

    expect(documentation).not.toMatch(/partner-restricted|partner access required|authorized local/i)
    expect(overviewText).toContain(
      'The official upstream EAGLE-I release is public and licensed under CC BY 4.0.'
    )
    expect(overviewText).toContain(
      "This repository's tracked transformed and event-joined CSV lineage remains unproven"
    )
    expect(detailText).toContain('52 tracked group/merged/with_events CSV derivatives')
    expect(detailText).toContain(
      'their parent release version, transformation chain, and event-join source remain unproven'
    )
    expect(detailText).toContain(
      'This personal/public site does not redistribute those tracked derivatives.'
    )
    expect(detailText).toContain(
      'Reviewed outputs and software checks do not establish upstream lineage or scientific validity.'
    )
  })

  it('sizes a 25-event LOEO chart to 1380px and allows horizontal scrolling', async () => {
    const source = await readView('ChartsView.vue')
    const layout = source.match(
      /Math\.max\((\d+),\s*(\d+)\s*\+\s*activeLoeo\.value\.length\s*\*\s*(\d+)\)/
    )
    const [, minimum, leftOffset, eventSpacing] = layout?.map(Number) ?? []
    const chartWidth = eventCount => Math.max(minimum, leftOffset + eventCount * eventSpacing)

    expect(chartWidth(25)).toBe(1380)
    expect(source).toMatch(/<svg[^>]+:viewBox=/)
    expect(source).toMatch(/\.chart-card\s*\{[^}]*overflow-x:\s*auto/s)
  })

  it('installs the same overview and map interactions for every map instance', async () => {
    const source = await readView('MapView.vue')
    const {
      configureMapInstance,
      configureOverviewInteractions,
      getOverviewLayerIds,
      isMapReadyForDetailLoad,
      styleSupportsLabels,
    } = await import('./MapView.vue')
    const registrations = []
    const fakeMap = {
      on: (...args) => registrations.push(args.slice(0, -1)),
    }
    const handlers = {
      zoom: () => {},
      moveend: () => {},
      click: () => {},
      mousemove: () => {},
      mouseleave: () => {},
    }

    configureMapInstance(fakeMap, handlers)
    configureOverviewInteractions(fakeMap, handlers)

    expect(registrations).toEqual([
      ['zoom'],
      ['moveend'],
      ['click'],
      ['mousemove'],
      ['mouseleave'],
      ['click', 'overview-dots'],
      ['mouseenter', 'overview-dots'],
      ['mouseleave', 'overview-dots'],
      ['click', 'overview-labels'],
      ['mouseenter', 'overview-labels'],
      ['mouseleave', 'overview-labels'],
    ])
    expect(source.match(/installMapInteractions\((?:map|replacementMap)\)/g)).toHaveLength(2)
    expect(source.match(/installOverviewInteractions\((?:map|replacementMap)\)/g)).toHaveLength(2)
    expect(source).toContain('async function tryAddEventLayers(ev, mapInstance = map)')
    expect(source).toContain('if (!mapInstance || map !== mapInstance || !mapInstance.isStyleLoaded()) return false')
    expect(source).toContain('tryAddEventLayers(event, replacementMap)')

    registrations.length = 0
    configureOverviewInteractions(fakeMap, handlers, getOverviewLayerIds({}))
    expect(registrations).toEqual([
      ['click', 'overview-dots'],
      ['mouseenter', 'overview-dots'],
      ['mouseleave', 'overview-dots'],
    ])
    expect(getOverviewLayerIds({ glyphs: 'https://example.test/{fontstack}/{range}.pbf' })).toEqual([
      'overview-dots',
      'overview-labels',
    ])
    expect(styleSupportsLabels({})).toBe(false)
    expect(styleSupportsLabels({ glyphs: 'https://example.test/{fontstack}/{range}.pbf' })).toBe(true)

    const loadingMap = {
      isStyleLoaded: () => false,
      getZoom: () => 10,
    }
    expect(isMapReadyForDetailLoad(loadingMap, loadingMap, 8, false)).toBe(false)
    loadingMap.isStyleLoaded = () => true
    expect(isMapReadyForDetailLoad(loadingMap, loadingMap, 8, false)).toBe(true)
    expect(isMapReadyForDetailLoad(loadingMap, {}, 8, false)).toBe(false)
  })

  it('omits every text layer when the active style has no glyph endpoint', async () => {
    const source = await readView('MapView.vue')
    const facilityLabels = source.slice(
      source.indexOf('// Text layers require a glyph endpoint.'),
      source.indexOf('// ── Toggle a layer group on/off ──')
    )

    expect(facilityLabels).toContain('if (styleSupportsLabels(mapInstance.getStyle()))')
    expect(facilityLabels).toContain('facilities-label-${ev.id}')
  })

  it('locks basemap switching before removing the current map and releases every outcome', async () => {
    const source = await readView('MapView.vue')
    const { canSwitchBasemap, resolveBasemapView } = await import('./MapView.vue')
    const switchBasemap = source.slice(
      source.indexOf('function switchBasemap(id)'),
      source.indexOf('// ── Fly to event ──')
    )

    expect(switchBasemap.indexOf('mapFading.value = true')).toBeGreaterThan(-1)
    expect(switchBasemap.indexOf('mapFading.value = true')).toBeLessThan(switchBasemap.indexOf('previousMap.remove()'))
    expect(switchBasemap.indexOf('map = null')).toBeLessThan(switchBasemap.indexOf('previousMap.remove()'))
    expect(switchBasemap).toContain('releaseMapInstance')
    expect(switchBasemap).toMatch(/catch \(error\)/)
    expect(canSwitchBasemap(null, 'satellite', 'positron', false)).toBe(true)
    expect(canSwitchBasemap(null, 'satellite', 'satellite', false)).toBe(true)
    expect(canSwitchBasemap(null, 'satellite', 'positron', true)).toBe(false)
    expect(resolveBasemapView(null, { center: [-82, 33], zoom: 4 })).toEqual({
      center: [-82, 33],
      zoom: 4,
    })
  })

  it('handles MapLibre errors and clears the map reference during unmount', async () => {
    const source = await readView('MapView.vue')
    const unmount = source.slice(source.indexOf('onUnmounted(() => {'), source.indexOf('// ── Data cache'))

    expect(source).toContain("map.on('error', handleMapError)")
    expect(unmount.indexOf('map = null')).toBeLessThan(unmount.indexOf('mapToRelease?.remove()'))
  })
})
