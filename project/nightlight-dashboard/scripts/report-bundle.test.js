import { describe, expect, it } from 'vitest'

import {
  assertNoExternalMapLibreCss,
  collectManifestFiles,
  resolveRouteBoundaries,
} from './report-bundle.mjs'

const createManifest = () => ({
  '_maplibre.js': {
    file: 'assets/maplibre.js',
    name: 'maplibre',
  },
  'index.html': {
    file: 'assets/index.js',
    isEntry: true,
    css: ['assets/index.css'],
  },
  'src/views/HomeView.vue': {
    file: 'assets/home.js',
    imports: ['index.html'],
    css: ['assets/home.css'],
  },
  'src/views/MapView.vue': {
    file: 'assets/map.js',
    imports: ['index.html', '_maplibre.js'],
    css: ['assets/map.css'],
  },
})

describe('bundle route boundaries', () => {
  it('collects static JavaScript and CSS dependencies for a route', () => {
    const files = collectManifestFiles(createManifest(), ['src/views/HomeView.vue'])

    expect([...files].sort()).toEqual([
      'assets/home.css',
      'assets/home.js',
      'assets/index.css',
      'assets/index.js',
    ])
  })

  it('keeps the isolated MapLibre chunk out of the initial home route', () => {
    const boundaries = resolveRouteBoundaries(createManifest())

    expect([...boundaries.homeFiles]).not.toContain('assets/maplibre.js')
    expect([...boundaries.mapIncrementalFiles].sort()).toEqual([
      'assets/map.css',
      'assets/map.js',
      'assets/maplibre.js',
    ])
  })

  it('fails when MapLibre leaks into the initial home route', () => {
    const manifest = createManifest()
    manifest['src/views/HomeView.vue'].imports.push('_maplibre.js')

    expect(() => resolveRouteBoundaries(manifest)).toThrow(
      'MapLibre leaked into the initial home-route payload',
    )
  })

  it('rejects an external MapLibre stylesheet in the application shell', () => {
    expect(() => assertNoExternalMapLibreCss(`
      <link rel="stylesheet" href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css">
    `)).toThrow('External MapLibre CSS leaked into the application shell')

    expect(() => assertNoExternalMapLibreCss('<link rel="stylesheet" href="/assets/index.css">'))
      .not.toThrow()
  })
})
