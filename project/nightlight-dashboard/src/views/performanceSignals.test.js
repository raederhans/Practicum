import { readFile } from 'node:fs/promises'
import { describe, expect, it } from 'vitest'

describe('dashboard runtime performance signals', () => {
  it('defers the below-fold map preview without replacing the real map link', async () => {
    const source = await readFile(new URL('./HomeView.vue', import.meta.url), 'utf8')

    expect(source).toMatch(/<RouterLink[^>]+to="\/map"[\s\S]*?<img[^>]+map_preview\.png[^>]+loading="lazy"/)
    expect(source).toMatch(/ref="mapPreviewContainer"[\s\S]*?<img\s+v-if="mapPreviewVisible"/)
    expect(source).not.toMatch(/\.\$el/)
    expect(source).toMatch(/new IntersectionObserver\([\s\S]*?mapPreviewVisible\.value\s*=\s*true[\s\S]*?rootMargin:\s*'0px'/)
  })

  it('exposes style, overview, detail, and basemap-restored readiness separately', async () => {
    const source = await readFile(new URL('./MapView.vue', import.meta.url), 'utf8')

    expect(source).toMatch(/:data-map-ready="mapReady \? 'true' : 'false'"/)
    expect(source).toMatch(/:data-map-style-ready="mapReady \? 'true' : 'false'"/)
    expect(source).toMatch(/:data-map-overview-ready="overviewReady \? 'true' : 'false'"/)
    expect(source).toMatch(/:data-map-detail-ready="detailReady \? 'true' : 'false'"/)
    expect(source).toMatch(/:data-map-basemap-restored="basemapRestored \? 'true' : 'false'"/)
    expect(source).toMatch(/:data-map-source-count="mapSourceCount"/)
    expect(source).toMatch(/:data-map-layer-count="mapLayerCount"/)
    expect(source).toMatch(/const mapReady\s*=\s*ref\(false\)/)
    expect(source).toMatch(/map\.on\('load',[\s\S]*?mapReady\.value\s*=\s*true[\s\S]*?overviewReady\.value\s*=\s*true/)
    expect(source).toMatch(/tryAddEventLayers[\s\S]*?detailReady\.value\s*=\s*true/)
    expect(source).toMatch(/replacementMap\.on\('load',[\s\S]*?mapReady\.value\s*=\s*true[\s\S]*?basemapRestored\.value\s*=\s*true/)
    expect(source).toMatch(/onUnmounted\([\s\S]*?resetMapPerformanceSignals/)
    expect(source).toContain("import 'maplibre-gl/dist/maplibre-gl.css'")
  })

  it('keeps throttled preview, MapLibre-ready, external-settle, and p95 evidence distinct', async () => {
    const probe = await readFile(new URL('../../scripts/navigation-performance-probe.pw.js', import.meta.url), 'utf8')

    expect(probe).toMatch(/Network\.emulateNetworkConditions/)
    expect(probe).toMatch(/p3Scope/)
    expect(probe).toMatch(/attributeFilter:[\s\S]*?'data-map-ready'[\s\S]*?'data-map-detail-ready'/)
    expect(probe).toMatch(/value\s*>=\s*start/)
    expect(probe).toMatch(/mapPreviewLoaded/)
    expect(probe).toMatch(/data-map-ready/)
    expect(probe).toMatch(/signalName: 'mapOverviewReady'/)
    expect(probe).toMatch(/signalName: 'mapDetailReady'/)
    expect(probe).toMatch(/mapBasemapRestored/)
    expect(probe).toMatch(/HeapProfiler\.collectGarbage/)
    expect(probe).toMatch(/Emulation\.setCPUThrottlingRate/)
    expect(probe).toMatch(/Emulation\.setDeviceMetricsOverride/)
    expect(probe).toMatch(/runLifecycleStress/)
    expect(probe).toMatch(/runBasemapStress/)
    expect(probe).toContain("collectStressEvidence('lifecycle-stress'")
    expect(probe).toContain("collectStressEvidence('basemap-stress'")
    expect(probe).toContain("document.querySelector('.map-container') === null")
    expect(probe).toMatch(/failureProfile === 'external'/)
    expect(probe).toMatch(/profile === 'webgl'/)
    expect(probe).toContain("type: 'http-error'")
    expect(probe).toContain('visibleError:')
    expect(probe).toMatch(/networkQuiet/)
    expect(probe).toMatch(/p95/)
  })
})
