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

  it('exposes MapLibre load readiness separately from canvas construction and external settle', async () => {
    const source = await readFile(new URL('./MapView.vue', import.meta.url), 'utf8')

    expect(source).toMatch(/:data-map-ready="mapReady \? 'true' : 'false'"/)
    expect(source).toMatch(/const mapReady\s*=\s*ref\(false\)/)
    expect(source).toMatch(/map\.on\('load',[\s\S]*?mapReady\.value\s*=\s*true/)
    expect(source).toMatch(/replacementMap\.on\('load',[\s\S]*?mapReady\.value\s*=\s*true/)
    expect(source).toMatch(/onUnmounted\([\s\S]*?mapReady\.value\s*=\s*false/)
  })

  it('keeps throttled preview, MapLibre-ready, external-settle, and p95 evidence distinct', async () => {
    const probe = await readFile(new URL('../../scripts/navigation-performance-probe.pw.js', import.meta.url), 'utf8')

    expect(probe).toMatch(/Network\.emulateNetworkConditions/)
    expect(probe).toMatch(/p3Scope/)
    expect(probe).toMatch(/attributeFilter:\s*\['data-map-ready'\]/)
    expect(probe).toMatch(/value\s*>=\s*start/)
    expect(probe).toMatch(/mapPreviewLoaded/)
    expect(probe).toMatch(/data-map-ready/)
    expect(probe).toMatch(/signalName: 'mapReady'/)
    expect(probe).toMatch(/networkQuiet/)
    expect(probe).toMatch(/p95/)
  })
})
