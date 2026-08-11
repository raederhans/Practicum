async page => {
  const invocation = await page.evaluate(() => ({
    protocol: location.protocol,
    baseUrl: `${location.origin}${location.pathname}`,
    targetKind: new URLSearchParams(location.search).get('p3Target') ?? 'dashboard',
    runCount: new URLSearchParams(location.search).get('p3Runs') ?? '7',
    networkProfile: new URLSearchParams(location.search).get('p3Profile') ?? 'none',
    measurementScope: new URLSearchParams(location.search).get('p3Scope') ?? 'all',
    viewport: new URLSearchParams(location.search).get('p3Viewport') ?? '1365x768',
    devicePixelRatio: new URLSearchParams(location.search).get('p3Dpr') ?? '1',
    cpuThrottleRate: new URLSearchParams(location.search).get('p3Cpu') ?? '1',
    eventId: new URLSearchParams(location.search).get('p3Event') ?? 'uri-houston',
    routeCycles: new URLSearchParams(location.search).get('p3Cycles') ?? '3',
    basemapSwitches: new URLSearchParams(location.search).get('p3BasemapSwitches') ?? '5',
    failureProfile: new URLSearchParams(location.search).get('p3Failure') ?? 'none',
  }))
  const targetKind = invocation.targetKind
  const runCount = Number(invocation.runCount)
  const baseUrl = invocation.baseUrl
  const networkProfile = invocation.networkProfile
  const measurementScope = invocation.measurementScope
  const viewportMatch = /^(\d+)x(\d+)$/.exec(invocation.viewport)
  const viewport = viewportMatch
    ? { width: Number(viewportMatch[1]), height: Number(viewportMatch[2]) }
    : null
  const devicePixelRatio = Number(invocation.devicePixelRatio)
  const cpuThrottleRate = Number(invocation.cpuThrottleRate)
  const routeCycles = Number(invocation.routeCycles)
  const basemapSwitches = Number(invocation.basemapSwitches)
  const eventId = invocation.eventId
  const failureProfile = invocation.failureProfile
  const quietWindowMs = 750
  const quietTimeoutMs = 15_000
  const navigationTimeoutMs = 30_000

  if (!/^https?:$/.test(invocation.protocol)) {
    throw new Error('Open an HTTP(S) target URL with p3Target and p3Runs query parameters before running the probe')
  }
  if (!Number.isInteger(runCount) || runCount < 3) {
    throw new Error('P3_PERF_RUNS must be an integer of at least 3')
  }
  if (!['dashboard', 'deployment'].includes(targetKind)) {
    throw new Error('P3_PERF_TARGET must be dashboard or deployment')
  }
  if (!['none', 'slow4g'].includes(networkProfile)) {
    throw new Error('P3_PERF_PROFILE must be none or slow4g')
  }
  if (!['all', 'home', 'map'].includes(measurementScope)) {
    throw new Error('P3_PERF_SCOPE must be all, home, or map')
  }
  if (!viewport || viewport.width < 280 || viewport.height < 480) {
    throw new Error('P3_PERF_VIEWPORT must be WIDTHxHEIGHT and at least 280x480')
  }
  if (!Number.isFinite(devicePixelRatio) || devicePixelRatio < 1 || devicePixelRatio > 4) {
    throw new Error('P3_PERF_DPR must be between 1 and 4')
  }
  if (!Number.isFinite(cpuThrottleRate) || cpuThrottleRate < 1 || cpuThrottleRate > 20) {
    throw new Error('P3_PERF_CPU must be between 1 and 20')
  }
  if (!Number.isInteger(routeCycles) || routeCycles < 3 || routeCycles > 5) {
    throw new Error('P3_PERF_CYCLES must be between 3 and 5')
  }
  if (!Number.isInteger(basemapSwitches) || basemapSwitches !== 5) {
    throw new Error('P3_PERF_BASEMAP_SWITCHES must be exactly 5')
  }
  if (!['none', 'external', 'webgl'].includes(failureProfile)) {
    throw new Error('P3_PERF_FAILURE must be none, external, or webgl')
  }

  const normalizedBaseUrl = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`
  const samples = []
  const errors = []
  const browserLogs = []
  let currentNetworkTracker = null

  page.setDefaultTimeout(navigationTimeoutMs)
  await page.setViewportSize(viewport)

  page.on('console', message => {
    if (['warning', 'error'].includes(message.type())) {
      browserLogs.push({
        type: message.type(),
        text: message.text(),
        url: page.url(),
      })
    }
  })
  page.on('pageerror', error => {
    browserLogs.push({ type: 'pageerror', text: error.message, url: page.url() })
  })
  page.on('response', response => {
    if (response.status() >= 400) {
      browserLogs.push({
        type: 'http-error',
        status: response.status(),
        resourceUrl: response.url(),
        url: page.url(),
      })
    }
  })

  const shouldTrackRequest = request => /^https?:/.test(request.url())
  page.on('request', request => {
    if (!currentNetworkTracker || !shouldTrackRequest(request)) return
    currentNetworkTracker.inFlight.add(request)
    currentNetworkTracker.lastActivityAt = Date.now()
  })
  const finishRequest = (request, failed) => {
    if (!currentNetworkTracker || !currentNetworkTracker.inFlight.has(request)) return
    currentNetworkTracker.inFlight.delete(request)
    currentNetworkTracker.lastActivityAt = Date.now()
    if (failed) currentNetworkTracker.failedUrls.push(request.url())
  }
  page.on('requestfinished', request => finishRequest(request, false))
  page.on('requestfailed', request => finishRequest(request, true))

  await page.addInitScript(profile => {
    if (profile === 'webgl') {
      const originalGetContext = HTMLCanvasElement.prototype.getContext
      HTMLCanvasElement.prototype.getContext = function getContext(type, ...args) {
        if (type === 'webgl' || type === 'webgl2' || type === 'experimental-webgl') return null
        return originalGetContext.call(this, type, ...args)
      }
    }
    const state = {
      longTasks: [],
      signals: {},
    }
    window.__p3PerformanceProbe = state

    try {
      const longTaskObserver = new PerformanceObserver(list => {
        for (const entry of list.getEntries()) {
          state.longTasks.push({ startTime: entry.startTime, duration: entry.duration })
        }
      })
      longTaskObserver.observe({ type: 'longtask', buffered: true })
    } catch {
      state.longTaskUnsupported = true
    }

    const detectSignals = () => {
      if (state.signals.bodyAttached == null && document.body) {
        state.signals.bodyAttached = performance.now()
      }
      if (state.signals.homeAttached == null && document.querySelector('.hero__title')) {
        state.signals.homeAttached = performance.now()
      }
      if (state.signals.mapCanvasAttached == null && document.querySelector('.maplibregl-canvas')) {
        state.signals.mapCanvasAttached = performance.now()
      }
      if (state.signals.mapReady == null && document.querySelector('[data-map-ready="true"]')) {
        state.signals.mapReady = performance.now()
      }
      const mapSignalSelectors = {
        mapStyleReady: '[data-map-style-ready="true"]',
        mapOverviewReady: '[data-map-overview-ready="true"]',
        mapDetailReady: '[data-map-detail-ready="true"]',
        mapBasemapRestored: '[data-map-basemap-restored="true"]',
      }
      for (const [name, selector] of Object.entries(mapSignalSelectors)) {
        if (state.signals[name] == null && document.querySelector(selector)) {
          state.signals[name] = performance.now()
        }
      }
      const preview = document.querySelector('img[src*="map_preview.png"]')
      if (state.signals.mapPreviewLoaded == null && preview?.complete && preview.naturalWidth > 0) {
        state.signals.mapPreviewLoaded = performance.now()
      }
    }

    new MutationObserver(detectSignals).observe(document, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: [
        'data-map-ready',
        'data-map-style-ready',
        'data-map-overview-ready',
        'data-map-detail-ready',
        'data-map-basemap-restored',
      ],
    })
    document.addEventListener('DOMContentLoaded', detectSignals, { once: true })
    document.addEventListener('load', event => {
      if (event.target instanceof HTMLImageElement && /map_preview\.png(?:$|\?)/.test(event.target.currentSrc)) {
        state.signals.mapPreviewLoaded = performance.now()
      }
    }, true)
  }, failureProfile)

  const context = page.context()
  const cdp = await context.newCDPSession(page)
  await cdp.send('Network.enable')
  await cdp.send('Performance.enable')
  await cdp.send('HeapProfiler.enable')
  await cdp.send('Emulation.setDeviceMetricsOverride', {
    width: viewport.width,
    height: viewport.height,
    deviceScaleFactor: devicePixelRatio,
    mobile: viewport.width <= 768,
  })
  await cdp.send('Emulation.setCPUThrottlingRate', { rate: cpuThrottleRate })
  const emulatedNetwork = networkProfile === 'slow4g' ? {
    offline: false,
    latency: 150,
    downloadThroughput: 200_000,
    uploadThroughput: 93_750,
    connectionType: 'cellular4g',
  } : null
  if (emulatedNetwork) {
    await cdp.send('Network.emulateNetworkConditions', emulatedNetwork)
  }
  if (failureProfile === 'external') {
    await page.route(/^https:\/\/(?:basemaps\.cartocdn\.com|server\.arcgisonline\.com)\//, route => (
      route.abort('failed')
    ))
  }

  const clearBrowserCache = async () => {
    await cdp.send('Network.setCacheDisabled', { cacheDisabled: false })
    await cdp.send('Network.clearBrowserCache')
  }

  const readCdpMetrics = async () => {
    const result = await cdp.send('Performance.getMetrics')
    return Object.fromEntries(result.metrics.map(metric => [metric.name, metric.value]))
  }

  const readHeapUsage = async ({ collectGarbage = false } = {}) => {
    if (collectGarbage) await cdp.send('HeapProfiler.collectGarbage')
    const usage = await cdp.send('Runtime.getHeapUsage')
    return {
      usedSize: usage.usedSize,
      totalSize: usage.totalSize,
      embedderHeapUsedSize: usage.embedderHeapUsedSize ?? null,
      backingStorageSize: usage.backingStorageSize ?? null,
    }
  }

  const metricDeltaMs = (before, after, name) => {
    const delta = after[name] - before[name]
    return Number.isFinite(delta) && delta >= 0 ? delta * 1000 : null
  }

  const beginNetworkTracking = () => {
    currentNetworkTracker = {
      startedAt: Date.now(),
      lastActivityAt: Date.now(),
      inFlight: new Set(),
      failedUrls: [],
    }
    return currentNetworkTracker
  }

  const waitForNetworkQuiet = async tracker => {
    while (Date.now() - tracker.startedAt < quietTimeoutMs) {
      const now = Date.now()
      if (tracker.inFlight.size === 0 && now - tracker.lastActivityAt >= quietWindowMs) {
        return {
          durationMs: now - tracker.startedAt,
          timedOut: false,
          failedUrls: [...tracker.failedUrls],
        }
      }
      await page.waitForTimeout(100)
    }

    return {
      durationMs: Date.now() - tracker.startedAt,
      timedOut: true,
      inFlightCount: tracker.inFlight.size,
      failedUrls: [...tracker.failedUrls],
    }
  }

  const resetDocumentProbe = async signalNames => page.evaluate(names => {
    const probe = window.__p3PerformanceProbe
    if (!probe) throw new Error('Performance probe init script did not run')
    probe.longTasks = []
    for (const name of names) probe.signals[name] = null
    performance.clearResourceTimings()
    return performance.now()
  }, Array.isArray(signalNames) ? signalNames : [signalNames])

  const collectPageMetrics = async ({ phaseStart, signalName, beforeCdp, beforeHeap, networkQuiet }) => {
    const browserMetrics = await page.evaluate(({ start, signal }) => {
      const probe = window.__p3PerformanceProbe ?? { longTasks: [], signals: {} }
      const navigation = performance.getEntriesByType('navigation')[0]
      const resources = performance.getEntriesByType('resource')
        .filter(entry => entry.startTime >= start)
        .map(entry => ({
          name: entry.name,
          initiatorType: entry.initiatorType,
          startTimeMs: entry.startTime - start,
          durationMs: entry.duration,
          responseStartMs: entry.responseStart - start,
          responseEndMs: entry.responseEnd - start,
          transferSize: entry.transferSize,
          encodedBodySize: entry.encodedBodySize,
          decodedBodySize: entry.decodedBodySize,
        }))

      const findLocalChunk = chunkName => resources.find(entry => (
        new RegExp(`/assets/${chunkName}-[^/]+\\.js(?:$|\\?)`).test(entry.name)
      )) ?? null
      const largestResources = [...resources]
        .sort((left, right) => right.encodedBodySize - left.encodedBodySize)
        .slice(0, 12)
      const longTasks = probe.longTasks.filter(entry => entry.startTime >= start)
      const signalValue = probe.signals[signal]
      const relativeSignals = Object.fromEntries(Object.entries(probe.signals).map(([name, value]) => [
        `${name}Ms`,
        Number.isFinite(value) && value >= start ? value - start : null,
      ]))

      return {
        navigation: navigation ? {
          responseStartMs: navigation.responseStart,
          responseEndMs: navigation.responseEnd,
          domInteractiveMs: navigation.domInteractive,
          domContentLoadedMs: navigation.domContentLoadedEventEnd,
          loadEventEndMs: navigation.loadEventEnd,
          transferSize: navigation.transferSize,
          encodedBodySize: navigation.encodedBodySize,
          decodedBodySize: navigation.decodedBodySize,
        } : null,
        signalMs: Number.isFinite(signalValue) ? signalValue - start : null,
        signalName: signal,
        signals: relativeSignals,
        resources: {
          count: resources.length,
          transferSize: resources.reduce((total, entry) => total + entry.transferSize, 0),
          encodedBodySize: resources.reduce((total, entry) => total + entry.encodedBodySize, 0),
          decodedBodySize: resources.reduce((total, entry) => total + entry.decodedBodySize, 0),
          homeView: findLocalChunk('HomeView'),
          mapView: findLocalChunk('MapView'),
          maplibre: findLocalChunk('maplibre'),
          mapPreview: resources.find(entry => /\/map_preview\.png(?:$|\?)/.test(entry.name)) ?? null,
          largestResources,
          externalHosts: [...new Set(resources.map(entry => new URL(entry.name).host)
            .filter(host => host !== location.host))].sort(),
        },
        mainThread: {
          longTaskCount: longTasks.length,
          longTaskTotalMs: longTasks.reduce((total, entry) => total + entry.duration, 0),
          maxLongTaskMs: Math.max(0, ...longTasks.map(entry => entry.duration)),
          longTasks,
          longTaskUnsupported: Boolean(probe.longTaskUnsupported),
        },
        mapRuntime: (() => {
          const container = document.querySelector('.map-container')
          return {
            canvasCount: document.querySelectorAll('.maplibregl-canvas').length,
            sourceCount: Number(container?.getAttribute('data-map-source-count') ?? 0),
            layerCount: Number(container?.getAttribute('data-map-layer-count') ?? 0),
            styleReady: container?.getAttribute('data-map-style-ready') === 'true',
            overviewReady: container?.getAttribute('data-map-overview-ready') === 'true',
            detailReady: container?.getAttribute('data-map-detail-ready') === 'true',
            basemapRestored: container?.getAttribute('data-map-basemap-restored') === 'true',
          }
        })(),
        documentTitle: document.title,
        url: location.href,
      }
    }, { start: phaseStart, signal: signalName })

    const afterCdp = await readCdpMetrics()
    browserMetrics.mainThread.taskDurationDeltaMs = metricDeltaMs(beforeCdp, afterCdp, 'TaskDuration')
    browserMetrics.mainThread.scriptDurationDeltaMs = metricDeltaMs(beforeCdp, afterCdp, 'ScriptDuration')
    const heapBeforeGc = await readHeapUsage()
    const heapPostGc = await readHeapUsage({ collectGarbage: true })
    browserMetrics.memory = {
      before: beforeHeap,
      afterBeforeGc: heapBeforeGc,
      afterPostGc: heapPostGc,
      usedDeltaBeforeGc: heapBeforeGc.usedSize - beforeHeap.usedSize,
      usedDeltaPostGc: heapPostGc.usedSize - beforeHeap.usedSize,
    }
    browserMetrics.workerCount = page.workers().length
    browserMetrics.networkQuiet = networkQuiet
    return browserMetrics
  }

  const routeUrl = route => `${normalizedBaseUrl}#${route}`

  const runDirectPhase = async ({ scenario, route, cacheState, selector, signalName, iteration }) => {
    await clearBrowserCache()
    await page.goto('about:blank')

    if (cacheState === 'warm') {
      const warmupTracker = beginNetworkTracking()
      await page.goto(routeUrl(route), { waitUntil: 'domcontentloaded' })
      await page.waitForSelector(selector, { state: 'attached' })
      await waitForNetworkQuiet(warmupTracker)
      currentNetworkTracker = null
      await page.goto('about:blank')
    }

    const beforeCdp = await readCdpMetrics()
    const beforeHeap = await readHeapUsage({ collectGarbage: true })
    const tracker = beginNetworkTracking()
    await page.goto(routeUrl(route), { waitUntil: 'domcontentloaded' })
    await page.waitForSelector(selector, { state: 'attached' })
    const networkQuiet = await waitForNetworkQuiet(tracker)
    currentNetworkTracker = null
    const metrics = await collectPageMetrics({
      phaseStart: 0,
      signalName,
      beforeCdp,
      beforeHeap,
      networkQuiet,
    })
    return { scenario, iteration, cacheState, navigationKind: 'direct', metrics }
  }

  const navigateHashAndWait = async ({ route, selector }) => {
    await page.evaluate(nextRoute => { location.hash = nextRoute }, route)
    await page.waitForSelector(selector, { state: 'attached' })
  }

  const runSpaMapPhase = async ({ scenario, cacheState, iteration }) => {
    await clearBrowserCache()
    await page.goto('about:blank')
    const homeTracker = beginNetworkTracking()
    await page.goto(routeUrl('/'), { waitUntil: 'domcontentloaded' })
    await page.waitForSelector('.hero__title', { state: 'attached' })
    await waitForNetworkQuiet(homeTracker)
    currentNetworkTracker = null

    if (cacheState === 'warm') {
      const warmupTracker = beginNetworkTracking()
      await navigateHashAndWait({ route: '/map', selector: '[data-map-overview-ready="true"]' })
      await waitForNetworkQuiet(warmupTracker)
      currentNetworkTracker = null
      await navigateHashAndWait({ route: '/', selector: '.hero__title' })
      await page.waitForFunction(() => !document.querySelector('.maplibregl-canvas'))
    }

    const phaseStart = await resetDocumentProbe([
      'mapCanvasAttached',
      'mapReady',
      'mapStyleReady',
      'mapOverviewReady',
      'mapDetailReady',
      'mapBasemapRestored',
    ])
    const beforeCdp = await readCdpMetrics()
    const beforeHeap = await readHeapUsage({ collectGarbage: true })
    const tracker = beginNetworkTracking()
    await navigateHashAndWait({ route: '/map', selector: '[data-map-overview-ready="true"]' })
    const networkQuiet = await waitForNetworkQuiet(tracker)
    currentNetworkTracker = null
    const metrics = await collectPageMetrics({
      phaseStart,
      signalName: 'mapOverviewReady',
      beforeCdp,
      beforeHeap,
      networkQuiet,
    })
    return { scenario, iteration, cacheState, navigationKind: 'spa', metrics }
  }

  const quantile = (sorted, fraction) => {
    if (!sorted.length) return null
    const index = (sorted.length - 1) * fraction
    const lower = Math.floor(index)
    const upper = Math.ceil(index)
    if (lower === upper) return sorted[lower]
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower)
  }

  const summarizeNumbers = values => {
    const sorted = values.filter(Number.isFinite).sort((a, b) => a - b)
    if (!sorted.length) return null
    const p25 = quantile(sorted, 0.25)
    const p75 = quantile(sorted, 0.75)
    return {
      count: sorted.length,
      median: quantile(sorted, 0.5),
      p95: sorted[Math.ceil(sorted.length * 0.95) - 1],
      min: sorted[0],
      max: sorted[sorted.length - 1],
      p25,
      p75,
      iqr: p75 - p25,
    }
  }

  const valueAt = (sample, path) => path.split('.').reduce((value, key) => value?.[key], sample.metrics)
  const summaryPaths = [
    'navigation.responseStartMs',
    'navigation.responseEndMs',
    'navigation.domInteractiveMs',
    'navigation.domContentLoadedMs',
    'navigation.loadEventEndMs',
    'signalMs',
    'signals.homeAttachedMs',
    'signals.mapPreviewLoadedMs',
    'signals.mapCanvasAttachedMs',
    'signals.mapReadyMs',
    'signals.mapStyleReadyMs',
    'signals.mapOverviewReadyMs',
    'signals.mapDetailReadyMs',
    'signals.mapBasemapRestoredMs',
    'networkQuiet.durationMs',
    'resources.transferSize',
    'resources.mapView.durationMs',
    'resources.mapView.responseEndMs',
    'resources.maplibre.durationMs',
    'resources.maplibre.responseEndMs',
    'resources.maplibre.transferSize',
    'mainThread.longTaskCount',
    'mainThread.longTaskTotalMs',
    'mainThread.maxLongTaskMs',
    'mainThread.taskDurationDeltaMs',
    'mainThread.scriptDurationDeltaMs',
    'memory.usedDeltaBeforeGc',
    'memory.usedDeltaPostGc',
    'workerCount',
    'mapRuntime.sourceCount',
    'mapRuntime.layerCount',
  ]

  const buildSummaries = () => Object.fromEntries(
    [...new Set(samples.map(sample => sample.scenario))].map(scenario => {
      const scenarioSamples = samples.filter(sample => sample.scenario === scenario)
      return [scenario, {
        successfulSamples: scenarioSamples.length,
        networkQuietTimeouts: scenarioSamples.filter(sample => sample.metrics.networkQuiet.timedOut).length,
        metrics: Object.fromEntries(summaryPaths.map(path => [
          path,
          summarizeNumbers(scenarioSamples.map(sample => valueAt(sample, path))),
        ])),
      }]
    }),
  )

  const runAndRecord = async operation => {
    try {
      samples.push(await operation())
    } catch (error) {
      currentNetworkTracker = null
      const mapState = await page.evaluate(() => {
        const container = document.querySelector('.map-container')
        return {
          styleReady: container?.getAttribute('data-map-style-ready') === 'true',
          overviewReady: container?.getAttribute('data-map-overview-ready') === 'true',
          detailReady: container?.getAttribute('data-map-detail-ready') === 'true',
          visibleError: document.querySelector('.map-data-error')?.textContent?.trim() ?? null,
        }
      }).catch(() => null)
      errors.push({
        message: error instanceof Error ? error.message : String(error),
        url: page.url(),
        mapState,
      })
    }
  }

  const collectStressSnapshot = async label => {
    const heap = await readHeapUsage({ collectGarbage: true })
    const runtime = await page.evaluate(() => {
      const container = document.querySelector('.map-container')
      return {
        url: location.href,
        canvasCount: document.querySelectorAll('.maplibregl-canvas').length,
        sourceCount: Number(container?.getAttribute('data-map-source-count') ?? 0),
        layerCount: Number(container?.getAttribute('data-map-layer-count') ?? 0),
        dataError: document.querySelector('.map-data-error')?.textContent?.trim() ?? null,
      }
    })
    return { label, heap, workerCount: page.workers().length, runtime }
  }

  const runLifecycleStress = async () => {
    await clearBrowserCache()
    await page.goto(routeUrl('/'), { waitUntil: 'domcontentloaded' })
    await page.waitForSelector('.hero__title')
    const snapshots = [await collectStressSnapshot('baseline-home')]
    for (let cycle = 1; cycle <= routeCycles; cycle += 1) {
      await navigateHashAndWait({
        route: `/map?event=${encodeURIComponent(eventId)}`,
        selector: '[data-map-detail-ready="true"]',
      })
      snapshots.push(await collectStressSnapshot(`cycle-${cycle}-detail`))
      await navigateHashAndWait({ route: '/', selector: '.hero__title' })
      await page.waitForFunction(() => !document.querySelector('.maplibregl-canvas'))
      await page.waitForFunction(() => document.querySelector('.map-container') === null)
      snapshots.push(await collectStressSnapshot(`cycle-${cycle}-home`))
    }
    const homeSnapshots = snapshots.filter(snapshot => snapshot.label.endsWith('-home'))
    const first = homeSnapshots[0].heap.usedSize
    const last = homeSnapshots.at(-1).heap.usedSize
    return {
      routeCycles,
      snapshots,
      postGcHomeDriftBytes: last - first,
      postGcHomeDriftRatio: first > 0 ? (last - first) / first : null,
      monotonicHomeGrowth: homeSnapshots.every((snapshot, index) => (
        index === 0 || snapshot.heap.usedSize >= homeSnapshots[index - 1].heap.usedSize
      )),
    }
  }

  const runBasemapStress = async () => {
    await page.goto(routeUrl(`/map?event=${encodeURIComponent(eventId)}`), { waitUntil: 'domcontentloaded' })
    await page.waitForSelector('[data-map-detail-ready="true"]')
    const sequence = ['satellite', 'positron', 'voyager', 'dark-nolbl', 'dark']
    const switches = []
    for (const id of sequence) {
      const phaseStart = await resetDocumentProbe('mapBasemapRestored')
      const tracker = beginNetworkTracking()
      const clicked = await page.evaluate(basemapId => {
        const button = document.querySelector(`[data-basemap-id="${basemapId}"]`)
        if (!(button instanceof HTMLButtonElement)) return false
        button.click()
        return true
      }, id)
      if (!clicked) throw new Error(`Could not find basemap control: ${id}`)
      await page.waitForSelector('[data-map-basemap-restored="true"]')
      const networkQuiet = await waitForNetworkQuiet(tracker)
      currentNetworkTracker = null
      const restoredMs = await page.evaluate(start => performance.now() - start, phaseStart)
      switches.push({ id, restoredMs, networkQuiet, snapshot: await collectStressSnapshot(`basemap-${id}`) })
    }
    return { requestedSwitches: basemapSwitches, switches }
  }

  const inspectDeployment = async () => {
    const inspect = async route => {
      await page.goto(routeUrl(route), { waitUntil: 'domcontentloaded' })
      await page.waitForSelector('body')
      await page.waitForTimeout(500)
      return page.evaluate(() => ({
        url: location.href,
        title: document.title,
        h1: [...document.querySelectorAll('h1')].map(element => element.textContent?.trim()),
        bodySnippet: document.body.innerText.slice(0, 800),
        dashboardSignals: {
          homeHero: Boolean(document.querySelector('.hero__title')),
          mapCanvas: Boolean(document.querySelector('.maplibregl-canvas')),
          dashboardText: document.body.innerText.includes('Can We See Generators'),
        },
        links: [...document.querySelectorAll('a[href]')].slice(0, 40).map(anchor => ({
          text: anchor.textContent?.trim(),
          href: anchor.href,
        })),
      }))
    }

    const root = await inspect('/')
    const mapHash = await inspect('/map')
    return {
      root,
      mapHash,
      dashboardEquivalent: root.dashboardSignals.homeHero && mapHash.dashboardSignals.mapCanvas,
    }
  }

  let deploymentInspection = null
  if (targetKind === 'deployment') {
    deploymentInspection = await inspectDeployment()
  }

  const dashboardEquivalent = targetKind === 'dashboard' || deploymentInspection.dashboardEquivalent
  let failureEvidence = null
  if (dashboardEquivalent && failureProfile !== 'none') {
    await clearBrowserCache()
    const tracker = beginNetworkTracking()
    await page.goto(routeUrl(`/map?event=${encodeURIComponent(eventId)}`), { waitUntil: 'domcontentloaded' })
    await page.waitForSelector('body')
    const networkQuiet = await waitForNetworkQuiet(tracker)
    currentNetworkTracker = null
    const mapState = await page.evaluate(() => {
      const container = document.querySelector('.map-container')
      return {
        canvasCount: document.querySelectorAll('.maplibregl-canvas').length,
        styleReady: container?.getAttribute('data-map-style-ready') === 'true',
        overviewReady: container?.getAttribute('data-map-overview-ready') === 'true',
        detailReady: container?.getAttribute('data-map-detail-ready') === 'true',
        visibleError: document.querySelector('.map-data-error')?.textContent?.trim() ?? null,
      }
    })
    await navigateHashAndWait({ route: '/', selector: '.hero__title' })
    failureEvidence = {
      profile: failureProfile,
      observationWindow: networkQuiet,
      mapState,
      recoveredToHome: await page.locator('.hero__title').isVisible(),
    }
  } else if (dashboardEquivalent) {
    for (let iteration = 1; iteration <= runCount; iteration += 1) {
      if (measurementScope !== 'map') {
        await runAndRecord(() => runDirectPhase({
          scenario: 'home-direct-cold', route: '/', cacheState: 'cold',
          selector: '.hero__title', signalName: 'homeAttached', iteration,
        }))
        await runAndRecord(() => runDirectPhase({
          scenario: 'home-direct-warm', route: '/', cacheState: 'warm',
          selector: '.hero__title', signalName: 'homeAttached', iteration,
        }))
      }
      if (measurementScope !== 'home') {
        await runAndRecord(() => runDirectPhase({
          scenario: 'map-overview-direct-cold', route: '/map', cacheState: 'cold',
          selector: '[data-map-overview-ready="true"]', signalName: 'mapOverviewReady', iteration,
        }))
        await runAndRecord(() => runDirectPhase({
          scenario: 'map-overview-direct-warm', route: '/map', cacheState: 'warm',
          selector: '[data-map-overview-ready="true"]', signalName: 'mapOverviewReady', iteration,
        }))
        await runAndRecord(() => runDirectPhase({
          scenario: 'map-detail-direct-cold', route: `/map?event=${encodeURIComponent(eventId)}`,
          cacheState: 'cold', selector: '[data-map-detail-ready="true"]',
          signalName: 'mapDetailReady', iteration,
        }))
        await runAndRecord(() => runDirectPhase({
          scenario: 'map-detail-direct-warm', route: `/map?event=${encodeURIComponent(eventId)}`,
          cacheState: 'warm', selector: '[data-map-detail-ready="true"]',
          signalName: 'mapDetailReady', iteration,
        }))
        await runAndRecord(() => runSpaMapPhase({
          scenario: 'home-to-map-spa-cold', cacheState: 'cold', iteration,
        }))
        await runAndRecord(() => runSpaMapPhase({
          scenario: 'home-to-map-spa-warm', cacheState: 'warm', iteration,
        }))
      }
    }
  } else {
    for (let iteration = 1; iteration <= runCount; iteration += 1) {
      await runAndRecord(() => runDirectPhase({
        scenario: 'deployed-root-direct-cold', route: '/', cacheState: 'cold',
        selector: 'body', signalName: 'bodyAttached', iteration,
      }))
      await runAndRecord(() => runDirectPhase({
        scenario: 'deployed-root-direct-warm', route: '/', cacheState: 'warm',
        selector: 'body', signalName: 'bodyAttached', iteration,
      }))
    }
  }

  let stressEvidence = null
  if (dashboardEquivalent && targetKind === 'dashboard' && failureProfile === 'none') {
    const collectStressEvidence = async (phase, operation) => {
      try {
        return { status: 'completed', evidence: await operation() }
      } catch (error) {
        const failure = {
          phase,
          message: error instanceof Error ? error.message : String(error),
          url: page.url(),
        }
        errors.push(failure)
        return { status: 'failed', failure }
      }
    }
    stressEvidence = {
      lifecycle: await collectStressEvidence('lifecycle-stress', runLifecycleStress),
      basemaps: await collectStressEvidence('basemap-stress', runBasemapStress),
    }
  }

  const browser = context.browser()
  const environment = await page.evaluate(() => ({
    userAgent: navigator.userAgent,
    language: navigator.language,
    timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    devicePixelRatio,
  }))

  await cdp.detach()

  return JSON.stringify({
    schemaVersion: 2,
    measuredAt: new Date().toISOString(),
    targetKind,
    baseUrl: normalizedBaseUrl,
    protocol: {
      requestedRunsPerScenario: runCount,
      viewport,
      devicePixelRatio,
      cpuThrottleRate,
      eventId,
      routeCycles,
      basemapSwitches,
      failureProfile,
      browserName: browser?.browserType().name() ?? 'unknown',
      browserVersion: browser?.version() ?? 'unknown',
      cacheMethod: 'CDP Network.clearBrowserCache; warm measurements follow one unmeasured warm-up',
      networkProfile,
      measurementScope,
      emulatedNetwork,
      quietWindowMs,
      quietTimeoutMs,
      signalLimitations: {
        homeAttached: 'DOM attachment of .hero__title',
        bodyAttached: 'DOM attachment of body for non-equivalent deployment context only',
        mapCanvasAttached: 'DOM attachment of .maplibregl-canvas; this is not MapLibre style load',
        mapReady: 'data-map-ready=true set synchronously inside the active MapLibre load event handler',
        mapStyleReady: 'same MapLibre style load boundary as mapReady, retained under an explicit name',
        mapOverviewReady: 'overview source/layers and interactions installed after style load',
        mapDetailReady: 'selected event sources/layers validated and installed; external tiles may still be settling',
        mapBasemapRestored: 'replacement style plus cached app-owned layers and visibility restored',
        mapPreviewLoaded: 'load completion of map_preview.png when the browser requests it; null means it was not fetched in the measured phase',
        networkQuiet: 'No tracked HTTP(S) request in flight for the quiet window; third-party map traffic may affect it',
        scriptAttribution: 'CDP task/script duration and Long Tasks are page-level, not per-chunk execution attribution',
      },
    },
    environment,
    deploymentInspection,
    samples,
    summaries: buildSummaries(),
    stressEvidence,
    failureEvidence,
    errors,
    browserLogs,
  }, null, 2)
}
