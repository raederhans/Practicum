async page => {
  const invocation = await page.evaluate(() => ({
    protocol: location.protocol,
    baseUrl: `${location.origin}${location.pathname}`,
    targetKind: new URLSearchParams(location.search).get('p3Target') ?? 'dashboard',
    runCount: new URLSearchParams(location.search).get('p3Runs') ?? '7',
  }))
  const targetKind = invocation.targetKind
  const runCount = Number(invocation.runCount)
  const baseUrl = invocation.baseUrl
  const viewport = { width: 1365, height: 768 }
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

  await page.addInitScript(() => {
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
    }

    new MutationObserver(detectSignals).observe(document, { childList: true, subtree: true })
    document.addEventListener('DOMContentLoaded', detectSignals, { once: true })
  })

  const context = page.context()
  const cdp = await context.newCDPSession(page)
  await cdp.send('Network.enable')
  await cdp.send('Performance.enable')

  const clearBrowserCache = async () => {
    await cdp.send('Network.setCacheDisabled', { cacheDisabled: false })
    await cdp.send('Network.clearBrowserCache')
  }

  const readCdpMetrics = async () => {
    const result = await cdp.send('Performance.getMetrics')
    return Object.fromEntries(result.metrics.map(metric => [metric.name, metric.value]))
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

  const resetDocumentProbe = async signalName => page.evaluate(name => {
    const probe = window.__p3PerformanceProbe
    if (!probe) throw new Error('Performance probe init script did not run')
    probe.longTasks = []
    probe.signals[name] = null
    performance.clearResourceTimings()
    return performance.now()
  }, signalName)

  const collectPageMetrics = async ({ phaseStart, signalName, beforeCdp, networkQuiet }) => {
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
        documentTitle: document.title,
        url: location.href,
      }
    }, { start: phaseStart, signal: signalName })

    const afterCdp = await readCdpMetrics()
    browserMetrics.mainThread.taskDurationDeltaMs = metricDeltaMs(beforeCdp, afterCdp, 'TaskDuration')
    browserMetrics.mainThread.scriptDurationDeltaMs = metricDeltaMs(beforeCdp, afterCdp, 'ScriptDuration')
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
    const tracker = beginNetworkTracking()
    await page.goto(routeUrl(route), { waitUntil: 'domcontentloaded' })
    await page.waitForSelector(selector, { state: 'attached' })
    const networkQuiet = await waitForNetworkQuiet(tracker)
    currentNetworkTracker = null
    const metrics = await collectPageMetrics({
      phaseStart: 0,
      signalName,
      beforeCdp,
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
      await navigateHashAndWait({ route: '/map', selector: '.maplibregl-canvas' })
      await waitForNetworkQuiet(warmupTracker)
      currentNetworkTracker = null
      await navigateHashAndWait({ route: '/', selector: '.hero__title' })
      await page.waitForFunction(() => !document.querySelector('.maplibregl-canvas'))
    }

    const phaseStart = await resetDocumentProbe('mapCanvasAttached')
    const beforeCdp = await readCdpMetrics()
    const tracker = beginNetworkTracking()
    await navigateHashAndWait({ route: '/map', selector: '.maplibregl-canvas' })
    const networkQuiet = await waitForNetworkQuiet(tracker)
    currentNetworkTracker = null
    const metrics = await collectPageMetrics({
      phaseStart,
      signalName: 'mapCanvasAttached',
      beforeCdp,
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
      errors.push({
        message: error instanceof Error ? error.message : String(error),
        url: page.url(),
      })
    }
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
  if (dashboardEquivalent) {
    for (let iteration = 1; iteration <= runCount; iteration += 1) {
      await runAndRecord(() => runDirectPhase({
        scenario: 'home-direct-cold', route: '/', cacheState: 'cold',
        selector: '.hero__title', signalName: 'homeAttached', iteration,
      }))
      await runAndRecord(() => runDirectPhase({
        scenario: 'home-direct-warm', route: '/', cacheState: 'warm',
        selector: '.hero__title', signalName: 'homeAttached', iteration,
      }))
      await runAndRecord(() => runDirectPhase({
        scenario: 'map-direct-cold', route: '/map', cacheState: 'cold',
        selector: '.maplibregl-canvas', signalName: 'mapCanvasAttached', iteration,
      }))
      await runAndRecord(() => runDirectPhase({
        scenario: 'map-direct-warm', route: '/map', cacheState: 'warm',
        selector: '.maplibregl-canvas', signalName: 'mapCanvasAttached', iteration,
      }))
      await runAndRecord(() => runSpaMapPhase({
        scenario: 'home-to-map-spa-cold', cacheState: 'cold', iteration,
      }))
      await runAndRecord(() => runSpaMapPhase({
        scenario: 'home-to-map-spa-warm', cacheState: 'warm', iteration,
      }))
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

  const browser = context.browser()
  const environment = await page.evaluate(() => ({
    userAgent: navigator.userAgent,
    language: navigator.language,
    timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    devicePixelRatio,
  }))

  await cdp.detach()

  return JSON.stringify({
    schemaVersion: 1,
    measuredAt: new Date().toISOString(),
    targetKind,
    baseUrl: normalizedBaseUrl,
    protocol: {
      requestedRunsPerScenario: runCount,
      viewport,
      browserName: browser?.browserType().name() ?? 'unknown',
      browserVersion: browser?.version() ?? 'unknown',
      cacheMethod: 'CDP Network.clearBrowserCache; warm measurements follow one unmeasured warm-up',
      quietWindowMs,
      quietTimeoutMs,
      signalLimitations: {
        homeAttached: 'DOM attachment of .hero__title',
        bodyAttached: 'DOM attachment of body for non-equivalent deployment context only',
        mapCanvasAttached: 'DOM attachment of .maplibregl-canvas; this is not MapLibre style load',
        networkQuiet: 'No tracked HTTP(S) request in flight for the quiet window; third-party map traffic may affect it',
        scriptAttribution: 'CDP task/script duration and Long Tasks are page-level, not per-chunk execution attribution',
      },
    },
    environment,
    deploymentInspection,
    samples,
    summaries: buildSummaries(),
    errors,
    browserLogs,
  }, null, 2)
}
