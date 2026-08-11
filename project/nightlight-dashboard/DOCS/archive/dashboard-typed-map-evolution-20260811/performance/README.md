# MapLibre measurement evidence

## Scope and protocol

These measurements describe the local Nightlight Dashboard preview built from this task worktree. They do not describe the current GitHub Pages product, which builds `nightlight-public`, and they are not a production-user performance claim.

- Browser: Playwright Chromium headless, one task-owned browser at a time.
- Preview: production Vite output at `127.0.0.1:54741/Practicum/`.
- Formal matrix: six cells, seven cold plus seven warm samples for overview and detail in every cell; three route cycles; five basemap switches; largest event `uri-houston`.
- Milestones: canvas attachment, style-ready, overview-ready, detail-ready, basemap-restored, and external network settle are retained separately.
- Failures are retained. No timeout was shortened to make a result pass.

## Pre-change formal matrix

The six formal cells retain 266 timing samples in the adjacent `pre-css-*.json` files. Values below are milliseconds. `External detail p50` is network-settle duration, not detail-ready duration.

| Cell | Samples | Timing errors | Overview cold p50 / p95 | Overview warm p50 | Detail cold p50 / p95 | Detail warm p50 | External detail p50 | Max long-task p95 | Lifecycle / basemap stress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Desktop baseline, 1365×768, DPR1 | 56 | 0 | 512 / 1,000 | 252 | 2,408 / 2,445 | 2,139 | 2,790 | 91 | completed / completed |
| Desktop Slow 4G | 42 | 0 | 5,372 / 8,798 | 785 | 9,308 / 9,372 | 950 | 13,295 | 85 | completed / completed |
| Desktop CPU×4 | 42 | 0 | 888 / 4,266 | 905 | 2,952 / 2,992 | 2,977 | 3,401 | 322 | completed / completed |
| 320×640, DPR2, CPU×4, Slow 4G | 42 | 0 | 5,518 / 6,147 | 1,352 | 9,551 / 9,582 | 1,722 | 12,629 | 364 | completed / completed |
| 390×844, DPR3, CPU×4, Slow 4G | 42 | 2 stress errors | 5,604 / 5,703 | 1,595 | 9,590 / 9,689 | 3,732 | 12,924 | 347 | failed / failed |
| 768×1024, DPR2, CPU×4, Slow 4G | 42 | 2 stress errors | 5,604 / 7,531 | 1,556 | 9,603 / 9,700 | 3,768 | 13,601 | 350 | failed / failed |

The 390 and 768 cells completed all 42 formal timing samples but timed out at 30 seconds while independently entering detail mode for lifecycle and basemap stress. The probe was then corrected to preserve those stress failures and already-completed samples instead of throwing away the entire cell.

## Failure and recovery evidence

- External-resource injection: one basemap URL failed; canvas existed but style/overview/detail readiness remained false; navigation recovered to the visible home route.
- WebGL injection: canvas existed but style/overview/detail readiness remained false; navigation recovered to the visible home route.
- External and WebGL profiles contain zero formal timing samples by design.
- Browser logs retain WebGL software-renderer stalls and external HTTP failures. A later diagnostic identified the repeated font `404` as a Google Fonts `woff2` URL, not an app data artifact.

## Threshold decisions

| Boundary | Evidence | Decision |
| --- | --- | --- |
| Route lifecycle / memory | Corrected three-cycle snapshots remove the canvas and report zero sources/layers on home, but the post-GC heap proxy still grows after loading MapLibre and its worker. Completed formal stress cells show 67.0%–93.3% drift; a fresh-page recheck shows cycle 1 to cycle 3 growth of 61.5%. | Threshold hit. No worker clearing, LRU, source eviction, or cache eviction was added. MapLibre documents its shared worker pool as warm-reuse infrastructure and advises clearing prewarmed resources only when the app will not return to a map; this SPA can return. The proxy is not proof of a leak. |
| Long tasks | CPU×4 and mobile pressure cells repeatedly exceed 100 ms, with cell maxima p95 of 322–364 ms. | Threshold hit, but the probe attributes these at page level. MapLibre/script duration is below the 20% worker-tuning admission threshold relative to style-ready p95, so worker tuning is a no-op. |
| Source/layer growth | Stable formal counts are 3 sources / 96 layers for overview and 6 / 104 for detail; home after corrected transition wait is 0 / 0. | No source eviction or LRU change. |
| Basemap switching | Completed cells restore all five requested styles; style-dependent layer counts vary as expected. Throttled 390/768 stress entry failures remain retained. | No lifecycle rewrite or alternative renderer. |
| MapLibre CSS | Before the change, all 14 home cold/warm samples requested the global unpkg CSS. It transferred 0 bytes in this environment but occupied 0–63.5 ms of the resource timeline (median 53.2 ms). After the change, all 14 home samples contain zero MapView, MapLibre, unpkg, or MapLibre CSS requests. | Move CSS to the lazy MapView package import. Treat this primarily as ownership correction because transfer savings were 0 bytes and warm/external timing moved within or against run-to-run noise. |

## Post-change evidence and recovered detail race

The first post-CSS baseline retained 14/14 home samples but only 52/56 total samples. Two further map-only runs retained 41/42 each. All missing samples were warm detail-ready timeouts. The final diagnostic recorded `styleReady=true`, `overviewReady=true`, no visible data error, and a separate Google Fonts `404`.

Static inspection found that `addEventLayers` ran after the map `load` event but rejected the current map when `isStyleLoaded()` temporarily became false while newly added overview sources were still loading. That API reports aggregate source-cache readiness, not whether the current loaded style can accept app-owned sources and layers. Removing only that redundant aggregate gate retained the existing current-map identity guard.

The post-fix desktop map run retains 42/42 samples with zero timing errors:

| Scenario | Samples | Signal p50 / p95 | External settle p50 | Long-task p95 |
| --- | ---: | ---: | ---: | ---: |
| Overview direct cold | 7 | 363.6 / 441.9 | 2,779 | 98 |
| Overview direct warm | 7 | 296.7 / 327.1 | 2,414 | 99 |
| Detail direct cold | 7 | 431.5 / 680.2 | 2,863 | 101 |
| Detail direct warm | 7 | 374.4 / 403.9 | 2,668 | 92 |
| Home-to-map SPA cold | 7 | 357.3 / 375.9 | 2,693 | 92 |
| Home-to-map SPA warm | 7 | 161.7 / 243.4 | 2,274 | 89 |

## Bundle and home resource boundary

- Home manifest closure: 275,855 raw / 67,627 gzip bytes across five local files.
- Map incremental closure: 913,706 raw / 239,907 gzip bytes.
- Route-local MapView CSS: 77,021 raw / 11,405 gzip bytes.
- MapLibre JS: 803,051 raw / 217,871 gzip bytes and still above the existing 500 kB minified warning threshold.
- Pre/post home cold signal p50: 75.7 ms to 37.7 ms; warm p50: 24.6 ms to 29.1 ms. External settle became slightly slower in the post run. These mixed local measurements support the request-boundary correction, not a general user-performance claim.

## Canonical files

- `pre-css-desktop-baseline.json`
- `pre-css-desktop-slow4g.json`
- `pre-css-desktop-cpu4.json`
- `pre-css-320-slow4g-cpu4.json`
- `pre-css-390-slow4g-cpu4.json`
- `pre-css-768-slow4g-cpu4.json`
- `failure-external.json`
- `failure-webgl.json`
- `pre-css-lifecycle-recheck.json`
- `post-css-desktop-baseline.json`
- `post-css-desktop-map-retry.json`
- `post-css-desktop-map-final-attempt.json`
- `post-fix-desktop-map.json`

The incomplete post-CSS runs are intentionally retained alongside the successful post-fix run.
