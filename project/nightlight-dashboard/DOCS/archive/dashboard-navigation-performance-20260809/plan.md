# Dashboard Navigation Performance Plan

## Objective

Measure production-build home and map-route loading in a real Chromium browser, distinguish cold and warm behavior across repeated runs, and decide whether the existing isolated `803,051 B` MapLibre chunk creates a user-visible bottleneck with a safe, minimal remedy.

## Scope and ownership

- Owner: P3 dashboard performance lane.
- Writable scope: `project/nightlight-dashboard/**` only.
- Base: detached `223fc653dba2768dad99df9d032beaedd9234d6a`.
- This lane is not the integration owner and will not change the Git index, refs, branches, worktree topology, or remotes.
- Product code changes are allowed only if repeated browser evidence shows a clear bottleneck and a low-risk fix preserves current behavior and project conventions.

## Protocol

1. Keep Vite's production build, hash router, existing route-level dynamic imports, and isolated `maplibre` manual chunk unchanged for the baseline.
2. Use one isolated Playwright CLI session with Chrome's Chromium engine and a fixed `1365 x 768` viewport.
3. Run seven measured samples per local scenario:
   - direct home, cold browser HTTP cache;
   - direct home, warm browser HTTP cache;
   - direct map, cold browser HTTP cache;
   - direct map, warm browser HTTP cache;
   - home-to-map SPA navigation with MapLibre cold;
   - repeated home-to-map SPA navigation with MapLibre/module cache warm.
4. Clear the browser HTTP cache before each cold sample and before each warm-up/measurement pair. Warm samples are measured only after an unmeasured warm-up under the same browser session.
5. Capture Navigation Timing, resource timing for route and MapLibre chunks, CDP task/script-duration deltas, Long Tasks, a DOM attachment signal, and bounded network quiet. Report median, min/max, p25/p75, and IQR; do not use a single run as the conclusion.
6. Treat `.maplibregl-canvas` attachment only as a map-construction proxy. Treat network quiet only as a resource-settle proxy. Neither is the MapLibre `load` event, because the application does not expose its map instance or an explicit ready marker.
7. Inspect the live GitHub Pages root and `/#/map`. Run equivalent dashboard scenarios only if the deployed structure is actually the dashboard; otherwise record non-equivalence and measure only the reachable public root as contextual evidence.
8. Compare browser evidence with the manifest bundle contract: MapLibre remains exactly isolated from home, and direct/incremental map payloads remain reproducible.

## Decision gate

- Modify product code only when the repeated results identify a stable, meaningful user-visible delay and a small change can directly reduce it without loading MapLibre on home, weakening behavior, or gaming Vite's warning.
- Otherwise retain product output and deliver the measurement harness plus an evidence-backed justified no-op.

## Validation and closeout

- Run the focused harness smoke, production bundle analyzer, full dashboard tests, local production preview measurements, and live-site structure check.
- Stop only the preview/browser processes created and positively identified by this lane.
- Remove lane-owned `dist`, `node_modules`, browser output, logs, and other temporary artifacts after evidence is captured.
- Finish with a diff and clean ownership report relative to `223fc653`.
