# Dashboard Performance and Architecture Task

## Current status

`ready-for-integration` — product runtime is a measured no-op; reproducible bundle reporting and route-boundary regression coverage are implemented and freshly verified.

## Checklist

- [x] Confirm starting SHA, detached branch state, clean dashboard status, and path restrictions.
- [x] Map router dynamic imports, MapLibre import ownership, and Vite manual chunking.
- [x] Scan listener ownership and protect externally owned ports.
- [x] Capture baseline manifest, exact chunk sizes, warning, and route-loading payloads.
- [x] Evaluate the smallest credible low-risk improvement.
- [x] Run focused tests and final production build.
- [x] Use a newly scanned free high port for route smoke if needed; stop only the exact lane-owned PID.
- [x] Remove temporary logs, manifests/build outputs, and browser/test artifacts before handoff.
- [x] Report final diff, Git state, overlap risk, commands/results, and integration method.

## Validation evidence

| Command or check | Result |
| --- | --- |
| `git status --short --branch` | Initial `## HEAD (no branch)`; no dashboard changes. |
| `git rev-parse HEAD` | `a79921b2becb81388762b10a365744d014026198`. |
| Router/import inspection | Six lazy routes; MapLibre import only in lazy `MapView.vue`; existing `manualChunks.maplibre`. |
| Listener scan | Protected `5174/PID 9956`, `5175/PID 25772`; released `43189` not reused. |
| Fresh preview pre-start scan | At `2026-08-09 21:36:43 +08:00`, dedicated high port `55473` was free; `5174` and `5175` still had their external owners. |
| First `npm ci` attempt | Failed with `EBUSY` because its active log was incorrectly placed inside the directory `npm ci` clears; installed nothing and left no npm process. Log was moved outside `node_modules` before the successful retry. |
| `npm ci` | Exit `0`; 110 packages added, 111 audited, 0 vulnerabilities. npm reported an environment policy notice for the unapproved `esbuild@0.25.12` postinstall script; Vite builds still completed successfully. |
| Baseline `npm test` | Exit `0`; 17/17 tests passed. |
| Baseline `npm run build -- --manifest` | Exit `0`; warning remained; MapLibre `803,051 B` minified / `217,871 B` Vite gzip. |
| No-manual-chunk experiment | Exit `0`; MapView became one `831.05 kB` chunk and the warning remained. Temporary config was removed. |
| `npx vitest run scripts/report-bundle.test.js` | Exit `0`; 3/3 boundary tests passed. |
| `npm test` after change | Exit `0`; 20/20 tests passed across 3 files. |
| `npm run build` after change | Exit `0`; 52 modules transformed; standard production build passed with the unchanged MapLibre warning. |
| `npm run analyze:bundle` | Exit `0`; home `275,707/67,531 B`, direct map `1,107,280/293,110 B`, incremental map `847,202/230,640 B`, MapLibre absent from home. |
| Baseline/final asset comparison | No differences in emitted asset lines, content hashes, or byte sizes. Runtime payload change: exactly zero. |
| Preview HTTP smoke | Exact PID `75140`, port `55473`; `/Practicum/`, Home JS, Map JS, and MapLibre JS returned `200` with expected content types and sizes. PID stopped; port free. |
| Fresh completion gate | `npm test && npm run analyze:bundle` equivalent sequential run exited `0`; 20/20 tests and manifest-based report passed. |
| Cleanup | Exact ignored `dist/` and `node_modules/` targets removed after `git clean -ndX` preview; task-specific temp logs removed; no browser artifacts were created; `55473` remains free. |

## Open risks and remaining work

- MapLibre remains a `803,051 B` minified map-route dependency and keeps Vite's 500 kB warning. Reducing it likely requires a dependency/version or worker-loading decision outside this low-risk lane.
- HTTP/base-path smoke was run, but no real WebGL/browser rendering or throttled-network timing was performed; existing map behavior is covered by the unchanged bundle plus existing unit regressions, not a fresh browser interaction run.
- The analyzer reports transfer-size direction using gzip over emitted files; it does not predict CDN cache hits, HTTP compression configuration, device parse time, or runtime WebGL performance.
- Integration, push, and shared-record updates remain owned by the primary task.
