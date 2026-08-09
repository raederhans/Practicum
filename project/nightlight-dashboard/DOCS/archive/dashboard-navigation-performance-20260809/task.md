# Dashboard Navigation Performance Task

## Current status

`integrated` — candidate `90dea17f112fb641d164d1e46f9a7d3d75f1b529` was integrated into `main` as `7db66b9`. The performance/runtime decision remains a justified no-op; routing, MapLibre ownership, Vite/chunk configuration, and emitted runtime assets remain unchanged. The P1-evidence documentation correction is included without claiming a performance improvement or new scientific validation.

## Checklist

- [x] Verify cwd, repository root, detached HEAD, clean Git state, and applicable instructions.
- [x] Map Vite/router/import behavior and preserve the `803,051 B` MapLibre boundary contract.
- [x] Scan listeners and reserve a currently free high-port candidate without touching external owners.
- [x] Define fixed-browser, fixed-viewport, repeated cold/warm protocol and explicit proxy limitations.
- [x] Add and smoke-check the reusable Playwright CLI measurement harness.
- [x] Install dependencies and run the production manifest analyzer.
- [x] Re-scan the dedicated port, start the production preview, verify exact ownership, and run local measurements.
- [x] Inspect and measure the actually deployed GitHub Pages structure without assuming equivalence.
- [x] Decide between a minimal product improvement and justified product no-op from repeated evidence.
- [x] Run final focused/full validation, stop lane processes, and remove generated artifacts.
- [x] Record raw and summarized evidence, final diff, Git state, risks, and integration recommendation.

## Evidence log

| Check | Result |
| --- | --- |
| `Get-Location` / `git rev-parse --show-toplevel` | Both resolve to the rebuilt P3 worktree. |
| `git rev-parse HEAD` | `223fc653dba2768dad99df9d032beaedd9234d6a`. |
| `git status --short --branch` | `## HEAD (no branch)` with no changes. |
| Applicable instructions | Only `C:\Users\raede\.codex\AGENTS.md`; no repository-local override or root `lessons learned.md`. |
| Static route/import map | Six lazy Vue routes; MapLibre is owned by lazy `MapView.vue`; existing manual chunk isolates it. |
| First listener scan | `5173`–`5176` externally owned; candidate `54731` free. |
| First two harness invocations | No samples produced: Playwright CLI's `run-code` sandbox exposes neither Node `process.env` nor the Node `URL` global, despite returning exit `0`. Configuration remains explicit in the invocation URL but is now parsed inside `page.evaluate`, where browser URL APIs are available. |
| Three-run harness smoke | 18/18 local scenario samples succeeded, no harness errors or network-quiet timeouts; fixed headless Chromium `151.0.7922.76`, `1365 x 768`. The smoke exposed non-bundle home resources, so the final probe now retains largest-resource evidence instead of attributing aggregate transfer to MapLibre. |
| `npm ci` | Exit `0`; 110 packages added, 111 audited, 0 vulnerabilities. Existing environment policy notice remains for unapproved `esbuild@0.25.12` install script; the Vite binary ran successfully. |
| `npm run analyze:bundle` | Exit `0`; home `275,707 / 67,531 B`, direct map `1,107,280 / 293,110 B`, incremental map `847,202 / 230,640 B`, isolated MapLibre `803,051 / 217,871 B`. Warning remains. |
| Preview ownership | Immediate scan found `54731` free. Exact Vite PID/listener PID `60496`, bound only to `127.0.0.1:54731`; `/Practicum/` returned `200`. |
| Formal local browser run | 42/42 samples succeeded: six scenarios x seven runs, 0 harness errors, 0 browser warnings/errors, 0 network-quiet timeouts. |
| Formal deployment inspection/run | 14/14 public-root context samples succeeded. Live root is `Nightlight Disaster Observatory`; neither root nor `/#/map` contains dashboard signals, so `dashboardEquivalent=false`. |
| Final `npm test` | Exit `0`; 3/3 test files and 20/20 tests passed. |
| Final `npm run analyze:bundle` | Exit `0`; 52 modules transformed; identical asset hashes/sizes and the same MapLibre warning/boundary report. |
| Documentation-accuracy follow-up | P1 fresh evidence corrected the prior EAGLE-I classification: the official upstream release is public under CC BY 4.0, while the parent release/version, transformation chain, and event-join source for this repository's 52 tracked group/merged/with_events derivatives remain unproven. The personal/public site does not redistribute those tracked derivatives, and reviewed outputs/software checks do not establish upstream lineage or scientific validity. |
| Documentation contract test | Focused RED-to-GREEN coverage added in `dashboardViews.test.js`; final short file run passed 6/6 tests. This is wording/lineage-boundary validation, not performance or scientific validation. |
| Process shutdown | Named Playwright session closed; CLI daemon PID `57884` absent. Preview command/PID/port reverified, exact PID `60496` stopped; `54731` free. |
| Artifact cleanup | Lane-owned `dist`, `node_modules`, `.playwright-cli`, and `C:\Users\raede\AppData\Local\Temp\codex-practicum-p3-navperf-223fc65-20260809-2225` removed. Two broad multi-target cleanup attempts were policy-blocked before execution; four single-target explicit removals then succeeded. |

## Formal measurement summary

All values are milliseconds except transfer bytes. `signal` is `.hero__title` attachment on home, `.maplibregl-canvas` attachment on map, and `body` attachment only for the non-equivalent deployment context. Parentheses show min–max; IQR is reported where it materially describes dispersion.

| Scenario, seven runs | Signal median (range; IQR) | MapLibre response end / fetch duration median | Network quiet median (range; IQR) | Median resource transfer | Median script duration | Long-task median |
| --- | --- | --- | --- | --- | --- | --- |
| Home direct cold | `92.3` (`75.7–170.9`; `75.35`) | absent | `987` (`895–1,188`; `133.5`) | `955,815 B` | `3.139` | `0` |
| Home direct warm | `20.1` (`18.0–21.7`; `1.5`) | absent | `811` (`810–825`; `7`) | `1,800 B` | `2.714` | `0` |
| Map direct cold | `116.7` (`110.4–122.5`; `4.1`) | `93.2 / 21.0` | `4,969` (`1,527–5,585`; `912.5`) | `296,863 B` | `28.767` | `0` |
| Map direct warm | `52.0` (`47.2–54.2`; `5.45`) | `21.2 / 4.1`, cache transfer `300 B` | `955` (`939–2,806`; `1,838.5`) | `2,400 B` | `13.191` | `0` |
| Home to map SPA cold | `44.9` (`43.0–46.9`; `1.05`) | `21.3 / 18.8` | `5,029` (`1,568–5,897`; `1,024`) | `232,440 B` | `24.964` | `0` |
| Home to map SPA warm | `11.7` (`8.9–13.1`; `0.65`) | no new resource entry; module already loaded | `2,732` (`2,700–5,329`; `495.5`) | `300 B` | `11.859` | `0` |

The complete selected per-sample rows are in `measurements.csv`. The two CLI result logs additionally retained resource lists during analysis but are temporary and must be deleted at closeout.

### Deployment context, not a dashboard comparison

- Live root: `https://raederhans.github.io/Practicum/#/`, title `Overview | Nightlight Disaster Observatory`, captured H1 text `Reading recoveryin the dark.`
- Attempted dashboard hash: `https://raederhans.github.io/Practicum/#/map`; no map canvas or dashboard route content appeared.
- Public root cold DCL median `33.2 ms` (seven runs); warm DCL median `8.0 ms`. These values describe a different product and are not used to judge dashboard architecture.

## Decision and confidence

### Performance/runtime justified no-op

No dashboard route, MapLibre import/ownership boundary, Vite/manual-chunk configuration, or emitted performance/runtime asset is changed by the performance phase.

1. The architecture contract already works: home never requests the MapLibre chunk, while cold map navigation requests it exactly once.
2. On local production preview, cold SPA navigation fetched the `217,871 B` compressed MapLibre resource in median `18.8 ms`, completed it by median `21.3 ms`, and attached the canvas by median `44.9 ms`. The chunk is measurable, but it is not the multi-second phase.
3. Map resource settle is the slow and dispersed phase: cold SPA median `5,029 ms` with `1,568–5,897 ms` range. Samples commonly recorded failed CARTO vector-tile requests, while canvas construction and browser logs remained successful. That implicates third-party map-resource behavior/animation, not a safe local chunk-boundary repair.
4. Median Long Tasks were `0 ms` in all six local scenarios. Page-level script-duration proxies were tens of milliseconds, not seconds. The available APIs cannot attribute all execution to MapLibre, but they do not support an architecture rewrite here.
5. Cold home loaded `map_preview.png` eagerly (`885,131 B` encoded; `885,431 B` transfer) and therefore transferred `955,815 B` total. In all cold samples the image began at or after the home DOM signal, so localhost timing does not prove it delays first presentation. Changing loading behavior based only on bytes would violate the task's evidence gate; it is retained as a separately measurable follow-up candidate for throttled-network testing.
6. Removing or reshaping `manualChunks` solely to hide the `803 kB` warning would not address the measured external-resource settle and would weaken the explicit home/map boundary.

Confidence is **high** for bundle ownership, cache behavior, raw transfer, and local chunk timing; **medium** for headless-Chromium canvas-construction timing; and **low/not established** for real-user Internet dashboard timing because the dashboard is not deployed and the app exposes no explicit MapLibre `load` ready signal.

## Documentation accuracy addendum

After the performance measurement completed, P1 supplied fresh evidence that corrected the old EAGLE-I access classification. The official upstream release is public and licensed under CC BY 4.0. That does not prove the lineage of this repository's 52 tracked group/merged/with_events transformed or event-joined CSV derivatives: their parent release/version, transformation chain, and event-join source remain unproven.

The user-visible documentation now states that this personal/public site does not redistribute those tracked derivatives and that reviewed outputs, checksums, artifact hashes, or software tests do not establish upstream lineage or scientific validity. This addendum is a documentation/source-lineage accuracy correction only. It is not a performance optimization, a new data admission decision, or new scientific validation.

## Unverified risks

- Headed browsers, mobile devices, CPU/network throttling, and different GPU/WebGL stacks were not measured.
- The dashboard has no online artifact at the supplied Pages URL, so no CDN dashboard cold/warm comparison exists.
- `.maplibregl-canvas` can precede style completion; bounded network quiet can include failed/retried tiles and ongoing animation requests.
- CDP `TaskDuration`/`ScriptDuration` are page-level deltas, not per-file parse/execute attribution.
- The large home preview image may matter on constrained networks even though it did not delay the local DOM signal; that needs a separately authorized throttled visual/interaction experiment before product change.

## Files and integration boundary

### Measurement implementation

- `scripts/navigation-performance-probe.pw.js`: reusable Playwright CLI probe for fixed-viewport cold/warm direct and SPA route measurements, deployment equivalence inspection, Resource/Navigation Timing, Long Tasks, CDP task/script deltas, network quiet, and median/range/IQR summaries.

### Lane-local evidence and handoff

- `DOCS/archive/dashboard-navigation-performance-20260809/plan.md`: protocol, decision gate, and closeout criteria.
- `DOCS/archive/dashboard-navigation-performance-20260809/context.md`: verified base, architecture/deployment context, limitations, and live-process contract.
- `DOCS/archive/dashboard-navigation-performance-20260809/task.md`: measured report, decision, commands/results, risks, and handoff state.
- `DOCS/archive/dashboard-navigation-performance-20260809/measurements.csv`: 42 local dashboard rows plus 14 non-equivalent deployment-context rows; timing fields rounded to three decimals.

### Documentation accuracy and contract follow-up

- `src/views/DocsView.vue`: replaces the stale partner-restricted/authorized-local summary with the public CC BY 4.0 upstream and unproven tracked-derivative lineage boundary.
- `src/views/DocsDetailView.vue`: applies the same distinction to the detailed data-source, Stage 3, reproducibility, and reference sections without changing scientific results.
- `src/views/dashboardViews.test.js`: locks the corrected user-visible classification and rejects the stale wording; the focused view-test file passes 6/6.

Relative to `223fc653dba2768dad99df9d032beaedd9234d6a`, the total candidate contains eight dashboard-owned files: five new performance measurement/evidence files and three documentation/contract-test modifications. The documentation follow-up changes user-visible source text, so the candidate is no longer accurately described as “product source unchanged”; however, no router, MapView, MapLibre, Vite/chunk, package manifest/lockfile, or performance runtime asset is modified.

The integration owner should preserve two reviewable responsibilities: commit the five performance measurement/evidence additions as one change, and commit the three documentation accuracy/contract files as a separate change. No cherry-pick SHA exists because this lane was forbidden to stage or commit.

Overlap risk remains low and dashboard-local. The five performance paths are new; the three documentation follow-up paths are existing tracked files and therefore have a higher textual-conflict risk if another lane edits the same documentation or view-contract test before integration.

## Open risks

- Headless Chrome may have different WebGL/GPU behavior from a user's headed browser.
- Third-party CARTO/style/tile responses may add variance or fail independently of dashboard code.
- The app exposes no explicit MapLibre `load`/ready marker, so canvas attachment and network quiet must remain clearly labeled proxies.
- The live GitHub Pages artifact may not contain this dashboard at all.

## Integration verification

- Main reran the dashboard suite: 21/21 tests passed.
- Main reran `npm run analyze:bundle`: MapLibre remained isolated from the home closure at `803,051 B` raw / `217,871 B` gzip.
- Main checked the reusable probe syntax and the selected CSV: 56 rows, 42 local, 14 deployment-context, zero network-quiet timeouts, and no home row with MapLibre transfer.
- The combined full Python suite exposed one stale root contract that still required `partner-restricted`; the integration owner repaired that test in `588c9aa`. Release review later caught a different public-repository distribution overstatement in the detailed callout; `c72e26e` now distinguishes Pages packaging from public Git tracking. The final full suite passed 112 tests plus 7 subtests.
