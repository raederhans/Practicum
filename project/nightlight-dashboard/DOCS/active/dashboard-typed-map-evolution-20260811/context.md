# Context

## Current truth

- Worktree: `C:\Users\raede\.codex\worktrees\2622\Practicum`
- Exclusive implementation/live-test owner: this delegated Dashboard task (`019fef6d-8206-7402-a53f-1c9b5e52edea`).
- Starting revision: exact clean detached `HEAD` at `6b3de4ee97c5391084538bec84db3b1a1f4e05ed`.
- Applicable scope: Dashboard, one new Dashboard-only non-deploy CI workflow, and this task record.
- Dashboard `node_modules` was absent at census start.
- Current Pages deployment builds `nightlight-public`, not this Dashboard; no production-performance claim is permitted.

## Decisions and deviations

| Time | Evidence or decision | Impact |
| --- | --- | --- |
| 2026-08-11 start | Exact baseline matched and worktree was clean. | Writing is authorized inside the delegated scope. |
| 2026-08-11 start | Architecture task recommends local typed repository, threshold-gated Map work, and unchanged Public static boundary. | No full TS migration, shared God service, Public modification, or speculative runtime API. |
| 2026-08-11 start | Map `style load` is earlier than overview/detail readiness. | Formal probe must expose separate milestones instead of treating `map-ready` as fully task-ready. |
| 2026-08-11 start | Python legacy artifacts contain unresolved numeric fallback semantics. | Conformance may classify them, but this lane will not repair or reinterpret scientific values. |
| 2026-08-11 Phase 1 | Broad browser census produced 358 diagnostics; 310 (86.6%) were implicit-`any`/`never` noise and 7 were alias-config errors. Node census produced 8 annotation-only diagnostics. | Fewer than two type-proven behavior defects and more than 70% noise triggered the stop condition. Keep strict checkJs only for loader, timeseries, and report-bundle; do not expand to large SFCs. |
| 2026-08-11 Phase 2 | All 56 JSON artifacts are unversioned. Facility artifacts contain 10 values equal to exporter fallback `0.5`; time series contain 4 paired null rows; probability GeoJSON also has legitimate model values equal to 0.5 with no provenance issue. | Treat all current shapes as `legacy-v0`; attach the facility fallback limitation to the artifact class, preserve paired null separately from zero, and do not infer that every numeric 0.5 is fallback. |
| 2026-08-11 Phase 2 | Scientific ambiguity was sent to coordination task `019febfc-6511-7051-89c9-60970113a4ea`. | Modeling/Data owns any producer change such as `null + reason` or a status field. |
| 2026-08-11 Phase 3 | Schema ambiguity and divergent direct-fetch error semantics justify a local typed repository despite the broad-TS stop. | Added narrow source port, HTTP source, results schema/repository/errors, validated-success cache, and first consumer only. No shared package or service locator. |
| 2026-08-11 science handoff | Modeling commit `c3b18c6613f8c915721021d5e45367f3b0ef5f02` defines producer schema `1.0.0`: only `available` may carry a finite `[0,1]` value; all other controlled statuses carry `null`, reason, and required lineage. Producer artifacts have not yet been generated. | Dashboard dual-read conformance branches explicitly on the v1 object versus legacy array, retains the legacy 0.5 limitation, and does not claim any v1 scientific availability. No Modeling files or values are changed here. |
| 2026-08-11 science handoff | Legacy reader exit requires 25/25 v1 regeneration, Dashboard adapter conformance, and one complete release retaining the legacy reader; the manifest must then prove no v0 production reference before a later major release removes it. | This pilot establishes only schema conformance. It does not start or complete the dual-read exit clock. |
| 2026-08-11 Phase 4 | Six formal pre-change cells retained 266 timing samples. 390/768 completed their formal timing samples but retained lifecycle/basemap stress-entry timeouts. External and WebGL profiles recovered to home without claiming map readiness. | Measurement evidence is canonical under `performance/`; failures and incomplete post-change runs remain present. |
| 2026-08-11 Phase 5 | Corrected transition wait shows zero canvas/source/layer state on home, while the post-GC heap proxy and one shared worker remain above the lifecycle threshold. Official MapLibre guidance makes per-route worker-pool clearing inappropriate for a SPA that can return to the map. | Worker clearing, LRU, source eviction, alternate renderer, and broad lifecycle rewrites are no-ops. |
| 2026-08-11 Phase 5 | The global unpkg CSS request moved to `MapView` package-local CSS. Fourteen post-change home samples request zero MapView/MapLibre/unpkg resources; transfer savings are zero in this environment and timing is mixed. | This is reported primarily as request ownership correction, not a production-user performance improvement. |
| 2026-08-11 Phase 5 | Three post-CSS attempts exposed repeatable warm detail-ready timeouts. `isStyleLoaded()` was incorrectly used after map load as an aggregate source-cache gate. | Removed only that redundant gate, retained current-map identity protection, and verified 42/42 post-fix map samples with zero timing errors. |
| 2026-08-11 Phase 6 | Existing workflow deploys only `nightlight-public`; no Dashboard-equivalent non-deploy CI existed. | Added a separate Ubuntu/Windows Dashboard workflow without changing deploy, Public, base path, or Pages configuration. |

## Live process ownership

| Process | Owner | Log path | State |
| --- | --- | --- | --- |
| Dependency install / short isolated checks | This task | Console output only | Not started |
| Pre/post production builds | This task only | Captured command output; transient runtime logs removed after validation | Completed |
| Dashboard preview server | This task only | Last owner PID `252060`, port `54741` | Stopped; owner process absent and port free |
| Browser / performance matrix | This task only | Canonical retained JSON under `performance/`; transient CLI/raw logs removed | Completed; no direct or named Playwright runner remains |

### Registered live-process contract

- Owner: this Dashboard task only. No other agent may start, poll, retry, stop, or interpret these processes.
- Build command: `npm.cmd run build` from `project/nightlight-dashboard`; shared output `dist/`, package cache `node_modules/`; success is exit `0`, failure is any nonzero exit.
- Preview command: `npm.cmd run preview -- --host 127.0.0.1 --port 54741 --strictPort` from `project/nightlight-dashboard`; exclusive listener `127.0.0.1:54741`; success requires the exact owner PID and HTTP `200` at `/Practicum/`; any pre-existing listener is a blocker, not takeover permission.
- Browser command surface: `npx.cmd --yes --package @playwright/cli playwright-cli --session nl-ts-map-20260811`; Chromium session/cache is task-owned; formal output is retained under the task `performance/` directory.
- Formal cells: desktop baseline `1365x768 DPR1 CPU1`; desktop Slow 4G; desktop CPU4; `320x640 DPR2 CPU4 Slow 4G`; `390x844 DPR3 CPU4 Slow 4G`; `768x1024 DPR2 CPU4 Slow 4G`. Every cell requests seven cold and seven warm overview/detail samples and retains all results. The baseline also measures home.
- Stress/failure: three route cycles, five basemap switches, post-GC heap proxy, worker/long-task/source/layer counts, largest event `uri-houston`, plus separate external-resource and WebGL failure/recovery observations.
- Stop conditions: same command/assumption fails three times; preview owner/port changes; formal result omits requested samples; browser timeout is retained as a failure and is not shortened to produce green; or output would cross task scope.
- Teardown: close the named browser session, stop only the recorded preview PID, confirm port `54741` is free, and remove only task-owned transient runtime output. Formal measurement evidence remains tracked.

## Handoff

No handoff yet. This task may commit within its isolated worktree but must not change main, merge, rebase, cherry-pick, push, deploy, or alter worktree topology.

## Next step

Complete the scoped audit, create and inspect reversible Lore commits, and report the committed integration order without changing main, remotes, or worktree topology.
