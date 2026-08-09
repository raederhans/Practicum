# Dashboard Performance and Architecture Context

Archive status: analyzer and route-boundary guard integrated; MapLibre warning remains an explicit future decision.

## Current truth

- Worktree: `C:\Users\raede\.codex\worktrees\3a29\Practicum`.
- Branch state: detached HEAD at `a79921b2becb81388762b10a365744d014026198`.
- Initial dashboard path status: clean.
- All six routes are lazy-loaded by `src/router/index.js`.
- `maplibre-gl` is imported only by the lazy `MapView.vue` route.
- `vite.config.js` already isolates `maplibre-gl` through `manualChunks`.
- Before any lane process, listeners included protected external owners at `5174/PID 9956` and `5175/PID 25772`; `43189` was free but is not reused.
- Baseline and final product assets have the same Vite content-hash names and exact byte sizes: home initial navigation is `275,707 B` minified / `67,531 B` gzip including the HTML document; direct map navigation is `1,107,280 B` / `293,110 B`; map navigation after home adds `847,202 B` / `230,640 B`.
- The isolated MapLibre chunk is `803,051 B` minified / `217,871 B` gzip. It is absent from the home-route closure and present only in the map-route closure.
- The Vite 500 kB warning remains. It is a map-route cost, not an initial-home-route leak.

## Decisions and deviations

| Time | Evidence or decision | Impact |
| --- | --- | --- |
| 2026-08-09 +08:00 | Use starting SHA `a79921b2` as the comparison base. | Later umbrella-doc commits are intentionally absent and must not be copied into this lane. |
| 2026-08-09 +08:00 | Measure before editing product/build code. | Avoids optimizing a warning without proving route impact. |
| 2026-08-09 +08:00 | Do not use ports 5174, 5175, or the released P2 port 43189. | Protects other task-owned processes and avoids stale ownership records. |
| 2026-08-09 +08:00 | If MapLibre is already absent from initial navigation, prefer a reproducible measurement/guardrail improvement or a justified product-code no-op. | Prevents unsafe async splitting and fake payload wins. |
| 2026-08-09 21:36:43 +08:00 | Fresh pre-start scan found high port `55473` free; protected ports remained `5174/PID 9956` and `5175/PID 25772`. | Reserve `127.0.0.1:55473` only for the P3 preview smoke; start with `--strictPort`. |
| 2026-08-09 +08:00 | Removing the existing manual chunk produced one `831.05 kB` MapView chunk and retained the warning. | Keep the existing MapLibre isolation; removal does not reduce map-route work. |
| 2026-08-09 +08:00 | Add a no-dependency `analyze:bundle` command plus a pure boundary test. | Makes route payloads reproducible and fails if MapLibre leaks into home without changing runtime bundles. |
| 2026-08-09 +08:00 | Do not move MapLibre into an `onMounted` import or switch to the CSP worker split. | The first only delays the same transfer and adds lifecycle races; the second adds worker URL/deployment behavior without evidence of a net gain. |

## Live process ownership

| Process | Owner | Log path | State |
| --- | --- | --- | --- |
| Baseline Vite bundle builder | P3 dashboard task | task-temporary baseline log; evidence transferred below | exited `0`; no live process |
| Final Vite bundle builder | P3 dashboard task | task-temporary final verification log; evidence transferred below | exited `0`; no live process |
| Final Vite preview / route smoke | P3 dashboard task | task-temporary preview logs; evidence transferred below | exact `node.exe` PID `75140` stopped after HTTP smoke; `127.0.0.1:55473` verified free; external PIDs `9956` and `25772` unchanged |

Shared build output is `project/nightlight-dashboard/dist`. Only this P3 task may start, monitor, interpret, or stop the listed processes. Any PID must be checked against its exact command line before stop.

## Handoff

Product runtime is a measured justified no-op: the current lazy route plus manual MapLibre chunk is already the safest boundary. The integration candidate is an architecture/tooling improvement only: `package.json`, `scripts/report-bundle.mjs`, `scripts/report-bundle.test.js`, and this lane-local record. The integration owner must review and integrate separately; passing tests do not mean merged completion.

## Next step

Integration owner: review the dashboard-only diff from base `a79921b2`, rerun `npm ci && npm test && npm run analyze:bundle`, and integrate the four responsibility groups without importing any shared-record changes from this detached worktree.
