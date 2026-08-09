# Dashboard Navigation Performance Context

## Verified starting state

- Worktree: `C:\Users\raede\.codex\worktrees\3a29\Practicum`.
- HEAD: `223fc653dba2768dad99df9d032beaedd9234d6a`.
- Git state: clean detached HEAD.
- Router: Vue hash history with lazy imports for every view.
- MapLibre ownership: statically imported only by lazy `MapView.vue` and isolated by `vite.config.js` as the `maplibre` chunk.
- Reproducible manifest baseline: MapLibre is `803,051 B` minified / `217,871 B` gzip and is absent from the home-route closure.
- The existing Vite `>500 kB` warning is expected and is not itself a performance diagnosis.

## Deployment topology

The repository deployment workflow builds `project/nightlight-public`, not `project/nightlight-dashboard`. The browser run must therefore inspect the live site before treating any route as comparable to the dashboard.

## Live-process contract

| Resource | Planned owner and command | Output/cache ownership | Success condition | Failure/stop condition |
| --- | --- | --- | --- | --- |
| Install | P3: `npm ci` from `project/nightlight-dashboard` | lane-owned `node_modules`; npm's normal user cache is read-only/shared | exit `0` | any non-zero exit; do not retry a repeated identical failure more than twice |
| Build/analyzer | P3: `npm run analyze:bundle` | lane-owned ignored `dist`; temporary command log outside the repository | analyzer exits `0`, reports exact route closures, home excludes MapLibre | non-zero exit or route-boundary contract failure |
| Preview | P3: `"C:\Program Files\nodejs\node.exe" node_modules\vite\bin\vite.js preview --host 127.0.0.1 --port 54731 --strictPort` | exact PID/command line and temporary stdout/stderr logs recorded by P3 | listener belongs to the launched Vite process and `/Practicum/` returns `200` | port ceases to be uniquely owned, process exits, or smoke fails; stop exact verified process only |
| Browser | P3 named Playwright session `p3-navperf-223fc65` | isolated CLI session; lane temp JSON/log output and `output/playwright/dashboard-navigation-performance-20260809` if created | fixed viewport, requested sample count, structured result returned | harness error, browser crash, or repeated scenario failure; close only this named session |

Listener scan at `2026-08-09T22:25:50.8141858+08:00` showed external listeners on ports `5173`, `5174`, `5175`, and `5176`; they are protected and will not be reused or stopped. Port `54731` was free at that scan. P3 must scan it again immediately before preview and select a different currently free high port if ownership changed.

The immediate pre-start scan also found `54731` free. The preview acquired `127.0.0.1:54731` at `2026-08-09T22:32:05.5764510+08:00`; launched PID and listener PID were both `60496`, and `/Practicum/` returned HTTP `200`.

## Measurement limitations

- Localhost measurements minimize network latency and are useful for route/module execution and cache-boundary behavior, not for predicting Internet download time.
- GitHub Pages is affected by CDN, geography, and current network conditions; its public-app timings are contextual unless the same dashboard is deployed.
- Resource Timing exposes fetch/cache timing but does not attribute JavaScript parse/execute cost to one script. CDP task/script deltas and Long Tasks are page-level proxies.
- Canvas attachment occurs before the MapLibre style `load` event. Bounded network quiet can be affected by map tiles, third-party servers, or animations.
