# P2/P3/Actions Execution Context

## Current truth

- Repository/worktree: isolated Codex worktree at the task path; absolute local path is intentionally not a product claim.
- HEAD/base: `ca8292040a402eae1d2e461708a4cc912867efcb`.
- Git state at start: detached HEAD, clean index and working tree.
- Repository-local `AGENTS.md` and root `lessons learned.md`: absent; the supplied top-level instructions apply.
- Shared worktree registry was read only. It identifies the prior P2/P3 release as integrated and leaves independent proxy evidence, throttled preview measurement, and Actions runtime maintenance as follow-up work.
- Agent role: independent execution owner, not integration owner. Git index/refs/branches/remotes/worktree topology and the registry are read-only.
- Writable ownership: the task-record directory, both Nightlight applications, their directly corresponding tests, and `.github/workflows/deploy-dashboard.yml` only.
- Existing measured follow-up: `project/nightlight-dashboard/public/map_preview.png` is `885,131 B` and was eagerly fetched on cold home loads; prior localhost evidence did not prove a first-presentation delay.

## Decisions and deviations

| Time | Evidence or decision | Impact |
| --- | --- | --- |
| 2026-08-10 +08:00 | Exact requested base is checked out and initially clean. | Evaluation and diff can remain pinned to `ca8292040`; no branch operation is needed. |
| 2026-08-10 +08:00 | The prior P2 plan explicitly permits technical and content proxy evidence but prohibits participant/usability/WCAG-conformance claims and self-assigned CDC scores. | The current phase will execute the proxy plan and retain the same strict claim boundary. |
| 2026-08-10 +08:00 | The prior P3 phase isolated MapLibre and measured multi-second external settle, but did not use headed/network-throttled runs and left the eager preview image unresolved. | This phase measures preview, MapLibre ready, and external settle separately before considering any product edit. |
| 2026-08-10 +08:00 | Official GitHub guidance says Node 20 action runtimes are deprecated; hosted runners began forcing Node 24 by default on 2026-06-16, while `setup-node`'s `node-version` input controls the application's Node separately. | Action runtime pins and the app test runtime must not be conflated; only official compatible action upgrades are candidates. |
| 2026-08-10 +08:00 | Ownership was narrowed before product edits: public route/view/component/style/router/app/general infrastructure became reserved for other tasks. | This task made no public product-source edit; its P2 assertions were merged into the existing P2 `proxy-evaluation.test.js`. |
| 2026-08-10 +08:00 | Native `loading=lazy` still fetched the 885 KB preview in 7/7 cold samples. Explicit viewport intersection avoided the request in 7/7 and reduced cold settle median from 6,843 ms to 2,390 ms. | Retain only the viewport-deferred candidate; do not claim p95 improvement because one retained candidate outlier set p95 to 6,733 ms. |
| 2026-08-10 +08:00 | Corrected map runs separated canvas, MapLibre `load`, and external settle. | Map ready is instrumented as `data-map-ready`; external/CARTO settle remains a noisy, separate proxy. |
| 2026-08-10 +08:00 | Official action releases and immutable tags support checkout 5.0.1, setup-node 5.0.0, upload-pages-artifact 5.0.0, and deploy-pages 5.0.0. | Update only the four full-SHA pins; keep project Node 20 input and do not deploy. |

## Live process ownership

This task explicitly uses the `$orchestrate-live-tests` single-owner contract. The primary agent in this task is the only owner of public/dashboard dependency installs, builds, `dist`, preview servers, headed Chromium/CDP sessions, throttling profiles, temporary browser caches, raw logs, and performance summaries. No other process may start, poll, retry, stop, or interpret these resources during the run.

| Process | Owner | Full command and cwd | Port/cache/output/log | Success signal | Failure/stop/cleanup |
| --- | --- | --- | --- | --- | --- |
| Public dependency restore and complete gate | This task | `npm ci`, then `npm run validate` from `project/nightlight-public` | npm cache is user-shared read/write by npm; output `project/nightlight-public/node_modules` and `dist`; command output captured into this record | install exit `0`; expected test count passes; build and required public boundary exit `0` | Stop after first actionable failure; no three identical retries; remove owned `node_modules` and `dist` after final evidence |
| Dashboard dependency restore/build/gates | This task | `npm ci`, `npm test`, `npm run analyze:bundle`, and final `npm run build` from `project/nightlight-dashboard` | output `project/nightlight-dashboard/node_modules`, `dist`, and `.vite`; command output captured into this record | exit `0`; existing bundle boundary remains explicit; product test suite passes | Stop after first actionable failure; no three identical retries; remove owned `node_modules`, `dist`, and `.vite` after final evidence |
| Public production preview | This task | `node node_modules/vite/bin/vite.js preview --host 127.0.0.1 --port 43241 --strictPort` from `project/nightlight-public` | `127.0.0.1:43241`; log `DOCS/active/p2-p3-solo-evidence-performance-20260810/.runtime/public-preview.log` | exact owned listener and HTTP 200; P2 browser matrix completes | Stop exact positively identified PID after P2 browser work or immediately on ownership mismatch; verify port free; delete `.runtime` |
| Dashboard production preview | This task | `node node_modules/vite/bin/vite.js preview --host 127.0.0.1 --port 43242 --strictPort` from `project/nightlight-dashboard` | `127.0.0.1:43242`; log `DOCS/active/p2-p3-solo-evidence-performance-20260810/.runtime/dashboard-preview.log` | exact owned listener and HTTP 200; all P3 samples complete | Stop exact positively identified PID after measurements or immediately on ownership mismatch; verify port free; delete `.runtime` |
| P2 headed Chromium browser | This task | Playwright CLI isolated session against `http://127.0.0.1:43241/` | isolated session name `practicum-p2-solo-20260810`; temporary browser profile/output under task `.runtime` when exposed by CLI | all route/viewport/keyboard/focus/console/network assertions are recorded | close owned session after checks; delete snapshots/traces/cache; do not touch unrelated browser processes |
| P3 headed throttled Chromium experiment | This task | One isolated Chromium launched headed; CDP `Network.emulateNetworkConditions` applies the recorded profile; probe runs serial cold/warm samples against `http://127.0.0.1:43242/` | isolated session/profile; raw results in task `.runtime`; durable selected rows in `measurements.csv` | each scenario reaches its explicit DOM/image/MapLibre/external-settle signal for at least seven valid samples; failures/timeouts remain data | stop on target/ownership mismatch or three repeated identical harness failures; close browser; delete profile, raw/transient logs, snapshots, traces, and `.runtime` after transferring durable evidence |

Reserved-port preflight on 2026-08-10: `43241`, `43242`, `43243`, and `43244` had no listeners. The task reserves `43241` and `43242`; it will recheck immediately before each launch.

## Handoff

Current state: `ready-for-integration`. The process contract was executed by one owner. Public preview PID `25144`, Dashboard preview PIDs `77288`, `81064`, `55868`, and `49516`, and the task's two named browser sessions were stopped/closed after their phases. Ports `43241` and `43242` were verified free. The task-owned public/dashboard `node_modules`, `dist`, and task `.runtime` were removed after durable evidence transfer. An unrelated browser session named `nightlight-ui-a-admission-final` was observed and deliberately left untouched.

No Git index, ref, branch, remote, worktree, registry, or deployment state changed. See `handoff.md` for the exact diff, validation commands, risks, and integration order.
