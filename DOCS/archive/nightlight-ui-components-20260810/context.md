# Nightlight Public UI Execution Context

## Current truth

- Worktree: `C:\Users\raede\.codex\worktrees\fa7d\Practicum`; detached exact HEAD/base `ca8292040a402eae1d2e461708a4cc912867efcb`.
- Git state at start: clean index and working tree. This lane is not integration owner and will not stage, commit, push, merge, rebase, cherry-pick, deploy, edit refs/index, or change worktree topology.
- Repository-local `AGENTS.md` and root `lessons learned.md`: absent; the user-supplied top-level instructions apply.
- Existing public primary routes are fixed at five: Overview, Study Atlas, Findings, Methods, Credits / Policy. No evidence currently authorizes a sixth.
- Worktree `0313` was inspected read-only at the same exact base. Its current changes are dashboard performance files plus `project/nightlight-public/tests/solo-evidence-gate.test.js`; none collide with this lane's reserved source files or task-record directory.
- The public release verifier uses an exact file allowlist. New component/test files would fail until another owner changes tooling, so this lane will first consolidate primitives inside already allowlisted source/test files.

## Decisions and deviations

| Time | Evidence or decision | Impact |
| --- | --- | --- |
| 2026-08-10 +08:00 | Classified as `complex`; invoked `$manage-task-records`, read `$integrate-worktrees`, and confirmed this lane is not integration owner. | Work proceeds with durable records and ends at `ready-for-integration`; Git and registry remain read-only. |
| 2026-08-10 +08:00 | Read-only `0313` scan found no collision in reserved UI source paths. | UI implementation may proceed; repeated collision checks will precede final handoff. |
| 2026-08-10 +08:00 | `scripts/verify-public.mjs` allowlists exact files and is owned outside this lane. | Prefer CSS/semantic primitives and existing tests; do not create unallowlisted files or edit the verifier. |
| 2026-08-10 +08:00 | Archive and current-source audit confirms prior fixes for mobile overflow, active navigation, route titles/H1 focus, Compare live-region duplication, outcome/admission language, and local-only runtime behavior. | Preserve those controls and close them as measured no-op unless fresh browser evidence regresses. |
| 2026-08-10 +08:00 | Current source lacks a forced-colors branch, explicit mode-to-panel relationships, a complete Methods admission/public-artifact timeline, consistent visible metric units, and route-local policy scanning for runtime/known limits. | These are the bounded implementation targets; hypotheses about real comprehension and assistive technology remain unmodified unknowns. |
| 2026-08-10 +08:00 | `AtlasView.vue` is a file hotspot, but state is local, comparison domain logic is already pure/tested, and new source files are not allowlisted. | Do not split Atlas in this lane; consolidate only real shared CSS/semantic primitives. |
| 2026-08-10 +08:00 | Browser admission found a repeatable 320 px overflow under WCAG text-spacing injection: browser-default `figure` margins plus large heading/card intrinsic widths. | Remove the default figure margin and add bounded mobile type/padding/wrap rules. Do not hide overflow or override user text spacing. |
| 2026-08-10 +08:00 | Final source/build/browser checks passed after the bounded responsive repair. | Candidate is `ready-for-integration`; this is not a commit, merge, release, WCAG certification, or human-understanding result. |

## Live process ownership

The `$orchestrate-live-tests` contract is active. The sole owner is **execution dialogue A in worktree `fa7d`**. Worktree `0313` separately owns ports `43241`/`43242` and its own outputs; this lane will not start, poll, retry, stop, or interpret those processes. This lane uses an isolated npm cache to avoid the user-shared cache during concurrent dependency work.

| Process | Owner | Full command and cwd | Port/cache/output/log | Success signal | Failure/stop/cleanup |
| --- | --- | --- | --- | --- | --- |
| Dependency restore | Dialogue A / `fa7d` only | `npm ci --cache C:\Users\raede\AppData\Local\Temp\practicum-nightlight-ui-components-20260810\npm-cache` from `project/nightlight-public` | Isolated npm cache under the named temp root; output `project/nightlight-public/node_modules`; logs `npm-ci.stdout.log` / `npm-ci.stderr.log` in the same temp root | Hidden owner process exits `0`; pinned dependencies present | Stop after the first actionable failure; no three identical retries; remove owned `node_modules` and temp cache/logs after final evidence |
| Targeted UI/route tests | Dialogue A / `fa7d` only | `npm test -- tests/routes.test.js tests/static-shell.test.js tests/evidence-passport.test.js tests/generalization-artifact.test.js` from `project/nightlight-public` | Uses lane `node_modules`; logs `targeted.stdout.log` / `targeted.stderr.log` under the named temp root | Exit `0`; all selected suites pass | One evidence-backed repair/retry per distinct failure; stop after three identical failures |
| Complete validation and production build | Dialogue A / `fa7d` only | `npm run validate` from `project/nightlight-public` | Output `project/nightlight-public/dist`; logs `validate.stdout.log` / `validate.stderr.log` under the named temp root | All Vitest tests, Vite production build, release manifest, and required public-boundary verifier exit `0` | Preserve exact failure; do not weaken verifier; remove `dist` after browser evidence |
| Production preview | Dialogue A / `fa7d` only | `node node_modules/vite/bin/vite.js preview --host 127.0.0.1 --port 43251 --strictPort` from `project/nightlight-public` | Verified-free `127.0.0.1:43251`; logs `preview.stdout.log` / `preview.stderr.log` under the named temp root | Exact owned listener and HTTP 200; browser matrix completes | Stop exact positively identified owner process tree on mismatch/completion; verify port free |
| Chromium browser matrix | Dialogue A / `fa7d` only | Isolated Playwright session against `http://127.0.0.1:43251/` | Dedicated session/profile/output under the named temp root where exposed; no shared browser process | Five routes and Atlas Compare checked at 320/373/375/768; keyboard/focus/history, 200% zoom/reflow, WCAG text spacing, forced-colors, console/network, and overflow facts recorded | Close only owned session; preserve failures; delete owned snapshots/traces/profile/logs |

## Final evidence

- Targeted Vitest: 53/53 passed after correcting one test-only false positive; no product fallback was added.
- Final `npm run validate`: 11/11 test files and 133/133 tests passed, Vite production build passed, the 11-file release manifest was produced, and `verify:public --require-dist` passed.
- Normal Chromium matrix: five routes at 320/373/375/768 px, 20/20 checks, no document overflow, one explicit route focus target, visible H1, and visible active navigation.
- Route/focus contract: the cold keyboard sequence remained Skip link -> identity -> Overview -> Study Atlas; keyboard navigation, browser back/forward, and Atlas Compare controls retained the expected focus target. No `scrollIntoView` exists.
- Reflow: five routes plus Atlas Compare passed at a 640 CSS-pixel viewport, used as the 1280-to-640 200%-zoom reflow equivalent, with no horizontal document overflow.
- WCAG text spacing: line-height 1.5, paragraph spacing 2em, letter spacing 0.12em, and word spacing 0.16em were injected through the browser debugging protocol; five routes at 320/375/768 px (15/15) plus Atlas Compare at 320 px had no overflow or clipped text.
- Forced colors: `forced-colors: active` matched in Chromium; five routes at 320/768 px (10/10) plus Atlas Compare had no overflow, status borders remained visible, and keyboard `:focus-visible` rendered a solid 3 px system highlight outline.
- Runtime boundary: console warnings 0, console errors 0, and external performance-resource origins 0.
- Diff review: `git diff --check` passed; changed product/test files are confined to the authorized App/styles/five views/static-shell test paths. No route, dependency, backend, analytics, external runtime request, store, package/lock, Vite, verifier, registry, Git state, or deployment mutation was added.
- Cross-lane recheck: `0313` now changes dashboard/workflow/proxy-evaluation and its own P2/P3 records; `aefe` changes P1 data/acquisition/tests; B lane `1335` changes policy/package/public-boundary/release tooling and new platform files. None overlaps this lane's changed files or task records.
- Live cleanup: all browser sessions owned by this lane were closed, the exact preview listener/launcher were stopped, and port `43251` was verified free. The environment policy rejected both explicit recursive deletion and a non-recursive file-only deletion attempt, so `project/nightlight-public/node_modules`, `project/nightlight-public/dist`, `.playwright-cli`, and `C:\Users\raede\AppData\Local\Temp\practicum-nightlight-ui-components-20260810` remain as ignored/local artifacts. No unsafe workaround was used.

## Handoff

Current state: `ready-for-integration` from exact base `ca8292040a402eae1d2e461708a4cc912867efcb`. Integration owner should review and integrate this lane together with the independent `0313`, `aefe`, and B-lane candidates, rerunning the combined gates because each lane was validated in isolation. This lane made no Git-state or deployment change.

## Unknowns retained

Screen-reader behavior, speech input, switch access, browsers other than the tested Chromium environment, OS-native high-contrast modes beyond browser forced-colors emulation, real-user task performance, and scientific understanding remain unverified. The implementation and automated evidence do not claim WCAG conformance, participant validation, scientific validation, merge, or release admission.
