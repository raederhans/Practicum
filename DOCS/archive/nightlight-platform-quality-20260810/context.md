# Nightlight Platform Quality Context

## Current truth

- Executor: independent conversation B in `C:\Users\raede\.codex\worktrees\1335\Practicum`; not an integration owner.
- Base and initial HEAD: `ca8292040a402eae1d2e461708a4cc912867efcb`, detached; initial `git status --short` was clean.
- No repository `AGENTS.md` or `lessons learned.md` exists inside this worktree; the user-supplied top-level instructions apply.
- Public app requires Node `>=20`; current host is Node `v22.23.0`, npm `11.18.0`.
- Locked direct versions: Vue `3.5.21`, Vue Router `4.5.1`, Vite `6.4.3`, Vitest `3.2.7`; esbuild install script is explicitly pinned to `0.25.12`.
- Initial worktree has no `project/nightlight-public/node_modules`; baseline reproduction therefore requires `npm ci` before tests/build.
- Current release entry is `npm run validate` = Vitest, Vite build, release manifest write, and public source/dist verification.
- Initial manifest schema v1 recorded stably sorted relative paths, bytes, and SHA-256 but did not bind base path or build contract; the B candidate is now schema v2 with a verified base-path/static-build contract.
- Current Vite build disables source maps and accepts only slash-delimited `VITE_BASE_PATH`; `index.html` has a local-only CSP meta and no-referrer meta.
- P1 `aefe` changes its own task records, `project/data/acquisition/`, `project/data/manifests/osm_modeled_event_scope_v1.json`, and `project/tests/test_authorized_source_acquisition.py`.
- P2/P3 `0313` changes its task/runtime evidence, two dashboard performance files plus one dashboard test, and `project/nightlight-public/tests/solo-evidence-gate.test.js`.
- UI candidate `fa7d` changes its own task records plus existing `src/App.vue`, `src/styles/main.css`, and five `src/views/*.vue` files; all `src/views`, `src/components`, `src/styles`, `src/router`, and `App.vue` remain prohibited for B.
- Retained `591a` is clean at older HEAD `223fc653dba2768dad99df9d032beaedd9234d6a` and is not an integration target.

## Decisions and deviations

| Time | Evidence or decision | Impact |
| --- | --- | --- |
| 2026-08-10 B0 | Registry is stale for the newly created lanes, so read-only Git status and untracked-file enumeration were used as the current changed-file source. | No same-file overlap is presently visible; recheck before handoff. |
| 2026-08-10 B0 | Existing public release history explicitly excludes backend, analytics, accounts, and runtime external requests. | B1 starts from static-only and requires affirmative trigger evidence to change it. |
| 2026-08-10 B0 | Existing 123/123, browser, CI, and Pages results belong to earlier exact SHAs. | They are context only and will not be reported as fresh B evidence. |
| 2026-08-10 B0 | Fresh baseline at the unchanged base passed 123/123 tests, built 11 files, and passed the source/dist verifier. | B changes can be compared against a current local baseline rather than historical evidence. |
| 2026-08-10 B1 | None of the four backend triggers has current evidence; reviewed aggregates are build-cycle snapshots and there is no write, large-query, or server-side access-control requirement. | Keep static-only; no API, database, login, analytics, telemetry, or mock was added. |
| 2026-08-10 B2 | Existing content validators are artifact-specific; a cross-artifact consumer value/error state has a separate responsibility. | Added one small `src/lib` contract; did not move or rewrite existing content/domain validation. |
| 2026-08-10 B4 | Similar status calculations in Evidence Passport validation and comparison serve separate trust boundaries; no swallowed runtime load error or duplicate transform was reproduced. | Large refactor, shared generic utility, service layer, state manager, and TypeScript migration remain evidence-backed no-ops. |
| 2026-08-10 B6 | Manifest schema v1 hashed files but did not bind the build base path or static architecture contract. | Schema v2 adds deterministic build metadata and verifies built local asset prefixes without claiming CI/deployment state. |
| 2026-08-10 B6 | Root workflow builds with `VITE_BASE_PATH` but verifies in a later step without that environment variable. | Manifest verification reads and cross-checks its own recorded base path; no workflow-owner file change or handoff is needed. |
| 2026-08-10 B8 | First-principles security review found prototype-inherited aggregate state and locale-dependent manifest sorting as avoidable trust assumptions. | Required own fields with `Object.hasOwn`; replaced locale sorting with fixed code-unit ordering and added regression evidence. |
| 2026-08-10 B8 | Final parallel-lane recheck shows UI `fa7d` edits only existing UI-owned files, P2/P3 `0313` adds only `tests/solo-evidence-gate.test.js` in the public app, and P1 remains data/Python-only. | No same-file overlap exists at handoff; any later UI-created source file must be added to the exact public allowlist during integration. |

## Live process ownership

| Process | Owner | Command / shared resources | Log path | State |
| --- | --- | --- | --- | --- |
| Baseline dependency install | Conversation B only | `npm ci`; writes this worktree's `project/nightlight-public/node_modules` and npm cache | `%TEMP%\nightlight-platform-quality-20260810-npm-ci.log` | Complete, exit 0; 73 packages, audit reported 0 vulnerabilities |
| Baseline/full validation | Conversation B only | `npm run validate`; reads `node_modules`, rewrites this worktree's `dist`, no ports/browser/database | `%TEMP%\nightlight-platform-quality-20260810-validate.log` | Complete; base 123/123 and final candidate 153/153, final exit 0 |
| Targeted tests | Conversation B only | `npm run test:platform`; isolated temp fixtures, no port/browser | `%TEMP%\nightlight-platform-quality-20260810-targeted.log` | Complete, exit 0; final 69/69 after verifier and own-property repairs |
| Pages-base determinism | Conversation B only | two serial `VITE_BASE_PATH=/Practicum/ npm run build` plus production verifier runs; rewrites only this worktree's `dist` | `%TEMP%\nightlight-platform-quality-20260810-pages-determinism.log` | Complete; both exits 0, identical final manifest SHA-256 `6fb4911118542a8f348281ce2dee59d5cc527029e3787735056dba68845a5edd` |

All B live processes are complete. Non-owner agents may read the four retained completed log snapshots, but `dist/release-manifest.json` is no longer present because B removed `dist` during cleanup. No P2/P3 ports, browser sessions, logs, checkpoints, or output directories were used.

## Handoff

State is `ready-for-integration`, not committed or released. Recommended order is UI A and P2/P3 first, then B as the platform gate, followed by one integrated `npm ci && npm run validate`; P1 is path-independent. If A adds a new public source file after this snapshot, the integration owner must update the exact allowlist and add a matching negative boundary test. The primary supervisor must stage/commit through its own authorized workflow and obtain immutable Actions `head_sha`, artifact, and Pages deployment evidence before making publication claims.

## Next step

Integration owner: review the 13-file B delivery, recheck late UI file additions, integrate after UI/P2/P3, reinstall dependencies, run the full candidate gate, and collect external CI/deployment evidence separately.

## Cleanup

- Agent-generated `project/nightlight-public/node_modules` and `dist` were removed after evidence capture with an exact ignored-path dry-run and cleanup.
- Four small `%TEMP%\nightlight-platform-quality-20260810-*.log` files remain because the execution policy rejected their deletion. They contain command output only, no credentials or product data; no process, port, browser, lock, or checkpoint remains.
