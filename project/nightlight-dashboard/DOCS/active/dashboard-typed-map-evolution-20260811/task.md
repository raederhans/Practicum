# Task

## Current status

Committed and ready for integration from this isolated detached worktree. No merge, rebase, cherry-pick, push, deploy, main update, or topology change was performed.

## Checklist

- [x] Confirm exact clean baseline `6b3de4ee97c5391084538bec84db3b1a1f4e05ed`.
- [x] Read the final report from architecture task `019fef1c-349e-7ee2-992e-405a8ce4e562`.
- [x] Read the required task-record, live-test ownership, and Lore commit skills.
- [x] Complete `NL-TS-01` and record diagnostic categories and go/no-go result.
- [x] Complete `NL-DATA-01` and report scientific-value ambiguities to the Modeling lane without changing them.
- [x] Complete or gate-stop `NL-SVC-01`.
- [x] Complete the formal `NL-PERF-01` matrix with retained samples and failure evidence.
- [x] Complete the `NL-PERF-02` CSS boundary and threshold-gated optimization decision.
- [x] Complete `NL-CI-01` if equivalent Dashboard CI is absent.
- [x] Run final scoped verification, security/generated-file audit, and live-resource cleanup.
- [x] Create and inspect reversible Lore commits and hand off recommended integration order.

## Validation evidence

| Command or check | Result |
| --- | --- |
| `git rev-parse HEAD` | Passed: exact required baseline. |
| `git status --short --branch` | Passed: clean detached worktree at start. |
| Repository `AGENTS.md` search | No more-specific repository file found; delegated top-level contract applies. |
| Architecture task read | Passed: final report read before implementation. |
| Broad browser census (`vue-tsc`, strict checkJs) | 358 diagnostics: 154 `TS7006`, 97 `TS2339`, 59 `TS7005`, 13 `TS18047`, 10 `TS2322`, 8 `TS7034`, 8 `TS7053`, 7 config `TS2307`, and 2 other. |
| Node report census (`tsc`, strict checkJs) | 8 diagnostics, all `TS7006` annotation noise. |
| Retained `npm run typecheck` | Passed: strict browser leaf/schema/repository check plus separate strict Node report check. |
| Legacy wire census | 56 JSON files, all unversioned: 51 arrays and 5 objects. Facilities: 25 files/6,225 rows/10 numeric `0.5` values/no nulls. Time series: 25 files/1,889 rows/4 paired nulls/no unpaired nulls. Probability: 25 files/61,903 features/no null or zero/94 values equal to 0.5. |
| Schema/loader/repository/view targeted tests | Passed: 4 files, 34 tests before adding the tracked-artifact conformance loop. |
| Modeling/Data ambiguity report | Sent to source coordination task with exporter line, artifact counts, and no-value-change boundary. |
| Producer v1 conformance | Added explicit `1.0.0` object versus legacy-array routing; genuine available `0.5`, null unavailable, count invariants, controlled reason/status, lineage, and unsupported-version tests pass. No producer artifact is claimed generated. |
| Formal MapLibre timing matrix | Six cells, 266 retained pre-change timing samples; seven cold plus seven warm overview/detail samples in every cell. 390/768 stress-entry failures remain explicit. |
| Failure recovery | External and WebGL injected profiles remain not-ready and recover to home; no failure profile is counted as timing success. |
| Map CSS/resource boundary | Fourteen post-change home samples contain zero MapView/MapLibre/unpkg requests. Analyzer places 77,021 raw / 11,405 gzip CSS bytes only in the map closure and rejects external MapLibre CSS in the shell. |
| Detail-ready race | Three incomplete post-CSS runs retained. Targeted aggregate-style gate repair then passed 42/42 map samples with zero timing errors. |
| Windows clean-install CI equivalent | Passed after `npm ci`: 49/49 tests, browser/Node typecheck, build, and bundle analyzer. Lockfile audit found 0 vulnerabilities; existing MapLibre >500 kB and esbuild install-script policy warnings remain explicit. |
| Workflow parsing | Passed with temporary pinned `js-yaml@4.1.0`; matrix contains Ubuntu/Windows and has no deploy/Public coupling. A real Ubuntu runner was not available locally. |
| Live teardown | Named CLI browser closed; all direct Playwright runners exited; last preview PID `252060` stopped; port `54741` confirmed free; transient raw/runtime/smoke files removed. |
| Scoped audit | `git diff --check` passed; 40 changed/new files before commits contained zero credential-pattern hits and no file above 1 MiB; ignored `dist`/`node_modules` were excluded; all task-owned live processes are absent. |

## Ready-for-integration commits

1. `79943770f3402f67d996383ca8677d3a3130f0a7` — bounded diagnostics and legacy/producer-v1 schema contracts.
2. `3b1308d31054423465f8a133fcdef858dbad4ee9` — app-local typed artifact repository and `ChartsView` consumer.
3. `eb052b78a43d1365f1689a7d15318ed8c876c4d1` — MapLibre measurement tooling and retained pre-change/failure evidence.
4. `220bcea8246ee983b8db030a295aa3067c78e7b8` — route-local CSS, threshold decisions, detail-ready race fix, and post evidence.
5. `9f58f0519bd998501db96c4344f6f586e2d1844d` — independent non-deploy Dashboard CI.

Integrate in the listed order. The first commit includes shared dev-tool pins used by later measurement work. The CI commit should be last so its full entry points exist when the workflow lands.

## Open risks and remaining work

- Broad TypeScript expansion stopped: at least 310/358 browser diagnostics (86.6%) were implicit-`any` or `never` inference noise, with no two type-proven behavior defects. The retained gate is intentionally limited to loader, timeseries, and bundle-report code.
- Modeling/Data still owns the future producer representation for facility fallback. This lane preserves a permanent legacy limitation and does not reinterpret values.
- Typed repository parity currently has one real consumer (`ChartsView`); shared-core admission is not met.
- Memory and long-task thresholds were crossed, but no low-risk admitted worker/LRU/source-eviction action was supported. The shared MapLibre worker is retained for SPA warm reuse; page-level long-task attribution did not meet the worker-tuning ratio.
- Ubuntu behavior is represented by parsed workflow configuration only; no real Ubuntu runner is available locally. The Windows clean-install equivalent gate passed.
- GitHub-hosted Ubuntu/Windows jobs, a physical-device browser, and any Dashboard deployment remain unrun. The existing MapLibre >500 kB warning and npm esbuild install-script-policy warning remain open evidence, not hidden failures.
