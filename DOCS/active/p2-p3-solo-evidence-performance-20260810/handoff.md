# Ready-for-Integration Handoff

## Conclusion

The package is ready for integration from exact base `ca8292040a402eae1d2e461708a4cc912867efcb` without a commit. P2 produces bounded owner-run/AI proxy evidence only; P3 retains a measured viewport-deferred preview change and a MapLibre-ready signal; Actions moves four official immutable pins to Node-24-capable releases. No Pages deployment, Git mutation, public product-source change, participant claim, usability claim, manual assistive-technology claim, WCAG-conformance claim, or scientific-validation claim was made.

## Exact changed files

Task records/evidence:

- `DOCS/active/p2-p3-solo-evidence-performance-20260810/plan.md`
- `DOCS/active/p2-p3-solo-evidence-performance-20260810/context.md`
- `DOCS/active/p2-p3-solo-evidence-performance-20260810/task.md`
- `DOCS/active/p2-p3-solo-evidence-performance-20260810/p2-evidence.md`
- `DOCS/active/p2-p3-solo-evidence-performance-20260810/p3-performance.md`
- `DOCS/active/p2-p3-solo-evidence-performance-20260810/measurements.csv`
- `DOCS/active/p2-p3-solo-evidence-performance-20260810/actions-maintenance.md`
- `DOCS/active/p2-p3-solo-evidence-performance-20260810/handoff.md`

P2 test only; no public product source:

- `project/nightlight-public/tests/proxy-evaluation.test.js`

P3 product/probe/test:

- `project/nightlight-dashboard/src/views/HomeView.vue`
- `project/nightlight-dashboard/src/views/MapView.vue`
- `project/nightlight-dashboard/scripts/navigation-performance-probe.pw.js`
- `project/nightlight-dashboard/src/views/performanceSignals.test.js`

Actions:

- `.github/workflows/deploy-dashboard.yml`

Tracked `git diff --stat` before task-record files were counted: 5 files, 183 insertions, 38 deletions. The remaining files above are new, untracked delivery/evidence files and the new Dashboard regression test. `git diff --check` passed. Integration owner should review `git diff --no-index /dev/null <untracked-file>` or stage only after reviewing these new files; this task did not stage anything.

## Validation commands and results

| Command/check | Result |
| --- | --- |
| public `npm ci` | Passed; 73 packages, 0 vulnerabilities |
| public baseline `npm test` | Passed; 11 files, 123/123 before the P2 gate |
| public final `npm run validate` | Passed; 11 files, 127/127; Vite production build; 11-file release manifest; required public source/dist boundary |
| public headed browser matrix | 20/20 route-width checks plus 4/4 Compare checks; Skip link and keyboard focus passed; zero console/page/request/HTTP failure |
| Dashboard `npm ci` | Passed; 66 packages, 0 vulnerabilities; npm warned only that the optional esbuild install script was not approved |
| Dashboard focused P3 gate | 3/3 passed |
| Dashboard final `npm test` | Passed; 4 files, 24/24 |
| Dashboard `npm run analyze:bundle` | Passed; MapLibre remains isolated from home initial route; size warning retained as context |
| corrected headed map experiment | Four scenarios × 7 samples; 28/28; zero probe error/timeout; median/p95 in `p3-performance.md` |
| home candidates | Native lazy 14/14 measured no-op; viewport-deferred 14/14, no pre-scroll preview request in all samples |
| final headed behavior smoke | no pre-scroll image/request; after-scroll image load; real focusable link; Enter → map; MapLibre ready |
| official Actions evidence | exact tags resolved against four `github.com/actions/*` repositories; pinned `action.yml` files show Node 24 or official composite upload-artifact v7 |
| Actions static gate | four expected full SHA pins occur exactly once; no shorthand action reference; `node-version: 20` unchanged; no insecure Node opt-out |
| CSV recomputation | 56 rows; eight groups of 7; settle medians/p95 match the report |
| `git diff --check` | Passed; line-ending notices only |

An initial P2 release-boundary run failed because the newly named standalone test file was not allowlisted. The assertions were merged into the existing allowed P2 proxy test and the standalone file was deleted; the final complete gate passed without changing the public-boundary scanner. A first map run produced null ready timestamps because the probe did not observe attributes; those timestamps were discarded, the probe was fixed/tested, and all map scenarios were rerun. These are harness/test-shape corrections, not hidden product passes.

## Unverified risks

- No participant, intended-audience, comprehension, task-success, SUS, usability-validation, manual screen-reader, manual assistive-technology, or WCAG-conformance evidence exists.
- P2 AI passes are non-human, same-session heuristic reviews; they are neither participants nor independent audience evidence.
- P3 is one local headed Chromium version under CDP emulation. Field, mobile-device, other-browser, CDN/Pages, user-perceived, and geographic performance are unknown.
- One retained deferred-image cold outlier prevents a p95 improvement claim.
- CARTO/external requests were variable and often failed; external settle is not application ready and does not diagnose external root cause.
- GitHub-hosted CI/Pages was intentionally not run. Runner admission, artifact upload, environment protection, and production deployment remain unverified.
- A diagnostic npm log outside the worktree may remain in the shared user npm cache; shared cache cleanup was out of scope.

## Cleanup status

- owned public/Dashboard preview servers stopped;
- ports `43241` and `43242` free;
- task-owned P2 and P3 browser sessions closed;
- unrelated browser session `nightlight-ui-a-admission-final` left untouched;
- public/dashboard `node_modules` and `dist` removed;
- task `.runtime`, snapshots, raw performance output, temporary probes, and preview logs removed;
- no Git index/ref/branch/remote/worktree/registry mutation and no deployment.

## Recommended integration and release order

1. Review task records and strict P2/P3 claim boundaries first.
2. Integrate the public P2 test-only change and task evidence; run public `npm ci && npm run validate`.
3. Integrate Dashboard readiness instrumentation, performance probe/test, and preview deferral; run `npm ci && npm test && npm run analyze:bundle`, then one production-preview smoke.
4. Integrate the four workflow pins; run the static full-SHA check and review permissions/triggers.
5. Commit in reviewable boundaries chosen by the integration owner; this package makes no commit recommendation that overrides repository Lore policy.
6. Push and let normal CI verify the candidate. Only after green CI should the integration owner decide whether to dispatch/allow Pages deployment.
7. After deployment, verify the exact deployed SHA, public boundary/release manifest, home preview behavior, map readiness, and workflow annotations. Do not promote P2 proxy evidence into human/usability/scientific claims.
