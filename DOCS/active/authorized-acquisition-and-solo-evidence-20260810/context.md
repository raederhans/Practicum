# Context — authorized acquisition and solo evidence phase

## Repository state at dispatch

- `main`: `ca8292040a402eae1d2e461708a4cc912867efcb`
- `origin/main`: `ca8292040a402eae1d2e461708a4cc912867efcb`
- Main checkout dirt: only user-owned `DOCS/archive/personal-project-evolution-research/` is untracked before these records were created.
- Unrelated registered worktree: `C:\Users\raede\.codex\worktrees\591a\Practicum` at `223fc653dba2768dad99df9d032beaedd9234d6a`.
- Empty historical directory shells are not valid worktrees and are not integration sources.

## Dispatch identities

| Lane | Task identity | Worktree | State |
| --- | --- | --- | --- |
| P1 authorized acquisition | `019fe950-a575-7b33-8f9b-dd65cca53bee` | `C:\Users\raede\.codex\worktrees\aefe\Practicum` | active |
| P2/P3 solo evidence | `019fe950-adf8-7600-b531-7ae219bc3df0` | `C:\Users\raede\.codex\worktrees\0313\Practicum` | active |

Both Codex worktrees were created from the fixed baseline; their tasks are active.

## Existing scientific and public boundaries

- Reviewed-modeling readiness is not independent scientific validation.
- The full-upstream mode currently has seven evidence blockers: VNP46A2, EAGLE-I lineage, immutable OSM input, TIGER archives, Miami-Dade ArcGIS identity, NLCD receipt, and WorldPop identity/license/checksum.
- The EAGLE-I tracked-tree receipt is LF-canonical: 52 files, 213,419,815 bytes, SHA-256 `5bcf40ce8a8791f405d4dc68a0cd85c6c010990f4ddecf2913443d05f52a9744`.
- P2 evidence is a personal proxy only. It cannot establish participant usability or manual assistive-technology validation.
- P3 previously measured MapLibre isolation as acceptable; external map-resource settling remained the slow/variable portion, and the home preview image remained an experiment candidate.

## Live-process ownership contracts

### P1 acquisition owner

- Owner: P1 task only.
- Resources: Earth Engine/API clients, task-specific virtual environment, download/cache/output directories, receipt logs.
- Start gate: credentials checked without disclosure; official source and license identified; size/quota/cost bounded; output path confirmed not to overwrite existing data.
- Success: immutable receipt/checksum/metadata plus acquisition validation, or a precise blocker with the attempted official path.
- Stop: cancel task-owned export/download, record partial output, delete or quarantine incomplete files, deactivate the task environment.

### P2/P3 runtime owner

- Owner: P2/P3 task only.
- Resources: public/dashboard dev servers, browser sessions, ports, `dist`, browser profiles, performance traces/logs.
- Start gate: ports and output directories recorded; no other owner uses them.
- Success: tests/build/boundary pass, controlled measurements collected, and any code change is supported by before/after evidence.
- Stop: close browser/server, release ports, remove task-owned generated artifacts, retain only intentionally tracked evidence.

## Integration owner

The primary task is the sole owner of staging, commits, merges/cherry-picks, branch/ref changes, pushes, registry reconciliation, Pages deployment observation, and worktree cleanup. Execution tasks return diffs and evidence only.

## Integration closeout facts — 2026-08-10

- The primary task was formally designated the only integration owner for four ready-for-integration packages.
- `git fetch --prune origin` completed through the normal Git credential path. The remote URL contains no embedded credential. `main` and `origin/main` remain equal at `ca8292040a402eae1d2e461708a4cc912867efcb` with divergence `0/0`.
- Package paths: UI `fa7d`, P2/P3 `0313`, P1 `aefe`, platform `1335`; retained unrelated worktree `591a` remains excluded.
- Changed-file counts: UI 11, P2/P3 14, P1 11, platform 13. Pairwise same-file overlap is zero.
- A redacted pattern scan found zero credential/private-key/API-key/JWT/embedded-credential URL matches in the 49 package files. No package file exceeds 5 MiB.
- Ignored residue is not an integration source: UI retains local `node_modules`, `dist`, and `.playwright-cli`; P1 retains its approximately 1.16 GB acquisition cache and Earth Engine environment. P2/P3 and platform report no repository-local generated residue.
- A P1 diagnostic outside the delivery diff previously exposed a GitHub credential environment value. The value was not copied, recorded, or re-read. Treat that credential as potentially compromised; remote operations use only the normal credential helper, and the user must revoke/rotate it after closeout.

## Combined live-process contract

The primary integration task is the sole owner. Execution tasks have stopped their processes and may not restart or interpret the combined lane.

| Process | Full command and cwd | Shared resources / log | Success and stop condition |
| --- | --- | --- | --- |
| Public dependency and candidate gates | `npm ci`, targeted Vitest, then `npm run validate` in `project/nightlight-public` | Main-owned `node_modules`, `dist`; isolated npm cache and logs under `%TEMP%\practicum-integration-20260810\` | Exit 0 with expected tests, build, schema-v2 manifest, source/dist verifier; stop after first actionable failure and never repeat the same unexplained failure three times. |
| Dashboard dependency and candidate gates | `npm ci`, targeted Vitest, `npm test`, `npm run analyze:bundle`, `npm run build` in `project/nightlight-dashboard` | Main-owned `node_modules`, `dist`, `.vite`; isolated npm cache/logs under the same temp root | Exit 0; MapLibre remains isolated from home; stop after first actionable failure. |
| Combined browser and performance smoke | production previews on reserved ports `43261` and `43262`, one isolated browser/session at a time | Main-owned ports, browser profile, preview/performance logs under the named temp root | Route/focus/overflow/console/public-boundary checks pass; Dashboard preview remains deferred before viewport entry and MapLibre ready is distinct. Stop exact owned PIDs, close the owned browser, verify ports free, then clean task-owned artifacts. |
| CI and Pages observation | normal `git push origin main`, then immutable GitHub run/deployment observation | GitHub-hosted runner/artifact/Pages; no local credential material | CI and Pages succeed for the exact candidate SHA. Do not force-push, change billing/permissions, or use a token literal. |

## Sequential integration evidence

- UI `fa7d`: 11 files applied without touching its index/ref; LF-canonical content matched the source package. Main targeted `static-shell.test.js` passed 20/20 after a fresh isolated-cache `npm ci`.
- P2/P3 `0313`: 14 files applied without touching its index/ref; LF-canonical content matched. Main public proxy gate passed 8/8 and Dashboard performance-signal gate passed 3/3 after an isolated-cache `npm ci`.
- Actions pins were independently checked against the official `actions/*` repositories: all four tag SHAs matched; checkout/setup-node/deploy-pages declare Node 24; upload-pages-artifact is an official composite pinned to the official upload-artifact v7.0.0 SHA. The first checker expected a literal `@v7` and produced a false negative; the corrected immutable-SHA check passed without a product edit.
- P1 `aefe`: 11 small files applied; LF-canonical content matched. The four ignored cache assets were rehashed in place and all declared byte counts and SHA-256 values matched. No raw/cache/credential file was copied. Main P1 targeted tests passed 30/30.
- Integration security review reproduced two narrow risks in `source_receipts.py`. Test-first failures proved that auth preflight enumerated the supplied environment and Overpass could read without a byte limit. The minimal repair now reads only named auth keys and caps each Overpass response at 64 MiB by default with an explicit CLI override and a post-fetch guard; both new tests passed.
- Platform `1335`: 13 files applied last; LF-canonical content matched. The first combined platform gate passed 69/69 across value/error, dependency/CSP, public allowlist, and release-manifest contracts.

## Aggregate pre-commit candidate evidence

- The project `.venv` is the supported Python environment. Its full gate passed `130 passed, 7 subtests`; reviewed-modeling exited 0 with 16 verified receipts, while full-upstream deliberately exited 1 with the same seven evidence blockers. A system-Python collection failure was dependency-context evidence, not a product failure.
- Public: `npm audit` reported zero vulnerabilities; `npm run validate` passed 13 files and 167 tests, built the site, and verified the source/dist boundary. Two independent Pages-base builds produced the same schema-v2 manifest SHA-256, `22d5da7cd40635e34e081fbf3e5823bbba7077c23e3bb36fdf4d647c2aee0ff2`.
- Dashboard: `npm audit` reported zero vulnerabilities; 24/24 tests, bundle analysis, and the production build passed. MapLibre remains isolated from the home initial chunk; its raw chunk remains approximately 803 kB and is not claimed as reduced.
- Browser: five public routes passed 320 px and 768 px title/active-navigation/overflow checks; route focus, forced-colors, local-only public origins, and zero console warning/error checks passed. Dashboard preview loading remained absent before scroll and began after scroll; the real map link accepted keyboard activation and reached distinct `data-map-ready=true` with a canvas and no data error.
- The supported minimum Slow 4G smoke completed three cold and three warm home samples with zero errors and zero browser logs. Home-attached signal medians were approximately 970 ms cold and 523 ms warm; no map-preview resource entered the measured home phases. This six-sample integration smoke does not replace the lane's 56-row controlled evidence.
- The first public preview omitted Vite's Pages base and produced local asset 404s. Restarting the exact owned preview with `--base /Practicum/` resolved the configuration error; this did not require a product change. One oversized browser matrix invocation ended without an assertion result, so it was split into shorter bounded checks that all passed.
- The EAGLE-I tracked-tree receipt now canonicalizes text to LF before byte counts and digests and has a regression test proving CRLF/LF stability. Public copy explicitly distinguishes what the Pages/site bundle omits from the fact that the public repository tracks 52 derivatives; tests reject the former inaccurate repository-wide non-redistribution statement.
- After browser verification, the owned Playwright session was closed, PIDs `38564` and `9424` were stopped after port-ownership checks, and ports 43261/43262 were confirmed free.
- The complete 54-file candidate totals about 566 kB: zero files exceed 5 MiB, zero protected personal-research paths are present, and a redacted candidate-wide secret-pattern check found zero matches. Node/Python syntax and `git diff --check` pass. The three protected personal-research files remain untracked and untouched.
- Final first-principles review remained bounded to reproducible defects. It confirmed both prior high-confidence findings are repaired: tracked-tree receipts use LF-canonical bytes with a CRLF/LF regression test, and public copy distinguishes the Pages artifact boundary from repository tracking. No additional reproducible blocking defect was found, so extensibility review stopped.
- Main-owned `node_modules`, `dist`, `.playwright-cli`, and named pre-commit temp logs were deleted after capture. They are generated/diagnostic artifacts and are not recoverable as files, but the gates are reproducible from lockfiles and the durable evidence above; the P1 acquisition cache was intentionally preserved in its lane pending verified owner-cache transfer.
