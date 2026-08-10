# Nightlight Public UI and Components Task

## Current status

`integrated` — committed as `e45c5ab`, admitted in exact product candidate `bd91194f85c6cc8ce1fc3d6ced80dd66d4bf6511`, and published by successful Pages run `31352969379`. Combined and live browser gates passed; human and manual assistive-technology claims remain out of scope.

## Checklist

- [x] Read supplied instructions and the complete `$manage-task-records`, `$integrate-worktrees`, and `$orchestrate-live-tests` workflow contracts.
- [x] Pin exact detached base/HEAD and clean initial state; confirm no repository-local `AGENTS.md` or root `lessons learned.md` exists.
- [x] Read the worktree registry and inspect `0313` Git state, task records, and changed-file list without modification.
- [x] Establish the only three task records and document writable/non-writable boundaries.
- [x] Read complete relevant Nightlight archive plans/evidence and current route source/tests.
- [x] Complete per-route task audit with defect/maintainability/hypothesis classification.
- [x] Implement evidence-backed route, primitive/token, focus, responsive, zoom/text-spacing, and forced-colors repairs.
- [x] Extend directly matching existing tests while respecting the exact public allowlist.
- [x] Establish the single-owner live-test contract and execute targeted/full/build/browser verification.
- [x] Review the diff, run bug and first-principles simplification checks, stop owned live resources, attempt filesystem cleanup, record the environment-policy blocker, and confirm no cross-lane collision.
- [x] Deliver the `ready-for-integration` package to the supervisor without any Git mutation.

## Validation evidence

| Command or check | Result |
| --- | --- |
| `git rev-parse HEAD` | `ca8292040a402eae1d2e461708a4cc912867efcb` |
| `git status --short` at start | Clean |
| `git worktree list --porcelain` | Relevant `fa7d` and `0313` worktrees both pinned to `ca8292040`; this lane detached |
| repository guidance scan | No repository-local `AGENTS.md`; no root `lessons learned.md` |
| `0313` read-only collision scan | Dashboard performance files and new `tests/solo-evidence-gate.test.js`; no reserved UI source collision |
| public release verifier inspection | Exact file allowlist; existing views/App/router/styles/tests are allowed, new files are not |
| baseline targeted Vitest attempt | Did not start because this clean worktree has no `node_modules` and `vitest` is unavailable; dependency restore is deferred to the recorded single-owner validation stage |
| isolated `npm ci` | 73 packages restored with 0 vulnerabilities under the lane-owned npm cache |
| targeted UI/domain suites | 53/53 passed; one initial 52/53 result was a test-only regex false positive and passed after the assertion was narrowed to actual event-rank logic |
| final `npm run validate` | 11/11 files, 133/133 tests, Vite build, 11-file release manifest, and `verify:public --require-dist` all passed |
| normal Chromium responsive matrix | Five routes x 320/373/375/768 px = 20/20; zero document overflow, one route focus target, visible H1 and active navigation |
| 200% reflow equivalent | Five routes plus Atlas Compare at 640 CSS px passed with zero document overflow |
| WCAG text-spacing injection | Five routes x 320/375/768 px = 15/15 plus Atlas Compare 320 px; zero document overflow and zero clipped text |
| forced-colors emulation | Five routes x 320/768 px = 10/10 plus Atlas Compare; media active, zero overflow, visible status borders, solid 3 px keyboard focus outline |
| runtime diagnostics | Console warnings 0; console errors 0; external resource origins 0 |
| `git diff --check` | Passed; line-ending messages are Git normalization warnings, not whitespace errors |
| final cross-lane scan | No changed-file overlap with `0313` P2/P3, `aefe` P1, or B lane `1335` platform-quality candidate |
| cleanup | Owned browsers closed; preview listener/launcher stopped; port `43251` free. Environment policy blocked recursive and file-only deletion, leaving named ignored/local `node_modules`, `dist`, `.playwright-cli`, cache, and log paths. |

## Five-route task audit

### Overview

| Class | Evidence | Decision |
| --- | --- | --- |
| Verified defect | The hero offers Atlas and Methods, while Findings is only discoverable lower on the page. The required primary decision path is Atlas or Findings. | Replace the secondary hero action with Findings and retain a visible Methods path in the research section. |
| Verified defect | The Overview R-squared strip says `descriptive R²` but omits the declared unit/range and the direct “not future accuracy” limitation at the number. | Add unit/range and limitation beside the displayed number. |
| Maintainability | The route already has purpose, research-question, public-boundary, semantic SVG text, and aggregate-only framing, but the difference between technical/proxy checks and human understanding is not exposed on the first route. | Add a compact native help disclosure; reuse the same disclosure primitive on another route. |
| Hypothesis | Visitors may still misread “Reading recovery” as a direct lived-recovery measure. | Preserve as unknown; existing nearby proxy/not-resilience text is source-backed, and no human comprehension evidence authorizes a larger rewrite. |

### Study Atlas

| Class | Evidence | Decision |
| --- | --- | --- |
| Verified control | Search, hazard filter, selected button state, Explore/Compare radios, native selectors, Evidence Passport, `Not assessed`, `Unavailable`, schema failure, measurement limits, and no-score boundary are already explicit in current source and tests. | Measured no-op for state management, scoring, route count, and core comparison structure. |
| Maintainability | The Explore/Compare radios do not expose an explicit `aria-controls` relationship or a concise nearby definition of `Evidence Passport`, `Not assessed`, and `Unavailable`. | Add stable panel IDs/control relationships and a native disclosure using the shared help primitive. Keep focus on the operated native control; do not auto-focus mode content. |
| Maintainability | Empty/unavailable states and status labels use several route-specific class bases although their semantics are shared. | Add shared `state-panel` and `status-badge` primitives while preserving route-specific modifiers. |
| Hypothesis | The two Explore polite regions may be verbose in some screen readers. | Do not change without assistive-technology evidence; record as unknown. |
| Hypothesis | Splitting the 500+ line view might improve readability. | No-op: state remains local and pure comparison logic is already extracted/tested; new files would require out-of-lane allowlist changes, and no independent reusable UI behavior is proven. |

### Findings

| Class | Evidence | Decision |
| --- | --- | --- |
| Verified defect | R-squared and AUC are role-labeled, but visible callouts/cards do not consistently expose their artifact-declared units/ranges; the ratio card also omits `unitless` context. | Render each metric's declared unit and add concise numerical interpretation rules, including AUC's 0.50 ranking reference. |
| Maintainability | The page defines roles across prose and a table, but key terms are distributed across sections rather than available in one progressive disclosure. | Add a native metric-reading disclosure using the shared definition primitive. |
| Verified control | Readiness/admission is not presented as recovery performance; recovery transport remains explicitly unavailable; R-squared and AUC are not merged into a shared scale. | Measured no-op; extend regression assertions rather than restyle. |

### Methods

| Class | Evidence | Decision |
| --- | --- | --- |
| Verified defect | The visible workflow ends at place-level modeling. Admission and public-artifact publication are described later but not shown as traceable timeline steps. | Extend the existing timeline to five steps: input boundary, processing, modeling, admission, public artifact. Do not invent new pipeline claims. |
| Maintainability | Raw/restricted inputs are mentioned at page level and in one output label, but the public/private transition is not scannable within the process sequence. | Label private inputs/outputs and public outputs at each relevant step. |
| Verified control | The artifact contract already requires role, lineage, SHA, cohort, lock, validation design, and explicit withholding. | Preserve; no new schema/tooling work. |

### Credits / Policy

| Class | Evidence | Decision |
| --- | --- | --- |
| Verified defect | Authorship, collaboration, EAGLE-I source, no endorsement, code license, and withheld artifacts are scannable, but `aggregate-only`, `local-assets-only`, `no analytics`, `no external runtime requests`, and known validation limits are only partially visible or confined to the global footer. | Add a concise policy facts section on this route, distinguishing user-activated attribution links from runtime requests. |
| Maintainability | Existing policy cards and withheld panel already provide the correct visual language. | Reuse those patterns; do not add a sixth route or new data-policy file. |
| Hypothesis | Visitors may want longer rights explanations for every upstream dataset. | Do not expand without a concrete rights gap; current notices remain the source of detail. |

## Cross-route primitive and focus decisions

- Reuse existing files only: shared CSS/semantic primitives for `page-lede`, `definition-disclosure`, `status-badge`, `data-table-wrap`, `state-panel`, and `focus-target`. Each must have at least two real consumers before being retained.
- Initial sequential focus remains Skip link → identity → primary navigation. No `scrollIntoView` is introduced.
- SPA route changes, including browser back/forward, continue to focus the entered route's explicit H1 focus target after the transition. Initial render remains unfocused.
- Atlas mode changes, presets, selects, and Swap retain focus on the native control the user operated; content changes are described by existing visible text/live summaries. No unexpected focus jump is added.
- Error-panel abstraction is a measured no-op because this static local-data application has no runtime fetching/error state. Empty and unavailable states are consolidated; invented error UI would be dead code.

## Exact ready-for-integration file set

- `DOCS/active/nightlight-ui-components-20260810/plan.md`
- `DOCS/active/nightlight-ui-components-20260810/context.md`
- `DOCS/active/nightlight-ui-components-20260810/task.md`
- `project/nightlight-public/src/App.vue`
- `project/nightlight-public/src/styles/main.css`
- `project/nightlight-public/src/views/OverviewView.vue`
- `project/nightlight-public/src/views/AtlasView.vue`
- `project/nightlight-public/src/views/FindingsView.vue`
- `project/nightlight-public/src/views/MethodsView.vue`
- `project/nightlight-public/src/views/CreditsView.vue`
- `project/nightlight-public/tests/static-shell.test.js`

## Component and primitive boundary

- Retained shared semantic/CSS primitives with multiple real consumers: `focus-target`, `page-lede`, `definition-disclosure`, `status-badge`, `data-table-wrap`, and `state-panel`, plus the route-local `policy-facts` layout.
- Retained the existing route-local Vue structure. Atlas/Compare splitting is a measured no-op: state is local, comparison logic is already isolated in pure tested modules, and new files would create an out-of-lane public allowlist dependency without proving a reusable behavior boundary.
- Error-panel abstraction is also a measured no-op because this release has no runtime fetching/error state; unavailable/empty semantics are represented without inventing dead error behavior.

## Remaining integration and verification risks

- The integration owner must combine this candidate with `0313`, `aefe`, and B lane `1335`, resolve any future drift, and rerun combined validation from the resulting exact SHA. Isolated lane passes are not combined-branch admission.
- B lane owns package, release manifest, verifier, policy/security, and new platform-boundary files. This lane deliberately created no new component/test file and requires no verifier allowlist request.
- Filesystem cleanup is an environment-policy blocker, not a product gate failure. The exact residual paths are recorded in `context.md`; the live processes and port are clean.
- Technical checks do not prove screen-reader, speech-input, switch-access, multi-browser, OS-native high-contrast, real-user comprehension, participant validation, or scientific validity.
- No commit, merge, push, deployment, release, registry update, or worktree cleanup outside this lane has been performed.
