# Generalization Autopsy Phase 1 Plan

## Goal

Deliver the first shippable personal-project increment: an accessible public shell, a fail-closed `Public Generalization Artifact v1`, and a one-page Generalization Autopsy experience inside the existing Findings route.

The product question is:

> Why can a disaster-recovery model look useful in known events yet fail to travel to the next event?

This phase must make model-role boundaries, negative results, data quality, and claim limits inspectable without publishing restricted raw data or implying community recovery rankings.

## Ownership and coordination

- Implementation owner: the newly created Codex project thread in its isolated worktree.
- Supervisor and future integration owner: the current root thread in `C:\Users\raede\Desktop\essay help master\Practicum`.
- The implementation thread may edit only its own worktree, run scoped validation, and prepare a delivery package.
- It must not push, deploy, merge into the supervisor branch, rewrite history, clean another worktree, or discard unrelated changes.
- It must report `ready-for-integration` with commit/branch state, diff summary, validation evidence, and remaining risks.

## Phase 1 scope

### Stage 0: accessibility admission gate

Complete the minimum prerequisites before expanding the analytical surface:

1. Mobile navigation
   - At 320, 373, 375, and 768 CSS pixels, every route remains discoverable.
   - The active route is visible after direct navigation and route changes.
   - Keyboard focus remains visible and usable.
   - No page-level horizontal overflow is introduced.
   - A horizontal navigation design is acceptable only if its scrollability is discoverable and the active item is automatically brought into view.

2. Text contrast
   - `--faint` must not carry essential normal text.
   - Essential normal text targets WCAG 2.2 AA contrast of at least 4.5:1.
   - Tiny metadata or navigation labels must be enlarged, promoted to `--muted`, or made non-essential.

3. Chart alternatives
   - Every chart that communicates a substantive result has a nearby plain-language summary.
   - Substantive chart data has a semantic table or equivalent accessible representation.
   - Color is not the only carrier of status or comparison.

### Stage 1: Public Generalization Artifact v1

Define and implement a versioned, build-time-only public contract. Prefer the narrowest existing data/content responsibility; do not add a runtime API or general data directory.

Every published value must include or inherit:

- artifact version;
- generated date;
- source artifact identifier and SHA-256 or equivalent immutable version evidence;
- event cohort;
- sample-lock or analysis version;
- validation design;
- model family;
- model role;
- metric name;
- metric type: `description`, `ranking`, or `calibration`;
- value and unit;
- quality/publication status;
- supported claim;
- unsupported claim;
- source/license/attribution status.

The contract and verifier must reject:

- facility names or coordinates;
- pixel, grid, ZIP-event, or other reversible row identifiers;
- probability surfaces, rasters, or geographic grids;
- raw or fine-grained time series;
- outage-duration or recovery-time records;
- model binaries or serialized estimators;
- credentials, local absolute paths, or runtime network access;
- unreviewed EAGLE-I-derived aggregates;
- metrics without a declared cohort, validation design, role, and source lineage;
- combinations capable of reconstructing restricted detail.

Use already reviewed aggregate sources when possible, including the existing canonical public-results manifest and model-role matrix. If publication rights for a candidate value are unclear, keep it unavailable and show the limitation; never fabricate, mock, or silently substitute another metric.

### Stage 2: one-page Generalization Autopsy MVP

Extend the existing Findings route rather than creating a platform or new primary route.

Required sections:

1. `The attractive result`
   - Explain the within-sample explanatory result.
   - Label it `explanatory`, `in-sample`, and `fixed-control` where applicable.
   - Do not present its R-squared as future-event accuracy.

2. `The harder test`
   - Explain leave-one-event-out or the applicable cross-event validation design.
   - Do not place unlike metrics such as R-squared and AUC on a shared quantitative scale.

3. `What improved / what failed`
   - Show the model-role matrix: explanatory, damage ranking, recovery transport, and secondary interpretation.
   - Separate ranking from calibration.
   - Make negative and inconclusive results first-class evidence.

4. Two or three evidence cards
   - Each card shows task, evidence, metric role, quality flag, supported claim, unsupported claim, cohort/version, and source lineage.
   - Use only values admitted by `Public Generalization Artifact v1`.

5. `Decision ledger`
   - Use explicit statuses such as `mainline`, `interpretive only`, `sensitivity only`, `repair first`, or `discontinued` only when the source evidence supports them.

6. Mandatory plain-language boundary

   > This is an analysis of model transport failure, not a ranking of community recovery.

Update Methods only as needed to explain the artifact, metric roles, and claim boundary. Preserve collaborator attribution and the existing authorship/provenance story.

## Expected implementation surface

Use repository evidence to confirm exact paths before editing. Likely surfaces include:

- `project/nightlight-public/src/content/` for the allowlisted public artifact;
- `project/nightlight-public/src/views/FindingsView.vue`;
- `project/nightlight-public/src/views/MethodsView.vue` if needed;
- `project/nightlight-public/src/styles/main.css`;
- `project/nightlight-public/DATA_POLICY.md` and provenance/credit surfaces only where the new contract requires it;
- `project/nightlight-public/scripts/verify-public.mjs` and its existing tests;
- existing public-site tests, extended rather than duplicated;
- a narrow offline generator under the existing modeling support/pipeline responsibility if machine generation is justified.

Do not modify README files unless a concrete build or contract requirement makes it unavoidable.

## Verification matrix

### Contract and scientific checks

- Schema validation covers required fields, enums, units, source lineage, and prohibited fields.
- Every displayed metric can be regenerated or traced to a fixed local artifact and hash/version.
- Cohort, target, sample lock, and validation design are compatible for every explicit comparison.
- Description, ranking, and calibration labels are tested and cannot be silently interchanged.
- Negative tests prove restricted or reconstructable records are rejected.

### UI and accessibility checks

- Existing route tests remain green.
- Findings sections, evidence cards, claim boundaries, tables, headings, and accessible SVG labels are tested.
- Keyboard focus and active navigation behavior are verified.
- 320/373/375/768/desktop browser checks cover all five routes.
- No page-level horizontal overflow and no hidden active route.
- No console errors or warnings attributable to the change.
- Reduced-motion behavior remains intact.

### Release checks

- Run the repository's supported targeted tests first.
- Run the complete public-site validation entry point, including test, build, public verifier, and release-manifest checks.
- Verify both GitHub Pages base-path behavior and Vercel/root-path behavior if the existing validation surface supports both.
- Confirm zero runtime external requests and no forbidden public artifacts.

If a long build, dev server, or browser smoke is needed, explicitly use `$orchestrate-live-tests`, record the single owner, port, logs, and teardown in `context.md`, and do not collide with existing services.

## Acceptance criteria

- Phase 0 accessibility defects are fixed and freshly verified.
- `Public Generalization Artifact v1` is versioned, fail-closed, tested, and documented at the contract boundary.
- Findings presents the Generalization Autopsy MVP with accurate role/metric language and accessible alternatives.
- All public values are traceable and publication-reviewed; unclear values stay withheld.
- No restricted raw data, fine-grained derivative, runtime API, mock, or silent fallback is introduced.
- Existing authorship and collaborator attribution remain truthful.
- Full supported validation is green, or an environmental gap is isolated and explicitly documented without claiming completion.
- The implementation owner updates `task.md` and `context.md`, then returns a complete `ready-for-integration` delivery package.

## Non-goals

- No Failure Atlas or event-map redesign.
- No Evidence Passport rollout across all events.
- No NASA/NOAA/FEMA/EIA ingestion or new external dataset.
- No facility ledger, exact locations, probability map, raw imagery, or daily recovery sequence.
- No live monitoring, analytics, accounts, backend, or runtime API.
- No community recovery, equity, resilience, policy-blind-spot, causal, or future-prediction claim.
- No deployment, remote push, main-branch merge, force push, history rewrite, or worktree cleanup.

## Stop conditions

Stop and report upward if:

- a required public metric lacks publication rights or traceable source lineage;
- the only implementation path would expose restricted/reconstructable data;
- cohort or validation incompatibility prevents an honest comparison;
- a product decision would change the agreed scientific claim boundary;
- unrelated user WIP overlaps the required file surface;
- three consecutive runs fail under the same assumption;
- credentials, external production authority, destructive cleanup, or remote deployment would be required.
