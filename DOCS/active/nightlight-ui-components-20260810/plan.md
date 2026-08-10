# Nightlight Public Five-Route UI and Components Plan

## Goal

From exact baseline `ca8292040a402eae1d2e461708a4cc912867efcb`, audit and make the smallest evidence-backed UI/UX, interaction, component-primitive, visual-system, route-focus, and responsive repairs across the five existing public routes: Overview, Study Atlas, Findings, Methods, and Credits / Policy. Deliver a verified `ready-for-integration` package without changing Git state or overlapping the P1/P2/P3/Actions lanes.

## Scope

- Task records: `DOCS/active/nightlight-ui-components-20260810/{plan,context,task}.md` only.
- Product: existing files under `project/nightlight-public/src/views/**`, `src/components/**`, `src/styles/**`, `src/router/**`, and `src/App.vue`.
- Tests: existing UI, route, focus, responsive, and accessibility-adjacent test files whose assertion ownership directly matches this lane.
- Read-only evidence: repository registry, archived Nightlight plans and evidence, current application source/content/domain modules, package/tooling configuration, and worktree `0313` state.

## Sources of truth

- User-supplied A0-A8 execution brief and acceptance criteria.
- Exact detached base/HEAD `ca8292040a402eae1d2e461708a4cc912867efcb`; initially clean.
- Current application source and tests.
- `DOCS/active/_worktree_registry.md` and relevant Nightlight archives.
- Read-only collision scan of `C:\Users\raede\.codex\worktrees\0313\Practicum`.
- Existing public-boundary exact file allowlist in `project/nightlight-public/scripts/verify-public.mjs`.

## Stages

- [x] Stage 0: load workflow contracts; pin base, role, writable boundaries, registry, five-route baseline, and other-worktree path collision state.
- [x] Stage 1: read existing Nightlight archive evidence and current source/tests; produce a per-route task audit separating verified defects, maintainability issues, and hypotheses.
- [x] Stage 2: implement minimal route copy, hierarchy, disclosure, semantic fallback, status, visual token, focus, and responsive/high-contrast repairs only where evidence supports them.
- [x] Stage 3: extend existing directly corresponding tests without creating non-allowlisted files or editing public-boundary tooling.
- [x] Stage 4: establish a single-owner live-process contract; run targeted tests, complete validation, production build, and controlled browser matrix at 320/373/375/768 px plus zoom/reflow, text spacing, keyboard/focus, and forced-colors checks.
- [x] Stage 5: review the full diff, search for regressions and semantic overclaims, run a first-principles simplification pass, stop owned live resources, attempt filesystem cleanup, record the environment-policy blocker, and publish the ready-for-integration handoff.

## Acceptance criteria

- All five existing routes have task-level audit results plus traceable changes or explicit measured no-op decisions; no sixth primary route is added.
- Overview states research purpose/scope and gives clear Atlas/Findings paths without equating proxy evidence with human validation.
- Atlas makes filters, selection, Compare Mode, Evidence Passport, `Not assessed`, and `Unavailable` discoverable without adding a score or ranking.
- Findings separates R-squared/AUC, readiness/admission, and recovery outcomes; every displayed number has range/unit/limitation context.
- Methods exposes an input-to-processing-to-admission-to-public-artifact trace and makes raw/restricted data boundaries discoverable.
- Credits / Policy exposes sources, rights, aggregate-only, local-assets-only, no analytics/external runtime requests, and known limitations in a scannable structure.
- Initial Tab order, skip link, primary navigation, SPA route H1 focus, browser history, Atlas tabs/Compare focus, and the prohibition on initial `scrollIntoView` are covered by code and fresh evidence.
- Technical checks cover 320/373/375/768 px, 200% zoom/reflow, WCAG text spacing, and forced-colors/high contrast. Screen reader, speech input, switch access, additional browsers, and real-user comprehension remain explicitly unknown unless actually tested.
- `Unavailable`, `Not assessed`, and errors are never represented as zero; readiness/admission is never described as recovery performance.
- No new dependencies, route, backend, analytics, external runtime request, global store, public-boundary tooling change, package/lockfile change, Vite change, deployment, or Git mutation.

## Non-goals

- No P1 data acquisition or modeling/data edits.
- No P2 proxy evidence execution or P3 performance/Actions maintenance.
- No scientific validation, recovery score/rank, policy recommendation, causal claim, participant/usability validation, or WCAG conformance certification.
- No Atlas state-management rewrite; extraction occurs only when at least two real consumers and a stable behavior boundary are demonstrated.
- No registry, personal-project research, other worktree, release/security tooling, package, lockfile, or Vite changes.

## Risks and constraints

- Public-boundary validation has an exact file allowlist. New component or test files would require a tooling edit owned by another lane, so the default is to reuse allowed files and CSS/semantic primitives; any unavoidable request will be handed to the supervisor rather than changed here.
- Worktree `0313` owns concurrent P2/P3/Actions work. Its state is read-only, and any newly detected overlap freezes only the collided file.
- Automated DOM/browser checks cannot establish assistive-technology support, real-user comprehension, or scientific validity.
- Browser checks share local ports, caches, build output, and logs; they run serially under one recorded owner and are cleaned before handoff.
