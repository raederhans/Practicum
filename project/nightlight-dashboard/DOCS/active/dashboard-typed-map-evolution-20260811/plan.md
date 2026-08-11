# Plan

## Goal

Deliver a committed, ready-for-integration Dashboard evolution pilot covering a bounded TypeScript diagnostic census, legacy artifact conformance boundary, an app-local typed repository for `ChartsView`, a reproducible MapLibre performance matrix, threshold-gated low-risk optimization, and independent non-deploy CI.

## Scope

- `project/nightlight-dashboard/**`
- One new Dashboard-only CI workflow
- This task record directory

The implementation follows the delegated six phases in strict order. A later phase only expands when the preceding diagnostic or schema gate supports it.

## Sources of truth

- Exact starting revision: `6b3de4ee97c5391084538bec84db3b1a1f4e05ed`
- Delegated execution brief from task `019febfc-6511-7051-89c9-60970113a4ea`
- Read-only architecture report from task `019fef1c-349e-7ee2-992e-405a8ce4e562`
- Current Dashboard source, artifacts, tests, package lock, and generated build manifest
- Fresh targeted tests, typecheck, build, bundle analysis, browser measurements, and audit evidence produced by this task

## Stages

- [x] Phase 1 — `NL-TS-01`: add pinned browser/Node typecheck entry points and classify real diagnostics as defect, schema ambiguity, annotation noise, or third-party issue.
- [x] Phase 2 — `NL-DATA-01`: inventory exporter and wire shapes, preserve zero/null/unavailable distinctions, and add conformance fixtures/tests plus a migration boundary.
- [x] Phase 3 — `NL-SVC-01`: only after go conditions, add Dashboard-local ports/repositories/schemas/errors and migrate `ChartsView` while retaining a rollback-compatible loader boundary.
- [x] Phase 4 — `NL-PERF-01`: extend ready-state instrumentation and run the owned cold/warm, network, CPU, viewport, DPR, route-cycle, basemap, memory, worker, long-task, layer/source, and failure matrix.
- [x] Phase 5 — `NL-PERF-02`: move MapLibre CSS to the lazy route and implement no other optimization unless the retained measurements cross a documented threshold.
- [x] Phase 6 — `NL-CI-01`: add independent non-deploy Ubuntu/Windows CI for Dashboard tests, typecheck, build, and bundle analysis when no equivalent workflow exists.
- [ ] Delivery — run scoped verification and audits, remove task-owned live resources and temporary output, retain formal measurement evidence, and create reversible Lore commits.

## Acceptance criteria

- Diagnostics include counts and evidence-backed go/no-go reasoning; no blanket `any`, undated `@ts-ignore`, or strictness-disabled shortcut.
- Legacy artifacts have explicit `legacy-v0` or versioned classification, missingness/fallback semantics, conformance fixtures, and migration rules; numeric fallbacks are not promoted to scientific truth.
- Repository behavior covers abort, network, 404, invalid JSON, invalid schema, unsupported version, validated-success caching, invalid non-caching, and unavailable-not-zero behavior.
- Performance evidence retains every formal sample, separates canvas/style/overview/detail/basemap/external milestones, and records failures without shortened timeouts.
- Home requests no MapLibre JS or CSS after the route-local CSS change; measured impact is described conservatively.
- Other worker/LRU/source-eviction/lifecycle/renderer work is either threshold-supported and verified or explicitly recorded as no-op.
- Independent Dashboard CI does not deploy and does not modify or couple the Public Pages workflow.
- Targeted tests plus Dashboard full tests, typecheck, build, bundle analysis, resource boundary, browser/failure recovery, workflow parsing/equivalent commands, diff checks, and secret/large/generated review are reported honestly.

## Non-goals

- No changes to `project/nightlight-public/**`, `project/modeling/**`, or `project/data/**`.
- No edits to the existing Public Pages deployment workflow or root `_worktree_registry.md`.
- No scientific reinterpretation or repair of Python model values.
- No cross-Public shared core before parity from two real consumers.
- No God service, service locator, speculative API, Dashboard deployment, Pages claim, or production-user performance claim.
- No main/ref/worktree topology changes, integration, push, or deployment.

## Risks and constraints

- Parallel Modeling/Data and Public App lanes may create unrelated changes elsewhere; this task must not absorb or revert them.
- Untracked research WIP must be preserved and excluded from commits.
- Dashboard lacks installed dependencies at the starting revision; missing packages are an environment prerequisite, not a product defect.
- External basemap and WebGL behavior can fail independently of application code and must remain separately classified.
- Browser/performance/build ownership is exclusive to this task and must be registered before each long or shared process.
