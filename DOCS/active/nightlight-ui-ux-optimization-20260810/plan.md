# Nightlight UI/UX Optimization Parallel Plan

## Goal

Execute the approved UI/UX optimization Phases A-D in four isolated Codex tasks, then integrate them sequentially into one verified candidate without weakening the aggregate-only, local-assets-only, scientific-interpretation, CSP, or release-verifier boundaries.

## Exact baseline

- Repository base before the local baseline commit: `d0ef4c33a837b2e7a04b59f0d2d24ee24d304f9d` (`main == origin/main`).
- Shared implementation baseline branch: `codex/nightlight-ui-ux-base`.
- The baseline includes the already verified development-CSP compatibility repair and `DOCS/nightlight-public-ui-ux-audit-20260810.md`.
- Unrelated untracked `DOCS/archive/personal-project-evolution-research/` is protected WIP and must not be staged, edited, copied into delivery commits, or deleted.

## Parallel lanes and ownership

| Lane | Scope | Exclusive writable product files | Test ownership | Live resources |
| --- | --- | --- | --- | --- |
| A | Accessibility and shell stabilization | `src/App.vue`, `src/main.js` | `tests/static-shell.test.js` | port 5181; phase-A temp/log root |
| B | Information hierarchy and long-form pages | `src/views/OverviewView.vue`, `FindingsView.vue`, `MethodsView.vue`, `CreditsView.vue` | `tests/generalization-artifact.test.js`, `tests/routes.test.js` | port 5182; phase-B temp/log root |
| C | Atlas workflow and URL state | `src/views/AtlasView.vue`, existing Atlas domain modules only when directly required | existing Atlas-specific test files, excluding files owned by A/B/D | port 5183; phase-C temp/log root |
| D | Visual system, responsive CSS, motion, and maintainability | `src/styles/main.css`, `index.html` | `tests/platform-boundary.test.js` | port 5184; phase-D temp/log root |

No lane may edit another lane's exclusive files. New source/test files are prohibited by default because the public verifier is fail-closed; an unavoidable new-file request must be reported to the integration owner instead of weakening the verifier.

## Shared markup and CSS contract

- Lane A may add `mobile-nav-toggle`, `site-nav--open`, and route/skip-focus state hooks; Lane D styles them.
- Lane B may add `page-summary`, `in-page-nav`, `content-section-nav`, and existing page-specific modifiers; Lane D styles them.
- Lane C may add `atlas-task-summary`, `atlas-mobile-view`, `event-selection-summary`, and existing comparison/Atlas modifiers; Lane D styles them.
- Lanes A-C must preserve existing public claims and class names where practical and report every new class to Lane D/integration owner.
- Lane D must not change Vue markup or interaction semantics. It provides graceful styles for both the current markup and declared new hooks.

## Stages

1. Create and verify the shared local baseline commit.
2. Dispatch A-D into four project worktrees from the exact baseline.
3. Each lane implements within ownership, records fresh targeted validation, and ends `ready-for-integration`; no child lane commits, pushes, merges, rebases, stages, or cleans worktrees unless the user follows up in that task and explicitly grants it main/integration ownership.
4. Primary integration owner inventories all four deliveries, computes file and semantic overlap, and integrates in order A -> B -> C -> D.
5. After each lane, run the smallest matching tests; after all lanes, run full public validation and one owned browser matrix.
6. Update registry/task records, preserve recovery references, and report push/deployment state truthfully.

## Acceptance criteria

- Phase A: Skip link works with hash history; mobile menu semantics and route focus behavior are browser-verifiable; no giant H1 debug-style outline once Lane D styles land.
- Phase B: inner-page hierarchy is compact; Findings figcaption warning is removed; long pages expose structured summaries/navigation without deleting scientific limitations.
- Phase C: Atlas mode/filter/event/comparison state is URL-backed and reload/back/forward safe; mobile task order is improved without introducing a score, ranking, precise map, or restricted field.
- Phase D: mobile horizontal primary-nav scrollbar is removed through an accessible menu presentation; functional type/control sizes improve; declared page/Atlas hooks are styled; motion remains reduced-motion safe; no new dependency.
- Five routes plus Atlas Compare retain no document-level horizontal overflow at 320/375/391/768/1440 CSS px.
- Cold keyboard flow, route focus, Atlas controls, 200% reflow, text spacing, forced colors, reduced motion, console warning/error, same-origin network, production build, and public verifier gates are explicitly reported.
- `Unavailable` and `Not assessed` never become zero; no recovery/safety/risk/similarity score or future-event claim is introduced.

## Non-goals

- No deployment, remote push, history rewrite, dependency upgrade, new primary route, analytics, backend/API, external runtime request, raw/fine-grained data, model change, or scientific-validation claim.
- No cleanup or mutation of retained worktree `591a`.
- No edit to protected research WIP.

## Risks and integration order

- A-D are parallel execution lanes, not four automatically mergeable releases. Their semantic dependency is A shell -> B content -> C Atlas -> D styling.
- D intentionally owns the monolithic stylesheet to avoid four-way text conflicts; integration may require adapting D selectors to the final A-C markup.
- The existing root dev server on port 5173 remains owned by the primary task and must not be polled, restarted, stopped, or interpreted by child lanes.
- Full validation currently has a known unrelated missing-evidence-file blocker; no lane may fabricate that file or weaken the gate.
