# Practicum P1-P3 Orchestration Plan

## Goal

Run three isolated implementation tasks for P1 scientific reproducibility, P2 user-understanding/accessibility validation, and P3 dashboard performance/architecture; supervise their evidence, then integrate only verified, non-overlapping deliverables into `main` in a controlled order.

## Base and authority

- Integration owner: the primary task that created this plan.
- Starting repository baseline: `main@a79921b2becb81388762b10a365744d014026198`, synchronized with `origin/main` before task creation.
- Execution tasks may edit only their assigned worktree paths and run scoped validation. They must not stage, commit, rebase, merge, push, edit refs, change worktree topology, update this umbrella record, or edit `_worktree_registry.md`.
- The primary task alone owns final review, Lore commits, sequential integration, push decisions, registry synchronization, and cleanup.

## Independent lanes

### P1: scientific reproducibility and provenance

- Owned product paths: `project/data/manifests/**`, `project/modeling/**`, `project/modeling_tracking/**`, `project/script/**`, and directly corresponding tests under `project/tests/**`.
- Do not edit `project/nightlight-public/**`, `project/nightlight-dashboard/**`, `.github/**`, or user-owned research records.
- Establish what can actually be reproduced from available public inputs, add machine-checkable receipts/entrypoints/tests where supported, and record restricted or missing-input blockers without fabricating completion.
- Passing software tests must remain separate from scientific validity and independent sign-off.

### P2: user understanding and accessibility

- Owned product paths: `project/nightlight-public/**` only, plus a lane-local task record if needed.
- Do not edit model outputs, data manifests, dashboard code, `.github/**`, the umbrella record, registry, or user-owned research records.
- Perform live technical accessibility and interpretation-risk QA, implement the smallest evidence-backed repairs/tests, and create an executable real-user study protocol/instrument.
- Do not invent participant results. A missing human study is an explicit remaining gate, not a reason to fabricate evidence.

### P3: dashboard performance and architecture

- Owned product paths: `project/nightlight-dashboard/**` only, plus a lane-local task record if needed.
- Do not edit public-app code, scientific pipelines/data, `.github/**`, the umbrella record, registry, or user-owned research records.
- Measure the existing MapLibre/Vite bundle and route-loading behavior, implement only a demonstrably useful low-risk improvement, and preserve dashboard behavior. A measured no-op is acceptable if the warning cannot be reduced safely.
- No new dependencies without explicit authorization.

## Live-process ownership

- P1 owns only Python/modeling processes started in its worktree. It must not mutate or substitute shared ignored raw-data caches without recording exact ownership and recovery.
- P2 owns any browser/dev-server process in its worktree and must use a dedicated port such as `5174` or an isolated preview process. It must close the process and remove generated browser artifacts before handoff.
- P3 owns any dashboard browser/dev-server process in its worktree and must use a different dedicated port such as `5175`. It must close the process and preserve baseline/final bundle evidence.
- The primary task will not duplicate or poll lane-owned live processes. It reads task status and completed evidence only.

## Stages

- [x] Stage 0: verify the clean integration baseline, load workflow rules, define path and live-resource ownership, and prepare this record.
- [x] Stage 1: create three isolated Codex worktree tasks and record thread/worktree identities.
- [x] Stage 2: supervise P1/P2/P3 until each is `ready-for-integration`, `blocked` with evidence, or intentionally no-op.
- [x] Stage 3: independently review each delivery package, changed paths, validation, scientific/product claims, and cross-lane overlap.
- [x] Stage 4: integrate eligible work in order P3, then P2, then P1; validate after each integration and resolve only evidence-backed conflicts.
- [ ] Stage 5: run final combined validation, decide whether any public-app change is authorized for production push, synchronize Git/registry, and clean eligible worktrees.

## Acceptance criteria

- Every lane starts from the same verified main baseline and has an isolated worktree.
- Each lane stays inside its file ownership and supplies a full delivery package: summary, changed files, relative diff, Git state, overlap, tests, risks, and recommended integration method.
- P1 never converts missing/restricted input evidence or passing code tests into a claim of full scientific reproduction.
- P2 provides technical accessibility evidence and an honest human-study boundary; no participant data is invented.
- P3 supplies before/after bundle evidence and does not trade correctness for a smaller warning.
- The primary task performs sequential review/integration; lane completion is not treated as merged completion.
- User-owned `DOCS/archive/personal-project-evolution-research/` remains untouched and untracked.
- Any external production deployment caused by a P2 public-app integration is handled as a distinct main-task authorization/verification decision.

## Non-goals

- Do not include P0 GitHub Actions Node-runtime maintenance in these three lanes.
- Do not perform a wholesale scientific-method rewrite, add dependencies, or publish restricted/fine-grained data.
- Do not fabricate users, raw inputs, successful reruns, exact-SHA sign-off, or performance gains.
- Do not let execution tasks commit, push, merge, clean worktrees, or update shared coordination files.

## Integration order and rationale

1. P3 first: narrow dashboard-only performance surface with the lowest semantic risk.
2. P2 second: public product behavior and acceptance tests, reviewed before any production-triggering push.
3. P1 last: broadest scientific/provenance surface and highest claim risk; it receives the strongest evidence and combined-test review.
