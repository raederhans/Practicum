# Practicum P1-P3 Orchestration Task

## Current status

`in-progress` — three isolated worktrees exist; P1 and P2 are active, while P3 worktree setup is complete but its real thread ID is still pending.

## Checklist

- [x] Load the task-record, worktree-integration, and live-test ownership workflows.
- [x] Verify the exact main/origin baseline, worktree topology, registry, and preserved user WIP.
- [x] Reconcile the previous plan with current code surfaces and scientific/accessibility boundaries.
- [x] Define disjoint P1/P2/P3 path ownership and live-resource ownership.
- [x] Create P1, P2, and P3 Codex worktree tasks from the project default branch.
- [ ] Record task IDs, host IDs, worktree paths, branches, and initial status.
- [ ] Monitor the three lanes through compact task snapshots; route blockers and prevent scope drift.
- [ ] Receive full `ready-for-integration`, blocked, or no-op delivery packages from all lanes.
- [ ] Review each diff and validation independently; classify cross-lane overlap and integration order.
- [ ] Integrate eligible lanes sequentially with fresh validation after each.
- [ ] Run combined validation, decide the public deployment boundary, synchronize main/origin and registry, and clean eligible worktrees.

## Current risks

- P1 may prove that restricted or missing raw inputs prevent complete rerun. That is an expected evidence outcome, not failure by itself.
- P2 cannot claim real-user results without actual participants. It can implement technical fixes and prepare a study instrument, but must preserve the human-validation gap.
- P3 may find MapLibre's library payload is already isolated and not safely reducible. It must not introduce brittle splitting or regress map behavior merely to silence a warning.
- A P2 change under `project/nightlight-public/**` would trigger the existing Pages workflow when pushed to `main`; final production synchronization therefore remains a primary-task release decision.
- The user research directory is unrelated untracked WIP and must never enter lane branches or integration commits.
