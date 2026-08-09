# Practicum P1-P3 Orchestration Task

## Current status

`complete` — P3, P2, and P1 were integrated into `main` in the planned order, passed fresh combined validation, synchronized to `origin/main`, and published through the existing workflow-mode GitHub Pages configuration.

## Checklist

- [x] Load the task-record, worktree-integration, and live-test ownership workflows.
- [x] Verify the exact main/origin baseline, worktree topology, registry, and preserved user WIP.
- [x] Reconcile the previous plan with current code surfaces and scientific/accessibility boundaries.
- [x] Define disjoint P1/P2/P3 path ownership and live-resource ownership.
- [x] Create P1, P2, and P3 Codex worktree tasks from the project default branch.
- [x] Record task IDs, host IDs, worktree paths, branches, and initial status.
- [x] Monitor the three lanes through compact task snapshots; route blockers and prevent scope drift.
- [x] Receive full `ready-for-integration`, blocked, or no-op delivery packages from all lanes.
- [x] Review each diff and validation independently; classify cross-lane overlap and integration order.
- [x] Integrate eligible lanes sequentially with fresh validation after each.
- [x] Run combined validation, decide the public deployment boundary, synchronize main/origin and registry, and clean eligible worktrees.

## Remaining risks

- P1 may prove that restricted or missing raw inputs prevent complete rerun. That is an expected evidence outcome, not failure by itself.
- P2 cannot claim real-user results without actual participants. It can implement technical fixes and prepare a study instrument, but must preserve the human-validation gap.
- P3 may find MapLibre's library payload is already isolated and not safely reducible. It must not introduce brittle splitting or regress map behavior merely to silence a warning.
- The public change was deployed successfully, but real-participant and manual assistive-technology evidence still does not exist.
- The user research directory is unrelated untracked WIP and must never enter lane branches or integration commits.
- The stopped duplicate P3 directory `C:\Users\raede\.codex\worktrees\b19a\Practicum` is no longer registered as a Git worktree, but Windows policy blocked recursive deletion of its generated dependency/build residue; do not treat it as an active lane or integration source.
- The canonical P3 directory `C:\Users\raede\.codex\worktrees\3a29\Practicum` is also no longer registered; Windows removed the Git topology entry but denied deleting the directory body. P1/P2 directories were deleted completely.

## Release evidence

- `main` release SHA: `2a47f363dc768c7dbb43f11f3457159c6d6d8a80`.
- GitHub Pages run `31317312185`: build and deploy succeeded for the exact release SHA.
- Pages configuration: workflow mode, public, HTTPS enforced.
- Live URL: `https://raederhans.github.io/Practicum/`.
- Live release manifest: 11/11 files matched their deployed byte counts and SHA-256 values.
- Live browser: initial focus remained on `BODY`; Atlas navigation produced a unique title, focused H1, visible 3px outline, no 320px overflow, and zero console errors/warnings.
