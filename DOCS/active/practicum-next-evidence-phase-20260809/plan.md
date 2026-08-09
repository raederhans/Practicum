# Practicum Next Evidence Phase Plan

## Goal

Continue the original P1, P2, and P3 Codex tasks from synchronized `main@223fc653dba2768dad99df9d032beaedd9234d6a`, with fresh isolated worktrees and evidence boundaries that remain honest about scientific reproducibility, proxy user research, and measured performance.

## Authority and integration boundary

- Integration owner: the primary task that created this record.
- P1, P2, and P3 are execution owners only. They may edit their assigned paths and run scoped validation, but must not stage, commit, merge, rebase, push, change refs, alter worktree topology, edit this umbrella record, or edit `DOCS/active/_worktree_registry.md`.
- The primary task alone reviews delivery packages, creates Lore commits, integrates sequentially, validates `main`, pushes, publishes if a product path triggers Pages, and cleans eligible worktrees.
- User-owned `DOCS/archive/personal-project-evolution-research/` remains untouched and untracked.

## P1: provenance and full-upstream readiness

- Owned paths: `project/data/manifests/**`, `project/modeling/**`, `project/modeling_tracking/**`, `project/script/**`, and directly corresponding `project/tests/**`.
- Audit each current `full-upstream` blocker against repository evidence and official source terms.
- Improve machine-readable receipts, immutable identifiers, license/access classifications, and fail-closed checks only where evidence supports them.
- Do not delete or publish restricted data, rewrite Git history, run a full model while preflight is red, or upgrade H4 beyond reviewed-output consistency.
- Deliver an exact blocker disposition: resolved, externally actionable, permission-gated, unavailable, or still ambiguous.

## P2: research-backed proxy evaluation

- Owned paths: `project/nightlight-public/**` only, including a new lane-local independent next-phase plan.
- Replace the infeasible participant study with an explicitly labeled proxy evaluation using current authoritative guidance, relevant research, static inspection, and technical browser evidence.
- Produce an approximate conclusion with source links, confidence by finding, supporting product evidence, limitations, and claims that remain prohibited.
- Do not invent participants, observations, completion rates, interviews, screen-reader users, or scientific comprehension evidence.
- Create a new independent next-phase plan that can be completed by one project owner without recruiting participants.

## P3: real performance timing and architecture decision

- Owned paths: `project/nightlight-dashboard/**` only.
- Measure production-like home and map navigation timing under a documented repeatable environment using existing tooling and no new dependency.
- Separate transfer, route navigation, MapLibre load/initialization, and any available browser timing signals; report medians and run-to-run variation.
- Preserve current route isolation unless measurements demonstrate a meaningful bottleneck and a low-risk change has before/after evidence.
- A measured no-op remains valid. Bundle size alone is not proof of user-visible latency.

## Live-process ownership

- P1 alone owns any Python/modeling process it starts. It must run fail-closed preflight first and must not mutate shared ignored raw-data caches.
- P2 alone owns any public-app dev server or browser session it starts, on a dedicated high port selected after checking listeners.
- P3 alone owns dashboard build, preview, browser, performance logs, and a dedicated high port. P3 is the sole performance-process owner; the primary task and other lanes do not duplicate its runs.
- Each lane must stop its processes and clean only lane-owned generated artifacts before handoff.

## Stages

- [x] Verify synchronized main, preserved user WIP, old task identities, and existing archive evidence.
- [x] Unarchive the original three Codex tasks and recreate their exact isolated worktree paths at current main.
- [x] Dispatch P1, P2, and P3 with updated scope, ownership, validation, and stop conditions.
- [x] Supervise each task to `ready-for-integration`, evidence-backed `blocked`, or justified no-op.
- [x] Review diffs, citations, measurements, claims, path overlap, and live-process cleanup independently.
- [ ] Integrate eligible deliveries sequentially, validate on main, push, update records, and clean worktrees. Integration, local validation, and P2/P3 lane archival are complete; push, Pages verification, umbrella archival, and cleanup remain.

## Acceptance criteria

- All three original task IDs resume in valid isolated worktrees at the same base SHA.
- P1 makes no full-reproduction or licensing claim without exact evidence and performs no history rewrite.
- P2 clearly distinguishes proxy confidence from participant evidence and supplies a genuinely independent next plan.
- P3 has a reproducible measurement protocol and does not infer latency from bundle bytes alone.
- Each delivery package includes changed files, relative diff, Git state, overlap, exact commands/results, unverified risks, and recommended integration method.
- The primary task independently verifies eligible changes before any completion, push, or deployment claim.

## Non-goals

- Do not create a replacement user cohort, synthetic interview transcript, or invented usability score.
- Do not add dependencies, broaden scientific claims, rewrite remote history, expose restricted data, or modify GitHub Actions in these three lanes.
- Do not reuse the earlier completed archive as an active handoff surface.
