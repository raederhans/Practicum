# Practicum P1-P3 Orchestration Context

## Current truth

- Primary integration owner: this task.
- Baseline: local `main == origin/main == a79921b2becb81388762b10a365744d014026198` before lane creation.
- Existing Git topology before lane creation: one local branch and one registered worktree, both `main`.
- Preserved user WIP: three untracked files under `DOCS/archive/personal-project-evolution-research/`; excluded from every lane and integration action.
- Pages remains live from product SHA `70dae02`; the later `a79921b` commit is documentation-only with zero public-app/workflow diff.
- Existing evidence: dashboard 17/17 plus build with an approximately 803 kB MapLibre chunk warning; public app 112/112 plus live five-route verification; Python 96 passed plus 7 subtests and focused Generalization/H4 8/8 zero-skip.
- Scientific boundary: H4 is reviewed-output consistency, not full upstream reproduction; the historical independent sign-off remains tied to `992fe58`.

## Lane matrix

| Lane | Thread / host | Worktree / branch | Owned paths | Live resources | State | Handoff requirement |
| --- | --- | --- | --- | --- | --- | --- |
| P1 reproducibility | `019fe697-b3b0-7782-a3e8-f6fb37853c26` / `local` | Removed worktree; candidate `f184220`, integrated as `1d2a0b1` | Modeling, manifests, scripts, matching Python tests | No remaining lane-owned process or generated test artifact | `integrated` | Main: 106 passed + 7 subtests; reviewed-output receipts pass, full-upstream exits 1, and full-run exits 2 before model loading |
| P2 understanding/accessibility | `019fe697-c99b-7273-b3a6-18e9e58c6a13` / `local` | Removed worktree; candidate `9ef90ae`, integrated as `e7e95be` | `project/nightlight-public/**` | Main smoke used 43217 and released it; browser closed; 5174/5175 untouched | `integrated` | Main: 119/119, production build/boundary, BODY on first load, route H1 focus/title/outline, four widths without overflow, zero console errors |
| P3 performance/architecture | `019fe6b2-49d7-7511-8a7b-da806bae3020` / `local` | Worktree registration removed; candidate `1ecd6c9`, integrated as `82d4f97` | `project/nightlight-dashboard/**` | No remaining lane-owned process; 5174/5175 excluded | `integrated` | Main: 20/20 and analyze:bundle; MapLibre remains isolated from home at 803,051 B raw / 217,871 B gzip; runtime assets unchanged |

## Shared-process rules

- Lane owners may run short isolated tests directly in their worktree.
- Any long pipeline, build, dev server, browser run, or checkpoint builder has exactly one lane owner and a stable log path recorded in that lane's commentary/task record.
- The primary task supervises task status through Codex task snapshots. It does not start, stop, retry, or interpret a lane-owned live process while the lane is active.
- Cross-lane combined validation starts only after every lane has stopped its live processes and reached a terminal handoff state.

## Decisions

| Time | Decision | Reason |
| --- | --- | --- |
| 2026-08-09 20:53 +08:00 | Use three separate Codex worktree tasks, not same-directory forks or native subagents. | The user explicitly requested different sessions; each lane needs visible ownership and independent Git state. |
| 2026-08-09 20:53 +08:00 | Keep P1, P2, and P3 on disjoint product directories and forbid shared registry/umbrella edits. | Prevent file conflicts and make integration evidence auditable. |
| 2026-08-09 20:53 +08:00 | Accept honest blocked/no-op lane results. | Restricted data, absent human participants, or non-actionable bundle warnings must not be converted into fabricated progress. |
| 2026-08-09 20:53 +08:00 | Defer production push decisions for P2 public-app changes to the primary task. | A normal main push may automatically deploy GitHub Pages and therefore needs a separate release review. |
| 2026-08-09 20:56 +08:00 | All three creation requests returned and three detached worktrees exist at exact baseline `a79921b`; P1 and P2 have real active thread IDs, while P3 still exposes only its setup client ID. | Register P1/P2 as active and P3 as setup-pending; do not create a duplicate P3 task or treat its worktree as an active execution lane until the real thread appears. |
| 2026-08-09 21:25 +08:00 | The original P3 setup materialized as thread `019fe6b2-49d7-7511-8a7b-da806bae3020` in `3a29`. A second request had already been issued after the user identified the missing start; it materialized in `b19a` and was stopped as a duplicate no-op. | Keep `3a29` as the sole canonical P3 lane. The duplicate thread is archived; its Git worktree registration was removed. Recursive disk deletion of the generated `b19a` residue was blocked by policy and remains an explicit cleanup gap. |
| 2026-08-09 21:56 +08:00 | P3, P2, and P1 were committed and integrated in the planned order as `82d4f97`, `e7e95be`, and `1d2a0b1`; fresh main validation passed for each lane. | Authorize the existing Pages workflow release. Keep full-upstream modeling blocked and keep the real-user study result tables blank. |

## Next action

Push the release-ready main branch, wait for the Pages workflow, verify the live site and exact deployment SHA, then close the umbrella task.
