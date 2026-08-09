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
| P1 reproducibility | `019fe697-b3b0-7782-a3e8-f6fb37853c26` / `local` | `C:/Users/raede/.codex/worktrees/3fa1/Practicum`; detached `a79921b` | Modeling, manifests, scripts, matching Python tests | No remaining lane-owned process or generated test artifact | `ready-for-integration` | 104 passed, 2 expected private-source skips, 7 subtests; reviewed-output receipts pass, full-upstream and modeling entrypoint fail closed with explicit blockers |
| P2 understanding/accessibility | `019fe697-c99b-7273-b3a6-18e9e58c6a13` / `local` | `C:/Users/raede/.codex/worktrees/3ab1/Practicum`; detached `a79921b` | `project/nightlight-public/**` | Lane used 43189; browser/server stopped and port released; 5174/5175 untouched | `ready-for-integration` | 119/119 public tests, production-preview browser evidence, minimal focus/title repairs, and an executable blank human-study protocol; no fake participants |
| P3 performance/architecture | `019fe6b2-49d7-7511-8a7b-da806bae3020` / `local` | `C:/Users/raede/.codex/worktrees/3a29/Practicum`; detached `a79921b` | `project/nightlight-dashboard/**` | Lane-owned build/browser/server on a newly verified free high port; 5174/5175 excluded | `active` | Baseline/final metrics, behavior tests, minimal improvement or justified no-op |

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

## Next action

Supervise the canonical P3 baseline-to-final measurement. Then independently review all three delivery packages before sequential integration.
