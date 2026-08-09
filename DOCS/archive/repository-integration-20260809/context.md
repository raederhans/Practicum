# Repository Integration Context

## Current truth

- Integration owner: primary agent in the current task.
- Registered worktrees: only `C:/Users/raede/Desktop/essay help master/Practicum` on `main`.
- Local branches: only `main`; it matches `origin/main` before this documentation-only closeout commit.
- Integrated product commit: `1a45d73b4fd94b476a4773523ee50b81a481bbbb` (`992fe58` candidate plus the minimal reviewed-source hash repair).
- Teammate reference remains `teammate/main@1f63e190ce280852d68945dbfce486075adda69b`; no push or merge was made to that remote.
- Preserved user WIP: untracked `DOCS/archive/personal-project-evolution-research/` with three unchanged files. Generated Playwright snapshots/logs were removed; primary `node_modules/` and `dist/` remain ignored local caches.
- The Codex-managed paths `C:/Users/raede/.codex/worktrees/c24c/Practicum` and `C:/Users/raede/.codex/worktrees/814c/Practicum` are no longer Git worktrees. Windows/Codex left empty, zero-child directory shells after unregistering them.

## Decisions and deviations

| Time | Evidence or decision | Impact |
| --- | --- | --- |
| 2026-08-09 | `codex/atlas-compare-audit-remediation@a7a577e` is an ancestor of `codex/personal-project-sync`. | Do not merge it separately; retain its SHA as recovery evidence until cleanup. |
| 2026-08-09 | `codex/generalization-autopsy-phase1@5607c07` is not an ancestor, but integrated commit `64acae7` has the same intent plus LF-canonical provenance repair and registry updates. | Treat `5607c07` as superseded only after final validation of the integrated version. |
| 2026-08-09 | `push-clean-modeling` and `push-clean-modeling-docs` both point to `5136851`, already an ancestor of `main`. | They are historical branch-name cleanup candidates, not merge candidates. |
| 2026-08-09 | `modeling-6events@bf1a7e3` has no merge base with local `main` and tracks `teammate/main`. | Preserve as a separate-history reference unless its content is proven fully represented elsewhere; never ordinary-merge it. |
| 2026-08-09 | Primary checkout has overlapping untracked app directories. | Perform integration in the clean worktree and compare exact content before synchronizing the primary checkout. |
| 2026-08-09 | File-by-file comparison found that the primary app directories contained only ignored dependency/build outputs and no incoming tracked source. | Fast-forwarding `main` was safe; no user source was overwritten. |
| 2026-08-09 | The full Python suite exposed a stale Phase 1 `study.js` hash after `a7a577e` added the reviewed hazard-family metadata. | Existing regression failed first; commit `1a45d73` updated only the two copies of that reviewed hash. |
| 2026-08-09 | `5607c07` was superseded rather than merged. | Recovery patch saved at `C:/Users/raede/.codex/integration-backups/Practicum/20260809-repository-integration/0001-Make-public-limits-on-model-generalization-inspectab.patch`; stale untracked registry also backed up there. |
| 2026-08-09 | Atlas, personal-project, legacy modeling, teammate-history, and original Generalization local branches all met their recovery gates. | Six local branches and three linked worktrees were removed; only `main` remains. |
| 2026-08-09 | GitHub run `31309394968` completed its build job successfully but deploy failed with Pages API 404. | Code/CI artifact is green; production Pages was not enabled or retried because that changes external publication state. |

## Live process ownership

| Process | Owner | Log path | State |
| --- | --- | --- | --- |
| Dashboard dependency install, tests, and build | Primary integration owner | `cache/logs/repository-integration-20260809/dashboard-*.log` | Complete: install/audit clean, 17/17 tests, production build passed; 803 kB MapLibre chunk warning retained as performance debt. |
| Public app dependency install, focused tests, and full validation | Primary integration owner | `cache/logs/repository-integration-20260809/public-*.log` | Complete: 36/36 Compare/Passport, 19/19 Generalization artifact, 112/112 full validate, 11-file release manifest, source/dist boundary passed. |
| Python repository and restricted provenance/readiness gates | Primary integration owner | `cache/logs/repository-integration-20260809/python-*.log` | Complete: isolated result 94 passed/2 expected skips/7 subtests; primary evidence worktree 96 passed/7 subtests and focused 8/8 with zero skips. |
| GitHub Actions Pages workflow | GitHub Actions; monitored read-only by primary integration owner | `https://github.com/raederhans/Practicum/actions/runs/31309394968` | Complete: build/test/base-path verification/artifact upload passed; deploy failed because the Pages API returned 404 and the repository Pages site is not enabled. |

## Handoff

No active handoff. Continue future work from `main`, create a new isolated worktree only for a new bounded feature, and do not recreate any deleted branch from UI/task-card state without checking Git history and this archive.

## Next step

Use the current `main` baseline for the next roadmap phase. If public GitHub Pages publication is desired, obtain explicit authorization to enable Pages, rerun the existing workflow, then perform post-deployment routing, headers/CSP, console/network, and mobile checks.
