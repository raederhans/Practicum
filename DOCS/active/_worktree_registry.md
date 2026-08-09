# Worktree Registry

| Worktree / path | Task | Base branch / commit | Current branch / HEAD | Goal | State | Hotspots | Tests | Overlap | Order | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `C:\Users\raede\Desktop\essay help master\Practicum` | P1-P3 orchestration owner | `main` / `2d8112104ab57ef7f6452ef99dc05ff9bbfb4427` | `main`; synchronized with `origin/main` at the orchestration checkpoint | Supervise P1 reproducibility, P2 understanding/accessibility, and P3 dashboard performance; own review/integration/push/cleanup | `in-progress` | Shared task records, registry, Git topology, integration and production-deployment gate | P2 evidence received; P1/P3 supervision active; combined validation deferred until remaining lane processes stop | User evolution-research WIP remains untracked and excluded; duplicate P3 task archived | 0 | Receive P1/P3 delivery packages, review all three, then integrate P3 -> P2 -> P1 |
| `C:\Users\raede\.codex\worktrees\3fa1\Practicum` | P1 scientific reproducibility | `main` / `a79921b2becb81388762b10a365744d014026198` | detached `a79921b`; thread `019fe697-b3b0-7782-a3e8-f6fb37853c26` | Audit and improve machine-checkable public/restricted reproducibility without overstating scientific validation | `ready-for-integration` | `project/data/manifests`, modeling, modeling tracking, scripts, corresponding Python tests | 104 passed, 2 expected skips, 7 subtests; no lane-owned process; full-upstream remains explicitly blocked | Disjoint from P2/P3; EAGLE-I restricted path is Git-tracked and requires separate owner review; no full-reproduction claim | 3 | Primary owner reviews atomic 5-file change and integrates after P3/P2 |
| `C:\Users\raede\.codex\worktrees\3ab1\Practicum` | P2 understanding/accessibility | `main` / `a79921b2becb81388762b10a365744d014026198` | detached `a79921b`; thread `019fe697-c99b-7273-b3a6-18e9e58c6a13` | Audit live interpretation/accessibility, implement minimal repairs/tests, and prepare an honest real-user study instrument | `ready-for-integration` | `project/nightlight-public/**`; lane used 43189 and released it | 119/119 tests; 24/24 local route/width checks; owned browser/server stopped and artifacts removed | Disjoint from P1/P3; public-app changes require primary production review before main push | 2 | Primary owner reviews and integrates after P3; no participant findings claimed |
| `C:\Users\raede\.codex\worktrees\3a29\Practicum` | P3 dashboard performance/architecture | `main` / `a79921b2becb81388762b10a365744d014026198` | detached `a79921b`; thread `019fe6b2-49d7-7511-8a7b-da806bae3020` | Measure the MapLibre bundle and implement only a safe evidence-backed performance improvement or justified no-op | `in-progress` | `project/nightlight-dashboard/**`; new verified-free high port only; 5174/5175 excluded | Baseline-first build and manifest measurement active | Disjoint from P1/P2; duplicate `b19a` task stopped and archived | 1 | Reach evidence-backed ready/no-op handoff with before/after metrics and teardown |

States: `in-progress`, `blocked`, `ready-for-review`, `ready-for-integration`, `integrated`, `abandoned`.

## Delivery package

- Summary: `codex/personal-project-sync` integrated; Atlas branch absorbed; original Generalization branch superseded by the LF-safe integrated version; stale public `study.js` hash repaired; local/remote main synchronized; workflow-driven GitHub Pages enabled and published.
- Files: 439 candidate files plus the two-line provenance hash repair and closeout records. Generated Playwright artifacts were removed; research WIP was preserved.
- Diff from base: candidate was 13 commits ahead/0 behind `main`; one narrow repair commit followed after full Python validation exposed the stale pointer.
- Commit and branch state: only local `main` remains; normal pushes only; no rebase, force-push, teammate-remote write, or history rewrite.
- Divergence from main: none after closeout; `main` is the sole registered worktree branch.
- Overlap and conflict risk: incoming app sources did not overlap primary WIP; apparent app directories contained only ignored dependencies/build output.
- Validation evidence: see `DOCS/archive/repository-integration-20260809/task.md` and ignored logs under `cache/logs/repository-integration-20260809/`.
- Publication evidence: run `31310854793` succeeded at `70dae02`; the live 11-file manifest matched byte length and SHA-256; five hash routes rendered with zero console errors/warnings and no page-level overflow at mobile/desktop widths.
- Unverified risks: no full raw-input scientific reproduction; no new independent exact-SHA sign-off; no manual assistive-technology study; pinned Actions now emit Node 20 deprecation warnings while GitHub forces Node 24.
- Recommended integration method: future work starts from `main` and uses one bounded feature worktree with explicit ownership and closeout gates.

## Cleanup evidence

- Removed Git worktrees: `c24c` Generalization source, `814c` Atlas remediation, and the temporary repository integration worktree.
- Deleted local branches: `codex/personal-project-sync`, `codex/atlas-compare-audit-remediation`, `codex/generalization-autopsy-phase1`, `modeling-6events`, `push-clean-modeling`, `push-clean-modeling-docs`, and the temporary integration branch.
- Recovery: `a7a577e`, `992fe58`, `5136851`, and `bf1a7e3` remain reachable through `main` or `teammate/main`. The superseded `5607c07` patch is preserved under `C:/Users/raede/.codex/integration-backups/Practicum/20260809-repository-integration/`.
- Non-Git residue: empty `c24c/Practicum` and `814c/Practicum` directory shells may remain until the Codex/Windows handle owner releases them; neither is registered and neither contains files.
