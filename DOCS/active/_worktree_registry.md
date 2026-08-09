# Worktree Registry

| Worktree / path | Task | Base branch / commit | Current branch / HEAD | Goal | State | Hotspots | Tests | Overlap | Order | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `C:\Users\raede\Desktop\essay help master\Practicum` | P1-P3 orchestration owner | `main` / `a79921b2becb81388762b10a365744d014026198` | `main`; synchronized with `origin/main` before orchestration record | Supervise P1 reproducibility, P2 understanding/accessibility, and P3 dashboard performance; own review/integration/push/cleanup | `in-progress` | Shared task records, registry, Git topology, integration and production-deployment gate | Reads lane-completed evidence only while tasks are active; combined validation deferred until lane processes stop | User evolution-research WIP remains untracked and excluded; primary task does not edit lane product paths during execution | 0 | Monitor P1/P2, resolve P3 setup identity, then receive three delivery packages |
| `C:\Users\raede\.codex\worktrees\3fa1\Practicum` | P1 scientific reproducibility | `main` / `a79921b2becb81388762b10a365744d014026198` | detached `a79921b`; thread `019fe697-b3b0-7782-a3e8-f6fb37853c26` | Audit and improve machine-checkable public/restricted reproducibility without overstating scientific validation | `in-progress` | `project/data/manifests`, modeling, modeling tracking, scripts, corresponding Python tests | Lane-owned scoped Python validation; final evidence pending | Disjoint from P2/P3 product paths; must not edit registry, public app, dashboard, or research WIP | 3 | Reach evidence-backed ready-for-integration, blocker, or justified no-op handoff |
| `C:\Users\raede\.codex\worktrees\3ab1\Practicum` | P2 understanding/accessibility | `main` / `a79921b2becb81388762b10a365744d014026198` | detached `a79921b`; thread `019fe697-c99b-7273-b3a6-18e9e58c6a13` | Audit live interpretation/accessibility, implement minimal repairs/tests, and prepare an honest real-user study instrument | `in-progress` | `project/nightlight-public/**`; browser/server on dedicated port 5174 | Lane-owned technical/browser validation; final evidence pending | Disjoint from P1/P3; public-app changes require primary production review before main push | 2 | Reach ready-for-integration with no fabricated participant evidence and all live processes stopped |
| `C:\Users\raede\.codex\worktrees\3a29\Practicum` | P3 dashboard performance/architecture | `main` / `a79921b2becb81388762b10a365744d014026198` | detached `a79921b`; setup client `client-new-thread:df45b861-8ca2-46d2-8ca6-3a0f39b55b49` | Measure the MapLibre bundle and implement only a safe evidence-backed performance improvement or justified no-op | `in-progress` | `project/nightlight-dashboard/**`; build/server on dedicated port 5175 | Worktree exists; real thread and validation evidence pending | Disjoint from P1/P2; do not duplicate task while setup client is materializing | 1 | Resolve real thread ID, then run baseline-first performance lane |

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
