# Worktree Registry

| Worktree / path | Task | Base branch / commit | Current branch / HEAD | Goal | State | Hotspots | Tests | Overlap | Order | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `C:\Users\raede\Desktop\essay help master\Practicum` | P1-P3 orchestration owner | `main` / `2d8112104ab57ef7f6452ef99dc05ff9bbfb4427` | `main@1d2a0b151ad32a3858be82decf3f81b51cc6f4b8`; ahead of `origin/main` pending release | Integrated P3 performance guardrail, P2 route accessibility/study instrument, and P1 fail-closed reproducibility boundary | `ready-for-review` | Pages workflow release and scientific/public claim boundaries | Dashboard 20/20 + bundle analysis; public 119/119 + build/boundary/browser smoke; Python 106 + 7 subtests; expected full-upstream/full-run blockers verified | User research WIP remains untracked; P1/P2 directories removed; 3a29 and b19a are unregistered Windows-policy residue only | 0 | Push main, verify Pages exact SHA/live routes, then archive the orchestration record |

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
