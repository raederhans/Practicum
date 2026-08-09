# Worktree Registry

| Worktree / path | Task | Base branch / commit | Current branch / HEAD | Goal | State | Hotspots | Tests | Overlap | Order | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `C:\Users\raede\Desktop\essay help master\Practicum` | Repository integration and public-release closeout | `main` / `1e3bcdade293b7e4c87ec4e00807cf3e86711bbc` | `main`; synchronized with `origin/main` after closeout | Consolidate the verified personal-project line, retire obsolete worktrees/branches, and publish the public observatory | `integrated` | Git topology, public provenance artifacts, Pages workflow/settings, task records, ignored local evidence | Dashboard 17/17 + build; public 112/112 + 11-file verifier; Python 96 + 7 subtests; focused Generalization/H4 8/8 zero-skip; Pages run 31310854793 and live artifact/browser checks passed | Only user evolution-research docs remain untracked; Pages is live at `https://raederhans.github.io/Practicum/` from exact published SHA `70dae02` | 1 | Continue from `main`; create a new isolated worktree only for a new bounded feature |

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
