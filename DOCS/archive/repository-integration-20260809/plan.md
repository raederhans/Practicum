# Repository Integration Plan

## Goal

Safely consolidate the Practicum repository's completed personal-project work into `main`, verify the exact integrated state, synchronize local and `origin`, and remove only worktrees or branches that have proven recovery paths and no unique deliverable.

## Scope

- Audit all Git worktrees, local branches, remote refs, divergence, ownership, dirty state, and file overlap.
- Integrate the completed `codex/personal-project-sync` line into `main` in dependency order.
- Treat already-absorbed and superseded feature branches as cleanup candidates only after exact ancestry/content checks.
- Run project-supported validation on the integrated commit, including the SHA-sensitive public and provenance gates.
- Push the verified integrated state to `origin/main`, reconcile local `main`, and clean eligible worktrees/branches.
- Preserve untracked research material and any other unowned WIP.

## Sources of truth

- `git worktree list --porcelain`, per-worktree `git status --short --branch`, commit ancestry, and diffs.
- `main`, `origin/main`, `codex/personal-project-sync`, and exact candidate SHA `992fe58cebee5efa99619a1881b3a4e2832facc1`.
- `DOCS/active/_worktree_registry.md` after the candidate is integrated.
- Existing Atlas, Generalization Autopsy, public-release, and personal-project task records.
- Fresh project validation output from the final integrated commit.

## Stages

- [x] Stage 1: establish authority, inspect workflow rules, refresh remotes, and capture initial worktree/branch truth.
- [x] Stage 2: classify overlap, WIP ownership, ancestry, supersession, and the exact integration order.
- [x] Stage 3: integrate the candidate into a clean integration worktree and reconcile task/registry truth.
- [x] Stage 4: run focused and full validation under a single live-test owner.
- [x] Stage 5: create the required provenance repair, push verified `main`, and synchronize the primary local checkout without overwriting WIP.
- [x] Stage 6: clean eligible worktrees/branches, re-audit Git state, and publish the next-stage roadmap.

## Acceptance criteria

- Every registered worktree and local branch is classified as integrated, superseded, retained with reason, or unsafe to merge.
- No untracked or unowned content is overwritten or deleted.
- The integrated candidate has fresh targeted and full validation evidence; scientific claims remain narrower than software-test claims.
- `main` and `origin/main` resolve to the same verified commit, or any local-checkout exception is explicit and caused only by preserved WIP.
- Removed worktrees/branches have a retained commit hash and proof that their deliverable is in `main` or superseded.
- The live worktree registry and final Git facts agree.

## Non-goals

- Do not merge the teammate repository wholesale or claim complete raw-data reproducibility.
- Do not push to the `teammate` remote, deploy to production, install analytics, or alter scientific methodology.
- Do not delete the untracked `DOCS/archive/personal-project-evolution-research/` material.

## Risks and constraints

- The apparent untracked dashboard/public-app overlap was proven to contain only ignored `node_modules/` and `dist/`; no source file was overwritten.
- The integrated Atlas signature remains historically bound to `992fe58`. The later `1a45d73` repair changed only the Generalization artifact's reviewed `study.js` hash; fresh public, Python, and zero-skip H4 gates passed, but no new independent scientific re-sign was claimed.
- `modeling-6events` had no merge base with local `main` but was an ancestor of `teammate/main`; its local branch was removed without merging separate history.
- GitHub Actions built, tested, verified, and uploaded the Pages artifact successfully, but deployment returned 404 because Pages is not enabled for this repository. Enabling production Pages remains an external authorization decision.
