# Practicum Next Evidence Phase Task

## Current status

`completed` — P3, P2, and P1 were reviewed, integrated, independently verified, pushed, and published. Lane records and this umbrella record are archived; the three lane worktree registrations and original Codex tasks are closed.

## Checklist

- [x] Load task-record, worktree-integration, and live-test ownership workflows.
- [x] Verify main/origin, worktree topology, archived evidence, and protected user WIP.
- [x] Unarchive the original P1/P2/P3 task IDs.
- [x] Recreate clean detached worktrees at the tasks' original exact paths.
- [x] Dispatch P1 provenance/readiness work.
- [x] Dispatch P2 research-backed proxy evaluation and independent next plan.
- [x] Dispatch P3 controlled real-timing measurements and architecture decision.
- [x] Supervise progress and resolve scope, evidence, or live-process conflicts.
- [x] Review delivery packages and integrate only eligible changes.
- [x] Validate combined main, synchronize remote, update records, and clean worktrees.

## Required handoff state

Each lane must finish as `ready-for-integration`, evidence-backed `blocked`, or justified no-op. A narrative report without current Git state, exact evidence, limitations, and cleanup status is not a complete handoff.

## Integrated delivery state

- P3 candidate `90dea17f112fb641d164d1e46f9a7d3d75f1b529` became `main@7db66b9`: reusable navigation-performance probe, 56 selected samples, a performance-runtime no-op decision, and corrected EAGLE-I documentation boundaries.
- P2 candidate `223631a02356eaa99866be17278f417a4174cbbc` became `main@232577e`: authoritative proxy-evidence report, solo-feasible next plan, and two browser-reproduced focus repairs.
- P1 candidate `5ede841441a4b616ac99249409f9a7d6f642e49e` became `main@a410ae2`: controlled upstream-blocker dispositions, corrected public EAGLE-I metadata, deterministic 52-file receipt, and fail-closed validation.
- Combined validation found one stale repository-level test that still required `partner-restricted`; `main@588c9aa` aligns that test with the corrected public-upstream/unproven-lineage contract.
- Release review then found two cross-boundary errors before push. `main@e35cd66` makes the 52-file tree receipt LF-canonical and stable across Windows/Linux checkout line endings; `main@c72e26e` distinguishes the Pages artifact, which does not bundle those derivatives, from the public repository, which currently tracks them.

## Release and cleanup evidence

- Product/release SHA: `e18b91a32cd92d379e4328889df1c0139f43ccee`, pushed to `origin/main` without force or history rewrite.
- GitHub Pages run: `31322447014`, build and deploy both succeeded for the exact release SHA.
- Pages artifact: `github-pages` artifact `9040560802`; its 1,804-byte release manifest exactly matched the live manifest by SHA-256 `f505d3989097d742fac6008a329eddcf1d566cf8d361d715edb824b10b32d259`.
- Live artifact: all 11 declared files matched their published byte lengths and SHA-256 values. Online Chromium reproduced Skip-link-first keyboard order, Atlas H1 route focus, 320px no-overflow behavior, zero console warnings/errors, and nine observed requests returning HTTP 200.
- Original P1, P2, and P3 Codex tasks were archived. Their Git worktree registrations were removed; Windows retained three empty, unregistered directory shells with zero items and no `.git` metadata because the Codex app still held directory handles.
- The unrelated `DOCS/archive/personal-project-evolution-research/` directory remained untracked, unstaged, and untouched.
- Remaining maintenance note: GitHub Actions reported a non-blocking Node 20 deprecation annotation because GitHub currently forces the pinned actions to run on Node 24.
